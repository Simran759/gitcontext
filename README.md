# GitContext

A minimal, **end-to-end RAG developer copilot** over a code repository. It indexes
source code and docs, then answers questions about the repo with **hybrid retrieval
+ reranking** and **citations** back to real files — and it runs **100% locally, with
no API key** (local embeddings, local reranker, local LLM).

This is a learning scaffold: every pipeline stage is its own small, commented module,
so you can read it top to bottom and learn how each piece works. It works with zero
setup (built-in fallbacks) and upgrades to real models when you install one package.

---

## Quick start

```powershell
cd gitcontext
python -m venv .venv ; .venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
pip install sentence-transformers                      # real models (optional but recommended)

python -m gitcontext.cli index sample_repo             # OFFLINE: full build
python -m gitcontext.cli ask "How does a user log in?" # ONLINE: ask a question
python -m gitcontext.cli ask --multi "how to sign in"  # multi-query (query translation)
python -m gitcontext.cli reindex sample_repo           # incremental refresh (only changed files)
python -m gitcontext.eval eval/questions.json 3        # Recall@K, Precision@K, MRR, latency

uvicorn gitcontext.api:app --reload                    # web chat UI at http://127.0.0.1:8000/
```

Point it at any Python repo: `python -m gitcontext.cli index <path/to/repo>`.

### Example output

```
$ python -m gitcontext.cli index sample_repo
[embed] backend: sentence-transformers (semantic)
[index] 9 chunks from sample_repo
[index] saved to .gitcontext_index

$ python -m gitcontext.cli ask "How does a user log in?"
=== CITATIONS ===
 - auth.py (AuthService)
 - README.md (Authentication)
 - auth.py (issue_token)
 ...
=== ANSWER ===
[local:Qwen/Qwen2.5-0.5B-Instruct] According to the provided code, a user logs in
using the `authenticate_user` method within the `AuthService` class. It looks up the
user, hashes the supplied password with SHA-256, compares it to the stored hash, and
returns a session token on success, otherwise raises a ValueError. [auth.py]

$ python -m gitcontext.eval eval/questions.json 3
[HIT ] How does a user log in?
[HIT ] How are passwords hashed?
[HIT ] What happens when you divide by zero?
[HIT ] How is the running total updated?
[HIT ] cannot divide by zero
Recall@3 = 5/5 = 100.00%
```

The answer is grounded only in the retrieved chunks and tagged with the backend used.

---

## How it works

Two phases. **Indexing** happens once (offline); **querying** happens per question (online).

```
INDEXING (offline, re-run when code changes)
  repo ──► ingest ──► chunk (AST / headings) ──► embed ──► Vector store
                                                   └──────► BM25 index
                                        (+ metadata: file_path, symbol, lines, source)

QUERYING (online, per question)
  question ──► embed ──► vector search ─┐
                       ► BM25 search ───┴─► RRF fuse ──► rerank ──► top-k
                                                              └─► grounded prompt ──► local LLM ──► answer + citations
```

### File-by-file

**Indexing**

| File | Stage | What it does |
| --- | --- | --- |
| `ingest.py` | 1. Ingestion | Walks the repo, yields code + doc files, skips binaries/vendored dirs. |
| `chunk.py` | 2. Chunking | **Structure-aware**: splits Python by function/class via the `ast` module, Markdown by heading; fixed-size fallback otherwise. Injects a `# file / # symbol` header and attaches metadata. |
| `embed.py` | 3. Embeddings | Text → vectors. `sentence-transformers` (semantic) if installed, else a pure-numpy hashing fallback so it always runs. |
| `git_meta.py` | 3b. Metadata enrich | Shells out to `git log` for each file's last commit: `commit`, `author`, `last_modified`, and `pr` (parsed from the commit message). Merged into chunk metadata. |
| `git_history.py` | 1b. History ingest | Ingests each **commit as its own document** (`source="commit"`, or `"pr"` when it references a PR number), so questions can be answered from history, not just current code. |
| `store.py` | 4. Vector store | Persists the vector matrix + chunk metadata; searches by cosine similarity (unit vectors → one matrix multiply); `remove_by_file` supports incremental re-indexing. |
| `bm25_index.py` | 6a. Keyword index | BM25 sparse index for exact tokens (error strings, identifiers). |

**Querying**

| File | Stage | What it does |
| --- | --- | --- |
| `query_expand.py` | 6a'. Query translation | Uses the local model to rewrite the question into alternative search queries (multi-query / RAG-fusion). |
| `retrieve.py` | 6b. Hybrid retrieval | Runs vector + BM25 search and fuses with **RRF**; `multi_query_search` fuses across several query rewrites. |
| `rerank.py` | 7. Reranking | Optional **cross-encoder** re-sorts the fused candidates by true relevance; no-op fallback. |
| `generate.py` | 8. Generation | Builds a grounded prompt and calls a **local assistant** (Ollama → local transformers → extractive). No API key. |

**Glue / interfaces / evaluation**

| File | What it does |
| --- | --- |
| `filters.py` | Metadata filter predicate `matches(meta, flt)` — pure and unit-tested; used by the store to scope searches. |
| `config.py` | All tunable knobs (top-K values, RRF constant, chunk size), env-overridable. |
| `pipeline.py` | `build_index()` (full offline build), `reindex()` (incremental), `GitContext.query()` (online). |
| `api.py` | FastAPI backend: `GET /` (web UI), `POST /index`, `POST /reindex`, `POST /query`. |
| `webui.py` | Self-contained HTML chat page served at `/` (vanilla JS, no CDN). |
| `cli.py` | `index`, `reindex`, and `ask` commands. |
| `eval.py` | Retrieval report (**Recall@K, Precision@K, MRR, latency**) and `--answers` mode (**LLM-as-judge** correctness + faithfulness). |
| `judge.py` | LLM-as-judge: scores an answer 1-5 on correctness and faithfulness using the local model (pure, testable parser). |
| `sample_repo/`, `eval/questions.json` | Demo repo + labelled eval set. |

### The index on disk (`.gitcontext_index/`)

`vectors.npy` + `chunks.json` (vector store), `bm25.json` (keyword index), and
`state.json` — a per-file content hash + the repo's commit, which is how
`reindex` detects exactly which files changed since the last build.

### Incremental re-indexing (staleness)

`reindex` hashes every file, compares to `state.json`, and for each **changed/new**
file it deletes the file's old chunks (`store.remove_by_file`), re-chunks, re-embeds,
and re-adds them; **deleted** files are just removed. BM25 is rebuilt over the current
corpus. A GitHub push webhook would simply call `POST /reindex`. Example:

```
$ python -m gitcontext.cli reindex sample_repo
[reindex] changed/new=1  deleted=0
  - removed 3 old chunk(s) from calculator.py
  + added 4 fresh chunk(s)
[reindex] done; 10 chunks now indexed
```

### Multi-query retrieval (query translation / RAG-fusion)

A user's wording may not match the code. With multi-query on, the local model
rewrites the question into N alternatives; each is retrieved with hybrid search,
and the rankings are fused with RRF — so a chunk relevant to several rephrasings
wins. Enable per-call (`ask --multi "..."`) or globally (`set GITCONTEXT_MULTI_QUERY=1`).

```
$ python -m gitcontext.cli ask --multi "how can a person sign in"
=== EXPANDED QUERIES ===
 ? how can a person sign in
 ? how to sign in online
 ? sign me in quickly
 ? how do I sign in?
... (auth.py:AuthService ranks #1)
```

### Commit / PR history ingestion

With a git repo, each commit is indexed as its own document (scoped to the
indexed directory). Ask *"what changed in PR #142?"* and the commit chunk is
retrieved alongside code. Tagged `source="pr"` when the message references a PR,
else `source="commit"`. True PR discussion threads need the GitHub API (an
extension). Controlled by `GITCONTEXT_INGEST_COMMITS` / `GITCONTEXT_COMMIT_LIMIT`.

### Metadata filtering

Scope a query to a subset of the index by passing a filter dict (`filters.py`):

- `{"source": "code"}` or `{"source": ["commit", "pr"]}` — by document type
- `{"type": "function"}` — only function chunks
- `{"path_prefix": "auth/"}` / `{"path_contains": "service"}` — by location
- `{"author": "simran"}` — by git author

The filter restricts **both** the vector and BM25 halves (the store computes the
allowed row set once and masks the rest). CLI: `ask --source code` / `ask --path auth/`;
API: `{"question": "...", "filter": {"source": "docs"}}`; web UI: the source dropdown.

### Web UI

`uvicorn gitcontext.api:app --reload` then open `http://127.0.0.1:8000/`. A search box
POSTs to `/query` and renders the grounded answer plus clickable citations (which now
show `commit · author · PR · date` when the repo is under git). `/docs` still gives the
Swagger developer UI.

---

## What is unique in this RAG (the differentiators)

Most "chat with your docs" demos stop at *split-by-500-tokens → embed → cosine search → LLM*.
GitContext deliberately does five things beyond that:

1. **Structure-aware code chunking.** Chunks are whole functions/classes parsed from the
   Python **AST**, not arbitrary token windows — so retrieval returns complete, runnable
   units of logic instead of half a function.
2. **Context injection into every chunk.** Each chunk is prefixed with `# file: ... / # symbol: ...`
   so an isolated function is self-describing. This improves both embedding quality and the
   LLM's understanding, and powers precise citations.
3. **True hybrid retrieval with RRF.** Dense (meaning) **and** BM25 (exact tokens) run in
   parallel and are fused by rank. Try `ask "cannot divide by zero"` — the exact error string
   is found by BM25 even when semantics would blur it.
4. **Cross-encoder reranking.** A second, more accurate pass re-sorts candidates before the
   LLM — usually the single biggest quality lever in a RAG system.
5. **Fully local, no API key.** Local embeddings, local reranker, and a local LLM
   (Ollama or a small transformers model). The design also leaves clean seams for the
   "production" extensions below (Git metadata, incremental re-indexing).

---

## What we used and why

| Concern | Choice | Why this one |
| --- | --- | --- |
| Code parsing | Python `ast` | Standard library, zero deps, exact function/class boundaries. (`tree-sitter` is the multi-language upgrade.) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), hashing fallback | MiniLM is tiny, fast on CPU, strong semantic quality. The numpy fallback guarantees the project runs with no download. |
| Vector store | numpy matrix + cosine | Transparent and dependency-light so you can *see* the mechanism. Swap for Pinecone/Qdrant/Weaviate (HNSW/ANN) at scale. |
| Keyword search | `rank_bm25` (BM25Okapi) | Pure-Python, no server; covers the exact-match half of hybrid retrieval. |
| Fusion | Reciprocal Rank Fusion | Combines two rankings by **rank**, so it needs no score calibration between vector and BM25. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small, accurate cross-encoder; reads (query, chunk) jointly for true relevance. |
| Generation | Local-first: Ollama → `Qwen/Qwen2.5-0.5B-Instruct` → extractive | No paid account needed; instruct-tuned, runs on CPU; graceful fallback so nothing crashes. |
| Backend | FastAPI | Async, typed, auto-docs at `/docs`; a `/query` endpoint in a few lines. |
| Eval | Hand-rolled Recall@K | Smallest thing that makes retrieval quality measurable and repeatable. |

---

## Answer generation (local, no API key)

`generate.py` is **local-first** and needs no paid account. Backends tried in order:

1. **Ollama** — if a server runs at `localhost:11434`. Start it (`ollama serve`), pull a model
   (`ollama pull llama3.2`), and it's used automatically. Override with `GITCONTEXT_OLLAMA_MODEL`.
2. **Local transformers model** — otherwise a small instruct model runs on CPU, downloaded once
   and cached. Default `Qwen/Qwen2.5-0.5B-Instruct`; override with `GITCONTEXT_LOCAL_MODEL`
   (e.g. `Qwen/Qwen2.5-1.5B-Instruct` for better answers).
3. **Extractive fallback** — returns the top retrieved chunk if no model is available.

Opt-in Anthropic (not required): `set GITCONTEXT_LLM=anthropic` + `set ANTHROPIC_API_KEY=...`.

---

## How to improve the evaluation harness

`eval.py` measures **Recall@K** on a 5-question set — enough to show the idea, not enough to
drive real decisions. To make it a proper harness:

1. **Grow and diversify the eval set.** 50–100 questions covering conceptual, exact-string,
   multi-file, and "answer-not-in-repo" cases. A 5-question set can't separate two backends
   (here hashing and semantic both score 80%@1 / 100%@3).
2. **Add more retrieval metrics.** Precision@K (noise), **MRR / nDCG** (reward rank position,
   not just presence), and chunk-level recall (did the *right function* come back, not just
   the right file).
3. **Measure answer quality, not just retrieval.** Add an **LLM-as-judge** (or string/semantic
   match to a reference answer) for correctness, plus **citation accuracy** (do the cited files
   actually support the claim?) and **faithfulness** (is every sentence grounded?).
4. **Track latency and cost.** Record retrieval ms, rerank ms, and generation ms per query so
   quality gains don't silently make it unusable.
5. **Ablation mode.** Run the same set with each component toggled off (no BM25, no rerank,
   fixed-size chunking) to *prove* each piece earns its place.
6. **Regression gate in CI.** Fail the build if Recall@5 drops below a threshold, so a chunking
   or model change can't quietly regress quality.
7. **Mine production questions.** Log real queries, label the hard ones, and feed them back into
   the eval set — the feedback loop that actually improves a RAG system over time.

---

## Tests

A fast, offline `pytest` suite (no model downloads — `conftest.py` forces the hashing
embedder, no-op reranker, and no LLM) covers chunking, filters, the store, BM25,
retrieval/RRF/multi-query, git metadata + commit ingestion, and the end-to-end
pipeline incl. incremental re-indexing and filtered queries.

```bash
pip install pytest
python -m pytest            # 23 passed in ~2.5s
```

## Done vs. still to extend

**Now implemented:** structure-aware chunking · semantic embeddings + hashing fallback ·
hybrid retrieval (vector + BM25 + RRF) · **multi-query / RAG-fusion** · cross-encoder reranking ·
local no-key generation · Git metadata (commit/author/PR/date) · **commit/PR history ingestion** ·
incremental re-indexing · **metadata filtering** · web UI · Recall@K / Precision@K / MRR / latency ·
**pytest suite**.

**Still to extend (production pieces):**

- Ingest **PR/issue discussion threads** via the GitHub API (commit history is done; live discussions need a token).
- Swap `store.py` for a real vector DB (Qdrant/Pinecone) for millions of vectors.
- A stronger/external **judge model** (the local 0.5B judge is lenient).

See `docs/LEARN_RAG.md` for the full concept-to-code walkthrough.
