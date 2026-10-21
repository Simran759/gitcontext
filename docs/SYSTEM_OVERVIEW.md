# GitContext — System Overview

A single reference for *what the system does*, *the concepts behind it*, and
*what every file in the codebase does*. Pair this with `LEARN_RAG.md`
(concept → code) and `../README.md` (the quick start).

---

## 1. What the system does

GitContext is a **RAG (Retrieval-Augmented Generation) developer copilot** over a
code repository. You point it at a repo; it answers natural-language questions
about that repo and cites the exact files/commits its answer came from. It runs
**fully locally — no API key** (local embeddings, local reranker, local LLM).

It works in two phases:

```
INDEXING (offline, once; refreshed incrementally)
  repo ─► ingest files ─► chunk (AST/headings) ─► embed ─► vector store
        └► read git history ─► commit/PR chunks ─┘    └──► BM25 keyword index
                                     (+ metadata: file, symbol, commit, author, PR, date)

QUERYING (online, per question)
  question
    └►(optional) rewrite into N queries        [multi-query]
         └► vector search + BM25 search         [hybrid, optional metadata filter]
              └► RRF fuse ─► cross-encoder rerank ─► top-k
                   └► grounded prompt ─► local LLM ─► answer + citations

EVALUATION
  retrieval: Recall@K / Precision@K / MRR / latency
  answers  : LLM-as-judge correctness + faithfulness
```

Interfaces: a **CLI**, a **FastAPI** service, and a **web chat UI**.

---

## 2. Concepts and terms

| Term | What it means in this system |
| --- | --- |
| **RAG** | Retrieve relevant chunks at query time, then have an LLM answer *from them* — so it uses private/current data and can cite sources. |
| **Indexing vs querying** | Indexing (offline) builds the searchable index; querying (online) uses it. Keep them mentally separate. |
| **Chunk** | A small retrievable unit of text. Here: one function/class (code) or one heading section (docs) or one commit. |
| **Structure-aware chunking** | Splitting code on real boundaries using the **AST** (abstract syntax tree), not fixed token windows, so a chunk is a whole unit of logic. |
| **Context injection** | Prepending `# file: … / # symbol: …` to a chunk so it is self-describing in isolation. |
| **Embedding** | A vector of floats representing text *meaning*; similar meaning → nearby vectors. |
| **Cosine similarity** | Closeness of two vectors (1 = same direction). With unit vectors it's just a dot product. |
| **Vector store** | Holds embeddings + metadata and finds the nearest ones to a query vector. |
| **ANN / HNSW** | Approximate Nearest Neighbour search (a graph index) used by real vector DBs for speed at scale. |
| **BM25** | Classic keyword relevance score; rewards exact, rare term matches. The "sparse" retriever. |
| **Dense vs sparse** | Dense = vector/meaning search; sparse = BM25/keyword search. |
| **Hybrid retrieval** | Run dense + sparse together and merge — robust to both conceptual and exact-string questions. |
| **RRF (Reciprocal Rank Fusion)** | Merge ranked lists by `1/(k+rank)`. Uses rank not raw score, so it fuses incomparable scores — and also fuses multiple query rewrites. |
| **Bi-encoder** | The embedding model: encodes query and chunk *separately* (fast, used for retrieval). |
| **Cross-encoder** | The reranker: reads query+chunk *together* (accurate, slow, used on the top candidates). |
| **Reranking** | A second, precise pass that re-sorts retrieved candidates before the LLM. Usually the biggest quality lever. |
| **Multi-query / RAG-fusion** | Ask the LLM to rewrite the question into several searches, retrieve each, and RRF-fuse — lifts recall for vague wording. |
| **Metadata filtering** | Restrict retrieval to chunks whose metadata matches (e.g. `source=docs`, `path_prefix=auth/`). |
| **Grounding** | Forcing the LLM to answer only from the retrieved context, and to say when it doesn't know — this suppresses hallucination. |
| **Citation** | Pointing each answer at its source (`file_path:symbol`, commit, PR) using chunk metadata. |
| **Incremental re-indexing** | On change, re-embed only the files whose content hash changed; delete stale chunks. Keeps a live repo fresh cheaply. |
| **Staleness** | When the index no longer matches the code; solved by incremental re-indexing (a push webhook would trigger it). |
| **Recall@K** | Of the files that should be retrieved, fraction present in the top-K. The headline retrieval metric. |
| **Precision@K** | Of the K retrieved chunks, fraction that are actually relevant (noise control). |
| **MRR** | Mean Reciprocal Rank — rewards putting the right chunk *high* (rank 1 → 1.0, rank 2 → 0.5). |
| **LLM-as-judge** | Using an LLM to score answer quality (correctness, faithfulness) when you have no gold answers. |
| **Faithfulness** | Whether every claim in the answer is supported by the retrieved context. |

---

## 3. What each file does

### The pipeline package (`gitcontext/`)

| File | Role | What it does |
| --- | --- | --- |
| `__init__.py` | setup | Package marker; silences cosmetic ML logs by setting env vars/log levels *before* heavy libs import. |
| `config.py` | config | All tunable knobs (top-K, RRF constant, chunk size, multi-query, commit ingestion), each env-overridable. |
| `quiet.py` | setup | `quiet_ml()` disables transformers/HuggingFace progress bars + load reports (called right before a model loads). |
| `ingest.py` | 1. ingest | Walks the repo, yields each indexable file as a `RawDoc(path, text, source)`; skips binaries/vendored dirs. |
| `chunk.py` | 2. chunk | Structure-aware chunking: Python by function/class via `ast`, markdown by heading, fixed-size fallback; injects file/symbol header; attaches metadata. |
| `embed.py` | 3. embed | Text→vectors. `SentenceTransformerEmbedder` (semantic) or offline `HashingEmbedder` fallback; `get_embedder()` chooses (env `GITCONTEXT_EMBED=hashing` forces fallback). |
| `git_meta.py` | 3b. metadata | Shells out to `git log` for a file's last commit → `commit`, `author`, `last_modified`, `pr` (parsed from message). |
| `git_history.py` | 1b. history | `commit_chunks()` turns each commit into its own document (`source="pr"` if it references a PR, else `"commit"`). |
| `store.py` | 4. vector store | `VectorStore`: add/search by cosine, persist to disk, `remove_by_file`/`remove_by_source` (for reindex), `allowed_indices`/masked search (for filtering). |
| `bm25_index.py` | 6a. keyword | `BM25Index`: tokenize + BM25Okapi scoring; `search(..., allowed)` honours metadata filters. |
| `filters.py` | filtering | Pure `matches(meta, flt)` predicate (exact / list / path_prefix / path_contains / AND across keys). |
| `query_expand.py` | 6a'. expand | `expand_query()` uses the local LLM to rewrite a question into alternative search queries (multi-query). |
| `retrieve.py` | 6b. retrieve | `hybrid_search` (vector+BM25 fused by RRF, optional filter), `multi_query_search` (RRF across query rewrites), generalized `_rrf`. |
| `rerank.py` | 7. rerank | `CrossEncoderReranker` (accurate) or `NoopReranker` fallback; `get_reranker()` chooses (env `GITCONTEXT_RERANK=noop`). |
| `generate.py` | 8. generate | Builds the grounded prompt; `complete(system,user)` calls the local LLM (Ollama→transformers→none), shared by answering and expansion; `generate()` returns answer+citations or extractive fallback. |
| `judge.py` | eval | `judge_answer()` LLM-scores an answer 1-5 on correctness + faithfulness; `_parse_scores()` is a pure, tested parser. |
| `pipeline.py` | glue | `build_index()` (full offline build + state/hashes + commit ingest), `reindex()` (incremental), `GitContext.query()` (online: expand→retrieve→rerank→generate, with optional filter). |
| `api.py` | interface | FastAPI app: `GET /` (web UI), `POST /index`, `POST /reindex`, `POST /query` (accepts `multi_query` + `filter`). |
| `webui.py` | interface | Self-contained HTML/CSS/JS chat page served at `/` (search box, source dropdown, renders answer + citations). |
| `cli.py` | interface | Commands: `index`, `reindex`, `ask` (with `--multi`, `--source`, `--path`). |
| `eval.py` | eval | `evaluate()` retrieval report (Recall@K/Precision@K/MRR/latency); `evaluate_answers()` answer-quality via LLM-judge (`--answers`). |

### Supporting files

| Path | What it is |
| --- | --- |
| `sample_repo/` | Tiny demo repo to index (`auth.py`, `calculator.py`, `README.md`). |
| `eval/questions.json` | Labelled eval questions, each with the `gold_files` it should retrieve. |
| `tests/` | `pytest` suite: `test_chunk`, `test_filters`, `test_store`, `test_bm25`, `test_retrieve`, `test_git`, `test_pipeline`, `test_judge`. |
| `conftest.py` | Forces offline/deterministic backends for tests (hashing embedder, no-op reranker, no LLM). |
| `pytest.ini` | Points pytest at `tests/`. |
| `requirements.txt` | Core deps (numpy, rank-bm25, fastapi, uvicorn, pydantic) + optional (sentence-transformers, anthropic, pytest). |
| `.gitcontext_index/` | Generated index: `vectors.npy`, `chunks.json`, `bm25.json`, `state.json` (git-ignored). |
| `docs/` | `LEARN_RAG.md` (concept → code), `SYSTEM_OVERVIEW.md` (this file). |

---

## 4. Commands

```bash
python -m gitcontext.cli index sample_repo                 # full build
python -m gitcontext.cli reindex sample_repo               # incremental refresh
python -m gitcontext.cli ask "How does a user log in?"     # ask
python -m gitcontext.cli ask --multi "how to sign in"      # multi-query
python -m gitcontext.cli ask --source docs "login flow"    # metadata filter
python -m gitcontext.eval eval/questions.json 3            # retrieval metrics
python -m gitcontext.eval --answers eval/questions.json    # answer-quality (LLM-judge)
uvicorn gitcontext.api:app --reload                        # web UI at :8000/
python -m pytest                                           # 28 tests, offline
```

---

## 5. One query, end to end

Question: **"what happens when you divide by zero?"**

1. (optional) expand into rewrites; else use as-is.
2. Embed the query; run **vector search** (meaning) and **BM25** (the exact phrase `cannot divide by zero`).
3. **RRF-fuse** the two rankings (and across rewrites if multi-query).
4. **Cross-encoder rerank** the fused candidates; keep top-k.
5. Build a **grounded prompt** from those chunks (each tagged with its `file_path:symbol`).
6. **Local LLM** answers using only that context; metadata becomes **citations**.
7. Result: the `calculator.py:divide` chunk is retrieved and the answer explains the `ZeroDivisionError`, cited to `[calculator.py]`.
