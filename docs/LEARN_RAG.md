# Learn RAG by Reading This Codebase

This is the main learning document. It teaches **Retrieval-Augmented Generation
(RAG) as used in real systems**, by walking each concept and then showing the
*actual code in this repo* that implements it. Read it top to bottom with the
`gitcontext/` source open beside you.

> Two docs, two jobs: **this file** = concept → real code; `SYSTEM_OVERVIEW.md`
> = system map + per-file table.

**How to read:** each section is *Concept → Why it exists → The code here → See it run.*

---

## 0. The big picture

A plain LLM can't see your private/current data and has a limited context window.
RAG fixes both: **offline** you index a corpus; **online** you retrieve the few
relevant pieces and let the LLM answer *from them*, with citations.

```
INDEXING (offline)   repo → chunk → embed → store   (+ keyword index, + metadata)
QUERYING (online)    question → retrieve → rerank → LLM → answer + citations
```

Everything below is one of those boxes, in order.

---

## 1. Ingestion — get the raw material

**Concept.** Read every file worth indexing (code + docs), skip the rest.
**Why.** You can only retrieve what you ingest; junk in = junk out.

**The code** (`ingest.py`):
```python
def ingest(repo_path):
    for p in sorted(Path(repo_path).rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        ext = p.suffix.lower()
        source = "code" if ext in CODE_EXTS else "docs" if ext in DOC_EXTS else None
        if source and (text := p.read_text(encoding="utf-8")).strip():
            yield RawDoc(path=str(p.relative_to(repo)).replace("\\","/"), text=text, source=source)
```
`rglob("*")` walks everything; `SKIP_DIRS` drops `.git`, `node_modules`, etc.; each file becomes a `RawDoc(path, text, source)`. `yield` makes it a stream (low memory).

**Real-life note.** In production you'd also pull commit history, PR threads, and
incident reports — this repo does commits (section 6) and leaves PR *threads* as a
GitHub-API extension.

---

## 2. Chunking — split into retrievable units (the step that caps quality)

**Concept.** Break documents into small pieces. **The key idea**: cut on *meaningful
boundaries*, not arbitrary token counts.
**Why.** A whole 2,000-line file embeds to a blurry average; a half-function chunk
can't answer a question. One function = one clean idea = good retrieval.

**The code** (`chunk.py`): Python is parsed into an **AST** and split per function/class:
```python
tree = ast.parse(doc.text)
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = "\n".join(lines[node.lineno-1 : node.end_lineno])
        header = f"# file: {doc.path}\n# {kind}: {node.name}\n"      # context injection
        chunks.append(Chunk(header + body, {"file_path":doc.path, "symbol":node.name, ...}))
```
Two real-life techniques here: **structure-aware chunking** (AST boundaries) and
**context injection** (the `# file / # symbol` header makes an isolated chunk
self-describing). Markdown is split by heading; anything else falls back to
overlapping fixed-size windows.

**See it run.** `index` prints the chunk count; each chunk's metadata carries its symbol.

---

## 3. Embeddings — turn text into vectors of *meaning*

**Concept.** A model maps text → a fixed-length vector; similar meaning → nearby vectors.
**Why.** This is what lets "sign in" match `authenticate_user` — matching *meaning*, not words.

**The code** (`embed.py`): two backends behind one interface.
```python
class SentenceTransformerEmbedder:     # real semantic model (all-MiniLM-L6-v2)
    def encode(self, texts): return self.model.encode(texts, normalize_embeddings=True)

class HashingEmbedder:                 # offline fallback: hash words into a fixed vector
    ...  # L2-normalized so cosine == dot product
```
Vectors are **L2-normalized**, so cosine similarity becomes a plain dot product
(important for the fast search in section 4).

**Real-life note.** Swap MiniLM for a code-specialized model (e.g. `jina-code`) for
better code retrieval; the interface stays the same.

---

## 4. Vector store — find the nearest vectors fast

**Concept.** Store all vectors; given a query vector, return the closest chunks.
**Why.** This is the "semantic search" engine.

**The code** (`store.py`):
```python
def search(self, query_vec, k, allowed=None):
    scores = self.vectors @ query_vec          # one matrix multiply = all similarities
    if allowed is not None:                    # metadata filtering (section 11)
        masked = np.full(scores.shape, -np.inf); masked[list(allowed)] = scores[list(allowed)]; scores = masked
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]
```
Because vectors are unit-length, `self.vectors @ query_vec` *is* cosine similarity
for every chunk at once. `argsort` takes the top-k.

**Real-life note.** This numpy store is a teaching stand-in for **Pinecone / Qdrant /
Weaviate**, which add **ANN (HNSW)** indexes so search stays fast over millions of
vectors. The interface (add / search / filter) is the same idea.

---

## 5. Metadata — the quiet feature that enables citations, filtering, freshness

**Concept.** Store structured labels next to each chunk (`file_path`, `symbol`,
`commit`, `author`, `pr`, `last_modified`, `source`).
**Why.** Metadata powers three things: **citations**, **filtering**, and **staleness** detection.

**The code** (`git_meta.py`): ask git for each file's last commit.
```python
out = _git(["log","-1","--date=short", f"--format={_FMT}", "--", rel_path], repo_path)
commit, author, date, subject = out.split("\x1f")
pr = int(_PR_RE.search(subject).group(1))   # "#142" → 142
```
These fields get merged into every chunk's metadata in `pipeline._chunks_for_doc`.

---

## 6. Keyword search (BM25) + commit history

**Concept.** Dense vectors are weak at *exact* strings (error messages, identifiers).
BM25 is classic keyword relevance — great at exact tokens.
**Why.** You need both: meaning *and* exact matching.

**The code** (`bm25_index.py`):
```python
scores = self.bm25.get_scores(tokenize(query))   # BM25 score for every chunk
ranked = sorted(candidates, key=lambda i: -scores[i])[:k]
```
**Commit history as documents** (`git_history.py`): each commit becomes its own chunk
so you can ask "what changed in PR #142?"
```python
text = f"# commit {commit[:10]} by {author} on {date}\n{subject}\n\n{body}"
metadata = {"source": "pr" if pr else "commit", "commit": commit[:10], "pr": pr, ...}
```

**See it run.** `ask "cannot divide by zero"` → BM25 surfaces the exact-match chunk first.

---

## 7. Hybrid retrieval + RRF — merge meaning and keywords

**Concept.** Run dense + sparse in parallel and fuse the two ranked lists.
**Why.** Robust across conceptual *and* exact-string questions.

**The code** (`retrieve.py`): **Reciprocal Rank Fusion** merges by rank, not raw score.
```python
def _rrf(ranked_lists, k=RRF_K):
    scores = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)   # 1/(k+rank)
    return sorted(scores.items(), key=lambda x: -x[1])

def hybrid_search(query, embedder, store, bm25, flt=None):
    allowed = store.allowed_indices(flt)
    dense  = store.search(embedder.encode([query])[0], TOP_K_DENSE, allowed)
    sparse = bm25.search(query, TOP_K_SPARSE, allowed)
    return [(i, s, store.chunks[i]) for i, s in rrf_fuse(dense, sparse)]
```
RRF uses **ranks**, so it fuses incomparable scores (cosine vs BM25) with no calibration.

---

## 8. Multi-query / RAG-fusion — don't bet on one phrasing

**Concept.** Ask the LLM to rewrite the question into several searches; retrieve each; RRF them.
**Why.** A user's wording may not match the code; multiple rephrasings widen recall.

**The code** (`query_expand.py` + `retrieve.multi_query_search`):
```python
queries = expand_query(question, n=3)          # LLM writes 3 alternatives (+ original)
per_query = [[(i,s) for i,s,_ in hybrid_search(q, ...)] for q in queries]
fused = _rrf(per_query)                         # same RRF, now across queries
```
Notice the **reuse**: the exact same `_rrf` fuses retrievers (section 7) *and* query
rewrites. Off by default (`--multi` to enable) because it costs an extra LLM call.

---

## 9. Reranking — a precise second pass

**Concept.** Retrieval is fast but coarse. A **cross-encoder** re-reads each
(query, chunk) pair *together* and re-sorts; keep the best few.
**Why.** Biggest single quality lever in most RAG systems.

**The code** (`rerank.py`):
```python
class CrossEncoderReranker:
    def rerank(self, query, candidates, top_n):
        scores = self.model.predict([[query, c[2]["text"]] for c in candidates])
        return [c for c,_ in sorted(zip(candidates, scores), key=lambda x:-x[1])[:top_n]]
```
**Bi-encoder vs cross-encoder:** retrieval uses a bi-encoder (query & chunk embedded
*separately* — fast); the reranker is a cross-encoder (query+chunk *together* —
accurate but slow, so only run on the ~top candidates). Pattern: **retrieve wide, rerank narrow.**

---

## 10. Generation — grounded, cited answers (fully local)

**Concept.** Put the top chunks in the prompt and tell the LLM to answer *only* from them, with citations.
**Why.** Grounding is what suppresses hallucination; metadata makes citations real.

**The code** (`generate.py`):
```python
SYSTEM = ("You are a codebase assistant. Answer ONLY from the context below. "
          "Cite sources as [file_path]. If the answer isn't in the context, say so.")
def complete(system, user, max_new_tokens=256):
    # local-first: Ollama → local transformers (Qwen2.5-0.5B) → None. No API key.
```
The LLM is a **reasoner over retrieved facts**, not a memory bank. Backends are
local, so no paid account is needed; if none is available, an extractive fallback
returns the top chunk.

---

## 11. Metadata filtering — scope the search

**Concept.** Restrict retrieval to chunks matching a filter (`source=docs`, `path_prefix=auth/`).
**Why.** "Recent *auth* changes" shouldn't search the calculator or the docs.

**The code** (`filters.py` + `store.allowed_indices`):
```python
def matches(meta, flt):      # pure, unit-tested predicate
    for key, want in (flt or {}).items():
        if key == "path_prefix" and not (meta.get("file_path") or "").startswith(want): return False
        elif isinstance(want, (list,tuple,set)) and meta.get(key) not in want: return False
        elif not isinstance(want,(list,tuple,set)) and meta.get(key) != want: return False
    return True
```
The filter is computed once into an allowed-index set and applied to **both** the
vector and BM25 halves (so fusion can't resurface filtered-out chunks).

---

## 12. Staleness — incremental re-indexing

**Concept.** Code changes; re-embed only what changed instead of rebuilding everything.
**Why.** A stale index gives confidently-wrong answers; full rebuilds are slow/expensive.

**The code** (`pipeline.reindex`): compare content hashes stored in `state.json`.
```python
changed = [p for p,h in current_hashes.items() if old.get(p) != h]   # new or modified
deleted = [p for p in old if p not in current_hashes]
for p in set(changed) | set(deleted): store.remove_by_file(p)        # drop stale chunks
store.add(embedder.encode([c["text"] for c in new_chunks]), new_chunks)  # re-embed changed
```
A GitHub **push webhook** would just call `reindex()` (exposed as `POST /reindex`).

**See it run.** Edit a file, `reindex` → it reports exactly which file's chunks were replaced.

---

## 13. Evaluation — how you know it works

**Concept.** Measure retrieval *and* answer quality; iterate against numbers, not vibes.
**Why.** You can't improve what you don't measure; Recall@K is the north star.

**Retrieval metrics** (`eval.py`): Recall@K (is a gold file in the top-K?), Precision@K
(how much of the top-K is relevant?), MRR (is the right chunk ranked high?), latency.

**Answer quality — LLM-as-judge** (`judge.py`): with no gold answers, ask an LLM to
score 1–5 on **correctness** and **faithfulness**.
```python
_SYS = 'rate ... Respond with ONLY compact JSON: {"correctness": N, "faithfulness": N}.'
```
`_parse_scores` robustly extracts the numbers (tested in `tests/test_judge.py`).

**See it run.** `python -m gitcontext.eval eval/questions.json 3` and `--answers`.

---

## 14. Interfaces — CLI, API, web UI

- **CLI** (`cli.py`): `index`, `reindex`, `ask` (`--multi`, `--source`, `--path`).
- **API** (`api.py`): FastAPI `GET /` (UI), `POST /index|reindex|query` (+ Swagger at `/docs`).
- **Web UI** (`webui.py`): one self-contained HTML page — search box, source filter, cited answers.

---

## 15. How one question flows end to end

Question: **"what happens when you divide by zero?"**
1. (optional) expand into rewrites — section 8.
2. embed query; vector search (meaning) + BM25 (the exact phrase) — sections 3,4,6.
3. RRF-fuse the rankings — section 7.
4. cross-encoder rerank → top-k — section 9.
5. build grounded prompt from those chunks — section 10.
6. local LLM answers; chunk metadata becomes citations — sections 5,10.
→ `calculator.py:divide` is retrieved; the answer explains `ZeroDivisionError`, cited `[calculator.py]`.

---

## 16. From this project to a production system

| Here (learning) | Production swap |
| --- | --- |
| numpy cosine store | Pinecone / Qdrant / Weaviate (ANN/HNSW) |
| MiniLM embeddings | code-specialized embeddings, batched on GPU |
| local Qwen 0.5B | a larger local model (Ollama) or a hosted LLM |
| commit chunks only | GitHub API for PR/issue discussion threads |
| content-hash reindex | push-webhook-triggered incremental indexing |
| 5-question eval set | 50–100 questions + CI regression gate |

The *architecture* is the same; only the components scale up. That's the point of RAG:
swappable stages behind stable interfaces.
