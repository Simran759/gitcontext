"""Glue: the two phases wired together, plus incremental re-indexing.

    build_index(repo_path)   -> OFFLINE full build: ingest, chunk, embed, store, bm25
    reindex(repo_path)       -> OFFLINE incremental: only re-embed changed/deleted files
    GitContext(...).query(q) -> ONLINE : retrieve, rerank, generate

The index directory holds four things:
    vectors.npy / chunks.json   (the vector store)
    bm25.json                   (the keyword index)
    state.json                  (per-file content hash + repo commit, for staleness)
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .config import (INDEX_DIR, TOP_K_FINAL, MULTI_QUERY, MULTI_QUERY_N,
                     INGEST_COMMITS, COMMIT_LIMIT)
from .ingest import ingest
from .chunk import chunk_doc
from .embed import get_embedder
from .store import VectorStore
from .bm25_index import BM25Index
from .retrieve import hybrid_search, multi_query_search
from .rerank import get_reranker
from .generate import generate
from .git_meta import file_git_meta, repo_head
from .git_history import commit_chunks


# --------------------------------------------------------------------------
# Index state (used to detect what changed since the last index)
# --------------------------------------------------------------------------
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _state_path(index_dir) -> Path:
    return Path(index_dir) / "state.json"


def _load_state(index_dir) -> dict:
    p = _state_path(index_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"files": {}, "commit": None}


def _save_state(index_dir, files: dict, commit) -> None:
    _state_path(index_dir).write_text(
        json.dumps({"files": files, "commit": commit}, indent=2), encoding="utf-8")


def _chunks_for_doc(doc, repo_path) -> list[dict]:
    """Chunk one file and merge in its git metadata (commit/author/pr/date)."""
    gmeta = file_git_meta(repo_path, doc.path)        # {} if not a git repo
    out = []
    for ch in chunk_doc(doc):
        out.append({"text": ch.text, "metadata": {**ch.metadata, **gmeta}})
    return out


# --------------------------------------------------------------------------
# OFFLINE: full build
# --------------------------------------------------------------------------
def build_index(repo_path, index_dir=INDEX_DIR) -> int:
    embedder, name = get_embedder()
    print(f"[embed] backend: {name}")

    chunks, hashes = [], {}
    for doc in ingest(repo_path):
        hashes[doc.path] = _sha1(doc.text)            # remember each file's content
        chunks.extend(_chunks_for_doc(doc, repo_path))
    print(f"[index] {len(chunks)} code/doc chunks from {repo_path}")

    if INGEST_COMMITS:                                # commit/PR history as documents
        cc = commit_chunks(repo_path, COMMIT_LIMIT)
        if cc:
            chunks.extend(cc)
            print(f"[index] + {len(cc)} commit/PR chunks")

    vectors = embedder.encode([c["text"] for c in chunks])
    store = VectorStore()
    store.add(vectors, chunks)
    store.save(index_dir)

    BM25Index.build([c["text"] for c in chunks]).save(index_dir)
    _save_state(index_dir, hashes, repo_head(repo_path))
    print(f"[index] saved to {index_dir}")
    return len(chunks)


# --------------------------------------------------------------------------
# OFFLINE: incremental re-index (staleness handling)
# --------------------------------------------------------------------------
def reindex(repo_path, index_dir=INDEX_DIR) -> int:
    """Re-embed only files whose content changed; drop deleted files.

    Compares each file's current content hash to the hash saved at last index.
    Returns the number of new chunks added.
    """
    if not _state_path(index_dir).exists():
        print("[reindex] no existing index; building from scratch")
        return build_index(repo_path, index_dir)

    old = _load_state(index_dir)["files"]
    current_docs, current_hashes = {}, {}
    for doc in ingest(repo_path):
        current_docs[doc.path] = doc
        current_hashes[doc.path] = _sha1(doc.text)

    changed = [p for p, h in current_hashes.items() if old.get(p) != h]   # new or modified
    deleted = [p for p in old if p not in current_hashes]                 # removed files

    if not changed and not deleted:
        print("[reindex] index already up to date")
        return 0
    print(f"[reindex] changed/new={len(changed)}  deleted={len(deleted)}")

    store = VectorStore.load(index_dir)
    embedder, _ = get_embedder()

    for p in set(changed) | set(deleted):             # remove stale chunks first
        n = store.remove_by_file(p)
        if n:
            print(f"  - removed {n} old chunk(s) from {p}")

    new_chunks = []
    for p in changed:
        new_chunks.extend(_chunks_for_doc(current_docs[p], repo_path))
    if new_chunks:
        store.add(embedder.encode([c["text"] for c in new_chunks]), new_chunks)
        print(f"  + added {len(new_chunks)} fresh chunk(s)")

    if INGEST_COMMITS:                                # refresh commit/PR history
        store.remove_by_source({"commit", "pr"})
        cc = commit_chunks(repo_path, COMMIT_LIMIT)
        if cc:
            store.add(embedder.encode([c["text"] for c in cc]), cc)
            print(f"  ~ refreshed {len(cc)} commit/PR chunk(s)")

    store.save(index_dir)
    # BM25 statistics are corpus-wide, so rebuild it over the current chunk set.
    BM25Index.build([c["text"] for c in store.chunks]).save(index_dir)
    _save_state(index_dir, current_hashes, repo_head(repo_path))
    print(f"[reindex] done; {len(store.chunks)} chunks now indexed")
    return len(new_chunks)


# --------------------------------------------------------------------------
# ONLINE: query
# --------------------------------------------------------------------------
class GitContext:
    def __init__(self, index_dir=INDEX_DIR):
        self.store = VectorStore.load(index_dir)
        self.bm25 = BM25Index.load(index_dir)
        self.embedder, _ = get_embedder()
        self.reranker, self.rerank_name = get_reranker()

    def query(self, question: str, top_k: int = TOP_K_FINAL, answer: bool = True,
              multi_query: bool = None, flt: dict | None = None) -> dict:
        use_mq = MULTI_QUERY if multi_query is None else multi_query
        result: dict = {"question": question}

        if use_mq:
            from .query_expand import expand_query
            queries = expand_query(question, MULTI_QUERY_N)
            result["expanded_queries"] = queries
            candidates = multi_query_search(queries, self.embedder, self.store, self.bm25, flt=flt)
        else:
            candidates = hybrid_search(question, self.embedder, self.store, self.bm25, flt=flt)

        top = self.reranker.rerank(question, candidates, top_k)
        chunks = [c[2] for c in top]
        result["citations"] = [c["metadata"] for c in chunks]
        if answer:
            result["answer"], _ = generate(question, chunks)
        else:
            result["chunks"] = chunks
        return result
