"""Step 6b - Hybrid retrieval with Reciprocal Rank Fusion (RRF).

    score(d) = sum over lists of 1 / (k + rank_in_list)

A chunk ranked high in any list rises; high in several wins. RRF needs no score
calibration because it uses ranks, not raw scores - which is why it can fuse
vector scores with BM25 scores, and also fuse across multiple query rewrites.
"""
from __future__ import annotations
from typing import List, Tuple
from .config import TOP_K_DENSE, TOP_K_SPARSE, RRF_K


def _rrf(ranked_lists: List[List[Tuple[int, float]]], k: int = RRF_K) -> List[Tuple[int, float]]:
    """Fuse any number of ranked lists of (index, score) by reciprocal rank."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rrf_fuse(dense, sparse, k: int = RRF_K) -> List[Tuple[int, float]]:
    """Two-list convenience wrapper (vector + BM25)."""
    return _rrf([dense, sparse], k)


def hybrid_search(query, embedder, store, bm25,
                  dense_k: int = TOP_K_DENSE, sparse_k: int = TOP_K_SPARSE,
                  flt: dict | None = None):
    """One query -> fused candidate list of (index, fused_score, chunk_dict).

    `flt` optionally restricts both searches to chunks whose metadata matches
    (e.g. {"source": "code"} or {"path_prefix": "auth/"}).
    """
    allowed = store.allowed_indices(flt)
    qv = embedder.encode([query])[0]
    dense = store.search(qv, dense_k, allowed)      # by meaning
    sparse = bm25.search(query, sparse_k, allowed)  # by exact words
    fused = rrf_fuse(dense, sparse)
    return [(idx, score, store.chunks[idx]) for idx, score in fused]


def multi_query_search(queries, embedder, store, bm25, flt: dict | None = None):
    """Several query rewrites -> fuse each query's hybrid ranking with RRF.

    Each query produces its own hybrid (vector+BM25) ranking; we then RRF those
    rankings together, so a chunk relevant to several rephrasings wins.
    """
    per_query = []
    for q in queries:
        hits = hybrid_search(q, embedder, store, bm25, flt=flt)
        per_query.append([(idx, score) for idx, score, _ in hits])
    fused = _rrf(per_query)
    return [(idx, score, store.chunks[idx]) for idx, score in fused]
