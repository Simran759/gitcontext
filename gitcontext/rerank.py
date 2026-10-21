"""Step 7 - Reranking (optional cross-encoder).

A cross-encoder reads (query, chunk) together and scores true relevance, far
more accurately than the fast bi-encoder used for retrieval. It is slow, so we
only run it on the fused candidates, then keep the top N.

If sentence-transformers isn't installed we skip reranking and keep the fusion
order - the pipeline still works.
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from .quiet import quiet_ml
        quiet_ml()
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query, candidates, top_n):
        pairs = [[query, c[2]["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_n]]


class NoopReranker:
    def rerank(self, query, candidates, top_n):
        return candidates[:top_n]


def get_reranker():
    # Set GITCONTEXT_RERANK=noop to skip the cross-encoder (used by tests).
    import os
    if os.getenv("GITCONTEXT_RERANK") == "noop":
        return NoopReranker(), "noop (forced)"
    try:
        return CrossEncoderReranker(), "cross-encoder"
    except Exception:
        return NoopReranker(), "noop (fusion order kept)"
