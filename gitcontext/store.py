"""Step 4 - Vector store.

A tiny in-memory vector index persisted to disk with numpy. It stores the
embedding matrix plus each chunk's text and metadata, and searches by cosine
similarity (vectors are L2-normalized, so cosine == dot product).

Real systems swap this for Pinecone / Qdrant / Weaviate, which add
approximate-nearest-neighbor (HNSW) search for millions of vectors.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Set, Tuple
from .filters import matches


class VectorStore:
    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[dict] = []  # row i of self.vectors describes chunks[i]

    def add(self, vectors: np.ndarray, chunks: List[dict]):
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def _retain(self, keep: List[int]) -> int:
        """Keep only the given row indices; return how many were removed."""
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0
        self.chunks = [self.chunks[i] for i in keep]
        if keep and self.vectors is not None:
            self.vectors = self.vectors[keep]          # fancy-index the rows we keep
        else:
            self.vectors = None
        return removed

    def remove_by_file(self, file_path: str) -> int:
        """Drop every chunk from one file (used by incremental re-indexing)."""
        return self._retain([i for i, c in enumerate(self.chunks)
                             if c["metadata"].get("file_path") != file_path])

    def remove_by_source(self, sources) -> int:
        """Drop every chunk whose metadata.source is in `sources` (e.g. commits)."""
        sources = set(sources)
        return self._retain([i for i, c in enumerate(self.chunks)
                             if c["metadata"].get("source") not in sources])

    def allowed_indices(self, flt: dict | None) -> Optional[Set[int]]:
        """Row indices whose metadata matches the filter (None = no filter)."""
        if not flt:
            return None
        return {i for i, c in enumerate(self.chunks) if matches(c["metadata"], flt)}

    def search(self, query_vec: np.ndarray, k: int,
               allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        scores = self.vectors @ query_vec           # cosine similarity per chunk
        if allowed is not None:
            # Keep allowed rows at their score; push everything else to -inf.
            masked = np.full(scores.shape, -np.inf, dtype=scores.dtype)
            keep = list(allowed)
            if keep:
                masked[keep] = scores[keep]
            scores = masked
        idx = np.argsort(-scores)[:k]               # top-k by score
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        vecs = self.vectors if self.vectors is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(path / "vectors.npy", vecs)
        (path / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        store.chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        return store
