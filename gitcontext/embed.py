"""Step 3 - Embeddings.

Turns text into vectors. Two backends:

1. `sentence-transformers` (real semantic embeddings) if installed -
   recommended; shows true meaning-based retrieval.
2. A pure-numpy hashing bag-of-words fallback that needs no download, so the
   project always runs. It captures word overlap, not deep meaning.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np
from typing import List

_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]+")


class HashingEmbedder:
    """Deterministic and offline. Hashes tokens into a fixed vector space."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms  # L2-normalized, so dot product == cosine similarity


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from .quiet import quiet_ml
        quiet_ml()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def encode(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def get_embedder():
    """Prefer real semantic embeddings; fall back to the offline hasher.

    Set GITCONTEXT_EMBED=hashing to force the offline backend (used by tests).
    """
    import os
    if os.getenv("GITCONTEXT_EMBED") == "hashing":
        return HashingEmbedder(), "hashing (forced)"
    try:
        return SentenceTransformerEmbedder(), "sentence-transformers (semantic)"
    except Exception:
        return HashingEmbedder(), "hashing fallback (install sentence-transformers for semantic search)"
