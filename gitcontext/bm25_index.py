"""Step 6a - Keyword (sparse) index using BM25.

Catches exact tokens - error strings, function names - that dense vector
search blurs away. Persisted as the tokenized corpus so it rebuilds on load.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\S")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(text)]


class BM25Index:
    def __init__(self, corpus_tokens: List[List[str]]):
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def build(cls, texts: List[str]) -> "BM25Index":
        return cls([tokenize(t) for t in texts])

    def search(self, query: str, k: int, allowed=None) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        candidates = range(len(scores)) if allowed is None else [i for i in allowed]
        ranked = sorted(candidates, key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in ranked]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "bm25.json").write_text(json.dumps(self.corpus_tokens), encoding="utf-8")

    @classmethod
    def load(cls, path):
        tokens = json.loads((Path(path) / "bm25.json").read_text(encoding="utf-8"))
        return cls(tokens)
