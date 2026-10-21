"""Query translation / expansion (the "RAG-Fusion" idea).

A user's phrasing may not match how the code is written. So we ask the local
model to rewrite the question into a few alternative search queries, retrieve
for each, and fuse the results. This lifts recall for awkwardly-phrased questions.

If no local model is available, we return just the original question, so
retrieval still works (it simply skips expansion).
"""
from __future__ import annotations
from typing import List
from .generate import complete

_SYS = ("You rewrite a user's question into alternative search queries for a code "
        "search engine. Output ONLY the queries, one per line, no numbering, no prose.")


def _clean(line: str) -> str:
    # strip bullets / numbering / quotes that small models like to add
    return line.strip().lstrip("-*0123456789.)").strip().strip('"').strip()


def expand_query(question: str, n: int = 3) -> List[str]:
    user = f"Question: {question}\n\nGive {n} alternative search queries, one per line."
    text, _ = complete(_SYS, user, max_new_tokens=128)

    queries = [question]                       # always keep the original
    if text:
        for line in text.splitlines():
            q = _clean(line)
            if q and q.lower() not in (x.lower() for x in queries):
                queries.append(q)
    return queries[: n + 1]                    # original + up to n rewrites
