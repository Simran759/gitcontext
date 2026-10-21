"""Retrieval: RRF fusion, hybrid search, filtering, and multi-query."""
from gitcontext.embed import HashingEmbedder
from gitcontext.store import VectorStore
from gitcontext.bm25_index import BM25Index
from gitcontext.retrieve import rrf_fuse, hybrid_search, multi_query_search


def test_rrf_prefers_consensus():
    dense = [(1, 0.9), (2, 0.8), (3, 0.1)]
    sparse = [(2, 5.0), (1, 4.0), (4, 1.0)]
    fused = rrf_fuse(dense, sparse)
    # 1 and 2 appear in both lists, so they take the top two spots.
    assert {fused[0][0], fused[1][0]} == {1, 2}


def _corpus():
    emb = HashingEmbedder()
    chunks = [
        {"text": "login authenticate user password token", "metadata": {"file_path": "auth.py", "source": "code"}},
        {"text": "add two numbers return sum", "metadata": {"file_path": "calc.py", "source": "code"}},
        {"text": "commit message fixed login bug", "metadata": {"file_path": "commit:1", "source": "commit"}},
    ]
    s = VectorStore()
    s.add(emb.encode([c["text"] for c in chunks]), chunks)
    b = BM25Index.build([c["text"] for c in chunks])
    return emb, s, b


def test_hybrid_filter_excludes_other_sources():
    emb, s, b = _corpus()
    res = hybrid_search("login", emb, s, b, flt={"source": "code"})
    assert res and all(c["metadata"]["source"] == "code" for _, _, c in res)


def test_multi_query_returns_fused_results():
    emb, s, b = _corpus()
    res = multi_query_search(["login", "authenticate user password"], emb, s, b)
    assert res[0][2]["metadata"]["file_path"] == "auth.py"
