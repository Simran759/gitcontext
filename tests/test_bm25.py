"""BM25 keyword index: exact-token retrieval and filtering."""
from gitcontext.bm25_index import BM25Index


def test_exact_string_ranks_first():
    texts = [
        "def divide raises ZeroDivisionError cannot divide by zero",
        "def add return the sum of a and b",
        "class AuthService handles user login",
    ]
    idx = BM25Index.build(texts)
    res = idx.search("cannot divide by zero", 3)
    assert res[0][0] == 0            # the chunk containing the exact phrase wins


def test_allowed_restricts_candidates():
    idx = BM25Index.build(["alpha shared", "beta shared", "gamma"])
    res = idx.search("shared", 3, allowed={1, 2})
    assert all(i in (1, 2) for i, _ in res)
