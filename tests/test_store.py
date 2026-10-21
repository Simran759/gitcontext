"""Vector store: nearest-neighbour search, deletion, and filtering."""
from gitcontext.store import VectorStore
from gitcontext.embed import HashingEmbedder


def _store():
    emb = HashingEmbedder()
    chunks = [
        {"text": "login authenticate user password token", "metadata": {"file_path": "auth.py", "source": "code"}},
        {"text": "add two numbers return sum", "metadata": {"file_path": "calc.py", "source": "code"}},
        {"text": "commit fixed login bug", "metadata": {"file_path": "commit:abc", "source": "commit"}},
    ]
    s = VectorStore()
    s.add(emb.encode([c["text"] for c in chunks]), chunks)
    return emb, s


def test_search_returns_nearest_by_meaning():
    emb, s = _store()
    q = emb.encode(["login password"])[0]
    res = s.search(q, 3)
    assert s.chunks[res[0][0]]["metadata"]["file_path"] == "auth.py"


def test_remove_by_file():
    emb, s = _store()
    assert s.remove_by_file("calc.py") == 1
    assert s.vectors.shape[0] == 2
    assert all(c["metadata"]["file_path"] != "calc.py" for c in s.chunks)


def test_remove_by_source():
    emb, s = _store()
    assert s.remove_by_source({"commit"}) == 1
    assert all(c["metadata"]["source"] != "commit" for c in s.chunks)


def test_allowed_indices_and_filtered_search():
    emb, s = _store()
    allowed = s.allowed_indices({"source": "code"})
    assert allowed == {0, 1}
    # Even though the query text matches the commit chunk, the filter excludes it.
    q = emb.encode(["commit fixed login bug"])[0]
    res = s.search(q, 3, allowed)
    assert all(i in allowed for i, _ in res)
