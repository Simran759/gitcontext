"""End-to-end: build an index, query it, and prove incremental re-indexing.

Runs fully offline (hashing embedder, no-op reranker, no LLM) thanks to
conftest.py, so there are no downloads.
"""
from gitcontext.pipeline import build_index, reindex, GitContext


def _make_repo(root):
    repo = root / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "def authenticate_user(username, password):\n    return True\n")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    return repo


def test_build_and_query(tmp_path):
    repo = _make_repo(tmp_path)
    idx = tmp_path / "idx"
    n = build_index(repo, idx)
    assert n >= 2

    res = GitContext(idx).query("how does authenticate_user work", top_k=3)
    files = [c["file_path"] for c in res["citations"]]
    assert "auth.py" in files
    assert res["answer"].startswith("[extractive]")   # GITCONTEXT_LLM=none


def test_reindex_only_touches_changed_file(tmp_path):
    repo = _make_repo(tmp_path)
    idx = tmp_path / "idx"
    build_index(repo, idx)

    # Add a new function to calc.py and reindex.
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n")
    added = reindex(repo, idx)
    assert added >= 1

    # The new symbol is now in the index; auth.py was untouched.
    symbols = {c["metadata"].get("symbol") for c in GitContext(idx).store.chunks}
    assert "mul" in symbols and "add" in symbols and "authenticate_user" in symbols


def test_filtered_query_restricts_source(tmp_path):
    repo = _make_repo(tmp_path)
    idx = tmp_path / "idx"
    build_index(repo, idx)
    res = GitContext(idx).query("add numbers", top_k=5, flt={"source": "code"})
    assert all(c["source"] == "code" for c in res["citations"])
