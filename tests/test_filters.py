"""Metadata filter predicate."""
from gitcontext.filters import matches


def test_empty_filter_always_true():
    assert matches({"source": "code"}, None)
    assert matches({"source": "code"}, {})


def test_exact_match():
    assert matches({"source": "code"}, {"source": "code"})
    assert not matches({"source": "docs"}, {"source": "code"})


def test_list_membership():
    assert matches({"source": "pr"}, {"source": ["commit", "pr"]})
    assert not matches({"source": "code"}, {"source": ["commit", "pr"]})


def test_path_prefix_and_contains():
    assert matches({"file_path": "auth/service.py"}, {"path_prefix": "auth/"})
    assert not matches({"file_path": "calc.py"}, {"path_prefix": "auth/"})
    assert matches({"file_path": "auth/service.py"}, {"path_contains": "service"})


def test_multiple_keys_are_anded():
    meta = {"source": "code", "type": "function"}
    assert matches(meta, {"source": "code", "type": "function"})
    assert not matches(meta, {"source": "code", "type": "class"})
