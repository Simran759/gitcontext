"""Structure-aware chunking: functions/classes (AST) and markdown headings."""
from gitcontext.ingest import RawDoc
from gitcontext.chunk import chunk_doc


def test_python_splits_by_function_and_class():
    code = "def foo():\n    return 1\n\nclass Bar:\n    def m(self):\n        return 2\n"
    chunks = chunk_doc(RawDoc(path="x.py", text=code, source="code"))
    symbols = {c.metadata["symbol"] for c in chunks}
    assert "foo" in symbols and "Bar" in symbols
    types = {c.metadata["type"] for c in chunks}
    assert "function" in types and "class" in types
    assert all(c.metadata["file_path"] == "x.py" for c in chunks)


def test_context_injection_header_present():
    chunks = chunk_doc(RawDoc(path="x.py", text="def foo():\n    return 1\n", source="code"))
    assert "# file: x.py" in chunks[0].text


def test_markdown_splits_by_heading():
    md = "# Title\nintro\n\n## Section A\nbody a\n\n## Section B\nbody b\n"
    chunks = chunk_doc(RawDoc(path="r.md", text=md, source="docs"))
    headings = {c.metadata.get("symbol") for c in chunks}
    assert "Section A" in headings and "Section B" in headings


def test_non_python_non_doc_falls_back_to_fixed():
    # a .txt treated as docs still chunks; empty-ish still yields something
    chunks = chunk_doc(RawDoc(path="notes.txt", text="hello world " * 50, source="docs"))
    assert len(chunks) >= 1
