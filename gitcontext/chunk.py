"""Step 2 - Structure-aware chunking.

Code is split by function/class using Python's `ast` module, so each chunk is
one logical unit instead of an arbitrary token window. Markdown is split by
heading. Anything else falls back to fixed-size word windows with overlap.

Each chunk carries metadata (file path, symbol, line numbers) which later
powers filtering and citations.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import List
from .config import CHUNK_WORDS, CHUNK_OVERLAP
from .ingest import RawDoc


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _fixed(text: str, meta: dict) -> List[Chunk]:
    """Fallback: overlapping fixed-size windows."""
    words = text.split()
    chunks, step = [], max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for i in range(0, max(1, len(words)), step):
        window = words[i:i + CHUNK_WORDS]
        if not window:
            break
        chunks.append(Chunk(" ".join(window), {**meta, "symbol": None}))
    return chunks


def _chunk_python(doc: RawDoc) -> List[Chunk]:
    lines = doc.text.splitlines()
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return _fixed(doc.text, {"file_path": doc.path, "source": doc.source, "type": "file"})

    chunks: List[Chunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = node.lineno - 1, getattr(node, "end_lineno", node.lineno)
            body = "\n".join(lines[start:end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            # Context injection: make the isolated chunk self-describing.
            header = f"# file: {doc.path}\n# {kind}: {node.name}\n"
            chunks.append(Chunk(header + body, {
                "file_path": doc.path, "source": doc.source, "type": kind,
                "symbol": node.name, "start_line": node.lineno, "end_line": end,
            }))
    return chunks or _fixed(doc.text, {"file_path": doc.path, "source": doc.source, "type": "file"})


def _chunk_markdown(doc: RawDoc) -> List[Chunk]:
    chunks: List[Chunk] = []
    current: List[str] = []
    heading = None

    def flush():
        text = "\n".join(current).strip()
        if text:
            chunks.append(Chunk(f"# file: {doc.path}\n" + text, {
                "file_path": doc.path, "source": doc.source,
                "type": "doc-section", "symbol": heading,
            }))

    for line in doc.text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            current = [line]
            heading = line.lstrip("# ").strip()
        else:
            current.append(line)
    flush()
    return chunks or _fixed(doc.text, {"file_path": doc.path, "source": doc.source, "type": "doc"})


def chunk_doc(doc: RawDoc) -> List[Chunk]:
    if doc.path.endswith(".py"):
        return _chunk_python(doc)
    if doc.source == "docs":
        return _chunk_markdown(doc)
    return _fixed(doc.text, {"file_path": doc.path, "source": doc.source, "type": "file"})
