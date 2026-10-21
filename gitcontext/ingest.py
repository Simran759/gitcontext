"""Step 1 - Ingestion.

Walk a repository on disk and yield its files as raw documents. We only read
text we can index (code + docs) and skip binaries and vendored directories.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union
from .config import CODE_EXTS, DOC_EXTS

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             "dist", "build", ".gitcontext_index"}


@dataclass
class RawDoc:
    path: str      # repo-relative path - used as metadata and as a citation
    text: str
    source: str    # "code" | "docs"


def ingest(repo_path: Union[str, Path]) -> Iterator[RawDoc]:
    """Yield one RawDoc per indexable file under repo_path."""
    repo = Path(repo_path)
    for p in sorted(repo.rglob("*")):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        ext = p.suffix.lower()
        if ext in CODE_EXTS:
            source = "code"
        elif ext in DOC_EXTS:
            source = "docs"
        else:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            yield RawDoc(
                path=str(p.relative_to(repo)).replace("\\", "/"),
                text=text,
                source=source,
            )
