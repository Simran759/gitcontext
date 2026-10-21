"""Metadata filtering.

A filter is a small dict of constraints applied to a chunk's metadata, e.g.

    {"source": "code"}                     # only source code
    {"source": ["commit", "pr"]}           # history only (value may be a list)
    {"type": "function"}                   # only function chunks
    {"path_prefix": "auth/"}               # only files under auth/
    {"path_contains": "service"}           # file_path contains "service"
    {"author": "simran"}                   # only simran's code/commits

Rules:
- A list/tuple/set value matches if the metadata value is in it (OR).
- Several keys in one filter must all hold (AND).
- `path_prefix` / `path_contains` are special keys that test `file_path`.
Keeping this pure and separate makes it trivial to unit-test.
"""
from __future__ import annotations


def matches(meta: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    for key, want in flt.items():
        if key == "path_prefix":
            if not (meta.get("file_path") or "").startswith(want):
                return False
        elif key == "path_contains":
            if want not in (meta.get("file_path") or ""):
                return False
        else:
            val = meta.get(key)
            if isinstance(want, (list, tuple, set)):
                if val not in want:
                    return False
            elif val != want:
                return False
    return True
