"""Command-line interface.

    python -m gitcontext.cli index <repo_path>      # full build
    python -m gitcontext.cli reindex <repo_path>    # incremental refresh (only changed files)
    python -m gitcontext.cli ask "your question here"
    python -m gitcontext.cli ask --multi "your question"         # multi-query retrieval
    python -m gitcontext.cli ask --source code "your question"   # filter: code only
    python -m gitcontext.cli ask --path auth/ "your question"    # filter: files under auth/
"""
from __future__ import annotations
import sys
from .pipeline import build_index, reindex, GitContext


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(__doc__)
        return

    cmd, rest = argv[0], argv[1:]
    if cmd == "index" and rest:
        build_index(rest[0])
    elif cmd == "reindex" and rest:
        reindex(rest[0])
    elif cmd == "ask" and rest:
        multi = "--multi" in rest
        flt, words, i = {}, [], 0
        tokens = [a for a in rest if a != "--multi"]
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--source" and i + 1 < len(tokens):
                flt["source"] = tokens[i + 1]; i += 2
            elif tok == "--path" and i + 1 < len(tokens):
                flt["path_prefix"] = tokens[i + 1]; i += 2
            else:
                words.append(tok); i += 1
        res = GitContext().query(" ".join(words), multi_query=multi, flt=flt or None)
        if flt:
            print("=== FILTER ===", flt)
        if res.get("expanded_queries"):
            print("=== EXPANDED QUERIES ===")
            for q in res["expanded_queries"]:
                print(" ?", q)
        print("\n=== CITATIONS ===")
        for c in res["citations"]:
            print(" -", c.get("file_path"), f"({c.get('symbol')})" if c.get("symbol") else "")
        print("\n=== ANSWER ===")
        print(res["answer"])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
