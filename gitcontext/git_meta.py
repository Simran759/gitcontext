"""Git metadata for each file: commit, author, last-modified date, PR number.

We shell out to `git` (no extra dependency). For each file we ask git for the
*last commit that touched it*. The PR number is parsed from the commit subject
(GitHub writes "... (#142)" or "Merge pull request #142 ..."). If the path is
not inside a git repo, or the file isn't committed yet, we return {} and the
pipeline simply omits those fields - nothing breaks.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from typing import Optional

# \x1f (unit separator) is a safe delimiter: it never appears in commit text.
_FMT = "%H%x1f%an%x1f%ad%x1f%s"
_PR_RE = re.compile(r"#(\d+)")


def _git(args, cwd) -> Optional[str]:
    """Run a git command; return stdout (stripped) or None on any failure."""
    try:
        r = subprocess.run(["git", *args], cwd=str(cwd),
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def repo_head(repo_path) -> Optional[str]:
    """The current commit hash of the repo, or None if not a git repo."""
    return _git(["rev-parse", "HEAD"], repo_path)


def is_git_repo(repo_path) -> bool:
    return _git(["rev-parse", "--is-inside-work-tree"], repo_path) == "true"


def file_git_meta(repo_path, rel_path: str) -> dict:
    """Return {commit, author, last_modified, pr} for the file's last commit."""
    out = _git(["log", "-1", "--date=short", f"--format={_FMT}", "--", rel_path],
               repo_path)
    if not out:
        return {}
    parts = out.split("\x1f")
    if len(parts) != 4:
        return {}
    commit, author, date, subject = parts
    pr_match = _PR_RE.search(subject)
    return {
        "commit": commit[:10],         # short hash, like git's default display
        "author": author,
        "last_modified": date,         # YYYY-MM-DD
        "pr": int(pr_match.group(1)) if pr_match else None,
    }
