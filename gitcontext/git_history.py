"""Ingest commit history (and PR references) as first-class documents.

Each commit becomes its own chunk - subject + body + author/date - so questions
like "why was retries added?" or "what changed in PR #142?" can be answered from
history, not just current code.

Note: true PR *discussion threads* live on GitHub and need its API. Here we
ingest the commit record and tag a chunk as source="pr" when its message
references a PR number (e.g. "(#142)"); plain commits are source="commit".
Commits are scoped to the indexed directory (`git log -- .`).
"""
from __future__ import annotations
from typing import List
from .git_meta import _git, is_git_repo, _PR_RE

# Record separator \x1e between commits, unit separator \x1f between fields.
_REC = "\x1e"
_FMT = "%H%x1f%an%x1f%ad%x1f%s%x1f%b"


def commit_chunks(repo_path, limit: int = 200) -> List[dict]:
    if not is_git_repo(repo_path):
        return []
    out = _git(["log", f"-{limit}", "--date=short",
                f"--pretty=format:{_FMT}{_REC}", "--", "."], repo_path)
    if not out:
        return []

    chunks: List[dict] = []
    for record in out.split(_REC):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x1f")
        if len(parts) < 4:
            continue
        commit, author, date, subject = parts[0], parts[1], parts[2], parts[3]
        body = parts[4].strip() if len(parts) > 4 else ""

        pr_match = _PR_RE.search(f"{subject} {body}")
        pr = int(pr_match.group(1)) if pr_match else None

        text = f"# commit {commit[:10]} by {author} on {date}\n{subject}"
        if body:
            text += f"\n\n{body}"

        chunks.append({"text": text, "metadata": {
            "source": "pr" if pr else "commit",
            "type": "commit",
            "file_path": f"commit:{commit[:10]}",   # gives citations something to show
            "symbol": subject[:60],
            "commit": commit[:10],
            "author": author,
            "last_modified": date,
            "pr": pr,
        }})
    return chunks
