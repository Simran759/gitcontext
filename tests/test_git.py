"""Git metadata + commit/PR ingestion (needs a real git repo in a temp dir)."""
import shutil
import subprocess
import pytest

from gitcontext.git_meta import file_git_meta
from gitcontext.git_history import commit_chunks

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _init_repo(path, message="Add auth service (#142)"):
    (path / "auth.py").write_text("def authenticate_user():\n    return True\n")
    for args in (["init", "-q"],
                 ["config", "user.email", "t@test.com"],
                 ["config", "user.name", "Tester"],
                 ["add", "auth.py"],
                 ["commit", "-q", "-m", message]):
        subprocess.run(["git", *args], cwd=str(path), check=True,
                       capture_output=True, text=True)


def test_file_git_meta_parses_fields(tmp_path):
    _init_repo(tmp_path)
    m = file_git_meta(tmp_path, "auth.py")
    assert m["author"] == "Tester"
    assert m["pr"] == 142                 # parsed from "(#142)"
    assert len(m["commit"]) == 10


def test_commit_chunks_tags_pr(tmp_path):
    _init_repo(tmp_path)
    cc = commit_chunks(tmp_path)
    assert len(cc) == 1
    md = cc[0]["metadata"]
    assert md["source"] == "pr" and md["pr"] == 142
    assert md["type"] == "commit"
