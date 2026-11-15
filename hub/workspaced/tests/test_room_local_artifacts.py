"""git_ops.py — room-local artefacts must not reach the client's repository.

Field case (room "Blog"): an agent without root ran ``apt-get download`` +
``dpkg -x`` into ``/workspace/.legion/libroot`` to get Chromium's libraries, and
the next auto-commit shipped ~150 ``.deb`` / ``.so`` files into the repo.

1. ``.legion/`` is an allowlist: only ``run.json`` and ``services.json`` are
   ever committed; a room that already tracks other ``.legion/`` content drops
   it from the index on its next commit (the working copy stays).
2. Newly added machine binaries (``.deb``/``.rpm``/``.so*``/ELF) anywhere else
   are kept out of the commit and ignored locally; ``git_commit`` reports them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "local-artifacts-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _g(root: Path, *args: str) -> str:
    return git_ops._run(list(args), cwd=root).stdout.strip()


def _write(root: Path, rel: str, body: bytes | str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, str):
        p.write_text(body, encoding="utf-8")
    else:
        p.write_bytes(body)


def _tracked(root: Path) -> list[str]:
    return _g(root, "ls-files").splitlines()


def test_legion_dir_is_an_allowlist(repo):
    room_id, root = repo
    _write(root, ".legion/run.json", '{"checks": []}')
    _write(root, ".legion/services.json", '{"services": []}')
    _write(root, ".legion/libroot/_debs/libx11-6_amd64.deb", b"!<arch>\n")
    _write(root, ".legion/environments/py/venv/pyvenv.cfg", "home = /usr\n")
    _write(root, ".legion/skills/xlsx/scripts/recalc.py", "print(1)\n")
    _write(root, ".legion/logs/abc.log", "out\n")
    _write(root, ".legion/local/bin/tool", "#!/bin/sh\n")
    _write(root, "index.md", "# hi\n")
    git_ops.git_commit(room_id, "work")

    tracked = _tracked(root)
    assert "index.md" in tracked
    assert ".legion/run.json" in tracked
    assert ".legion/services.json" in tracked
    assert not [t for t in tracked if t.startswith(".legion/")
                and t not in git_ops._TRACKED_LEGION_FILES]
    assert _g(root, "status", "--porcelain") == ""


def test_legacy_tracked_legion_content_is_untracked_on_next_commit(repo):
    room_id, root = repo
    _write(root, ".legion/run.json", '{"checks": []}')
    _write(root, ".legion/libroot/lib/libbz2.so.1.0.4", b"\x7fELF....")
    # a room older than the rule: the junk got force-added and committed
    git_ops._run(["add", "-f", "--", ".legion"], cwd=root)
    git_ops._run(["commit", "-m", "legacy"], cwd=root)
    assert ".legion/libroot/lib/libbz2.so.1.0.4" in _tracked(root)

    _write(root, "post.md", "new post\n")
    git_ops.git_commit(room_id, "next")

    tracked = _tracked(root)
    assert ".legion/libroot/lib/libbz2.so.1.0.4" not in tracked
    assert ".legion/run.json" in tracked
    assert "post.md" in tracked
    # the agent's files are still on disk
    assert (root / ".legion/libroot/lib/libbz2.so.1.0.4").exists()


def test_new_machine_binaries_are_held_back_and_reported(repo):
    room_id, root = repo
    _write(root, "libs/_debs/libnss3_3.87_amd64.deb", b"!<arch>\n")
    _write(root, "libs/usr/lib/libX11.so.6.4.0", b"\x7fELF....")
    _write(root, "libs/usr/bin/gio-querymodules", b"\x7fELF\x02\x01")
    _write(root, "site/logo.png", b"\x89PNG\r\n\x1a\n")
    _write(root, "site/index.html", "<h1>hi</h1>")

    report: dict = {}
    sha = git_ops.git_commit(room_id, "work", report=report)

    assert sha is not None
    tracked = _tracked(root)
    assert "site/index.html" in tracked
    assert "site/logo.png" in tracked
    assert not [t for t in tracked if t.startswith("libs/")]
    assert sorted(report["held_back"]) == [
        "libs/_debs/libnss3_3.87_amd64.deb",
        "libs/usr/bin/gio-querymodules",
        "libs/usr/lib/libX11.so.6.4.0",
    ]
    # ignored locally from now on — the agent's own `git add -A` skips them
    assert _g(root, "status", "--porcelain") == ""
    exclude = (root / ".git/info/exclude").read_text()
    assert git_ops._HOLD_BACK_HEADER in exclude
    assert "/libs/usr/lib/libX11.so.6.4.0" in exclude.splitlines()

    # a second commit does not report them again
    report2: dict = {}
    git_ops.git_commit(room_id, "again", report=report2)
    assert report2["held_back"] == []


def test_already_tracked_binary_is_left_alone(repo):
    room_id, root = repo
    _write(root, "vendor/libfoo.so", b"\x7fELF v1")
    git_ops._run(["add", "-f", "--", "vendor/libfoo.so"], cwd=root)
    git_ops._run(["commit", "-m", "vendored on purpose"], cwd=root)

    _write(root, "vendor/libfoo.so", b"\x7fELF v2")
    report: dict = {}
    git_ops.git_commit(room_id, "update vendored lib", report=report)

    assert report["held_back"] == []
    assert _g(root, "show", "HEAD:vendor/libfoo.so") == "\x7fELF v2"


def test_gitignore_literal_escapes_patterns():
    assert git_ops._gitignore_literal("a/b*[1].so") == "/a/b\\*\\[1\\].so"
    assert git_ops._gitignore_literal("#x/!y") == "/\\#x/\\!y"
