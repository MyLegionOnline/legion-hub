"""internal design notes T3: room disk measurement and "Clean dependencies"."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app import disk_usage


@pytest.fixture
def room(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(disk_usage, "workspace_root", lambda room_id: tmp_path / room_id)
    root = tmp_path / "room-1"
    root.mkdir()
    return root


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_measure_counts_everything(room):
    _write(room / "src" / "app.js", 10_000)
    _write(room / "node_modules" / "lib" / "index.js", 50_000)
    _write(room / ".git" / "objects" / "ab", 5_000)
    result = disk_usage.measure_room_disk("room-1")
    assert result["exists"] and not result["partial"]
    assert result["bytes"] >= 65_000


def test_measure_missing_room_is_zero(room):
    assert disk_usage.measure_room_disk("room-2") == {
        "room_id": "room-2", "bytes": 0, "partial": False, "exists": False,
    }


def test_measure_rejects_traversal(room):
    with pytest.raises(ValueError):
        disk_usage.measure_room_disk("../etc")


def test_measure_budget_marks_partial(room, monkeypatch):
    _write(room / "a" / "b", 10)
    monkeypatch.setattr(disk_usage, "MEASURE_BUDGET_SEC", -1.0)
    assert disk_usage.measure_room_disk("room-1")["partial"]


def test_clean_removes_dependency_dirs_outside_git(room):
    _write(room / "src" / "app.js", 100)
    _write(room / "node_modules" / "lib" / "index.js", 50_000)
    _write(room / "packages" / "web" / "node_modules" / "x.js", 20_000)
    _write(room / "api" / ".venv" / "lib" / "site.py", 20_000)
    _write(room / "src" / "__pycache__" / "a.pyc", 1_000)
    result = disk_usage.clean_dependencies("room-1")
    assert sorted(result["removed"]) == [
        "api/.venv", "node_modules", "packages/web/node_modules", "src/__pycache__",
    ]
    assert result["freed_bytes"] >= 91_000
    assert (room / "src" / "app.js").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_clean_keeps_directories_tracked_by_git(room):
    _write(room / "build" / "deploy.sh", 10)
    _write(room / "dist" / "bundle.js", 10)
    subprocess.run(["git", "init", "-q", str(room)], check=True)
    subprocess.run(["git", "-C", str(room), "add", "build/deploy.sh"], check=True)
    result = disk_usage.clean_dependencies("room-1")
    assert result["removed"] == ["dist"]
    assert (room / "build" / "deploy.sh").exists()


def test_symlinked_dependency_dir_is_not_followed(room, tmp_path):
    outside = tmp_path / "outside" / "node_modules"
    _write(outside / "x.js", 10)
    try:
        (room / "node_modules").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    disk_usage.clean_dependencies("room-1")
    assert (outside / "x.js").exists()


def test_git_internals_are_never_candidates(room):
    _write(room / ".git" / "node_modules" / "keep", 10)
    assert disk_usage.find_dependency_dirs(room, None) == []


def test_unreadable_git_tracking_refuses_to_clean(room):
    """A broken repository: tracked files unknown, so nothing is deleted."""
    _write(room / ".git" / "HEAD", 3)
    _write(room / "node_modules" / "x.js", 10)
    with pytest.raises(RuntimeError):
        disk_usage.clean_dependencies("room-1")
    assert (room / "node_modules" / "x.js").exists()


def test_data_volume_usage_reports_the_filesystem(tmp_path, monkeypatch):
    class _Stat:
        f_frsize = 4096
        f_blocks = 1000
        f_bfree = 300
        f_bavail = 250

    monkeypatch.setattr(disk_usage.os, "statvfs", lambda path: _Stat(), raising=False)
    result = disk_usage.data_volume_usage(tmp_path)
    assert result["size_bytes"] == 1000 * 4096
    assert result["used_bytes"] == 700 * 4096
    # What an unprivileged writer can still take, not the root reserve.
    assert result["free_bytes"] == 250 * 4096
