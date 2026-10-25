"""internal design notes T11: room archive, restore and archive delete."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from app import room_archive


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    store_dir = tmp_path / "archive"
    monkeypatch.setattr(room_archive, "workspace_root", lambda room_id: data / room_id)
    monkeypatch.setattr(room_archive, "WORKSPACE_DATA_DIR", str(data))
    monkeypatch.setattr(room_archive, "STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setattr(room_archive, "_store", room_archive._DirStore(str(store_dir)))
    monkeypatch.setattr(room_archive, "chown_workspace_tree", lambda room_id: None)
    return data, store_dir


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_archive_then_restore_round_trip(env):
    data, store_dir = env
    room = data / "room-1"
    _write(room / "src" / "app.py", "print(1)")
    _write(room / "notes.txt", "untracked")
    _write(room / ".legion" / "generation.json", '{"epoch": "e1"}')
    _write(room / "node_modules" / "lib" / "index.js", "dep")
    _write(room / "build" / "keep.txt", "tracked build")
    _git(room, "init", "-q")
    _git(room, "add", "src/app.py", "build/keep.txt")
    _git(room, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init")

    result = room_archive.archive_room("room-1")
    assert result["archived"] and result["bytes"] > 0
    assert result["skipped"] == ["node_modules"]
    assert not room.exists()
    assert (store_dir / "room-1.tar.gz").is_file()
    assert room_archive.archive_info("room-1")["exists"]

    restored = room_archive.unarchive_room("room-1")
    assert restored["restored"]
    assert (room / "src" / "app.py").read_text() == "print(1)"
    assert (room / "notes.txt").read_text() == "untracked"
    assert (room / "build" / "keep.txt").read_text() == "tracked build"
    assert (room / ".legion" / "generation.json").read_text() == '{"epoch": "e1"}'
    assert not (room / "node_modules").exists()
    log = subprocess.run(["git", "-C", str(room), "log", "--oneline"], capture_output=True, text=True)
    assert "init" in log.stdout
    # The archive is dropped once the room is back.
    assert not (store_dir / "room-1.tar.gz").exists()


def test_empty_room_is_only_noted(env):
    data, store_dir = env
    _write(data / "room-1" / ".legion" / "generation.json", "{}")
    result = room_archive.archive_room("room-1")
    assert result["empty"] and not result["archived"]
    assert not (data / "room-1").exists()
    assert not store_dir.exists() or not any(store_dir.iterdir())
    assert room_archive.archive_room("room-2")["empty"]


def test_restore_over_metadata_only_dir(env):
    data, _ = env
    _write(data / "room-1" / "a.txt", "a")
    room_archive.archive_room("room-1")
    # Reading the generation of the archived room recreates its metadata.
    _write(data / "room-1" / ".legion" / "generation.json", '{"epoch": "new"}')
    assert room_archive.unarchive_room("room-1")["restored"]
    assert (data / "room-1" / "a.txt").read_text() == "a"


def test_restore_never_overwrites_files(env):
    data, store_dir = env
    _write(data / "room-1" / "a.txt", "a")
    room_archive.archive_room("room-1")
    _write(data / "room-1" / "b.txt", "new work")
    with pytest.raises(RuntimeError):
        room_archive.unarchive_room("room-1")
    assert (data / "room-1" / "b.txt").read_text() == "new work"
    assert (store_dir / "room-1.tar.gz").is_file()


def test_restore_without_archive(env):
    assert room_archive.unarchive_room("room-9") == {
        "room_id": "room-9", "restored": False, "missing": True, "bytes": 0,
    }


def test_failed_store_keeps_the_room(env, monkeypatch):
    data, _ = env
    _write(data / "room-1" / "a.txt", "a")

    def boom(local, key):
        raise OSError("store down")

    monkeypatch.setattr(room_archive._store, "put", boom)
    with pytest.raises(OSError):
        room_archive.archive_room("room-1")
    assert (data / "room-1" / "a.txt").read_text() == "a"


def test_restore_rejects_escaping_members(env, tmp_path):
    _, store_dir = env
    evil = tmp_path / "evil.txt"
    evil.write_text("evil")
    store_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(store_dir / "room-1.tar.gz", "w:gz") as tar:
        tar.add(str(evil), arcname="../escaped.txt")
    with pytest.raises(tarfile.TarError):
        room_archive.unarchive_room("room-1")
    assert not (tmp_path / "data" / "escaped.txt").exists()
    assert not (tmp_path / "data" / "room-1").exists()


def test_delete_archive(env):
    data, store_dir = env
    _write(data / "room-1" / "a.txt", "a")
    room_archive.archive_room("room-1")
    assert room_archive.delete_archive("room-1") == {"room_id": "room-1", "deleted": True}
    assert room_archive.delete_archive("room-1") == {"room_id": "room-1", "deleted": False}


def test_rejects_traversal(env):
    with pytest.raises(ValueError):
        room_archive.archive_room("../etc")


def test_plans_under_legion_are_not_an_empty_room(env):
    data, store_dir = env
    _write(data / "room-1" / ".legion" / "generation.json", "{}")
    _write(data / "room-1" / ".legion" / "plans" / "plan.md", "# plan")
    result = room_archive.archive_room("room-1")
    assert result["archived"] and not result["empty"]
    room_archive.unarchive_room("room-1")
    assert (data / "room-1" / ".legion" / "plans" / "plan.md").read_text() == "# plan"


def test_export_zip_has_files_and_history_without_dependencies(env):
    import zipfile

    data, _ = env
    room = data / "room-1"
    _write(room / "src" / "app.py", "print(1)")
    _write(room / "node_modules" / "lib" / "index.js", "dep")
    _write(room / ".legion" / "generation.json", "{}")
    _write(room / ".legion" / "plans" / "plan.md", "# plan")
    _git(room, "init", "-q")
    _git(room, "add", "src/app.py")
    _git(room, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init")

    path = room_archive.export_zip("room-1")
    try:
        names = set(zipfile.ZipFile(path).namelist())
    finally:
        path.unlink()
    assert "src/app.py" in names and ".legion/plans/plan.md" in names
    assert any(n.startswith(".git/") for n in names)
    assert not any(n.startswith("node_modules/") for n in names)
    assert ".legion/generation.json" not in names
    assert room.is_dir()  # an export leaves the room alone
    assert room_archive.export_zip("room-2") is None
