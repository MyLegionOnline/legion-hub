"""Stage 2 — private room files remain discoverable to users and agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import file_ops, generation


@pytest.fixture
def isolated_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)
    return "userfiles-stage2", tmp_path / "userfiles-stage2"


def test_tree_exposes_only_userfiles_dot_directory(isolated_volume):
    room_id, root = isolated_volume
    (root / ".userfiles").mkdir(parents=True)
    (root / ".userfiles" / "reference.md").write_text("reference", encoding="utf-8")
    (root / ".userfiles" / ".hidden.txt").write_text("hidden", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("internal", encoding="utf-8")
    (root / ".legion").mkdir()
    (root / ".legion" / "internal.log").write_text("internal", encoding="utf-8")
    (root / ".vscode").mkdir()
    (root / ".vscode" / "settings.json").write_text("{}", encoding="utf-8")

    paths = {entry["path"] for entry in file_ops.get_tree(room_id)}

    assert "/.userfiles/reference.md" in paths
    assert "/.userfiles/.hidden.txt" not in paths
    assert "/.git/config" not in paths
    assert "/.legion/internal.log" not in paths
    assert "/.vscode/settings.json" not in paths


def test_private_files_bypass_sync_ignore_but_remain_metadata_only_in_snapshot(
    isolated_volume,
):
    room_id, root = isolated_volume
    (root / ".userfiles").mkdir(parents=True)
    (root / ".gitignore").write_text(".userfiles/\n", encoding="utf-8")
    (root / ".userfiles" / "notes.md").write_text("hello", encoding="utf-8")
    (root / ".userfiles" / "image.bin").write_bytes(b"\xff\x00\x01")

    written = file_ops.write_file(room_id, ".userfiles/created.txt", "created")
    snapshot = file_ops.get_snapshot(room_id)
    by_path = {entry["path"]: entry for entry in snapshot["files"]}

    assert written["sync_ignored"] is False
    assert by_path["/.userfiles/notes.md"]["binary"] is False
    assert by_path["/.userfiles/image.bin"]["binary"] is True
    assert snapshot["previews"] == {}
    assert file_ops.read_file(room_id, ".userfiles/notes.md") == "hello"
