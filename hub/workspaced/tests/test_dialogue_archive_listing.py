"""internal design notes Stage 8 — list_dialogue_archive_files."""
from __future__ import annotations

from pathlib import Path

import pytest

from app import file_ops, generation


@pytest.fixture
def isolated_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "pd-room"
    root = tmp_path / room_id
    (root / ".legion" / "plans" / "goal-1-attachments").mkdir(parents=True)
    (root / ".legion" / "drafts").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / ".legion" / "plans" / "goal-1.md").write_text("x", encoding="utf-8")
    (root / ".legion" / "plans" / "goal-1-attachments" / "01-nginx.conf").write_text(
        "y", encoding="utf-8"
    )
    (root / ".legion" / "drafts" / "sess-a.md").write_text("z", encoding="utf-8")
    (root / ".legion" / "git.lock").write_text("", encoding="utf-8")
    (root / "src" / "main.py").write_text("code", encoding="utf-8")
    return room_id


def test_lists_only_plans_and_drafts(isolated_volume):
    paths = {e["path"] for e in file_ops.list_dialogue_archive_files(isolated_volume)}
    assert paths == {
        ".legion/plans/goal-1.md",
        ".legion/plans/goal-1-attachments/01-nginx.conf",
        ".legion/drafts/sess-a.md",
    }


def test_entries_carry_mtime(isolated_volume):
    entries = file_ops.list_dialogue_archive_files(isolated_volume)
    assert all(isinstance(e["mtime"], float) and e["mtime"] > 0 for e in entries)


def test_missing_dirs_return_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)
    assert file_ops.list_dialogue_archive_files("nonexistent-room") == []
