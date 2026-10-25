"""Stage 8 volume/generation integration tests for workspaced."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import file_ops, generation
from app.path_contract import WorkspacePathError, normalize_workspace_path


@pytest.fixture
def isolated_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage8-room"
    return room_id, tmp_path / room_id


def test_stage8_write_and_read_accept_legacy_and_canonical_paths(isolated_volume):
    room_id, root = isolated_volume
    result = file_ops.write_file(room_id, "/workspace/src/a.js", "one")
    assert result["path"] == "/src/a.js"
    assert file_ops.read_file(room_id, "src/a.js") == "one"
    assert file_ops.read_file(room_id, "/workspace/src/a.js") == "one"
    assert (root / "src" / "a.js").read_text(encoding="utf-8") == "one"


def test_stage8_file_operations_reject_traversal(isolated_volume):
    room_id, _ = isolated_volume
    with pytest.raises(ValueError):
        file_ops.write_file(room_id, "../outside.txt", "no")
    assert file_ops.read_file(room_id, "../outside.txt") is None
    with pytest.raises(WorkspacePathError):
        normalize_workspace_path("/workspace/../../outside.txt")


def test_stage8_snapshot_is_volume_first_and_excludes_service_metadata(isolated_volume):
    room_id, root = isolated_volume
    # Simulate a file created by a shell process rather than by write_file.
    (root / "src").mkdir(parents=True)
    (root / "src" / "from-bash.js").write_text("console.log(1)", encoding="utf-8")
    (root / ".legion").mkdir(parents=True)
    (root / ".legion" / "internal.log").write_text("hidden", encoding="utf-8")

    snapshot = file_ops.get_snapshot(room_id)
    paths = {entry["path"] for entry in snapshot["files"]}
    assert "/src/from-bash.js" in paths
    assert "/.legion/internal.log" not in paths
    assert snapshot["generation"]["epoch"]
    assert snapshot["generation"]["revision"] >= 0


def test_read_file_data_uri_mime_comes_from_magic_bytes_not_extension(isolated_volume):
    """A JPEG saved as ``.png`` must be labeled image/jpeg in the data URI.

    Downstream consumers (browser preview, the vision MCP validator, media
    reference resolution) trust the declared type; deriving it from the
    extension alone lets a misnamed file poison every one of them.
    """
    room_id, root = isolated_volume
    root.mkdir(parents=True)
    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01])
    png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D])
    (root / "mockup.png").write_bytes(jpeg)          # JPEG bytes, .png name
    (root / "real.png").write_bytes(png)
    (root / "note.jpg").write_bytes(png)             # PNG bytes, .jpg name
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01\x02\x03\x80\x81not utf-8")

    assert file_ops.read_file(room_id, "mockup.png").startswith("data:image/jpeg;base64,")
    assert file_ops.read_file(room_id, "real.png").startswith("data:image/png;base64,")
    assert file_ops.read_file(room_id, "note.jpg").startswith("data:image/png;base64,")
    assert file_ops.read_file(room_id, "blob.bin").startswith("data:application/octet-stream;base64,")


def test_stage8_generation_revision_is_monotonic_and_epoch_changes_on_replace(
    isolated_volume,
):
    room_id, root = isolated_volume
    first = generation.ensure_generation(room_id)
    second = generation.ensure_generation(room_id)
    assert second == first

    (root / "a.txt").write_text("a", encoding="utf-8")
    changed = generation.sync_generation(room_id)
    assert changed["epoch"] == first["epoch"]
    assert changed["revision"] == first["revision"] + 1
    assert generation.sync_generation(room_id) == changed

    replaced = generation.replace_volume_generation(room_id)
    assert replaced["epoch"] != changed["epoch"]
    assert replaced["revision"] == 0


def test_stage8_snapshot_retries_when_generation_changes_during_read(isolated_volume, monkeypatch):
    room_id, root = isolated_volume
    root.mkdir(parents=True)
    (root / "a.txt").write_text("before", encoding="utf-8")
    original = file_ops._snapshot_files
    calls = {"count": 0}

    def mutating_snapshot(current_room: str):
        calls["count"] += 1
        result = original(current_room)
        if calls["count"] == 1:
            (root / "a.txt").write_text("after", encoding="utf-8")
        return result

    monkeypatch.setattr(file_ops, "_snapshot_files", mutating_snapshot)
    snapshot = file_ops.get_snapshot(room_id)
    assert calls["count"] >= 2
    assert snapshot["consistent"] is True
    assert snapshot["files"][0]["fingerprint"]
