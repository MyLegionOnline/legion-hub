"""
Tests for the durable, DB-backed read manifest that survives repeated
tool_limit resumes.

Root cause covered: load_paused_history's compact_note carries a manifest of
already-read files, but that manifest is derived by re-scanning raw chat
messages. Once a resumed session is itself compacted a second time, the
earlier compact_note (plain text, not structured tool_use/tool_result blocks)
is invisible to that scan, so the list of already-read files is silently
dropped and the agent re-reads files it already inspected in the first
session. build_task_read_manifest sidesteps this by reading
context_state.read_ledger directly off every past session row for the task,
independent of how the chat history was trimmed.
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.context_observations import (
    build_task_read_manifest,
    format_read_manifest_note,
)


def _session_row(context_state: dict[str, Any] | None) -> MagicMock:
    row = MagicMock()
    row.context_state = context_state
    return row


def _db_returning(rows: list[Any]) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    db.execute = AsyncMock(return_value=result)
    return db


class TestBuildTaskReadManifest:
    @pytest.mark.asyncio
    async def test_no_sessions_returns_empty(self):
        db = _db_returning([])
        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        assert manifest == []

    @pytest.mark.asyncio
    async def test_single_session_ledger_surfaced(self):
        row = _session_row({
            "read_ledger": {
                "blog/models.py": {
                    "fingerprint": "fp1",
                    "ranges": [[1, 120]],
                    "observation_refs": ["obs-1"],
                }
            }
        })
        db = _db_returning([row])
        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        assert manifest == [
            {
                "path": "blog/models.py",
                "fingerprint": "fp1",
                "ranges": [[1, 120]],
                "observation_refs": ["obs-1"],
            }
        ]

    @pytest.mark.asyncio
    async def test_survives_two_resume_cycles_across_three_sessions(self):
        """Simulates the exact failure: session 1 reads files, hits tool_limit;
        session 2 (after resume) reads more files, hits tool_limit again;
        session 3 (after a second resume) must still see everything from
        session 1 AND session 2, even though neither session's raw chat
        history is consulted here."""
        session1 = _session_row({
            "read_ledger": {
                "blog/models.py": {"fingerprint": "fp1", "ranges": [[1, 80]], "observation_refs": ["obs-1"]},
                "blog/views.py": {"fingerprint": "fp2", "ranges": [[1, 50]], "observation_refs": ["obs-2"]},
            }
        })
        session2 = _session_row({
            "read_ledger": {
                "config/settings.py": {"fingerprint": "fp3", "ranges": [[1, 200]], "observation_refs": ["obs-3"]},
            }
        })
        session3 = _session_row(None)  # session 3 hasn't read anything yet
        db = _db_returning([session1, session2, session3])

        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        paths = {entry["path"] for entry in manifest}
        assert paths == {"blog/models.py", "blog/views.py", "config/settings.py"}

    @pytest.mark.asyncio
    async def test_overlapping_ranges_merge_same_fingerprint(self):
        session1 = _session_row({
            "read_ledger": {"a.py": {"fingerprint": "fp1", "ranges": [[1, 50]], "observation_refs": ["r1"]}}
        })
        session2 = _session_row({
            "read_ledger": {"a.py": {"fingerprint": "fp1", "ranges": [[40, 100]], "observation_refs": ["r2"]}}
        })
        db = _db_returning([session1, session2])

        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        assert manifest == [
            {"path": "a.py", "fingerprint": "fp1", "ranges": [[1, 100]], "observation_refs": ["r1", "r2"]}
        ]

    @pytest.mark.asyncio
    async def test_fingerprint_change_resets_ranges(self):
        """A later session re-read the file after it changed on disk — old
        ranges described stale content and must not be merged with the new
        read as if they cover the same bytes."""
        session1 = _session_row({
            "read_ledger": {"a.py": {"fingerprint": "fp-old", "ranges": [[1, 500]], "observation_refs": ["r1"]}}
        })
        session2 = _session_row({
            "read_ledger": {"a.py": {"fingerprint": "fp-new", "ranges": [[1, 20]], "observation_refs": ["r2"]}}
        })
        db = _db_returning([session1, session2])

        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        assert manifest == [
            {"path": "a.py", "fingerprint": "fp-new", "ranges": [[1, 20]], "observation_refs": ["r1", "r2"]}
        ]

    @pytest.mark.asyncio
    async def test_malformed_context_state_ignored(self):
        rows = [_session_row(None), _session_row({"read_ledger": "not-a-dict"}), _session_row({})]
        db = _db_returning(rows)
        manifest = await build_task_read_manifest(db, task_id=uuid.uuid4())
        assert manifest == []


class TestFormatReadManifestNote:
    def test_empty_manifest_returns_empty_string(self):
        assert format_read_manifest_note([]) == ""

    def test_includes_path_ranges_and_observation_hint(self):
        text = format_read_manifest_note([
            {"path": "blog/models.py", "fingerprint": "fp1", "ranges": [[1, 80]], "observation_refs": ["obs-1"]},
        ])
        assert "blog/models.py" in text
        assert "1-80" in text
        assert "observation_read(ref='obs-1')" in text
        assert "durable" in text.lower()

    def test_entry_without_refs_has_no_observation_hint(self):
        text = format_read_manifest_note([
            {"path": "x.py", "fingerprint": "fp1", "ranges": [[1, 10]], "observation_refs": []},
        ])
        assert "x.py" in text
        assert "observation_read" not in text
