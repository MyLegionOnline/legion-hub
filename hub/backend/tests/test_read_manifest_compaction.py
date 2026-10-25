"""
Tests for the read-manifest surfaced in load_paused_history's compact_note.

Root cause covered: on content-heavy tasks the agent burnt through the token
budget re-reading the same files after every resume because the compaction
kept only a summary + tail. The manifest now lists every file already
inspected so the resumed session can skip the redundant reads.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.proxy_session import (
    _MANIFEST_PREVIEW_CHARS,
    _extract_read_manifest,
    _format_manifest,
    load_paused_history,
)


# --- helpers ----------------------------------------------------------------


def _anthropic_read_pair(call_id: str, path: str, result_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "read_file", "input": {"path": path}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": result_text}
            ],
        },
    ]


def _anthropic_bash_pair(call_id: str, cmd: str, result_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "bash", "input": {"command": cmd}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": result_text}
            ],
        },
    ]


def _openai_read_pair(call_id: str, path: str, result_text: str) -> list[dict[str, Any]]:
    import json as _json
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": _json.dumps({"path": path})},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result_text},
    ]


# --- pure extractor ---------------------------------------------------------


class TestExtractReadManifest:
    def test_anthropic_read_file_pair(self):
        msgs = _anthropic_read_pair("t1", "/workspace/comic1-frame1.svg", "<svg>hello</svg>")
        manifest = _extract_read_manifest(msgs)
        assert manifest == [
            {"name": "read_file", "target": "/workspace/comic1-frame1.svg",
             "preview": "<svg>hello</svg>"}
        ]

    def test_anthropic_bash_cat_captured_full_command_as_target(self):
        msgs = _anthropic_bash_pair("b1", "cat /workspace/logo.svg", "SVG BYTES")
        manifest = _extract_read_manifest(msgs)
        assert len(manifest) == 1
        assert manifest[0]["name"] == "bash:cat"
        assert manifest[0]["target"] == "cat /workspace/logo.svg"
        assert manifest[0]["preview"] == "SVG BYTES"

    def test_bash_non_read_command_is_skipped(self):
        # `echo` isn't in the read allowlist — must not appear in the manifest.
        msgs = _anthropic_bash_pair("b2", "echo hello", "hello")
        assert _extract_read_manifest(msgs) == []

    def test_openai_tool_calls_shape(self):
        msgs = _openai_read_pair("call_1", "scripts.md", "# scripts")
        manifest = _extract_read_manifest(msgs)
        assert manifest == [
            {"name": "read_file", "target": "scripts.md", "preview": "# scripts"}
        ]

    def test_preview_is_truncated_to_limit(self):
        big = "X" * (_MANIFEST_PREVIEW_CHARS + 500)
        msgs = _anthropic_read_pair("t2", "big.svg", big)
        manifest = _extract_read_manifest(msgs)
        preview = manifest[0]["preview"]
        assert preview.startswith("X" * _MANIFEST_PREVIEW_CHARS)
        assert "+500 chars]" in preview
        assert len(preview) < len(big)

    def test_duplicate_reads_collapsed_keeping_latest(self):
        # Same path read twice → one entry with the SECOND result's preview.
        msgs = (
            _anthropic_read_pair("id1", "same.svg", "FIRST_CONTENT")
            + _anthropic_read_pair("id2", "same.svg", "SECOND_CONTENT")
        )
        manifest = _extract_read_manifest(msgs)
        assert len(manifest) == 1
        assert manifest[0]["target"] == "same.svg"
        assert manifest[0]["preview"] == "SECOND_CONTENT"

    def test_multiple_files_preserve_order(self):
        msgs = (
            _anthropic_read_pair("a", "one.svg", "1")
            + _anthropic_read_pair("b", "two.svg", "2")
            + _anthropic_read_pair("c", "three.svg", "3")
        )
        targets = [e["target"] for e in _extract_read_manifest(msgs)]
        assert targets == ["one.svg", "two.svg", "three.svg"]

    def test_tool_use_without_matching_result_still_listed(self):
        # If the pair was cut off mid-flight, still record the request.
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "orphan", "name": "read_file",
                 "input": {"path": "orphan.svg"}}
            ],
        }]
        manifest = _extract_read_manifest(msgs)
        assert manifest == [
            {"name": "read_file", "target": "orphan.svg", "preview": "[no result recorded]"}
        ]


# --- format helper ----------------------------------------------------------


class TestFormatManifest:
    def test_empty_manifest_returns_empty_string(self):
        assert _format_manifest([]) == ""

    def test_format_includes_header_and_all_entries(self):
        text = _format_manifest([
            {"name": "read_file", "target": "a.svg", "preview": "AAA"},
            {"name": "bash:cat", "target": "cat b.md", "preview": "BBB"},
        ])
        assert "Files already inspected" in text
        assert "a.svg" in text
        assert "AAA" in text
        assert "cat b.md" in text
        assert "BBB" in text


# --- load_paused_history integration ---------------------------------------


def _make_row(messages: list[dict[str, Any]], summary: str | None = None) -> MagicMock:
    row = MagicMock()
    row.context_snapshot = {"messages": messages}
    row.summary = summary
    return row


def _db_returning(row: Any) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = row
    db.execute = AsyncMock(return_value=result)
    return db


class TestLoadPausedHistoryManifest:
    @pytest.mark.asyncio
    async def test_short_history_returned_verbatim_without_manifest(self):
        """Below the compaction threshold nothing is trimmed and no manifest injected."""
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        db = _db_returning(_make_row(msgs))
        out = await load_paused_history(uuid.uuid4(), db)
        assert out == msgs

    @pytest.mark.asyncio
    async def test_large_history_gets_compact_note_with_manifest(self):
        """A history full of reads gets compacted; the compact_note lists every read file."""
        task_data = {"role": "user", "content": "Build index.html from these files"}
        # 20 reads (~ enough turns + bytes to trigger compaction).
        reads: list[dict[str, Any]] = []
        for i in range(20):
            reads.extend(
                _anthropic_read_pair(f"id{i}", f"/workspace/file{i}.svg", "X" * 3000)
            )
        tail_chat = [
            {"role": "assistant", "content": "planning next step"},
            {"role": "user", "content": "continue"},
        ]
        messages = [task_data] + reads + tail_chat
        db = _db_returning(_make_row(messages, summary="Read all frames, no HTML written yet"))

        out = await load_paused_history(uuid.uuid4(), db)
        assert out is not None
        # head + note + tail structure
        assert out[0] == task_data
        note = out[1]
        assert note["role"] == "user"
        content = note["content"]
        assert "CONTEXT COMPACTED" in content
        assert "Read all frames, no HTML written yet" in content
        assert "Files already inspected" in content
        # Every read file should be surfaced by target path.
        for i in range(20):
            assert f"/workspace/file{i}.svg" in content, f"file{i}.svg missing from manifest"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_paused_session(self):
        db = _db_returning(None)
        assert await load_paused_history(uuid.uuid4(), db) is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self):
        db = _db_returning(_make_row([]))
        assert await load_paused_history(uuid.uuid4(), db) is None
