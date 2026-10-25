"""Focused unit tests for workspace lifecycle stage 5."""

from __future__ import annotations

import uuid
from unittest.mock import ANY, AsyncMock, patch

import pytest

from app.core.context_observations import fingerprint
from app.core.tools import _session_state
from app.core.workspace_checkpoint import (
    _validate_fingerprints,
    extract_open_tool_calls,
)
from app.models.workspace import WorkspaceCheckpoint, WorkspaceManagedProcess


def test_extract_open_tool_calls_handles_anthropic_and_openai():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "ant-done",
                    "name": "read_file",
                    "input": {"path": "a.py"},
                },
                {
                    "type": "tool_use",
                    "id": "ant-open",
                    "name": "edit_file",
                    "input": {"path": "b.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "ant-done", "content": "ok"}
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "oai-open",
                    "function": {
                        "name": "bash",
                        "arguments": '{"cmd":"pytest"}',
                    },
                }
            ],
        },
    ]

    pending = extract_open_tool_calls(messages)

    assert {(item["id"], item["name"]) for item in pending} == {
        ("ant-open", "edit_file"),
        ("oai-open", "bash"),
    }


def test_read_state_can_move_to_a_new_provider_session():
    source = str(uuid.uuid4())
    target = str(uuid.uuid4())
    try:
        _session_state.mark_read(source, "/workspace/src/app.py", "fp-1")
        saved = _session_state.export_reads(source)
        _session_state.restore_reads(target, saved)

        assert _session_state.was_read(target, "src/app.py")
        assert _session_state.read_fingerprint(target, "src/app.py") == "fp-1"
    finally:
        _session_state.clear_session(source)
        _session_state.clear_session(target)


def test_identical_tool_error_circuit_opens_after_three_failures():
    session_id = str(uuid.uuid4())
    args = {"path": "missing.py"}
    try:
        for count in range(1, _session_state.MAX_IDENTICAL_TOOL_ERRORS + 1):
            state = _session_state.record_tool_result(
                session_id,
                "read_file",
                args,
                "Error: file not found",
            )
            assert state["count"] == count
        blocked = _session_state.repeated_tool_error(session_id, "read_file", args)
        assert blocked is not None
        assert blocked["category"] == "path"

        # Changing the arguments is not blocked by the previous signature.
        assert (
            _session_state.repeated_tool_error(
                session_id, "read_file", {"path": "other.py"}
            )
            is None
        )
    finally:
        _session_state.clear_session(session_id)


@pytest.mark.asyncio
async def test_fingerprint_validation_keeps_unchanged_and_invalidates_changed():
    room_id = uuid.uuid4()
    manifest = {
        "same.txt": {"fingerprint": fingerprint("same")},
        "/workspace/changed.txt": {"fingerprint": fingerprint("before")},
    }

    async def fake_read(_room_id: str, path: str):
        return {"same.txt": "same", "changed.txt": "after"}[path]

    invalidate = AsyncMock()
    with (
        patch(
            "app.core.workspaced_client.read_workspace_file",
            side_effect=fake_read,
        ),
        patch(
            "app.core.workspace_checkpoint.invalidate_path",
            invalidate,
        ),
    ):
        valid, stale = await _validate_fingerprints(
            room_id,
            manifest,
            AsyncMock(),
        )

    assert valid == {"same.txt": fingerprint("same")}
    assert [item["path"] for item in stale] == ["changed.txt"]
    invalidate.assert_awaited_once_with(
        ANY,
        room_id,
        "changed.txt",
        new_fingerprint=fingerprint("after"),
    )


def test_stage5_models_separate_managed_processes_from_arbitrary_background():
    workspace_id = uuid.uuid4()
    room_id = uuid.uuid4()
    process = WorkspaceManagedProcess(
        workspace_id=workspace_id,
        name="dev",
        command="npm run dev",
        ports=[3000],
        resume_policy="restart",
    )
    checkpoint = WorkspaceCheckpoint(
        workspace_id=workspace_id,
        room_id=room_id,
        reason="workspace_switch",
        pause_reason="workspace_switch",
        payload={"managed_processes": [{"name": "dev"}]},
    )

    assert process.resume_policy == "restart"
    assert process.ports == [3000]
    assert checkpoint.status == "created"
    assert checkpoint.payload["managed_processes"][0]["name"] == "dev"
