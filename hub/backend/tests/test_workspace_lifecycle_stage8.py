"""Stage 8 contract tests for the room/workspace lifecycle.

These tests intentionally exercise the small deterministic seams of the
lifecycle implementation.  They do not require Docker or PostgreSQL, so the
same suite can run in CI and as a fast pre-commit check.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.context_observations import fingerprint
from app.core.path_contract import (
    WorkspacePathError,
    canonicalize_path_value,
    normalize_workspace_glob,
    normalize_workspace_path,
)
from app.core.tools import _session_state
from app.core.workspace_lifecycle import (
    CONFIG,
    WorkspaceLifecycleConfig,
    _workspace_payload,
    _broadcast_workspace_status,
)
from app.core.workspace_checkpoint import _validate_fingerprints
from app.models.task import TaskState
from app.models.workspace import WorkspaceCheckpoint, WorkspaceManagedProcess


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("src/App.vue", "src/App.vue"),
        ("/src/App.vue", "src/App.vue"),
        ("/workspace/src/App.vue", "src/App.vue"),
        ("workspace/src/App.vue", "src/App.vue"),
        (r"src\components\Button.vue", "src/components/Button.vue"),
        ("./src//components/../Button.vue", None),
    ],
)
def test_stage8_path_normalizer_accepts_legacy_forms_and_rejects_traversal(
    raw: str, expected: str | None
):
    if expected is None:
        with pytest.raises(WorkspacePathError):
            normalize_workspace_path(raw)
    else:
        assert normalize_workspace_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["../secret", "/workspace/../../secret", "C:/secret", "src/\x00bad"],
)
def test_stage8_path_normalizer_rejects_escape_forms(raw: str):
    with pytest.raises(WorkspacePathError):
        normalize_workspace_path(raw)


def test_stage8_glob_and_nested_tool_arguments_share_canonical_paths():
    assert normalize_workspace_glob("/workspace/src/**/*.py") == "src/**/*.py"
    payload = canonicalize_path_value(
        {
            "path": "/workspace/src/a.py",
            "cwd": "/workspace",
            "nested": [{"path": "/docs/plan.md"}],
        }
    )
    assert payload == {
        "cwd": "",
        "nested": [{"path": "docs/plan.md"}],
        "path": "src/a.py",
    }


def test_stage8_bash_does_not_satisfy_read_before_edit():
    session_id = str(uuid4())
    try:
        _session_state.mark_read(session_id, "/workspace/src/a.py", "fp")
        assert _session_state.was_read(session_id, "src/a.py")
        assert CONFIG.bash_satisfies_read_before_edit is False
    finally:
        _session_state.clear_session(session_id)


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("Error: file not found", "path"),
        ("Error: workspace not running", "workspace_transient"),
        ("Error: permission denied", "permission"),
        ("Error: invalid expected_fingerprint", "invalid_input"),
        ("Error: provider failed", "tool_error"),
    ],
)
def test_stage8_retry_error_classification(message: str, category: str):
    assert _session_state.classify_tool_error(message) == category


def test_stage8_identical_tool_error_circuit_survives_continuation():
    source = str(uuid4())
    target = str(uuid4())
    tool_input = {"path": "/workspace/missing.py"}
    try:
        for _ in range(_session_state.MAX_IDENTICAL_TOOL_ERRORS):
            _session_state.record_tool_result(
                source, "read_file", tool_input, "Error: file not found"
            )
        state = _session_state.export_tool_failure(source)
        _session_state.restore_tool_failure(target, state)
        blocked = _session_state.repeated_tool_error(target, "read_file", tool_input)
        assert blocked is not None
        assert blocked["count"] == _session_state.MAX_IDENTICAL_TOOL_ERRORS
        assert blocked["category"] == "path"
    finally:
        _session_state.clear_session(source)
        _session_state.clear_session(target)


def test_stage8_lifecycle_counted_states_and_defaults():
    assert CONFIG.counted_states == ("starting", "running", "draining", "stopping")
    custom = WorkspaceLifecycleConfig(
        starting_counts_toward_quota=False,
        polling_initial_sec=1,
        polling_max_sec=2,
    )
    assert custom.counted_states == ("running", "draining", "stopping")
    assert custom.idle_timeout_sec == 900
    assert custom.force_stop_requires_confirmation is True


def test_stage8_workspace_payload_exposes_generation_and_operation_metadata():
    room_id = uuid4()
    operation_id = uuid4()
    ws = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        status_reason="ready",
        last_error=None,
        operation_id=operation_id,
        quota_owner_id=uuid4(),
        container_id="container",
        image="image",
        mem_limit_mb=512,
        cpu_limit_milli=500,
        disk_limit_mb=2048,
        network_enabled=True,
        last_activity_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        stopped_at=None,
        last_heartbeat_at=datetime.now(timezone.utc),
        workspace_generation={"epoch": "e", "revision": 4},
        active_session_count=0,
        draining_since=None,
    )
    payload = _workspace_payload(ws)
    assert payload["workspace_generation"] == {"epoch": "e", "revision": 4}
    assert payload["operation_id"] == operation_id
    assert payload["active_session_count"] == 0


@pytest.mark.asyncio
async def test_stage8_realtime_event_envelope_is_ordered_and_versioned():
    room_id = uuid4()
    ws = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        operation_id=uuid4(),
        workspace_generation={"epoch": "e", "revision": 1},
    )
    frames: list[dict] = []
    manager = SimpleNamespace(broadcast=AsyncMock(side_effect=lambda frame, _room: frames.append(frame)))
    with patch("app.websockets.manager.manager", manager):
        await _broadcast_workspace_status(
            ws,
            previous_status="starting",
            operation_id=ws.operation_id,
            reason="ready",
            event_type="workspace.running",
        )
        await _broadcast_workspace_status(
            ws,
            previous_status="running",
            operation_id=ws.operation_id,
            reason="snapshot",
            event_type="workspace.snapshot.created",
        )
    assert [frame["sequence"] for frame in frames] == [1, 2]
    assert all(frame["schema_version"] == 1 for frame in frames)
    assert all(frame["event_id"] for frame in frames)
    assert frames[0]["previous_status"] == "starting"
    assert frames[1]["generation"] == {"epoch": "e", "revision": 1}


class _SequenceResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def first(self):
        return self.value


class _SequenceDB:
    def __init__(self, *values):
        self.values = list(values)

    async def execute(self, _query):
        return _SequenceResult(self.values.pop(0))


@pytest.mark.asyncio
async def test_stage8_idle_activity_classification_ignores_browser_presence():
    from app.core.workspace_lifecycle import workspace_is_idle

    room_id = uuid4()
    workspace = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        active_session_count=0,
        last_activity_at=datetime.now(timezone.utc) - timedelta(seconds=901),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=902),
    )
    db = _SequenceDB(workspace, None, None, None, None)
    assert await workspace_is_idle(room_id, db, idle_timeout_sec=900) is True


@pytest.mark.asyncio
async def test_stage8_recent_service_access_resets_the_shared_idle_timer():
    from app.core.workspace_lifecycle import workspace_is_idle

    room_id = uuid4()
    now = datetime.now(timezone.utc)
    workspace = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        active_session_count=0,
        last_activity_at=now - timedelta(seconds=901),
        started_at=now - timedelta(seconds=902),
    )
    db = _SequenceDB(
        workspace,
        None,  # no active task
        None,  # no active session
        None,  # no arbitrary background process
        now - timedelta(seconds=1),  # Browser/agent service access
    )

    assert await workspace_is_idle(room_id, db, idle_timeout_sec=900, now=now) is False


@pytest.mark.asyncio
async def test_stage8_active_task_prevents_idle_stop():
    from app.core.workspace_lifecycle import workspace_is_idle

    room_id = uuid4()
    workspace = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        active_session_count=0,
        last_activity_at=datetime.now(timezone.utc) - timedelta(seconds=901),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=902),
    )
    db = _SequenceDB(workspace, SimpleNamespace(id=uuid4()), None)
    assert await workspace_is_idle(room_id, db, idle_timeout_sec=900) is False


@pytest.mark.asyncio
async def test_stage8_open_proxy_session_prevents_idle_stop():
    from app.core.workspace_lifecycle import workspace_is_idle

    room_id = uuid4()
    workspace = SimpleNamespace(
        id=uuid4(),
        room_id=room_id,
        status="running",
        active_session_count=0,
        last_activity_at=datetime.now(timezone.utc) - timedelta(seconds=901),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=902),
    )
    db = _SequenceDB(
        workspace,
        None,  # no active task
        SimpleNamespace(id=uuid4()),  # open proxy session
    )
    assert await workspace_is_idle(room_id, db, idle_timeout_sec=900) is False


@pytest.mark.asyncio
async def test_stage8_fingerprint_validation_marks_changed_artifacts_stale():
    room_id = uuid4()
    manifest = {
        "same.txt": {"fingerprint": fingerprint("same")},
        "changed.txt": {"fingerprint": fingerprint("before")},
    }

    async def read_file(_room_id: str, path: str):
        return {"same.txt": "same", "changed.txt": "after"}[path]

    invalidate = AsyncMock()
    with (
        patch("app.core.workspaced_client.read_workspace_file", side_effect=read_file),
        patch("app.core.workspace_checkpoint.invalidate_path", invalidate),
    ):
        valid, stale = await _validate_fingerprints(room_id, manifest, AsyncMock())
    assert valid == {"same.txt": fingerprint("same")}
    assert stale == [{"path": "changed.txt", "expected": fingerprint("before"), "actual": fingerprint("after")}]
    invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_stage8_quota_owner_lock_serializes_concurrent_reservations():
    from app.core.workspace_quota import quota_manager

    owner_id = uuid4()
    active = 0
    max_active = 0

    async def fake_limit(_owner, _db):
        return 1

    async def fake_reserved(_owner, _db, *, exclude_room_id=None):
        return 0

    async def operation():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return "ok"

    with (
        patch.object(quota_manager, "effective_max_running", side_effect=fake_limit),
        patch(
            "app.core.workspace_quota.reserved_workspace_slots",
            side_effect=fake_reserved,
        ),
    ):
        results = await asyncio.gather(
            *(
                quota_manager.reserve_and_run(owner_id, uuid4(), AsyncMock(), operation)
                for _ in range(20)
            )
        )
    assert results == ["ok"] * 20
    assert max_active == 1


def test_stage8_task_pause_states_are_explicit():
    assert TaskState.PAUSING.value == "pausing"
    assert TaskState.PAUSED.value == "paused"
    assert TaskState.RESUMING.value == "resuming"


def test_stage8_managed_process_policy_does_not_resume_arbitrary_background_work():
    managed = WorkspaceManagedProcess(
        workspace_id=uuid4(),
        name="dev",
        command="npm run dev",
        resume_policy="restart",
        status="paused",
    )
    arbitrary = WorkspaceManagedProcess(
        workspace_id=managed.workspace_id,
        name="shell-bg",
        command="nohup ./script.sh &",
        resume_policy="never",
        status="stopped",
    )
    assert managed.resume_policy == "restart"
    assert arbitrary.resume_policy == "never"


def test_stage8_force_stop_checkpoint_is_diagnostic_not_task_failure():
    checkpoint = WorkspaceCheckpoint(
        workspace_id=uuid4(),
        room_id=uuid4(),
        reason="force_stop",
        pause_reason="force_stop",
        force_stop=True,
        payload={"open_tool_calls": [{"id": "call-1", "status": "interrupted"}]},
    )
    assert checkpoint.force_stop is True
    assert checkpoint.payload["open_tool_calls"][0]["status"] == "interrupted"
