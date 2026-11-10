from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.auto_retry import (
    auto_retry_action,
    backoff_seconds,
    retry_after_seconds,
    schedule_auto_retry,
)
from app.core.loop_guard import STALL_RESTART_THRESHOLD
from app.models.task import Task, TaskState


@pytest.mark.parametrize(
    ("reason", "action"),
    [
        ("timeout", "retry"),
        ("orphaned_restart", "retry"),
        ("tool_calls", "resume"),
        ("context_window", "resume"),
        ("token_budget", None),
        ("prompt_cap", None),
        ("model_context_limit", None),
        ("repeated_tool_error", None),
        ("legacy_pipeline_stalled", None),
        ("pause_abandoned", None),
        ("dispatch_failed", None),
        ("no_chat_master", None),
        ("quota_exhausted", None),
        ("room_frozen", None),
        ("room_stopped", None),
        ("user", None),
        ("storage_full", None),
        ("disk_quota", None),
        ("image_switch", None),
        ("workspace_switch", None),
        ("blocked", None),
        ("rejected", None),
        ("cancelled", None),
        ("unknown", None),
    ],
)
def test_auto_retry_action_for_interruption(reason, action):
    assert auto_retry_action(reason, None) == action


@pytest.mark.parametrize("status", [None, 429, 500, 503, 599])
def test_transient_provider_errors_retry(status):
    assert auto_retry_action("provider_error", status) == "retry"


@pytest.mark.parametrize("status", [200, 400, 401, 402, 403, 404, 408, 409, 422])
def test_other_provider_statuses_do_not_retry(status):
    assert auto_retry_action("provider_error", status) is None


@pytest.mark.parametrize(("attempt", "base_delay"), [(1, 30), (2, 120), (3, 480), (10, 480)])
@pytest.mark.parametrize("jitter", [0.8, 1.0, 1.2])
def test_backoff_sequence_with_jitter(attempt, base_delay, jitter):
    with patch("app.core.auto_retry.random.uniform", return_value=jitter) as uniform:
        assert backoff_seconds(attempt) == pytest.approx(base_delay * jitter)

    uniform.assert_called_once_with(0.8, 1.2)


@pytest.mark.parametrize(("retry_after", "delay"), [(20, 24), (30, 30), (600, 600)])
def test_backoff_honors_retry_after_after_jitter(retry_after, delay):
    with patch("app.core.auto_retry.random.uniform", return_value=0.8):
        assert backoff_seconds(1, retry_after) == delay


def test_backoff_requires_a_positive_attempt_number():
    with pytest.raises(ValueError, match="Attempt must be at least 1"):
        backoff_seconds(0)


@pytest.mark.parametrize("value", ["invalid", "", "-1", "9" * 50])
def test_retry_after_rejects_invalid_headers(value):
    assert retry_after_seconds(value) is None


def test_retry_after_accepts_delay_and_http_date():
    assert retry_after_seconds("240") == 240
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "status", "action"),
    [
        ("provider_error", 503, "retry"),
        ("provider_error", 429, "retry"),
        ("provider_error", None, "retry"),
        ("provider_error", 401, None),
        ("timeout", None, "retry"),
        ("orphaned_restart", None, "retry"),
        ("tool_calls", None, "resume"),
        ("context_window", None, "resume"),
        ("token_budget", None, None),
        ("prompt_cap", None, None),
        ("model_context_limit", None, None),
        ("repeated_tool_error", None, None),
        ("legacy_pipeline_stalled", None, None),
        ("pause_abandoned", None, None),
        ("proxy_error", None, None),
    ],
)
async def test_schedule_auto_retry_for_interruption(reason, status, action):
    task = Task(room_id=uuid4(), title="Recover the task", state=TaskState.BLOCKED)
    db = AsyncMock()
    before = datetime.now(timezone.utc)
    with (
        patch("app.core.auto_retry.auto_retry_policy", return_value=(3, False)),
        patch("app.core.auto_retry.emit_event") as emit,
    ):
        notice = await schedule_auto_retry(task, reason, status, db, retry_after=180)

    # The caller announces the interruption first and emits the notice after.
    emit.assert_not_awaited()
    if action is None:
        assert task.auto_retry_at is None
        assert notice is None
        return
    assert task.auto_retry_reason == reason
    assert task.auto_retry_count == 0
    assert task.auto_retry_at >= before + timedelta(seconds=180 if status == 429 else 24)
    assert notice.event_type == "TASK_AUTO_RETRY_SCHEDULED"
    assert notice.metadata["action"] == action


@pytest.mark.asyncio
async def test_schedule_uses_the_unlocked_read_when_the_room_row_is_busy():
    task = Task(room_id=uuid4(), title="Recover the task", state=TaskState.BLOCKED)
    with patch("app.core.auto_retry.auto_retry_policy", return_value=(3, True)):
        notice = await schedule_auto_retry(task, "timeout", None, AsyncMock())

    assert task.auto_retry_at is not None
    assert notice.event_type == "TASK_AUTO_RETRY_SCHEDULED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "attempts", "stall_streak", "origin", "event_type"),
    [
        (None, 0, 0, None, None),
        (3, 0, STALL_RESTART_THRESHOLD, None, "TASK_AUTO_RETRY_SKIPPED"),
        (3, 0, 0, "direct_connector", None),
        (3, 3, 0, None, "TASK_AUTO_RETRY_EXHAUSTED"),
    ],
)
async def test_schedule_auto_retry_respects_policy(limit, attempts, stall_streak, origin, event_type):
    task = Task(
        room_id=uuid4(), title="Recover the task", state=TaskState.BLOCKED,
        auto_retry_count=attempts, stall_streak=stall_streak, origin=origin,
    )
    with patch("app.core.auto_retry.auto_retry_policy", return_value=(limit, False)):
        notice = await schedule_auto_retry(task, "timeout", None, AsyncMock())

    assert task.auto_retry_at is None
    if event_type is None:
        assert notice is None
        return
    assert notice.event_type == event_type
    if event_type == "TASK_AUTO_RETRY_EXHAUSTED":
        assert notice.metadata["attempt"] == attempts
    else:
        assert notice.metadata == {"reason": "timeout", "skip_reason": "no_progress"}
        assert "without progress" in notice.content


@pytest.mark.parametrize(
    ("state", "remaining_count"),
    [
        (TaskState.CANCELLED, 2),
        (TaskState.PAUSED, 2),
        (TaskState.WAITING_REVIEW, 0),
        (TaskState.COMPLETED, 0),
    ],
)
def test_task_state_change_clears_schedule(state, remaining_count):
    task = Task(
        room_id=uuid4(), title="Recover the task", state=TaskState.BLOCKED,
        auto_retry_count=2,
    )
    task.auto_retry_at = datetime.now(timezone.utc)
    task.auto_retry_reason = "timeout"

    task.state = state

    assert task.auto_retry_at is None
    assert task.auto_retry_reason is None
    assert task.auto_retry_count == remaining_count
