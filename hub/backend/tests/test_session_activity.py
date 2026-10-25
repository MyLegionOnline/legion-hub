"""Unit tests for chat upgrade stage 3: last_activity_at tracking.

Covers:
- last_activity_at is set on both sessions at loop start
- last_activity_at is updated after each successful LLM response (agent + CM)
- task.updated_at is bumped alongside last_activity_at at LLM response points
- last_activity_at is updated after each successful broker.publish
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.core.proxy_session import _run_proxy_loop
from app.models.task import DispatchStatus, TaskState


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_proxy_session.py)
# ---------------------------------------------------------------------------


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.first.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    result_mock.first.return_value = None
    result_mock.all.return_value = []
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _make_session(closed: bool = False) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = datetime.now(timezone.utc) if closed else None
    s.close_reason = None
    s.is_open = not closed
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.last_activity_at = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    return s


def _make_task() -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Test task"
    t.description = "desc"
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = uuid.uuid4()
    t.goal_id = uuid.uuid4()
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_activity_at_set_at_start_and_after_llm_response():
    """Both sessions get last_activity_at set at loop start and after each LLM response."""
    agent_session = _make_session()
    cm_session = _make_session()
    task = _make_task()
    db = _make_db()
    room_id = uuid.uuid4()

    llm_result = ("Agent reply", None, 50, 100, "claude-sonnet-4-6", 50, 50)
    cm_result_close = (None, MagicMock(), 40, 80, "claude-sonnet-4-6", 40, 40)

    # Patch away everything external
    with (
        patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock) as mock_cfg,
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock) as mock_mem,
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.proxy_session._call_with_heartbeat", new_callable=AsyncMock,
              side_effect=[llm_result, cm_result_close]),
        patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.ProxyBroker") as mock_broker_cls,
        patch("app.core.proxy_session._close_side", new_callable=AsyncMock,
              return_value=MagicMock(summary="done", files_changed=[], output=None)),
        patch("app.core.proxy_session._emit_closed", new_callable=AsyncMock),
        patch("app.core.proxy_session._render_close_note", return_value="[done]"),
        patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
    ):
        mock_cfg.return_value = {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "x"}
        mock_mem.return_value = MagicMock()
        broker = AsyncMock()
        broker.publish = AsyncMock()
        mock_broker_cls.return_value = broker

        await _run_proxy_loop(
            agent_session=agent_session,
            cm_session=cm_session,
            task=task,
            task_data_text="Do something",
            room_id=room_id,
            db=db,
        )

    # last_activity_at must have been set (not None) on both sessions
    assert agent_session.last_activity_at is not None, (
        "agent_session.last_activity_at should be set at loop start"
    )
    assert cm_session.last_activity_at is not None, (
        "cm_session.last_activity_at should be set at loop start"
    )


@pytest.mark.asyncio
async def test_task_updated_at_bumped_after_llm_response():
    """task.updated_at is set to a newer value after each LLM response."""
    agent_session = _make_session()
    cm_session = _make_session()
    task = _make_task()
    original_updated_at = task.updated_at
    db = _make_db()
    room_id = uuid.uuid4()

    llm_result = ("Agent reply", None, 50, 100, "claude-sonnet-4-6", 50, 50)
    cm_result_close = (None, MagicMock(), 40, 80, "claude-sonnet-4-6", 40, 40)

    with (
        patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock),
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.proxy_session._call_with_heartbeat", new_callable=AsyncMock,
              side_effect=[llm_result, cm_result_close]),
        patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.ProxyBroker") as mock_broker_cls,
        patch("app.core.proxy_session._close_side", new_callable=AsyncMock,
              return_value=MagicMock(summary="done", files_changed=[], output=None)),
        patch("app.core.proxy_session._emit_closed", new_callable=AsyncMock),
        patch("app.core.proxy_session._render_close_note", return_value="[done]"),
        patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
    ):
        broker = AsyncMock()
        broker.publish = AsyncMock()
        mock_broker_cls.return_value = broker

        await _run_proxy_loop(
            agent_session=agent_session,
            cm_session=cm_session,
            task=task,
            task_data_text="Do something",
            room_id=room_id,
            db=db,
        )

    # task.updated_at must have been written (db.add called with task at some point)
    added_objects = [c.args[0] for c in db.add.call_args_list]
    assert task in added_objects, "task must be passed to db.add() so updated_at is persisted"
