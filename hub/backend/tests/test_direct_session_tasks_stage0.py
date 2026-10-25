"""Stage 0 of internal design notes — data model + flow config.

Covers the foundation the later stages build on:
  * ``Task.origin`` / ``Task.origin_session_id`` exist, default to NULL, and
    reach the API through ``TaskOut``;
  * ``get_flow_config`` reads ``direct_session_mode`` /
    ``direct_session_auto_merge``, defaults them to the same behaviour a
    dispatched task gets, and clamps unknown values instead of trusting them;
  * the migrations that add the two columns are registered.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.core import room_git_flow
from app.db.session import _MIGRATIONS
from app.models.task import TASK_ORIGIN_DIRECT_SESSION, Task
from app.schemas.task import TaskOut


# ── flow config ─────────────────────────────────────────────────────────────

def test_direct_session_defaults_match_dispatched_task_rules():
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={}))
    assert cfg.direct_session_mode == "gate1"
    assert cfg.direct_session_auto_merge == "risk"


def test_direct_session_overrides_are_read():
    room = SimpleNamespace(git_config={
        "direct_session_mode": "branch_only",
        "direct_session_auto_merge": "normal_too",
    })
    cfg = room_git_flow.get_flow_config(room)
    assert cfg.direct_session_mode == "branch_only"
    assert cfg.direct_session_auto_merge == "normal_too"


def test_direct_session_off_is_a_valid_mode():
    room = SimpleNamespace(git_config={"direct_session_mode": "off"})
    assert room_git_flow.get_flow_config(room).direct_session_mode == "off"


def test_unknown_direct_session_values_fall_back_to_default():
    room = SimpleNamespace(git_config={
        "direct_session_mode": "yolo",
        "direct_session_auto_merge": "always",
    })
    cfg = room_git_flow.get_flow_config(room)
    assert cfg.direct_session_mode == "gate1"
    assert cfg.direct_session_auto_merge == "risk"


def test_blank_direct_session_values_fall_back_to_default():
    room = SimpleNamespace(git_config={
        "direct_session_mode": "  ",
        "direct_session_auto_merge": None,
    })
    cfg = room_git_flow.get_flow_config(room)
    assert cfg.direct_session_mode == "gate1"
    assert cfg.direct_session_auto_merge == "risk"


# ── Task.origin ─────────────────────────────────────────────────────────────

def test_task_origin_defaults_to_null():
    task = Task(room_id=uuid4(), title="t")
    assert task.origin is None
    assert task.origin_session_id is None


def test_task_origin_round_trips_into_task_out():
    session_id = uuid4()
    task = Task(
        room_id=uuid4(),
        title="Padding in the article",
        origin=TASK_ORIGIN_DIRECT_SESSION,
        origin_session_id=session_id,
    )
    out = TaskOut.model_validate(task)
    assert out.origin == "direct_session"
    assert out.origin_session_id == session_id


def test_task_out_keeps_origin_null_for_ordinary_tasks():
    out = TaskOut.model_validate(Task(room_id=uuid4(), title="t"))
    assert out.origin is None
    assert out.origin_session_id is None


# ── migrations ──────────────────────────────────────────────────────────────

def test_origin_columns_have_idempotent_migrations():
    ddl = " ".join(_MIGRATIONS)
    assert "ADD COLUMN IF NOT EXISTS origin VARCHAR" in ddl
    assert "ADD COLUMN IF NOT EXISTS origin_session_id UUID" in ddl
    assert "ix_task_origin_session" in ddl
