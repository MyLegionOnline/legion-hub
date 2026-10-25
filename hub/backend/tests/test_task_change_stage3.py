"""API contract tests for the task-change confirmation flow (stage 3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.chat import (
    confirm_plan_session,
    confirm_task_change_session,
    reject_task_change_session,
)
from app.core.task_change import TaskChangeConflictError
from app.models.message import PersistedEventType
from app.schemas.task_change import TaskChangeActionRequest


def _result(value):
    result = MagicMock()
    result.scalars.return_value.first.return_value = value
    return result


def _session_with_pending(case, *, open_: bool = True):
    session = SimpleNamespace(
        id=uuid4(),
        room_id=case.room_id,
        kind="user_cm_p1",
        goal_id=case.goal_id,
        is_open=open_,
        context_snapshot={
            "messages": [{"role": "assistant", "content": "proposal"}],
            "keep_me": True,
            "pending_task_change": case.proposal.model_dump(mode="json"),
        },
    )
    return session


def _db_for_session(session, *, member=True, extra_results=()):
    membership = object() if member else None
    return SimpleNamespace(
        execute=AsyncMock(
            side_effect=[_result(session), _result(membership), *extra_results]
        ),
        add=MagicMock(),
        flush=AsyncMock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_confirm_task_change_applies_stored_proposal_without_llm(task_change_e2e_case):
    case = task_change_e2e_case
    session = _session_with_pending(case)
    db = _db_for_session(session)
    user = SimpleNamespace(id=uuid4())
    applied = {
        "proposal_id": str(case.proposal.proposal_id),
        "cancelled_tasks": [{"task_id": str(case.source.id)}],
        "created_tasks": [{"task_id": str(uuid4()), "ref": "og-image"}],
        "dependency_rewrites": [],
        "dispatch": {},
    }

    with patch(
        "app.core.task_change.apply_task_change_proposal",
        new_callable=AsyncMock,
        return_value=applied,
    ) as apply:
        result = await confirm_task_change_session(
            session.id,
            TaskChangeActionRequest(proposal_id=case.proposal.proposal_id),
            db=db,
            current_user=user,
        )

    assert result["status"] == "confirmed"
    assert result["cancelled_tasks"] == applied["cancelled_tasks"]
    apply.assert_awaited_once()
    assert apply.await_args.args[0] is session
    assert apply.await_args.args[1].proposal_id == case.proposal.proposal_id


@pytest.mark.asyncio
async def test_confirm_task_change_rejects_wrong_proposal_id_without_apply(task_change_e2e_case):
    case = task_change_e2e_case
    session = _session_with_pending(case)
    db = _db_for_session(session)
    user = SimpleNamespace(id=uuid4())

    with patch(
        "app.core.task_change.apply_task_change_proposal",
        new_callable=AsyncMock,
    ) as apply:
        with pytest.raises(HTTPException) as caught:
            await confirm_task_change_session(
                session.id,
                TaskChangeActionRequest(proposal_id=uuid4()),
                db=db,
                current_user=user,
            )

    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "proposal_id_mismatch"
    apply.assert_not_awaited()
    assert "pending_task_change" in session.context_snapshot


@pytest.mark.asyncio
async def test_reject_task_change_clears_only_pending_and_emits_event(task_change_e2e_case):
    case = task_change_e2e_case
    session = _session_with_pending(case)
    db = _db_for_session(session)
    user = SimpleNamespace(id=uuid4())

    with patch("app.core.events.emit_event", new_callable=AsyncMock) as emit:
        result = await reject_task_change_session(
            session.id,
            TaskChangeActionRequest(proposal_id=case.proposal.proposal_id),
            db=db,
            current_user=user,
        )

    assert result == {
        "status": "rejected",
        "proposal_id": str(case.proposal.proposal_id),
    }
    assert "pending_task_change" not in session.context_snapshot
    assert session.context_snapshot["messages"]
    assert session.context_snapshot["keep_me"] is True
    assert emit.await_args.kwargs["event_type"] is PersistedEventType.TASK_CHANGE_REJECTED
    db.add.assert_called_once_with(session)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_task_change_returns_409_and_emits_conflict(task_change_e2e_case):
    case = task_change_e2e_case
    session = _session_with_pending(case)
    current_task = SimpleNamespace(
        id=case.source.id,
        title=case.source.title,
        state="in_progress",
        updated_at=case.source.updated_at,
    )
    db = _db_for_session(session, extra_results=(_result(current_task),))
    user = SimpleNamespace(id=uuid4())
    conflict = TaskChangeConflictError(
        "source task changed",
        task_id=case.source.id,
        actual_updated_at=current_task.updated_at,
        actual_state="in_progress",
    )

    with (
        patch(
            "app.core.task_change.apply_task_change_proposal",
            new_callable=AsyncMock,
            side_effect=conflict,
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock) as emit,
    ):
        with pytest.raises(HTTPException) as caught:
            await confirm_task_change_session(
                session.id,
                TaskChangeActionRequest(proposal_id=case.proposal.proposal_id),
                db=db,
                current_user=user,
            )

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "task_change_conflict"
    assert caught.value.detail["task"]["task_id"] == str(case.source.id)
    assert emit.await_args.kwargs["event_type"] is PersistedEventType.TASK_CHANGE_CONFLICT
    assert "pending_task_change" in session.context_snapshot


@pytest.mark.asyncio
async def test_confirm_task_change_retry_returns_durable_result(task_change_e2e_case):
    case = task_change_e2e_case
    applied = {
        "proposal_id": str(case.proposal.proposal_id),
        "cancelled_tasks": [],
        "created_tasks": [],
        "dependency_rewrites": [],
        "dispatch": {},
    }
    session = SimpleNamespace(
        id=uuid4(),
        room_id=case.room_id,
        is_open=True,
        context_snapshot={"last_applied_task_change": applied},
    )
    db = _db_for_session(session)
    user = SimpleNamespace(id=uuid4())

    with patch(
        "app.core.task_change.apply_task_change_proposal",
        new_callable=AsyncMock,
    ) as apply:
        result = await confirm_task_change_session(
            session.id,
            TaskChangeActionRequest(proposal_id=case.proposal.proposal_id),
            db=db,
            current_user=user,
        )

    assert result == {"status": "confirmed", **applied}
    apply.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_change_actions_require_room_membership(task_change_e2e_case):
    case = task_change_e2e_case
    session = _session_with_pending(case)
    db = _db_for_session(session, member=False)
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(HTTPException) as caught:
        await reject_task_change_session(
            session.id,
            TaskChangeActionRequest(proposal_id=case.proposal.proposal_id),
            db=db,
            current_user=user,
        )

    assert caught.value.status_code == 403
    assert "pending_task_change" in session.context_snapshot
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_plan_requires_room_membership():
    session = SimpleNamespace(
        id=uuid4(),
        room_id=uuid4(),
        is_open=True,
        context_snapshot={
            "pending_plan": {"goal_title": "SEO", "goal_description": "desc"}
        },
    )
    db = _db_for_session(session, member=False)
    user = SimpleNamespace(id=uuid4(), full_name="User", email="u@example.com")

    with pytest.raises(HTTPException) as caught:
        await confirm_plan_session(session.id, db=db, current_user=user)

    assert caught.value.status_code == 403
