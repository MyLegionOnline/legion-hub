"""Unit tests for internal design notes, stage 3.

Covers the deterministic code around the checkpoint primitive:

- emit_checkpoint_request: CHECKPOINT_REQUEST metadata contract
  ({task_id, checkpoint_kind, checkpoint_payload}).
- publish_unblocked_tasks: the single interception point between
  try_unblock_tasks and dispatch — checkpoint tasks are published as
  CHECKPOINT_REQUEST and NEVER dispatched; regular tasks get TASK_UNBLOCKED
  and, when assigned, go through request_task_dispatch.
- resolve_task_checkpoint endpoint: WAITING_APPROVAL-only, checkpoint-only,
  choice validation for select_variant, checkpoint_choice persisted,
  COMPLETED/REJECTED transitions, the choice appended to dependent task
  descriptions, try_unblock_tasks + publish on approve, and
  TASK_DEPENDENCY_FAILED for dependents on reject.
- change_task_state: a manual Approve on a checkpoint lands it in COMPLETED
  (TaskPanel Approve) also unblocks dependents.
- TaskOut exposes the checkpoint fields for the stage-4 card.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.tasks import (
    _format_checkpoint_choice_note,
    change_task_state,
    resolve_task_checkpoint,
)
from app.core.task_deps import emit_checkpoint_request, publish_unblocked_tasks
from app.models.task import DispatchStatus, Task, TaskState
from app.schemas.task import CheckpointResolve, TaskOut, TaskStateChange

NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)

VARIANTS = [
    {"id": "a", "label": "Variant A", "paths": ["/project/images/a1.png"]},
    {"id": "b", "label": "Variant B", "paths": ["/project/images/b1.png"], "note": "darker"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkpoint_task(**overrides) -> Task:
    defaults = dict(
        room_id=uuid.uuid4(),
        title="Design variant agreement",
        description="",
        state=TaskState.WAITING_APPROVAL,
        dispatch_status=DispatchStatus.READY,
        assigned_agent_id=None,
        checkpoint_kind="select_variant",
        checkpoint_payload={"prompt": "Choose a variant", "variants": VARIANTS},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _regular_task(**overrides) -> Task:
    defaults = dict(
        room_id=uuid.uuid4(),
        title="CSS foundation",
        description="Lay out the foundation",
        state=TaskState.ASSIGNED,
        dispatch_status=DispatchStatus.READY,
        assigned_agent_id=uuid.uuid4(),
    )
    defaults.update(overrides)
    return Task(**defaults)


def _result(all_rows=None, first_row=None, rows=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.all.return_value = all_rows or []
    m.scalars.return_value.first.return_value = first_row
    m.fetchall.return_value = rows or []
    return m


def _make_db(results: list) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.execute = AsyncMock(side_effect=results)
    return db


@contextmanager
def _approve_patches(task, *, unblock=None, publish=None):
    """Patch stack shared by the approve-path endpoint tests."""
    with (
        patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
        patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
        patch(
            "app.api.v1.tasks.try_unblock_tasks",
            new_callable=AsyncMock,
            return_value=unblock or [],
        ) as mock_unblock,
        patch(
            "app.api.v1.tasks.publish_unblocked_tasks",
            new_callable=AsyncMock,
            return_value=publish or [],
        ) as mock_publish,
        patch(
            "app.core.session_close.maybe_refresh_room_memory",
            new_callable=AsyncMock,
        ) as mock_memory,
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock) as mock_emit,
        patch(
            "app.api.v1.tasks._enrich_task",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(id=task.id),
        ),
    ):
        yield SimpleNamespace(
            unblock=mock_unblock, publish=mock_publish,
            memory=mock_memory, emit=mock_emit,
        )


# ---------------------------------------------------------------------------
# emit_checkpoint_request
# ---------------------------------------------------------------------------


class TestEmitCheckpointRequest:
    @pytest.mark.asyncio
    async def test_metadata_contract(self):
        task = _checkpoint_task()
        db = AsyncMock()

        with patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit:
            await emit_checkpoint_request(task.room_id, task, db)

        mock_emit.assert_awaited_once()
        kwargs = mock_emit.await_args.kwargs
        assert kwargs["event_type"] == "CHECKPOINT_REQUEST"
        assert kwargs["room_id"] == task.room_id
        assert kwargs["task_id"] == task.id
        assert kwargs["metadata"]["task_id"] == str(task.id)
        assert kwargs["metadata"]["checkpoint_kind"] == "select_variant"
        assert kwargs["metadata"]["checkpoint_payload"] == task.checkpoint_payload
        # payload.prompt is the visible question
        assert "Choose a variant" in kwargs["content"]

    @pytest.mark.asyncio
    async def test_falls_back_to_title_without_prompt(self):
        task = _checkpoint_task(checkpoint_payload={"variants": VARIANTS})
        db = AsyncMock()

        with patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit:
            await emit_checkpoint_request(task.room_id, task, db)

        assert task.title in mock_emit.await_args.kwargs["content"]


# ---------------------------------------------------------------------------
# publish_unblocked_tasks
# ---------------------------------------------------------------------------


class TestPublishUnblockedTasks:
    @pytest.mark.asyncio
    async def test_checkpoint_published_not_dispatched(self):
        checkpoint = _checkpoint_task()
        db = AsyncMock()

        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock
            ) as mock_dispatch,
        ):
            regular = await publish_unblocked_tasks(
                [checkpoint], checkpoint.room_id, db, source="test_unblock",
            )

        assert regular == []
        mock_dispatch.assert_not_awaited()
        mock_emit.assert_awaited_once()
        assert mock_emit.await_args.kwargs["event_type"] == "CHECKPOINT_REQUEST"
        # The checkpoint stays in WAITING_APPROVAL — the resolve endpoint moves it.
        assert checkpoint.state == TaskState.WAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_regular_task_unblocked_and_dispatched(self):
        task = _regular_task()
        db = AsyncMock()

        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.core.proxy_session.request_task_dispatch",
                new_callable=AsyncMock,
                return_value="dispatched",
            ) as mock_dispatch,
        ):
            regular = await publish_unblocked_tasks(
                [task], task.room_id, db,
                source="test_unblock",
                unblocked_by=uuid.uuid4(),
                predecessor_title="Concept generation",
            )

        assert regular == [task]
        assert mock_emit.await_args.kwargs["event_type"] == "TASK_UNBLOCKED"
        assert "Concept generation" in mock_emit.await_args.kwargs["content"]
        mock_dispatch.assert_awaited_once()
        assert mock_dispatch.await_args.kwargs["source"] == "test_unblock"

    @pytest.mark.asyncio
    async def test_regular_task_without_agent_not_dispatched(self):
        task = _regular_task(assigned_agent_id=None)
        db = AsyncMock()

        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock
            ) as mock_dispatch,
        ):
            regular = await publish_unblocked_tasks(
                [task], task.room_id, db, source="test_unblock",
            )

        assert regular == [task]
        mock_emit.assert_awaited_once()
        mock_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preflight_error_swallowed(self):
        from app.core.proxy_session import TaskDispatchPreflightError

        task = _regular_task()
        db = AsyncMock()

        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch(
                "app.core.proxy_session.request_task_dispatch",
                new_callable=AsyncMock,
                side_effect=TaskDispatchPreflightError("already_dispatched", "already"),
            ),
        ):
            regular = await publish_unblocked_tasks(
                [task], task.room_id, db, source="test_unblock",
            )

        assert regular == [task]

    @pytest.mark.asyncio
    async def test_mixed_batch_splits_checkpoints_from_regular(self):
        checkpoint = _checkpoint_task()
        task = _regular_task()
        db = AsyncMock()

        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock
            ),
        ):
            regular = await publish_unblocked_tasks(
                [checkpoint, task], task.room_id, db, source="test_unblock",
            )

        assert regular == [task]
        types = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert types == ["CHECKPOINT_REQUEST", "TASK_UNBLOCKED"]


# ---------------------------------------------------------------------------
# resolve_task_checkpoint — validation
# ---------------------------------------------------------------------------


class TestResolveValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_waiting_approval_state(self):
        task = _checkpoint_task(state=TaskState.COMPLETED)
        db = _make_db([])
        with (
            patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
            patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
        ):
            with pytest.raises(HTTPException) as exc:
                await resolve_task_checkpoint(
                    task.room_id, task.id,
                    CheckpointResolve(approve=True, choice_id="a"),
                    db, current_user=MagicMock(),
                )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_non_checkpoint_task(self):
        task = _regular_task(state=TaskState.WAITING_APPROVAL)
        db = _make_db([])
        with (
            patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
            patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
        ):
            with pytest.raises(HTTPException) as exc:
                await resolve_task_checkpoint(
                    task.room_id, task.id,
                    CheckpointResolve(approve=True),
                    db, current_user=MagicMock(),
                )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_select_variant_requires_choice_id(self):
        task = _checkpoint_task()
        db = _make_db([])
        with (
            patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
            patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
        ):
            with pytest.raises(HTTPException) as exc:
                await resolve_task_checkpoint(
                    task.room_id, task.id,
                    CheckpointResolve(approve=True),
                    db, current_user=MagicMock(),
                )
        assert exc.value.status_code == 422
        assert task.state == TaskState.WAITING_APPROVAL
        assert task.checkpoint_choice is None

    @pytest.mark.asyncio
    async def test_select_variant_unknown_choice_id_rejected(self):
        task = _checkpoint_task()
        db = _make_db([])
        with (
            patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
            patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
        ):
            with pytest.raises(HTTPException) as exc:
                await resolve_task_checkpoint(
                    task.room_id, task.id,
                    CheckpointResolve(approve=True, choice_id="nope"),
                    db, current_user=MagicMock(),
                )
        assert exc.value.status_code == 422
        assert task.checkpoint_choice is None


# ---------------------------------------------------------------------------
# resolve_task_checkpoint — approve
# ---------------------------------------------------------------------------


class TestResolveApprove:
    @pytest.mark.asyncio
    async def test_approve_select_variant_full_flow(self):
        task = _checkpoint_task()
        dependent = _regular_task(dispatch_status=DispatchStatus.BLOCKED)
        next_task = _regular_task(title="Next")
        db = _make_db([
            _result(rows=[(dependent.id,)]),   # TaskDependency lookup
            _result(all_rows=[dependent]),     # dependent tasks
        ])

        with _approve_patches(task, unblock=[next_task], publish=[next_task]) as mocks:
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=True, choice_id="b", note="we take the dark one"),
                db, current_user=MagicMock(),
            )

        # State machine + choice persisted
        assert task.state == TaskState.COMPLETED
        assert task.checkpoint_choice["approved"] is True
        assert task.checkpoint_choice["choice_id"] == "b"
        assert task.checkpoint_choice["variant"] == VARIANTS[1]
        assert task.checkpoint_choice["note"] == "we take the dark one"
        assert "resolved_at" in task.checkpoint_choice

        # Choice propagated to the dependent's description deterministically
        assert "[CHECKPOINT CHOICE]" in dependent.description
        assert "Variant B" in dependent.description
        assert "/project/images/b1.png" in dependent.description
        assert "we take the dark one" in dependent.description

        # Dependents unblocked + published (dispatch interception lives there)
        mocks.unblock.assert_awaited_once_with(task.id, db)
        mocks.publish.assert_awaited_once()
        assert mocks.publish.await_args.args[0] == [next_task]
        assert mocks.publish.await_args.kwargs["source"] == "checkpoint_resolve"

        mocks.memory.assert_awaited_once()

        event_types = [c.kwargs["event_type"] for c in mocks.emit.await_args_list]
        assert "CHECKPOINT_RESOLVED" in event_types
        assert "TASK_STATE_CHANGED" in event_types
        resolved = next(
            c for c in mocks.emit.await_args_list
            if c.kwargs["event_type"] == "CHECKPOINT_RESOLVED"
        )
        assert resolved.kwargs["metadata"]["approved"] is True
        assert resolved.kwargs["metadata"]["choice_id"] == "b"

    @pytest.mark.asyncio
    async def test_approve_confirm_kind_needs_no_choice(self):
        task = _checkpoint_task(
            checkpoint_kind="confirm",
            checkpoint_payload={"prompt": "Continue?"},
        )
        db = _make_db([_result(rows=[])])   # no dependents

        with _approve_patches(task):
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=True),
                db, current_user=MagicMock(),
            )

        assert task.state == TaskState.COMPLETED
        assert task.checkpoint_choice["approved"] is True
        assert "choice_id" not in task.checkpoint_choice

    @pytest.mark.asyncio
    async def test_approve_rechecks_goal_completion(self):
        """A final human-acceptance checkpoint may be the goal's last task."""
        task = _checkpoint_task(
            checkpoint_kind="confirm",
            checkpoint_payload={"prompt": "Accept?"},
            goal_id=uuid.uuid4(),
        )
        db = _make_db([_result(rows=[])])

        with (
            _approve_patches(task),
            patch(
                "app.core.goal_progress.maybe_complete_goal",
                new_callable=AsyncMock,
            ) as mock_goal,
        ):
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=True),
                db, current_user=MagicMock(),
            )

        assert task.state == TaskState.COMPLETED
        mock_goal.assert_awaited_once_with(task.goal_id, task.room_id, db)

    @pytest.mark.asyncio
    async def test_approve_with_empty_variants_accepts_any_choice(self):
        """LLM may omit variants — approve still records the choice_id."""
        task = _checkpoint_task(checkpoint_payload={"prompt": "?"})
        db = _make_db([_result(rows=[])])

        with _approve_patches(task):
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=True, choice_id="freeform"),
                db, current_user=MagicMock(),
            )

        assert task.state == TaskState.COMPLETED
        assert task.checkpoint_choice["choice_id"] == "freeform"
        assert "variant" not in task.checkpoint_choice

    @pytest.mark.asyncio
    async def test_terminal_dependent_description_untouched(self):
        task = _checkpoint_task()
        done = _regular_task(state=TaskState.COMPLETED, description="old")
        db = _make_db([
            _result(rows=[(done.id,)]),
            _result(all_rows=[done]),
        ])

        with _approve_patches(task):
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=True, choice_id="a"),
                db, current_user=MagicMock(),
            )

        assert done.description == "old"


# ---------------------------------------------------------------------------
# resolve_task_checkpoint — reject
# ---------------------------------------------------------------------------


class TestResolveReject:
    @pytest.mark.asyncio
    async def test_reject_notifies_dependents(self):
        task = _checkpoint_task()
        blocked_dep = _regular_task(dispatch_status=DispatchStatus.BLOCKED)
        db = _make_db([])

        with (
            patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock),
            patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task),
            patch(
                "app.api.v1.tasks.get_blocked_dependents",
                new_callable=AsyncMock,
                return_value=[blocked_dep],
            ),
            patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.api.v1.tasks._enrich_task",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id=task.id),
            ),
        ):
            await resolve_task_checkpoint(
                task.room_id, task.id,
                CheckpointResolve(approve=False, note="not that"),
                db, current_user=MagicMock(),
            )

        assert task.state == TaskState.REJECTED
        assert task.checkpoint_choice["approved"] is False
        assert task.checkpoint_choice["note"] == "not that"

        event_types = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert "TASK_DEPENDENCY_FAILED" in event_types
        assert "CHECKPOINT_RESOLVED" in event_types
        assert "TASK_STATE_CHANGED" in event_types
        dep_failed = next(
            c for c in mock_emit.await_args_list
            if c.kwargs["event_type"] == "TASK_DEPENDENCY_FAILED"
        )
        assert dep_failed.kwargs["metadata"]["blocked_because"] == str(task.id)
        resolved = next(
            c for c in mock_emit.await_args_list
            if c.kwargs["event_type"] == "CHECKPOINT_RESOLVED"
        )
        assert resolved.kwargs["metadata"]["approved"] is False


# ---------------------------------------------------------------------------
# _format_checkpoint_choice_note
# ---------------------------------------------------------------------------


class TestChoiceNote:
    def test_select_variant_note(self):
        task = _checkpoint_task()
        task.checkpoint_choice = {
            "approved": True,
            "choice_id": "b",
            "variant": VARIANTS[1],
            "note": "we take the dark one",
        }
        note = _format_checkpoint_choice_note(task)
        assert "Variant B" in note
        assert "id=b" in note
        assert "/project/images/b1.png" in note
        assert "darker" in note
        assert "we take the dark one" in note
        assert "ONLY this variant" in note

    def test_confirm_note(self):
        task = _checkpoint_task(checkpoint_kind="confirm", checkpoint_payload={})
        task.checkpoint_choice = {"approved": True}
        note = _format_checkpoint_choice_note(task)
        assert "confirmed" in note


# ---------------------------------------------------------------------------
# change_task_state: manual Approve also unblocks (TaskPanel fallback path)
# ---------------------------------------------------------------------------


class TestManualApproveUnblocks:
    @pytest.mark.asyncio
    async def test_waiting_approval_to_approved_unblocks(self):
        task = _checkpoint_task(goal_id=uuid.uuid4())
        db = _make_db([
            _result(first_row=object()),  # _verify_membership
            _result(first_row=task),      # _get_task
        ])

        with (
            patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
            patch(
                "app.api.v1.tasks.try_unblock_tasks",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_unblock,
            patch(
                "app.api.v1.tasks.publish_unblocked_tasks",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_publish,
            patch(
                "app.api.v1.tasks._enrich_task",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id=task.id),
            ),
            patch(
                "app.core.room_git_flow.maybe_open_promotion",
                new_callable=AsyncMock,
            ),
            patch(
                "app.core.goal_progress.maybe_complete_goal",
                new_callable=AsyncMock,
            ) as mock_goal,
        ):
            await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.APPROVED),
                db, current_user=MagicMock(),
            )

        # Approving a checkpoint finishes it — an APPROVED checkpoint never
        # counted as done for its goal, so the goal/<id> → dev gate never opened.
        assert task.state == TaskState.COMPLETED
        assert task.checkpoint_choice["approved"] is True
        mock_unblock.assert_awaited_once_with(task.id, db)
        mock_publish.assert_awaited_once()
        mock_goal.assert_awaited_once_with(task.goal_id, task.room_id, db)

    @pytest.mark.asyncio
    async def test_waiting_approval_to_rejected_does_not_unblock(self):
        task = _checkpoint_task()
        db = _make_db([
            _result(first_row=object()),
            _result(first_row=task),
        ])

        with (
            patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
            patch(
                "app.api.v1.tasks.try_unblock_tasks", new_callable=AsyncMock
            ) as mock_unblock,
            patch(
                "app.api.v1.tasks._enrich_task",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(id=task.id),
            ),
        ):
            await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.REJECTED),
                db, current_user=MagicMock(),
            )

        assert task.state == TaskState.REJECTED
        mock_unblock.assert_not_awaited()


# ---------------------------------------------------------------------------
# TaskOut exposes checkpoint fields (stage-4 card consumer contract)
# ---------------------------------------------------------------------------


class TestTaskOutCheckpointFields:
    def test_fields_exposed(self):
        task = _checkpoint_task()
        out = TaskOut.model_validate(task)
        assert out.checkpoint_kind == "select_variant"
        assert out.checkpoint_payload == {"prompt": "Choose a variant", "variants": VARIANTS}
        assert out.checkpoint_choice is None

    def test_defaults_for_regular_task(self):
        task = _regular_task()
        out = TaskOut.model_validate(task)
        assert out.checkpoint_kind is None
        assert out.checkpoint_payload == {}
        assert out.checkpoint_choice is None
