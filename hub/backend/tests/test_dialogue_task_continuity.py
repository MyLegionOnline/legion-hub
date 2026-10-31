"""One dialogue, one task — across rotation and across the board's buttons.

Two ways a single conversation used to splinter:

* rotation (the dialogue outlived its turn limit) replayed the turn into a
  fresh session that knew nothing of the task its predecessor had recorded —
  its first write opened a new branch off the working branch and the next
  finalization recorded a second task, titled after the carry-over bridge
  ("[Conversation continued: …");
* Start / Back to Work / Unblock on such a task went to the dispatcher, which
  sent a media-capable agent's task into image generation (it failed with
  "All images failed") — instead of back into a dialogue.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1.tasks import _REVIEW_REOPEN_FEEDBACK, change_task_state
from app.core import session_work
from app.core.proxy_session import _routes_to_image_generation
from app.models.task import TASK_ORIGIN_DIRECT_SESSION, DispatchStatus, TaskState
from app.schemas.task import TaskStateChange
from tests.test_task_dispatch_contract import _endpoint_db, _make_task


class _FakeDB:
    def __init__(self, task=None):
        self._task = task

    def add(self, _obj):
        pass

    async def flush(self):
        pass

    async def execute(self, _stmt):
        task = self._task
        return SimpleNamespace(scalar_one_or_none=lambda: task)


def _session(snapshot=None):
    return SimpleNamespace(
        id=uuid4(), room_id=uuid4(), kind="user_agent_p2",
        context_snapshot=dict(snapshot or {}),
    )


class TestRotationCarriesTheTask:
    @pytest.mark.asyncio
    async def test_open_task_moves_to_the_successor(self):
        task = SimpleNamespace(id=uuid4(), state=TaskState.BLOCKED, merge_commit=None)
        previous = _session({session_work.SNAPSHOT_TASK_ID: str(task.id)})
        successor = _session()

        await session_work.carry_dialogue_task(previous, successor, _FakeDB(task))

        assert successor.context_snapshot[session_work.SNAPSHOT_TASK_ID] == str(task.id)

    @pytest.mark.asyncio
    async def test_merged_task_is_not_carried(self):
        task = SimpleNamespace(id=uuid4(), state=TaskState.COMPLETED, merge_commit="abc")
        previous = _session({session_work.SNAPSHOT_TASK_ID: str(task.id)})
        successor = _session()

        await session_work.carry_dialogue_task(previous, successor, _FakeDB(task))

        assert session_work.SNAPSHOT_TASK_ID not in successor.context_snapshot

    @pytest.mark.asyncio
    async def test_no_task_no_carry(self):
        successor = _session()
        await session_work.carry_dialogue_task(_session(), successor, _FakeDB())
        assert successor.context_snapshot == {}


class TestCarryoverIsNotTheRequest:
    BRIDGE = (
        "[Conversation continued: the previous session was closed at the length limit. "
        "Its result:\nSession closed automatically. Last replies: [done]\n\nmore]\n\n"
    )

    def test_title_skips_the_bridge(self):
        title = session_work.derive_task_title(self.BRIDGE + "render the vertical cut")
        assert title == "render the vertical cut"

    def test_bridge_alone_falls_back(self):
        assert session_work.derive_task_title(self.BRIDGE) == session_work._FALLBACK_TITLE

    def test_description_skips_the_bridge(self):
        desc = session_work.derive_task_description(self.BRIDGE + "render it", "rendered")
        assert desc.startswith("render it")
        assert "Conversation continued" not in desc

    def test_ordinary_request_untouched(self):
        assert session_work.derive_task_title("make [x]\n\nbold") == "make [x] bold"


class TestMediaRouting:
    @staticmethod
    def _agent():
        return SimpleNamespace(llm_config={"media": {"capability": "image_generation"}})

    def test_planned_task_of_media_agent_goes_to_images(self):
        assert _routes_to_image_generation(SimpleNamespace(origin=None), self._agent())

    def test_dialogue_task_of_media_agent_does_not(self):
        task = SimpleNamespace(origin=TASK_ORIGIN_DIRECT_SESSION)
        assert not _routes_to_image_generation(task, self._agent())

    def test_text_agent_never(self):
        agent = SimpleNamespace(llm_config={})
        assert not _routes_to_image_generation(SimpleNamespace(origin=None), agent)


class TestBackToWorkOnDialogueTask:
    @pytest.mark.asyncio
    async def test_continued_in_a_dialogue_not_dispatched(self):
        task = _make_task(
            state=TaskState.WAITING_REVIEW,
            dispatch_status=DispatchStatus.DISPATCHED,
            origin=TASK_ORIGIN_DIRECT_SESSION,
        )
        db = _endpoint_db(task)
        user = MagicMock()
        user.id = uuid4()
        resume = AsyncMock()

        with patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), patch(
            "app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock,
        ), patch(
            "app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
        ) as preflight, patch(
            "app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
        ) as dispatch, patch(
            "app.core.waiting_reply.resume_direct_session_task", resume,
        ), patch("asyncio.create_task") as spawn:
            out = await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.IN_PROGRESS), db, user,
            )

        assert out.state == TaskState.IN_PROGRESS
        preflight.assert_not_awaited()
        dispatch.assert_not_awaited()
        spawn.assert_called_once()
        resume.assert_called_once_with(
            task.room_id, user.id, task.id, _REVIEW_REOPEN_FEEDBACK,
        )
        # the board must not start a second executor on the dialogue's branch
        assert task.dispatch_status == DispatchStatus.DISPATCHED
        spawn.call_args.args[0].close()
