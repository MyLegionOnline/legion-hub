"""Mid-task dialogue between an agent and ChatMaster inside a proxy session.

The proxy pair was always able to exchange plain-text turns, but nothing told
either side that the channel existed: the agent's prompt described close_session
as the only exit, and the CM's prompt described every incoming message as a
completed-work report. These tests pin the transport half of the fix — the
framing that makes a text turn readable as a question and its reply readable as
an answer, and the bounds that keep an unbounded dialogue from stalling silently.

All LLM and DB calls are stubbed, mirroring tests/test_proxy_session.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import proxy_session
from app.core.proxy_session import (
    _CM_ANSWER_NOTE,
    _DIALOGUE_FOOTER_MARKER,
    _MID_TASK_NOTE,
    _ensure_dialogue_footer,
    handle_task_created,
)
from app.models.task import DispatchStatus, TaskState


# ---------------------------------------------------------------------------
# Helpers (same shape as tests/test_proxy_session.py)
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
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _make_task(assigned_agent_id: uuid.UUID, goal_id: uuid.UUID) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Insert decor block into article_detail.html"
    t.description = "Pool of 4 images, index = article.id % 4."
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = assigned_agent_id
    t.goal_id = goal_id
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    t.rework_count = 0
    return t


def _claim_result() -> MagicMock:
    r = MagicMock()
    r.rowcount = 1
    return r


def _agent_result() -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.first.return_value = MagicMock(llm_config={})
    return r


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.first.return_value = None
    r.scalars.return_value.all.return_value = []
    r.first.return_value = None
    r.all.return_value = []
    return r


def _db_results(*results: MagicMock):
    yield from results
    while True:
        yield _empty_result()


def _make_room(room_id: uuid.UUID, master_agent_id: uuid.UUID) -> MagicMock:
    room = MagicMock()
    room.id = room_id
    room.coordinator_config = {"master_agent_id": str(master_agent_id)}
    return room


def _make_session(side: str) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = side
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.plan_id = uuid.uuid4()
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    return s


def _fake_emit_factory(captured: list[str]):
    async def _emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
        captured.append(event_type)
        return MagicMock()
    return _emit


def _text(reply: str) -> tuple:
    return (reply, None, 50, 40, "claude-haiku-4-5-20251001", 50, 50)


def _close(tool_input: dict) -> tuple:
    return (None, tool_input, 50, 40, "claude-haiku-4-5-20251001", 50, 50)


def _history(session: MagicMock) -> list[dict[str, Any]]:
    return (session.context_snapshot or {}).get("messages") or []


def _published(broker: AsyncMock) -> list[Any]:
    return [c.args[0] for c in broker.publish.await_args_list]


async def _run(llm_calls: list[tuple], emitted: list[str]) -> dict[str, Any]:
    """Drive one full proxy pair over the given LLM script."""
    room_id = uuid.uuid4()
    goal_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    cm_agent_id = uuid.uuid4()

    task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
    room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

    agent_session = _make_session("agent")
    agent_session.agent_id = agent_id
    agent_session.room_id = room_id
    cm_session = _make_session("chat_master")
    cm_session.agent_id = cm_agent_id
    cm_session.room_id = room_id

    db = _make_db()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    room_result = MagicMock()
    room_result.scalars.return_value.first.return_value = room
    db.execute = AsyncMock(
        side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result)
    )

    broker_instance = AsyncMock()

    with (
        patch("app.core.proxy_session.SessionRepository") as MockRepo,
        # The run_id/runtime_state fence has its own coverage in
        # test_proxy_stale_loop.py; here it would report the pair as superseded
        # because the stubbed db never returns a runtime row.
        patch("app.core.proxy_session._pair_closed_externally",
              new_callable=AsyncMock, return_value=False),
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
        patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock,
              side_effect=llm_calls),
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
        patch("app.core.proxy_session.ProxyBroker") as MockBroker,
        patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        patch("app.core.watchdog.block_stalled_task", new_callable=AsyncMock) as mock_block,
    ):
        repo_instance = AsyncMock()
        repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
        MockRepo.return_value = repo_instance
        MockBroker.return_value = broker_instance

        await handle_task_created(task.id, room_id, goal_id, plan_id, db)

    return {
        "task": task,
        "agent_session": agent_session,
        "cm_session": cm_session,
        "broker": broker_instance,
        "block_stalled_task": mock_block,
    }


# ---------------------------------------------------------------------------
# _ensure_dialogue_footer
# ---------------------------------------------------------------------------


class TestEnsureDialogueFooter:
    def test_appends_to_first_user_message(self):
        history = [{"role": "user", "content": "# Task: do the thing"}]
        out = _ensure_dialogue_footer(history, "ChatMaster")
        assert out[0]["content"].startswith("# Task: do the thing")
        assert _DIALOGUE_FOOTER_MARKER in out[0]["content"]
        assert "ChatMaster" in out[0]["content"]

    def test_appends_text_block_to_multimodal_message(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "# Task"},
                                         {"type": "image", "source": {}}]}
        ]
        out = _ensure_dialogue_footer(history, "CM")
        blocks = out[0]["content"]
        assert len(blocks) == 3
        assert _DIALOGUE_FOOTER_MARKER in blocks[-1]["text"]

    def test_is_idempotent(self):
        history = [{"role": "user", "content": "# Task"}]
        once = _ensure_dialogue_footer(history, "CM")
        twice = _ensure_dialogue_footer(list(once), "CM")
        assert twice[0]["content"].count(_DIALOGUE_FOOTER_MARKER) == 1

    def test_resumed_history_keeps_existing_footer(self):
        """A rework resume replays the old first turn — no second footer."""
        seeded = _ensure_dialogue_footer([{"role": "user", "content": "# Task"}], "CM")
        resumed = seeded + [
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "CHANGES_REQUESTED: fix it"},
        ]
        out = _ensure_dialogue_footer(list(resumed), "CM")
        assert _DIALOGUE_FOOTER_MARKER not in out[-1]["content"]

    def test_no_user_message_is_a_noop(self):
        history = [{"role": "assistant", "content": "orphan"}]
        assert _ensure_dialogue_footer(list(history), "CM") == history


# ---------------------------------------------------------------------------
# Question → answer → work → close
# ---------------------------------------------------------------------------


class TestMidTaskDialogue:
    @pytest.mark.asyncio
    async def test_question_and_answer_round_trip(self):
        """A text turn is framed as a question; the CM's text reply as an answer."""
        emitted: list[str] = []
        result = await _run(
            [
                _text("Acceptance says id=4 -> d.jpg, but id % 4 = 0 -> a.jpg. Which wins?"),
                _text("The rule wins: id=4 ships a.jpg. The acceptance line is wrong."),
                _close({"summary": "Implemented the tag with id % 4.", "tasks_update": []}),
                _close({"summary": "Matches the rule.", "plan_update": None}),
            ],
            emitted,
        )

        # The agent's first turn seeds the task; the CM sees it marked as mid-task.
        cm_user_turns = [m for m in _history(result["cm_session"]) if m["role"] == "user"]
        assert cm_user_turns[0]["content"].startswith(_MID_TASK_NOTE)
        assert "Which wins?" in cm_user_turns[0]["content"]
        # A question is not the completion notification — that shape comes later.
        assert "has completed their work" not in cm_user_turns[0]["content"]
        assert any("closed the session" in m["content"] for m in cm_user_turns[1:])

        # The agent sees the reply as an answer, not a verdict.
        agent_answers = [
            m for m in _history(result["agent_session"])
            if m["role"] == "user" and m["content"].startswith(_CM_ANSWER_NOTE)
        ]
        assert len(agent_answers) == 1
        assert "The rule wins" in agent_answers[0]["content"]

        kinds = [p.kind for p in _published(result["broker"])]
        assert "question" in kinds
        assert "answer" in kinds
        assert "TASK_APPROVED" in emitted
        assert "TASK_REJECTED" not in emitted
        assert result["task"].state == TaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_first_agent_turn_carries_the_dialogue_footer(self):
        emitted: list[str] = []
        result = await _run(
            [
                _close({"summary": "Done.", "tasks_update": []}),
                _close({"summary": "Fine.", "plan_update": None}),
            ],
            emitted,
        )
        first_turn = _history(result["agent_session"])[0]
        assert first_turn["role"] == "user"
        assert _DIALOGUE_FOOTER_MARKER in first_turn["content"]

    @pytest.mark.asyncio
    async def test_multiple_rounds_keep_the_session_open(self):
        """The dialogue is not one-shot: three rounds, then a normal delivery."""
        emitted: list[str] = []
        script: list[tuple] = []
        for i in range(3):
            script.append(_text(f"Question {i}?"))
            script.append(_text(f"Answer {i}."))
        script.append(_close({"summary": "Built it.", "tasks_update": []}))
        script.append(_close({"summary": "Approved.", "plan_update": None}))

        result = await _run(script, emitted)

        kinds = [p.kind for p in _published(result["broker"])]
        assert kinds.count("question") == 3
        assert kinds.count("answer") == 3
        answers = [
            m for m in _history(result["agent_session"])
            if m["role"] == "user" and m["content"].startswith(_CM_ANSWER_NOTE)
        ]
        assert len(answers) == 3
        assert "TASK_APPROVED" in emitted
        # Nothing was rejected or sent for rework just because the agent asked.
        assert "TASK_CHANGES_REQUESTED" not in emitted
        assert "TASK_REJECTED" not in emitted

    @pytest.mark.asyncio
    async def test_soft_limit_marks_the_answer_final(self):
        emitted: list[str] = []
        script: list[tuple] = []
        for i in range(2):
            script.append(_text(f"Question {i}?"))
            script.append(_text(f"Answer {i}."))
        script.append(_close({"summary": "Built it.", "tasks_update": []}))
        script.append(_close({"summary": "Approved.", "plan_update": None}))

        with patch.object(proxy_session, "_QA_SOFT_LIMIT", 2):
            result = await _run(script, emitted)

        answers = [
            m["content"] for m in _history(result["agent_session"])
            if m["role"] == "user" and m["content"].startswith(_CM_ANSWER_NOTE)
        ]
        assert len(answers) == 2
        assert "Treat this answer as final" not in answers[0]
        assert "Treat this answer as final" in answers[1]

    @pytest.mark.asyncio
    async def test_the_cm_is_told_to_correct_the_task_in_the_same_turn(self):
        """A ruling that invalidates the task text must be written back, not promised.

        Observed in production: CM answered "I'll correct the acceptance criterion
        in the task now" and its turn ended — no TASK_UPDATED, the wrong criterion
        stayed on the board and would be replayed on every rework.
        """
        assert "task_update(" in _MID_TASK_NOTE
        assert "THIS turn" in _MID_TASK_NOTE

    @pytest.mark.asyncio
    async def test_turn_cap_warns_then_blocks_the_task(self):
        """A pair that only ever talks must not become a silent dead end."""
        emitted: list[str] = []
        script = [_text("chatter")] * 10

        with patch.object(proxy_session, "_MAX_PROXY_TURNS", 2):
            result = await _run(script, emitted)

        answers = [
            m["content"] for m in _history(result["agent_session"])
            if m["role"] == "user" and m["content"].startswith(_CM_ANSWER_NOTE)
        ]
        assert any("force-closed" in a for a in answers)

        result["block_stalled_task"].assert_awaited_once()
        call = result["block_stalled_task"].await_args
        assert call.args[1] == "proxy_max_turns"
        assert call.kwargs["extra_meta"]["qa_rounds"] == 2


# ---------------------------------------------------------------------------
# The instrument itself: ChatMaster must be able to rewrite the task it owns
# ---------------------------------------------------------------------------


class TestChatMasterCanCorrectTheTask:
    def test_task_update_is_offered_to_the_cm_role(self):
        from app.core.tools import REGISTRY

        names = {t.name for t in REGISTRY.tools_for("cm")}
        assert "task_update" in names, "CM cannot fix a task it just ruled invalid"

    def test_task_update_accepts_a_new_description(self):
        from app.core.tools import REGISTRY

        schema = REGISTRY.get("task_update").input_schema
        assert "description" in schema["properties"]
        assert schema["required"] == ["task_id"]

    def test_cm_prompt_mandates_the_correction_in_the_same_turn(self):
        from app.core.memory_layer import _KIND_HEADERS

        cm_prompt = _KIND_HEADERS["cm_agent_proxy_cm"]
        assert "task_update(" in cm_prompt
        assert "MANDATORY, IN THIS SAME TURN" in cm_prompt
        # The failure mode is a promise, not a missing tool.
        assert "I will update the task" in cm_prompt
