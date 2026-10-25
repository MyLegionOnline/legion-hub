"""internal design notes Stage 4 — "@agent …" into the agent's running task.

- the deterministic shortcut (status phrases always, no-new-work text when the
  agent has exactly one running task) skips Jev and the LLM;
- the guard keeps interject_task serviceable and prefers it over a low
  confidence single-agent route;
- Jev is offered interject_task only when the addressee is running a task;
- the dispatcher queues the existing chat message and keeps the open session,
  and falls back to the old route when the task cannot take it.

Pure/mocked — no database required.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import router as router_mod
from app.core import router_jev as rj
from app.core.router import (
    OpenSessionContext,
    RouteDecision,
    _apply_interject_guard,
    _interject_shortcut,
)


def _agent_ctx(running=1, recent=("old task",), agent_id=None):
    running_rows = [
        {"id": str(uuid.uuid4()), "title": f"running {i}", "state": "in_progress",
         "description": "", "running": True}
        for i in range(running)
    ]
    recent_rows = [
        {"id": str(uuid.uuid4()), "title": t, "state": "completed",
         "description": "", "running": False}
        for t in recent
    ]
    return {
        "id": agent_id or str(uuid.uuid4()),
        "nickname": "front",
        "short_description": "",
        "tasks": running_rows + recent_rows,
        "running_tasks": [{"id": t["id"], "title": t["title"]} for t in running_rows],
    }


class TestShortcut:
    @pytest.mark.parametrize("text", [
        "@front check the status",
        "@front are you alive?",
        "@front what are you doing now?",
        "@front how long will it take?",
        "@front stuck?",
        "@front status?",
        "@front tell me which step you are on",
    ])
    def test_status_phrases_go_into_the_task(self, text):
        a = _agent_ctx()
        d = _interject_shortcut(text, [a], None)
        assert d is not None and d.route == "interject_task"
        assert str(d.matched_task_id) == a["running_tasks"][0]["id"]

    def test_no_new_work_goes_into_the_task(self):
        d = _interject_shortcut("@front shorten the scope, only the main page", [_agent_ctx()], None)
        assert d is not None and d.route == "interject_task"

    @pytest.mark.parametrize("text", [
        "@front also make a contacts page",
        "@front create a new task for the logo",
        "@front implement the pricing page",
    ])
    def test_new_work_is_left_to_the_model(self, text):
        assert _interject_shortcut(text, [_agent_ctx()], None) is None

    def test_needs_exactly_one_running_task(self):
        assert _interject_shortcut("@front status", [_agent_ctx(running=0)], None) is None
        assert _interject_shortcut("@front status", [_agent_ctx(running=2)], None) is None

    def test_needs_exactly_one_addressee(self):
        assert _interject_shortcut("@front @back status", [_agent_ctx(), _agent_ctx()], None) is None

    def test_dialogue_with_the_same_agent_keeps_non_status_text(self):
        a = _agent_ctx()
        os_ = OpenSessionContext(session_id=uuid.uuid4(), kind="user_agent_p2",
                                 peer_agent_id=uuid.UUID(a["id"]))
        assert _interject_shortcut("@front and what about the header next", [a], os_) is None
        assert _interject_shortcut("@front status?", [a], os_).route == "interject_task"

    def test_living_words_are_not_status(self):
        # "hungry" must not read as "hung"
        d = _interject_shortcut("@front hungry cats illustration in a modern style", [_agent_ctx()], None)
        assert d is not None and "no new work" in d.reasoning


def _d(route, conf=0.9, task=None):
    return RouteDecision(route=route, confidence=conf, reasoning="r",
                         matched_task_id=uuid.UUID(task) if task else None)


class TestGuard:
    def test_without_running_task_interject_becomes_p2(self):
        d = _apply_interject_guard(_d("interject_task", task=str(uuid.uuid4())), [])
        assert d.route == "single_agent_p2" and d.matched_task_id is None

    def test_wrong_task_id_is_replaced(self):
        run = str(uuid.uuid4())
        d = _apply_interject_guard(_d("interject_task", task=str(uuid.uuid4())), [run])
        assert d.route == "interject_task" and str(d.matched_task_id) == run

    def test_p1_onto_running_task_is_interject(self):
        run = str(uuid.uuid4())
        d = _apply_interject_guard(_d("single_agent_p1", task=run), [run])
        assert d.route == "interject_task" and str(d.matched_task_id) == run

    def test_low_confidence_single_agent_prefers_interject(self):
        run = str(uuid.uuid4())
        d = _apply_interject_guard(_d("single_agent_p2", conf=0.4), [run])
        assert d.route == "interject_task" and str(d.matched_task_id) == run

    def test_confident_new_work_stays(self):
        run = str(uuid.uuid4())
        d = _apply_interject_guard(_d("single_agent_p2", conf=0.9), [run])
        assert d.route == "single_agent_p2"
        other = str(uuid.uuid4())
        assert _apply_interject_guard(_d("single_agent_p1", task=other), [run]).route == "single_agent_p1"


class TestJev:
    def _ctx(self, agent):
        return {"text": "@front fix the padding", "agents_ctx": [agent], "goals_ctx": [],
                "waiting_ctx": [], "open_session": None, "mentioned_ids": [agent["id"]]}

    def test_option_offered_only_with_running_task(self):
        body, maps = rj.build_jev_request(self._ctx(_agent_ctx()), "jev")
        assert "interject_task" in body["questions"]["route"]["criteria"]
        assert body["state"]["mentioned_agent"]["executing_now"] == ["running 0"]
        body, _ = rj.build_jev_request(self._ctx(_agent_ctx(running=0)), "jev")
        assert "interject_task" not in body["questions"]["route"]["criteria"]

    def test_decide_maps_to_a_running_task(self):
        a = _agent_ctx()
        _, maps = rj.build_jev_request(self._ctx(a), "jev")
        # Jev picked the old, finished task as "the task" — still the running one.
        old_key = next(k for k, v in maps["task"].items() if v == a["tasks"][1]["id"])
        ans = rj.decide({"route": {"choice": "interject_task", "confidence": 0.9},
                         "task": {"choice": old_key}}, maps)
        assert ans["route"] == "interject_task"
        assert ans["task_id"] == a["running_tasks"][0]["id"]


@pytest.mark.asyncio
async def test_route_user_message_shortcut_skips_the_model():
    a = _agent_ctx()
    cm = SimpleNamespace(id=uuid.uuid4(), owner_id=uuid.uuid4(), nickname="cm",
                         system_role=None, llm_config={"provider": "anthropic", "model": "m"})
    gathered = router_mod._GatheredContext(
        [a], [], [uuid.UUID(a["id"])], cm.llm_config, cm, room_agents=[cm],
    )
    from app.models.room import Room, RoomType

    room = Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="r")
    room.room_type = RoomType.group
    room.coordinator_config = {"master_agent_id": str(cm.id)}
    db = AsyncMock()
    db.add = MagicMock()
    with (
        patch.object(router_mod, "_gather_context", new=AsyncMock(return_value=gathered)),
        patch.object(router_mod, "get_router_agent", new=AsyncMock(return_value=None)),
        patch.object(router_mod, "_call_router_llm", new=AsyncMock()) as llm,
        patch.object(router_mod, "emit_event", new=AsyncMock()) as emit,
    ):
        d = await router_mod.route_user_message(room, "@front check the status", db)
    assert d.route == "interject_task"
    assert str(d.matched_task_id) == a["running_tasks"][0]["id"]
    llm.assert_not_awaited()
    meta = next(c for c in emit.call_args_list
                if c.kwargs["event_type"] == "ROUTE_DECIDED").kwargs["metadata"]
    assert meta["engine"] == "shortcut"


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def _room():
    from app.models.room import Room, RoomType

    room = MagicMock(spec=Room)
    room.id = uuid.uuid4()
    room.room_type = RoomType.group
    room.coordinator_config = {}
    return room


def _repo_with(session):
    repo = MagicMock()
    repo.find_open_user_session = AsyncMock(return_value=session)
    return patch("app.repositories.session.SessionRepository", return_value=repo)


@pytest.mark.asyncio
async def test_dispatch_interject_keeps_the_open_session():
    from app.api.v1.chat import _dispatch_group_route

    task_id, msg_id = uuid.uuid4(), uuid.uuid4()
    decision = RouteDecision(route="interject_task", matched_task_id=task_id,
                             mentioned_agent_ids=[uuid.uuid4()], confidence=1.0, reasoning="r")
    open_session = MagicMock()
    with (
        _repo_with(open_session),
        patch("app.core.user_sessions.open_session_context", new=AsyncMock(return_value=None)),
        patch("app.api.v1.chat.route_user_message", new=AsyncMock(return_value=decision)),
        patch("app.api.v1.chat._handle_interject", new=AsyncMock(return_value=True)) as handle,
        patch("app.core.user_sessions.close_user_session_gracefully", new=AsyncMock()) as close,
        patch("app.api.v1.chat._handle_direct_agent_session", new=AsyncMock()) as direct,
    ):
        await _dispatch_group_route(_room(), MagicMock(id=uuid.uuid4()), "@front status?",
                                    AsyncMock(), message_id=msg_id)
    handle.assert_awaited_once()
    assert handle.await_args.args[-1] == msg_id
    close.assert_not_awaited()
    direct.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_refused_interject_falls_back_to_p1():
    from app.api.v1.chat import _dispatch_group_route

    task_id = uuid.uuid4()
    decision = RouteDecision(route="interject_task", matched_task_id=task_id,
                             mentioned_agent_ids=[uuid.uuid4()], confidence=1.0, reasoning="r")
    with (
        _repo_with(None),
        patch("app.api.v1.chat.route_user_message", new=AsyncMock(return_value=decision)),
        patch("app.api.v1.chat._handle_interject", new=AsyncMock(return_value=False)),
        patch("app.api.v1.chat._handle_direct_agent_session", new=AsyncMock()) as direct,
    ):
        await _dispatch_group_route(_room(), MagicMock(id=uuid.uuid4()), "@front status?", AsyncMock())
    sent = direct.await_args.args[3]
    assert sent.route == "single_agent_p1" and sent.matched_task_id == task_id


@pytest.mark.asyncio
async def test_handle_interject_tags_the_message_and_answers_with_the_verdict():
    from app.api.v1 import chat as chat_mod
    from app.core import task_interject, task_status_probe

    task = SimpleNamespace(id=uuid.uuid4(), room_id=uuid.uuid4(), title="Landing page",
                           assigned_agent_id=uuid.uuid4())
    room = SimpleNamespace(id=task.room_id)
    db = AsyncMock()
    res = MagicMock()
    res.scalars.return_value.first.return_value = task
    db.execute = AsyncMock(return_value=res)
    db.get = AsyncMock(return_value=SimpleNamespace(nickname="front"))
    row = SimpleNamespace(id=uuid.uuid4(), message_id=uuid.uuid4())
    msg_id = uuid.uuid4()
    decision = RouteDecision(route="interject_task", matched_task_id=task.id,
                             confidence=1.0, reasoning="r")
    with (
        patch.object(task_interject, "create", new=AsyncMock(return_value=row)) as create,
        patch.object(task_status_probe, "collect", new=AsyncMock(return_value={
            "verdict": {"level": "stuck", "text": "The model has not responded for 7 min."}})),
        patch("app.core.events.emit_event", new=AsyncMock()) as emit,
    ):
        ok = await chat_mod._handle_interject(room, SimpleNamespace(id=uuid.uuid4()),
                                              "@front status?", decision, db, msg_id)
    assert ok
    assert create.await_args.kwargs["existing_message_id"] == msg_id
    kw = emit.await_args.kwargs
    assert kw["event_type"] == "TASK_INTERJECTION_ROUTED"
    assert "@front" in kw["content"] and "The model has not responded" in kw["content"]
    assert kw["metadata"]["verdict"] == "stuck"


@pytest.mark.asyncio
async def test_handle_interject_refusal_returns_false():
    from app.api.v1 import chat as chat_mod
    from app.core import task_interject

    task = SimpleNamespace(id=uuid.uuid4(), room_id=uuid.uuid4(), title="t", assigned_agent_id=None)
    db = AsyncMock()
    res = MagicMock()
    res.scalars.return_value.first.return_value = task
    db.execute = AsyncMock(return_value=res)
    decision = RouteDecision(route="interject_task", matched_task_id=task.id,
                             confidence=1.0, reasoning="r")
    with patch.object(task_interject, "create",
                      new=AsyncMock(side_effect=task_interject.InterjectRefused("not_running", "x"))):
        ok = await chat_mod._handle_interject(SimpleNamespace(id=task.room_id),
                                              SimpleNamespace(id=uuid.uuid4()),
                                              "@front status?", decision, db, None)
    assert ok is False
