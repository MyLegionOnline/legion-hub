"""Jev routing (internal design notes Stage 3, core/router_jev.py + core/router.py).

Covers:
- build_jev_request: route options by mention count, goals, waiting tasks and
  the open session; ≥2 distinct mentions → no request; duplicate mentions count
  once; "mentions the same agent the conversation is with"
- decide: key maps → ids, p1 without a match → p2, risk hint
- route_user_message in Jev mode: a confident Jev answer skips the LLM, and all
  four escalations (multi_mention, low_confidence, waiting_guard, jev_error)
  reach the escalation LLM; inherit never calls Jev
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import router as router_mod
from app.core import router_jev as rj
from app.core.secrets.crypto import SecretValue
from app.models.room import Room, RoomType


def _agent_ctx(nick="dev", tasks=()):
    return {"id": str(uuid.uuid4()), "nickname": nick, "short_description": "",
            "tasks": [{"id": str(uuid.uuid4()), "title": t, "state": "COMPLETED",
                       "description": ""} for t in tasks]}


def _ctx(text="make a landing page", agents=(), goals=(), waiting=(), open_session=None,
         mentioned_ids=None):
    agents = list(agents)
    return {
        "text": text,
        "agents_ctx": agents,
        "goals_ctx": [{"id": str(uuid.uuid4()), "title": g} for g in goals],
        "waiting_ctx": [{"id": str(uuid.uuid4()), "title": w, "agent": "dev",
                         "kind": "review", "prompt": ""} for w in waiting],
        "open_session": open_session,
        "mentioned_ids": mentioned_ids if mentioned_ids is not None
        else [a["id"] for a in agents],
    }


def _options(body):
    return list(body["questions"]["route"]["criteria"])


class TestBuildJevRequest:
    def test_no_mentions_no_goals(self):
        body, _ = rj.build_jev_request(_ctx(), "typesafe/jev-1.13")
        assert body["model"] == "typesafe/jev-1.13"
        assert _options(body) == ["chatmaster_p2"]
        assert "goal" not in body["questions"] and "risk" in body["questions"]

    def test_no_mentions_goals_waiting_open_session(self):
        os_ = {"summary_so_far": "s", "last_assistant_excerpt": "a?",
               "has_open_question": True, "turns": 3, "with_agent_id": None,
               "with_agent": None, "with_chatmaster": True}
        body, maps = rj.build_jev_request(
            _ctx(goals=["G"], waiting=["W"], open_session=os_), "m")
        assert _options(body) == ["chatmaster_p1", "chatmaster_p2",
                                  "resume_waiting_task", "continue_session"]
        assert set(maps["goal"]) == {"g1"} and set(maps["task"]) == {"t1"}
        conv = body["state"]["open_conversation"]
        assert conv["with"].startswith("the ChatMaster")
        assert conv["assistant_asked_a_question"] is True

    def test_one_mention(self):
        a = _agent_ctx(tasks=["fix header"])
        body, maps = rj.build_jev_request(_ctx(agents=[a]), "m")
        assert _options(body) == ["single_agent_p1", "single_agent_p2"]
        assert maps["task"] == {"t1": a["tasks"][0]["id"]}
        assert body["state"]["mentioned_agent"]["nickname"] == "dev"

    def test_two_mentions_escalate(self):
        body, _ = rj.build_jev_request(_ctx(agents=[_agent_ctx("a"), _agent_ctx("b")]), "m")
        assert body is None

    def test_duplicate_mention_counts_once(self):
        a = _agent_ctx()
        body, _ = rj.build_jev_request(_ctx(agents=[a, a]), "m")
        assert body is not None and _options(body)[0] == "single_agent_p1"

    def test_mentions_the_conversation_peer(self):
        a = _agent_ctx()
        os_ = {"summary_so_far": "", "last_assistant_excerpt": "", "has_open_question": False,
               "turns": 2, "with_agent_id": a["id"], "with_agent": "dev",
               "with_chatmaster": False}
        body, _ = rj.build_jev_request(_ctx(agents=[a], open_session=os_), "m")
        conv = body["state"]["open_conversation"]
        assert conv["with"] == "agent @dev"
        assert conv["message_mentions_this_same_agent"] is True
        assert "continue_session" in _options(body)


class TestDecide:
    def test_maps_ids(self):
        maps = {"goal": {"g1": "G-ID"}, "task": {}}
        d = rj.decide({"route": {"choice": "chatmaster_p1", "confidence": 0.9,
                                 "probabilities": {"chatmaster_p1": 0.9}},
                       "goal": {"choice": "g1"}, "risk": {"choice": "risky"}}, maps)
        assert (d["route"], d["goal_id"], d["risk"]) == ("chatmaster_p1", "G-ID", "risky")

    @pytest.mark.parametrize("route,fallback", [
        ("chatmaster_p1", "chatmaster_p2"),
        ("single_agent_p1", "single_agent_p2"),
        ("resume_waiting_task", "chatmaster_p2"),
    ])
    def test_p1_without_match_falls_to_p2(self, route, fallback):
        d = rj.decide({"route": {"choice": route, "confidence": 0.8},
                       "goal": {"choice": "none"}, "task": {"choice": "none"}},
                      {"goal": {}, "task": {}})
        assert d["route"] == fallback

    def test_normal_risk_is_no_hint(self):
        d = rj.decide({"route": {"choice": "chatmaster_p2", "confidence": 0.8},
                       "risk": {"choice": "normal"}}, {"goal": {}, "task": {}})
        assert d["risk"] is None


# ---------------------------------------------------------------------------
# route_user_message in Jev mode
# ---------------------------------------------------------------------------


def _agent(nick, **llm):
    return SimpleNamespace(id=uuid.uuid4(), owner_id=uuid.uuid4(), nickname=nick,
                           system_role=None, llm_config=llm)


def _router(mode="jev", conf=0.6, **llm):
    return SimpleNamespace(
        id=uuid.uuid4(), owner_id=uuid.uuid4(), nickname="Router", system_role="router",
        llm_config={**llm, "router": {"mode": mode, "jev_min_confidence": conf,
                                      "secret_binding_id": str(uuid.uuid4())}},
    )


def _jev_reply(route="chatmaster_p2", conf=0.95, **answers):
    return ({"answers": {"route": {"choice": route, "confidence": conf,
                                   "probabilities": {route: conf}}, **answers},
             "usage": {"cost": 0.00006}}, 0.3)


def _llm_raw(route="chatmaster_p2"):
    return (json.dumps({"route": route, "confidence": 0.9, "reasoning": "llm"}),
            10, 5, "cheap")


async def _route(*, gathered, router, jev=None, jev_exc=None, secret="k",
                 open_session=None, text="make a landing page"):
    cm = gathered.room_agents[0] if gathered.room_agents else None
    room = Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="r")
    room.room_type = RoomType.group
    room.coordinator_config = {"master_agent_id": str(cm.id)} if cm else {}
    call_jev = AsyncMock(return_value=jev, side_effect=jev_exc)
    db = AsyncMock()
    db.add = MagicMock()
    with (
        patch.object(router_mod, "_gather_context", new=AsyncMock(return_value=gathered)),
        patch.object(router_mod, "get_router_agent", new=AsyncMock(return_value=router)),
        patch.object(router_mod, "resolve_agent_secret",
                     new=AsyncMock(return_value=SecretValue(secret) if secret else None)),
        patch.object(rj, "call_jev", new=call_jev),
        patch.object(router_mod, "_call_router_llm",
                     new=AsyncMock(return_value=_llm_raw())) as llm,
        patch.object(router_mod, "emit_event", new=AsyncMock()) as emit,
    ):
        decision = await router_mod.route_user_message(
            room, text, db, open_session=open_session)
    decided = next(c for c in emit.call_args_list if c.kwargs["event_type"] == "ROUTE_DECIDED")
    _route.last_log = db.add.call_args.args[0]
    return decision, call_jev, llm, decided.kwargs["metadata"]


def _gathered(agents_ctx=(), waiting=(), goals=()):
    cm = _agent("cm", provider="anthropic", model="expensive")
    return router_mod._GatheredContext(
        list(agents_ctx), [{"id": str(uuid.uuid4()), "title": g} for g in goals],
        [uuid.UUID(a["id"]) for a in agents_ctx], cm.llm_config, cm,
        waiting_ctx=[{"id": str(uuid.uuid4()), "title": w, "agent": "dev",
                      "kind": "review", "prompt": ""} for w in waiting],
        room_agents=[cm],
    )


class TestRouteWithJev:
    @pytest.mark.asyncio
    async def test_confident_jev_skips_llm(self):
        d, jev, llm, meta = await _route(gathered=_gathered(), router=_router(),
                                         jev=_jev_reply())
        assert d.route == "chatmaster_p2" and d.reasoning.startswith("jev:")
        jev.assert_awaited_once()
        llm.assert_not_awaited()
        assert meta["cost_usd"] == pytest.approx(0.00006)

    @pytest.mark.asyncio
    async def test_goal_match_maps_to_uuid(self):
        g = _gathered(goals=["Landing"])
        d, *_ = await _route(gathered=g, router=_router(),
                             jev=_jev_reply("chatmaster_p1", goal={"choice": "g1"}))
        assert d.route == "chatmaster_p1"
        assert str(d.matched_goal_id) == g.goals_ctx[0]["id"]

    @pytest.mark.asyncio
    async def test_low_confidence_escalates(self):
        d, jev, llm, _ = await _route(gathered=_gathered(), router=_router(conf=0.7),
                                      jev=_jev_reply(conf=0.65))
        jev.assert_awaited_once()
        llm.assert_awaited_once()
        assert d.reasoning.startswith("jev→llm (low_confidence)")

    @pytest.mark.asyncio
    async def test_multi_mention_never_calls_jev(self):
        g = _gathered(agents_ctx=[_agent_ctx("a"), _agent_ctx("b")])
        d, jev, llm, _ = await _route(gathered=g, router=_router(), jev=_jev_reply())
        jev.assert_not_awaited()
        llm.assert_awaited_once()
        assert "multi_mention" in d.reasoning

    @pytest.mark.asyncio
    async def test_waiting_task_guards_continue_session(self):
        osc = router_mod.OpenSessionContext(session_id=uuid.uuid4(), kind="user_cm_p2",
                                            turns=4, peer_is_chatmaster=True)
        d, jev, llm, _ = await _route(
            gathered=_gathered(waiting=["W"]), router=_router(),
            jev=_jev_reply("continue_session", conf=0.91), open_session=osc,
            text="let's continue with the task and check everything once more")
        llm.assert_awaited_once()
        assert "waiting_guard" in d.reasoning

    @pytest.mark.asyncio
    async def test_jev_error_escalates(self):
        d, jev, llm, _ = await _route(gathered=_gathered(), router=_router(),
                                      jev_exc=RuntimeError("Jev HTTP 503"))
        llm.assert_awaited_once()
        assert d.route == "chatmaster_p2" and "jev_error" in d.reasoning

    @pytest.mark.asyncio
    async def test_missing_jev_key_escalates(self):
        d, jev, llm, _ = await _route(gathered=_gathered(), router=_router(),
                                      jev=_jev_reply(), secret=None)
        jev.assert_not_awaited()
        llm.assert_awaited_once()
        assert "jev_error" in d.reasoning

    @pytest.mark.asyncio
    async def test_escalation_uses_router_model(self):
        _, _, llm, _ = await _route(
            gathered=_gathered(), router=_router(provider="openrouter", model="cheap"),
            jev=_jev_reply(conf=0.2))
        assert llm.await_args.args[1]["model"] == "cheap"

    @pytest.mark.asyncio
    async def test_inherit_never_calls_jev(self):
        d, jev, llm, _ = await _route(gathered=_gathered(), router=_router(mode="inherit"),
                                      jev=_jev_reply())
        jev.assert_not_awaited()
        llm.assert_awaited_once()
        assert not d.reasoning.startswith("jev")


class TestJevLogging:
    """Stage 4: engine / escalation_reason / Jev probabilities reach the
    ROUTE_DECIDED metadata and the router_decision_log row."""

    @pytest.mark.asyncio
    async def test_jev_decision(self):
        _, _, _, meta = await _route(gathered=_gathered(), router=_router(), jev=_jev_reply())
        assert (meta["engine"], meta["escalation_reason"]) == ("jev", None)
        assert meta["jev_confidence"] == 0.95 and meta["jev_latency_ms"] == 300
        log = _route.last_log
        assert (log.engine, log.escalation_reason, log.outcome) == ("jev", None, "llm")
        assert log.jev_probabilities == {"chatmaster_p2": 0.95}
        assert log.model_used == "typesafe/jev-1.13"

    @pytest.mark.asyncio
    async def test_low_confidence_keeps_probabilities(self):
        _, _, _, meta = await _route(gathered=_gathered(), router=_router(),
                                     jev=_jev_reply(conf=0.4))
        assert (meta["engine"], meta["escalation_reason"]) == ("llm", "low_confidence")
        assert meta["jev_confidence"] == 0.4
        assert _route.last_log.jev_probabilities == {"chatmaster_p2": 0.4}

    @pytest.mark.asyncio
    async def test_jev_error_has_no_probabilities(self):
        _, _, _, meta = await _route(gathered=_gathered(), router=_router(),
                                     jev_exc=RuntimeError("boom"))
        assert (meta["engine"], meta["escalation_reason"]) == ("llm", "jev_error")
        assert meta["jev_confidence"] is None
        assert _route.last_log.jev_probabilities is None

    @pytest.mark.asyncio
    async def test_inherit_is_plain_llm(self):
        _, _, _, meta = await _route(gathered=_gathered(), router=_router(mode="inherit"),
                                     jev=_jev_reply())
        assert (meta["engine"], meta["escalation_reason"]) == ("llm", None)
        assert _route.last_log.engine == "llm"

    @pytest.mark.asyncio
    async def test_shortcut(self):
        osc = router_mod.OpenSessionContext(session_id=uuid.uuid4(), kind="user_cm_p2", turns=2)
        _, jev, _, meta = await _route(gathered=_gathered(), router=_router(),
                                       jev=_jev_reply(), open_session=osc, text="ok")
        jev.assert_not_awaited()
        assert meta["engine"] == "shortcut" and _route.last_log.engine == "shortcut"


class TestDryRun:
    """Stage 5 "Test a route": the same decision, no events and no log row."""

    @pytest.mark.asyncio
    async def test_dry_run_reports_without_side_effects(self):
        g = _gathered()
        cm = g.room_agents[0]
        room = Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="r")
        room.room_type = RoomType.group
        room.coordinator_config = {"master_agent_id": str(cm.id)}
        db = AsyncMock()
        db.add = MagicMock()
        report: dict = {}
        with (
            patch.object(router_mod, "_gather_context", new=AsyncMock(return_value=g)),
            patch.object(router_mod, "get_router_agent", new=AsyncMock(return_value=_router())),
            patch.object(router_mod, "resolve_agent_secret",
                         new=AsyncMock(return_value=SecretValue("k"))),
            patch.object(rj, "call_jev", new=AsyncMock(return_value=_jev_reply(conf=0.3))),
            patch.object(router_mod, "_call_router_llm",
                         new=AsyncMock(return_value=_llm_raw())),
            patch.object(router_mod, "emit_event", new=AsyncMock()) as emit,
        ):
            d = await router_mod.route_user_message(
                room, "make a landing page", db, dry_run=True, report=report)
        emit.assert_not_awaited()
        db.add.assert_not_called()
        db.commit.assert_not_awaited()
        assert d.route == "chatmaster_p2"
        assert report["engine"] == "llm" and report["escalation_reason"] == "low_confidence"
        assert report["jev_route"] == "chatmaster_p2" and report["jev_confidence"] == 0.3
        assert report["jev_probabilities"] == {"chatmaster_p2": 0.3}
        assert report["model"] == "cheap"


class TestRouterTestEndpoint:
    @pytest.mark.asyncio
    async def test_returns_report_and_ownership(self):
        from app.api.v1 import router_check as api

        user = SimpleNamespace(id=uuid.uuid4())
        room = SimpleNamespace(id=uuid.uuid4(), creator_id=user.id)

        async def fake_route(room_, text, db, *, open_session, dry_run, report):
            assert dry_run is True and open_session is None
            report.update(route="chatmaster_p2", confidence=0.97, reasoning="jev: x",
                          engine="jev", escalation_reason=None, model="typesafe/jev-1.13",
                          latency_ms=310, cost_usd=0.00006, jev_confidence=0.97)

        with (
            patch.object(api, "_require_room_member", new=AsyncMock(return_value=room)),
            patch.object(router_mod, "route_user_message", new=fake_route),
        ):
            out = await api.test_route(
                api.RouterTestIn(room_id=room.id, text="hi", use_open_session=False),
                db=AsyncMock(), current_user=user)
        assert (out.route, out.engine, out.latency_ms, out.own_router) == (
            "chatmaster_p2", "jev", 310, True)

    @pytest.mark.asyncio
    async def test_non_member_is_refused(self):
        from fastapi import HTTPException
        from app.api.v1 import router_check as api

        with patch.object(api, "_require_room_member",
                          new=AsyncMock(side_effect=HTTPException(403, "no"))):
            with pytest.raises(HTTPException):
                await api.test_route(api.RouterTestIn(room_id=uuid.uuid4(), text="hi"),
                                     db=AsyncMock(), current_user=SimpleNamespace(id=uuid.uuid4()))
