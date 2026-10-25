"""The system Router agent (internal design notes Stage 1, core/router_agent.py).

Covers:
- router_settings: an unconfigured Router behaves as inherit; Jev defaults
- validate_router_block: accepted and refused shapes of llm_config.router
- ensure_router_agent: returns the existing Router, creates a missing one,
  survives a concurrent create
- the Router never joins a room (add_agent_to_room, the HTTP mapping)
- agents API: the Router cannot be deleted, renamed, given skills or invoked;
  a bad router block is a 422
- the Jev key purpose is a valid agent binding
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.api.v1 import agents as agents_api
from app.api.v1 import rooms as rooms_api
from app.core import room_members as rm
from app.core import router_agent as ra
from app.models.agent import Agent
from app.models.secret import SecretPurpose, secret_binding_purpose_check
from app.schemas.agent import AgentUpdate


def _router(**llm):
    return SimpleNamespace(
        id=uuid.uuid4(), owner_id=uuid.uuid4(), nickname="Router",
        system_role="router", agent_type="internal", llm_config=llm,
    )


class TestRouterSettings:
    def test_unconfigured_is_inherit(self):
        s = ra.router_settings(_router(router={"mode": None}))
        assert s.mode == "inherit" and s.configured is False
        assert s.jev_model == "typesafe/jev-1.13"
        assert s.jev_min_confidence == 0.6
        assert s.has_escalation_model is False

    def test_missing_block_is_inherit(self):
        s = ra.router_settings(_router())
        assert s.mode == "inherit" and s.configured is False

    def test_jev_with_escalation_model(self):
        s = ra.router_settings(_router(
            provider="openrouter", model="z-ai/glm-5.3-flash",
            router={"mode": "jev", "jev_min_confidence": 0.7, "jev_model": "~typesafe/jev-latest"},
        ))
        assert (s.mode, s.configured, s.jev_min_confidence, s.jev_model) == (
            "jev", True, 0.7, "~typesafe/jev-latest")
        assert s.has_escalation_model is True

    def test_is_router(self):
        assert ra.is_router(_router())
        assert not ra.is_router(SimpleNamespace(system_role=None))
        assert not ra.is_router(None)


class TestValidateRouterBlock:
    @pytest.mark.parametrize("block", [
        {"mode": None},
        {"mode": "inherit"},
        {"mode": "jev", "jev_model": "typesafe/jev-1.13", "jev_min_confidence": 0.6,
         "secret_binding_id": str(uuid.uuid4())},
        {"mode": "jev", "jev_min_confidence": 1},
    ])
    def test_accepted(self, block):
        ra.validate_router_block({"router": block})

    def test_no_block_is_fine(self):
        ra.validate_router_block({"provider": "openrouter"})

    @pytest.mark.parametrize("block", [
        {"mode": "llm"},
        {"mode": "jev", "jev_min_confidence": 1.5},
        {"mode": "jev", "jev_min_confidence": True},
        {"mode": "jev", "jev_model": ""},
        {"mode": "jev", "api_key": "sk-..."},
        "jev",
    ])
    def test_refused(self, block):
        with pytest.raises(ValueError):
            ra.validate_router_block({"router": block})


def _db_returning(first):
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = first
    db.execute = AsyncMock(return_value=result)
    return db


class TestEnsureRouterAgent:
    @pytest.mark.asyncio
    async def test_existing_is_returned(self):
        existing = _router()
        db = _db_returning(existing)
        assert await ra.ensure_router_agent(existing.owner_id, db) is existing
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_is_created_unconfigured(self):
        owner = uuid.uuid4()
        db = _db_returning(None)
        agent = await ra.ensure_router_agent(owner, db)
        created = db.add.call_args.args[0]
        assert isinstance(created, Agent) and agent is created
        assert created.system_role == "router"
        assert created.nickname == "Router"
        assert created.agent_type == "internal"
        assert created.owner_id == owner
        assert created.llm_config == {"router": {"mode": None}}
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_create_returns_winner(self):
        winner = _router()
        db = AsyncMock()
        db.add = MagicMock()
        miss, hit = MagicMock(), MagicMock()
        miss.scalars.return_value.first.return_value = None
        hit.scalars.return_value.first.return_value = winner
        db.execute = AsyncMock(side_effect=[miss, hit])
        db.commit = AsyncMock(side_effect=IntegrityError("insert", {}, Exception("dup")))
        assert await ra.ensure_router_agent(winner.owner_id, db) is winner
        db.rollback.assert_awaited_once()


class TestNeverInARoom:
    @pytest.mark.asyncio
    async def test_add_agent_to_room_refuses_router(self):
        router = _router()
        db = AsyncMock()
        db.add = MagicMock()
        db.get = AsyncMock(return_value=router)
        from app.models.room import RoomType

        room = SimpleNamespace(id=uuid.uuid4(), room_type=RoomType.group)
        with pytest.raises(rm.RoomAgentMembershipError) as exc:
            await rm.add_agent_to_room(room, router.id, router.owner_id, db)
        assert exc.value.code == "system_agent"
        db.add.assert_not_called()

    def test_http_mapping_is_conflict(self):
        err = rooms_api._membership_http_error(rm.RoomAgentMembershipError("system_agent", "x"))
        assert err.status_code == 409 and err.detail["code"] == "system_agent"


class TestAgentsApiGuards:
    @pytest.mark.asyncio
    async def test_delete_refused(self):
        router = _router()
        db = AsyncMock()
        with patch.object(agents_api, "_get_owned_agent", AsyncMock(return_value=router)):
            with pytest.raises(HTTPException) as exc:
                await agents_api.delete_agent(router.id, db=db, current_user=SimpleNamespace(id=router.owner_id))
        assert exc.value.status_code == 409
        db.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_rename_refused(self):
        router = _router()
        with patch.object(agents_api, "_get_owned_agent", AsyncMock(return_value=router)):
            with pytest.raises(HTTPException) as exc:
                await agents_api.update_agent(
                    router.id, AgentUpdate(nickname="Dispatcher"),
                    db=AsyncMock(), current_user=SimpleNamespace(id=router.owner_id),
                )
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_bad_router_block_is_422(self):
        router = _router()
        with patch.object(agents_api, "_get_owned_agent", AsyncMock(return_value=router)):
            with pytest.raises(HTTPException) as exc:
                await agents_api.update_agent(
                    router.id, AgentUpdate(llm_config={"router": {"mode": "gpt"}}),
                    db=AsyncMock(), current_user=SimpleNamespace(id=router.owner_id),
                )
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_mode_switch_is_saved(self):
        router = _router(router={"mode": None})
        db = AsyncMock()
        db.add = MagicMock()
        with patch.object(agents_api, "_get_owned_agent", AsyncMock(return_value=router)), \
             patch.object(agents_api, "flag_modified", MagicMock()), \
             patch.object(agents_api, "_with_skill_names", AsyncMock(return_value=[router])):
            out = await agents_api.update_agent(
                router.id, AgentUpdate(nickname="Router", llm_config={"router": {"mode": "jev"}}),
                db=db, current_user=SimpleNamespace(id=router.owner_id),
            )
        assert out is router
        assert router.llm_config == {"router": {"mode": "jev"}}
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skills_refused(self):
        router = _router()
        with patch.object(agents_api, "_require_owned_agent", AsyncMock(return_value=router)):
            with pytest.raises(HTTPException) as exc:
                await agents_api.assign_agent_skill(
                    router.id, uuid.uuid4(), db=AsyncMock(),
                    current_user=SimpleNamespace(id=router.owner_id),
                )
        assert exc.value.status_code == 409


def test_jev_key_purpose_is_a_valid_binding():
    assert SecretPurpose.LLM_ROUTER_JEV.value == "llm_router_jev"
    assert "llm_router_jev" in secret_binding_purpose_check()
    from app.core.secrets import store
    assert SecretPurpose.LLM_ROUTER_JEV in store._SINGLE_SECRET_PURPOSES


# ---------------------------------------------------------------------------
# Stage 2 — resolving the Router into a RouterBinding (core/router.py §3.1)
# ---------------------------------------------------------------------------

from app.core import router as router_mod  # noqa: E402
from app.models.room import Room, RoomType  # noqa: E402


def _room_agent(nickname, **llm):
    return SimpleNamespace(
        id=uuid.uuid4(), owner_id=uuid.uuid4(), nickname=nickname,
        system_role=None, llm_config=llm or {"provider": "opencode_go", "model": "glm-5.3"},
    )


def _group_room(master=None):
    room = Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="r")
    room.room_type = RoomType.group
    room.coordinator_config = {"master_agent_id": str(master.id)} if master else {}
    return room


async def _resolve(room, agents, router):
    with patch.object(router_mod, "get_router_agent", new=AsyncMock(return_value=router)) as m:
        binding = await router_mod._resolve_router(room, agents, db=MagicMock())
    m.assert_awaited_once()
    assert m.await_args.args[0] == room.creator_id  # the room owner's Router
    return binding


class TestResolveRouter:
    @pytest.mark.asyncio
    async def test_missing_router_is_inherit(self):
        cm = _room_agent("cm")
        b = await _resolve(_group_room(cm), [cm], None)
        assert (b.engine, b.llm_agent, b.router) == ("llm", cm, None)

    @pytest.mark.asyncio
    async def test_lookup_failure_is_inherit(self):
        cm = _room_agent("cm")
        room = _group_room(cm)
        with patch.object(router_mod, "get_router_agent",
                          new=AsyncMock(side_effect=RuntimeError("db down"))):
            b = await router_mod._resolve_router(room, [cm], db=MagicMock())
        assert (b.engine, b.llm_agent) == ("llm", cm)

    @pytest.mark.asyncio
    async def test_non_router_row_is_ignored(self):
        cm = _room_agent("cm")
        b = await _resolve(_group_room(cm), [cm], MagicMock())
        assert (b.engine, b.llm_agent, b.router) == ("llm", cm, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", [None, "inherit"])
    async def test_unconfigured_and_inherit_use_chatmaster(self, mode):
        cm = _room_agent("cm")
        router = _router(provider="openrouter", model="x/y", router={"mode": mode})
        b = await _resolve(_group_room(cm), [cm], router)
        assert (b.engine, b.llm_agent, b.router) == ("llm", cm, router)

    @pytest.mark.asyncio
    async def test_jev_escalates_to_router_model(self):
        cm = _room_agent("cm")
        router = _router(provider="openrouter", model="x/y", router={"mode": "jev"})
        b = await _resolve(_group_room(cm), [cm], router)
        assert (b.engine, b.llm_agent) == ("jev", router)
        assert b.settings.jev_min_confidence == 0.6

    @pytest.mark.asyncio
    async def test_jev_without_router_model_escalates_to_chatmaster(self):
        cm = _room_agent("cm")
        router = _router(router={"mode": "jev"})
        b = await _resolve(_group_room(cm), [cm], router)
        assert (b.engine, b.llm_agent) == ("jev", cm)

    @pytest.mark.asyncio
    async def test_jev_without_any_model_has_no_llm_agent(self):
        b = await _resolve(_group_room(), [], _router(router={"mode": "jev"}))
        assert (b.engine, b.llm_agent) == ("jev", None)

    @pytest.mark.asyncio
    async def test_direct_room_escalates_to_its_agent(self):
        sole = _room_agent("sole")
        room = _group_room()
        room.room_type = RoomType.direct
        b = await _resolve(room, [sole], _router(router={"mode": "jev"}))
        assert (b.engine, b.llm_agent) == ("jev", sole)


class TestRouteUserMessageUsesBinding:
    """route_user_message speaks with the binding's LLM agent."""

    @staticmethod
    async def _route(room, gathered, router):
        from app.core.secrets.crypto import SecretValue

        raw = '{"route": "chatmaster_p2", "confidence": 0.9, "reasoning": "t"}'
        db = AsyncMock()
        with (
            patch.object(router_mod, "_gather_context", new=AsyncMock(return_value=gathered)),
            patch.object(router_mod, "get_router_agent", new=AsyncMock(return_value=router)),
            patch.object(router_mod, "resolve_agent_secret",
                         new=AsyncMock(return_value=SecretValue("k"))) as secret,
            patch.object(router_mod, "_call_router_llm",
                         new=AsyncMock(return_value=(raw, 1, 1, "m"))) as llm,
            patch.object(router_mod, "emit_event", new=AsyncMock()),
        ):
            decision = await router_mod.route_user_message(room, "make a landing page", db)
        return decision, llm.await_args.args[1], secret.await_args.kwargs["agent_id"]

    @pytest.mark.asyncio
    async def test_jev_mode_routes_with_router_model(self):
        cm = _room_agent("cm", provider="anthropic", model="expensive")
        room = _group_room(cm)
        router = _router(provider="openrouter", model="cheap", router={"mode": "jev"})
        gathered = router_mod._GatheredContext(
            [], [], [], cm.llm_config, cm, room_agents=[cm])
        decision, cfg, agent_id = await self._route(room, gathered, router)
        assert decision.route == "chatmaster_p2"
        assert cfg["model"] == "cheap" and agent_id == router.id

    @pytest.mark.asyncio
    async def test_inherit_routes_with_chatmaster_model(self):
        cm = _room_agent("cm", provider="anthropic", model="expensive")
        room = _group_room(cm)
        router = _router(provider="openrouter", model="cheap", router={"mode": "inherit"})
        gathered = router_mod._GatheredContext(
            [], [], [], cm.llm_config, cm, room_agents=[cm])
        _, cfg, agent_id = await self._route(room, gathered, router)
        assert cfg["model"] == "expensive" and agent_id == cm.id


class TestStartupBackfill:
    """Stage 6: the startup backfill runs on every boot (and every --reload), so
    it must never create a second Router for an account."""

    @staticmethod
    def _stmts():
        from app.db.session import _MIGRATIONS
        return [s for s in _MIGRATIONS if "system_role" in s]

    def test_unique_index_one_router_per_owner(self):
        idx = next(s for s in self._stmts() if "CREATE UNIQUE INDEX" in s)
        assert "ON agent(owner_id) WHERE system_role = 'router'" in idx

    def test_insert_is_idempotent(self):
        ins = next(s for s in self._stmts() if s.lstrip().startswith("INSERT INTO agent"))
        assert "WHERE NOT EXISTS" in ins and "a.system_role = 'router'" in ins
        assert ins.rstrip().endswith("ON CONFLICT DO NOTHING")

    def test_backfilled_router_is_unconfigured(self):
        ins = next(s for s in self._stmts() if s.lstrip().startswith("INSERT INTO agent"))
        assert '{"router": {"mode": null}}' in ins and "'Router'" in ins
        assert ra.ROUTER_DESCRIPTION in ins
