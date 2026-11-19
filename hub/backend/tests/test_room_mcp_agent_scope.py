"""Room MCP agent scope — stage E1 (internal design notes).

DB part: ``room_mcp_tools`` filters by the agent's scope on a real
PostgreSQL schema; default Market MCPs are ordinary ``room_mcp_server`` rows
(§3.0); removing an agent or unassigning a server never widens access.

Unit part: the task proxy loop builds separate MCP sets for the executor and
for ChatMaster, each from its own scoped discovery.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.models import Agent, Room, RoomAgentLink, RoomUserLink, User
from app.models.market import (
    RoomMcpScope,
    RoomMcpScopeAgent,
    RoomMcpServer,
    UserDefaultProvision,
    UserInstalledItem,
)


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sessions():
    url = os.environ.get("DATABASE_URL")
    if not url or not (make_url(url).database or "").endswith("_test"):
        pytest.skip("DATABASE_URL must point to a PostgreSQL database ending in _test")

    schema = f"mcp_scope_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def _world(db, *, servers=("srv-a", "srv-b")):
    """A user, one room with agents A and B, and MCP servers attached to it."""
    user = User(email=f"{uuid4().hex}@example.test", hashed_password="x")
    db.add(user)
    await db.flush()
    room = Room(name="scope", creator_id=user.id)
    db.add(room)
    await db.flush()
    db.add(RoomUserLink(room_id=room.id, user_id=user.id))
    agents = []
    for nick in ("alpha", "beta"):
        agent = Agent(owner_id=user.id, nickname=nick, agent_uid=f"{nick}-{uuid4().hex[:6]}", token=uuid4().hex)
        db.add(agent)
        await db.flush()
        db.add(RoomAgentLink(room_id=room.id, agent_id=agent.id))
        agents.append(agent)
    items = {}
    for slug in servers:
        item = UserInstalledItem(
            user_id=user.id, type="mcp", name=slug, market_slug=slug,
            status="active", is_custom=False,
        )
        db.add(item)
        await db.flush()
        db.add(RoomMcpServer(room_id=room.id, user_installed_item_id=item.id, added_by=user.id))
        items[slug] = item
    await db.commit()
    return user, room, agents, items


@pytest.fixture
def fake_gateway(monkeypatch):
    """Every Market MCP answers discovery with one tool named ``ping``."""
    monkeypatch.setattr(
        "app.core.mcp_gateway._gateway_get",
        AsyncMock(return_value={"tools": [{"name": "ping", "description": "d", "inputSchema": {}}]}),
    )
    monkeypatch.setattr("app.core.mcp_gateway.emit_event", AsyncMock())
    monkeypatch.setattr("app.core.mcp_credits.get_wallet_balance_mcr", AsyncMock(return_value=10**9))


async def _slugs(db, room, user, agent_id):
    from app.core.mcp_gateway import room_mcp_tools

    _ant, _oai, bindings = await room_mcp_tools(room.id, user.id, db, agent_id=agent_id)
    return sorted({b.market_slug for b in bindings.values()})


async def _select(db, room, item, agent_ids):
    db.add(RoomMcpScope(room_id=room.id, user_installed_item_id=item.id, mode="selected"))
    await db.flush()
    for agent_id in agent_ids:
        db.add(RoomMcpScopeAgent(room_id=room.id, user_installed_item_id=item.id, agent_id=agent_id))
    await db.commit()


# ---------------------------------------------------------------------------
# room_mcp_tools scope filter
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.asyncio
async def test_no_scope_row_means_every_agent(sessions, fake_gateway):
    async with sessions() as db:
        user, room, (a, b), _items = await _world(db)
        assert await _slugs(db, room, user, a.id) == ["srv-a", "srv-b"]
        assert await _slugs(db, room, user, b.id) == ["srv-a", "srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_selected_scope_reaches_only_listed_agents(sessions, fake_gateway):
    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        await _select(db, room, items["srv-a"], [a.id])
        assert await _slugs(db, room, user, a.id) == ["srv-a", "srv-b"]
        assert await _slugs(db, room, user, b.id) == ["srv-b"]
        # No agent filter (Hub-internal callers) sees everything attached.
        assert await _slugs(db, room, user, None) == ["srv-a", "srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_explicit_all_mode_behaves_like_no_row(sessions, fake_gateway):
    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        db.add(RoomMcpScope(room_id=room.id, user_installed_item_id=items["srv-a"].id, mode="all"))
        await db.commit()
        assert await _slugs(db, room, user, b.id) == ["srv-a", "srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_selected_with_nobody_reaches_nobody(sessions, fake_gateway):
    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        await _select(db, room, items["srv-a"], [])
        assert await _slugs(db, room, user, a.id) == ["srv-b"]
        assert await _slugs(db, room, user, b.id) == ["srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_out_of_scope_server_is_never_discovered(sessions, monkeypatch):
    """The filter runs before discovery: no probe, no failure for that agent."""
    from app.core.mcp_gateway import room_mcp_capabilities

    gateway = AsyncMock(side_effect=RuntimeError("down"))
    monkeypatch.setattr("app.core.mcp_gateway._gateway_get", gateway)
    monkeypatch.setattr("app.core.mcp_gateway.emit_event", AsyncMock())
    async with sessions() as db:
        user, room, (a, b), items = await _world(db, servers=("srv-a",))
        await _select(db, room, items["srv-a"], [a.id])
        _ant, _oai, bindings, failures = await room_mcp_capabilities(room.id, user.id, db, agent_id=b.id)
    assert bindings == {}
    assert failures == []
    gateway.assert_not_awaited()


@pytest.mark.db
@pytest.mark.asyncio
async def test_removing_the_only_selected_agent_does_not_widen_to_all(sessions, fake_gateway, monkeypatch):
    from app.core.room_members import remove_agent_from_room

    monkeypatch.setattr("app.core.user_sessions.turn_in_progress", lambda _sid: False)
    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        await _select(db, room, items["srv-a"], [a.id])
        await remove_agent_from_room(room, a.id, db)
        await db.commit()

        scope = await db.get(RoomMcpScope, (room.id, items["srv-a"].id))
        assert scope is not None and scope.mode == "selected"
        left = (await db.execute(select(RoomMcpScopeAgent))).scalars().all()
        assert left == []
        assert await _slugs(db, room, user, b.id) == ["srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_unassigning_a_server_drops_its_scope(sessions, fake_gateway):
    from app.api.v1.rooms import assign_room_mcp_server, unassign_room_mcp_server

    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        await _select(db, room, items["srv-a"], [a.id])
        await unassign_room_mcp_server(room.id, items["srv-a"].id, db, user)
        assert await db.get(RoomMcpScope, (room.id, items["srv-a"].id)) is None
        assert (await db.execute(select(RoomMcpScopeAgent))).scalars().all() == []

        # Re-assigned → starts from "all agents".
        await assign_room_mcp_server(room.id, items["srv-a"].id, db, user)
        assert await _slugs(db, room, user, b.id) == ["srv-a", "srv-b"]


# ---------------------------------------------------------------------------
# §3.0 defaults are room rows
# ---------------------------------------------------------------------------


async def _default_item(db, user, slug="mcp-vision"):
    market_item_id = uuid4()
    item = UserInstalledItem(
        user_id=user.id, type="mcp", name=slug, market_slug=slug, market_item_id=market_item_id,
        status="active", is_custom=False,
    )
    db.add(item)
    db.add(UserDefaultProvision(user_id=user.id, market_item_id=market_item_id, market_slug=slug))
    await db.commit()
    return item


@pytest.mark.db
@pytest.mark.asyncio
async def test_ledger_alone_no_longer_reaches_agents(sessions, fake_gateway, monkeypatch):
    """A default with no room row is not offered: Disable really disables."""
    monkeypatch.setattr("app.core.mcp_gateway.legion_wallet.central", lambda: False)
    async with sessions() as db:
        user, room, (a, _b), _items = await _world(db, servers=())
        await _default_item(db, user)
        assert await _slugs(db, room, user, a.id) == []


@pytest.mark.db
@pytest.mark.asyncio
async def test_new_room_gets_default_mcps_only(sessions):
    from app.core.skill_defaults import attach_default_mcps_to_room

    async with sessions() as db:
        user, room, _agents, items = await _world(db, servers=("srv-a",))
        vision = await _default_item(db, user)
        # srv-a is not in the ledger, so a new room must not get it.
        await db.execute(delete(RoomMcpServer).where(RoomMcpServer.room_id == room.id))
        await attach_default_mcps_to_room(room.id, user.id, db)
        await db.commit()
        rows = (await db.execute(select(RoomMcpServer.user_installed_item_id).where(RoomMcpServer.room_id == room.id))).scalars().all()
        assert rows == [vision.id]
        # Idempotent.
        await attach_default_mcps_to_room(room.id, user.id, db)
        await db.commit()
        rows = (await db.execute(select(RoomMcpServer).where(RoomMcpServer.room_id == room.id))).scalars().all()
        assert len(rows) == 1
        assert items["srv-a"].id not in {r.user_installed_item_id for r in rows}


@pytest.mark.db
@pytest.mark.asyncio
async def test_first_default_install_reaches_every_room(sessions):
    from app.core.skill_defaults import attach_mcp_to_user_rooms

    async with sessions() as db:
        user, room, _agents, _items = await _world(db, servers=())
        second = Room(name="second", creator_id=user.id)
        db.add(second)
        await db.commit()
        vision = await _default_item(db, user)
        await attach_mcp_to_user_rooms(vision.id, user.id, db)
        await db.commit()
        await attach_mcp_to_user_rooms(vision.id, user.id, db)  # idempotent
        await db.commit()
        rooms = (await db.execute(select(RoomMcpServer.room_id).where(RoomMcpServer.user_installed_item_id == vision.id))).scalars().all()
        assert sorted(rooms) == sorted([room.id, second.id])


def _backfill_statements() -> list[str]:
    from app.db.session import _MIGRATIONS

    out = [sql for sql in _MIGRATIONS if "hub_data_migration" in sql]
    assert len(out) == 2, "expected the marker table and the guarded backfill"
    return out


@pytest.mark.db
@pytest.mark.asyncio
async def test_default_backfill_runs_exactly_once(sessions):
    async with sessions() as db:
        user, room, _agents, items = await _world(db, servers=("srv-a",))
        vision = await _default_item(db, user)
        # The ledger also lists skills; a non-MCP item must not get a row.
        skill_mid = uuid4()
        db.add(UserInstalledItem(user_id=user.id, type="skill", name="s", market_slug="skill-x",
                                 market_item_id=skill_mid, status="active", is_custom=False))
        db.add(UserDefaultProvision(user_id=user.id, market_item_id=skill_mid, market_slug="skill-x"))
        await db.commit()

        for sql in _backfill_statements():
            await db.execute(text(sql))
        await db.commit()
        rows = set((await db.execute(select(RoomMcpServer.user_installed_item_id).where(RoomMcpServer.room_id == room.id))).scalars().all())
        assert rows == {items["srv-a"].id, vision.id}

        # The user disables it; the next startup must not bring it back.
        await db.execute(delete(RoomMcpServer).where(RoomMcpServer.user_installed_item_id == vision.id))
        await db.commit()
        for sql in _backfill_statements():
            await db.execute(text(sql))
        await db.commit()
        rows = set((await db.execute(select(RoomMcpServer.user_installed_item_id).where(RoomMcpServer.room_id == room.id))).scalars().all())
        assert rows == {items["srv-a"].id}


# ---------------------------------------------------------------------------
# §3.5 scope API (stage 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def quiet_events(monkeypatch):
    emitted = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emitted)
    return emitted


async def _listed_scopes(db, room, user):
    from app.api.v1.rooms import list_room_mcp_servers

    with patch("app.api.v1.rooms.mcp_connection_status", AsyncMock(return_value=None)):
        out = await list_room_mcp_servers(room.id, db, user)
    return {row.name: row.agent_scope for row in out}


@pytest.mark.db
@pytest.mark.asyncio
async def test_list_reports_scope_per_server(sessions):
    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        await _select(db, room, items["srv-a"], [b.id, a.id])
        scopes = await _listed_scopes(db, room, user)
        assert scopes["srv-b"].mode == "all" and scopes["srv-b"].agent_ids == []
        assert scopes["srv-a"].mode == "selected"
        assert set(scopes["srv-a"].agent_ids) == {a.id, b.id}


@pytest.mark.db
@pytest.mark.asyncio
async def test_put_scope_selected_then_all(sessions, fake_gateway, quiet_events):
    from app.api.v1.rooms import set_room_mcp_server_scope
    from app.schemas.mcp_connection import McpAgentScopeIn

    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        item = items["srv-a"]
        out = await set_room_mcp_server_scope(
            room.id, item.id, McpAgentScopeIn(mode="selected", agent_ids=[a.id, a.id]), db, user
        )
        assert out.mode == "selected" and out.agent_ids == [a.id]
        assert await _slugs(db, room, user, a.id) == ["srv-a", "srv-b"]
        assert await _slugs(db, room, user, b.id) == ["srv-b"]
        event = quiet_events.await_args.kwargs
        assert event["event_type"] == "ROOM_MCP_SCOPE_CHANGED"
        assert event["metadata"] == {"installed_item_id": str(item.id), "mode": "selected", "agent_ids": [str(a.id)]}

        # Idempotent: the same body again leaves one agent row.
        await set_room_mcp_server_scope(room.id, item.id, McpAgentScopeIn(mode="selected", agent_ids=[a.id]), db, user)
        assert len((await db.execute(select(RoomMcpScopeAgent))).scalars().all()) == 1

        # "all" clears the list, even if the body carries agent ids.
        out = await set_room_mcp_server_scope(room.id, item.id, McpAgentScopeIn(mode="all", agent_ids=[a.id]), db, user)
        assert out.mode == "all" and out.agent_ids == []
        assert (await db.execute(select(RoomMcpScopeAgent))).scalars().all() == []
        assert await _slugs(db, room, user, b.id) == ["srv-a", "srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_put_scope_selected_nobody(sessions, fake_gateway, quiet_events):
    from app.api.v1.rooms import set_room_mcp_server_scope
    from app.schemas.mcp_connection import McpAgentScopeIn

    async with sessions() as db:
        user, room, (a, b), items = await _world(db)
        out = await set_room_mcp_server_scope(room.id, items["srv-a"].id, McpAgentScopeIn(mode="selected"), db, user)
        assert out.mode == "selected" and out.agent_ids == []
        assert await _slugs(db, room, user, a.id) == ["srv-b"]
        assert await _slugs(db, room, user, b.id) == ["srv-b"]


@pytest.mark.db
@pytest.mark.asyncio
async def test_put_scope_accepts_the_room_chat_master(sessions, quiet_events):
    from app.api.v1.rooms import set_room_mcp_server_scope
    from app.schemas.mcp_connection import McpAgentScopeIn

    async with sessions() as db:
        user, room, _agents, items = await _world(db)
        cm = Agent(owner_id=user.id, nickname="cm", agent_uid=f"cm-{uuid4().hex[:6]}", token=uuid4().hex)
        db.add(cm)
        await db.flush()
        room.coordinator_config = {"agent_id": str(cm.id)}
        await db.commit()
        out = await set_room_mcp_server_scope(
            room.id, items["srv-a"].id, McpAgentScopeIn(mode="selected", agent_ids=[cm.id]), db, user
        )
        assert out.agent_ids == [cm.id]


@pytest.mark.db
@pytest.mark.asyncio
async def test_put_scope_rejections(sessions, quiet_events):
    from fastapi import HTTPException

    from app.api.v1.rooms import set_room_mcp_server_scope
    from app.schemas.mcp_connection import McpAgentScopeIn

    async with sessions() as db:
        user, room, (a, _b), items = await _world(db)
        stranger = Agent(owner_id=user.id, nickname="out", agent_uid=f"out-{uuid4().hex[:6]}", token=uuid4().hex)
        db.add(stranger)
        await db.commit()

        # An agent that is not in the room → 422, nothing written.
        with pytest.raises(HTTPException) as exc:
            await set_room_mcp_server_scope(
                room.id, items["srv-a"].id, McpAgentScopeIn(mode="selected", agent_ids=[a.id, stranger.id]), db, user
            )
        assert exc.value.status_code == 422
        assert exc.value.detail == {"code": "agent_not_in_room", "agent_ids": [str(stranger.id)]}
        assert await db.get(RoomMcpScope, (room.id, items["srv-a"].id)) is None

        # Someone else's server → 403.
        other_user, _r, _ag, other_items = await _world(db, servers=("srv-x",))
        with pytest.raises(HTTPException) as exc:
            await set_room_mcp_server_scope(room.id, other_items["srv-x"].id, McpAgentScopeIn(mode="all"), db, user)
        assert exc.value.status_code == 403

        # Own server that is not attached to this room → 409.
        loose = UserInstalledItem(user_id=user.id, type="mcp", name="loose", market_slug="loose",
                                  status="active", is_custom=False)
        db.add(loose)
        await db.commit()
        with pytest.raises(HTTPException) as exc:
            await set_room_mcp_server_scope(room.id, loose.id, McpAgentScopeIn(mode="all"), db, user)
        assert exc.value.status_code == 409
        quiet_events.assert_not_awaited()


@pytest.mark.db
@pytest.mark.asyncio
async def test_unassign_then_reassign_lists_all(sessions, quiet_events):
    from app.api.v1.rooms import assign_room_mcp_server, set_room_mcp_server_scope, unassign_room_mcp_server
    from app.schemas.mcp_connection import McpAgentScopeIn

    async with sessions() as db:
        user, room, (a, _b), items = await _world(db)
        await set_room_mcp_server_scope(room.id, items["srv-a"].id, McpAgentScopeIn(mode="selected", agent_ids=[a.id]), db, user)
        await unassign_room_mcp_server(room.id, items["srv-a"].id, db, user)
        await assign_room_mcp_server(room.id, items["srv-a"].id, db, user)
        scopes = await _listed_scopes(db, room, user)
        assert scopes["srv-a"].mode == "all" and scopes["srv-a"].agent_ids == []


# ---------------------------------------------------------------------------
# Proxy loop: executor and ChatMaster get their own scoped sets (unit)
# ---------------------------------------------------------------------------


def _proxy_session() -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
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
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.last_activity_at = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    return s


def _binding(slug: str) -> MagicMock:
    b = MagicMock()
    b.market_slug = slug
    b.tool_name = "ping"
    return b


@pytest.mark.asyncio
async def test_proxy_loop_scopes_executor_and_chat_master_separately():
    from app.core.mcp_gateway import McpCapabilityFailure
    from app.core.proxy_session import _run_proxy_loop
    from app.models.task import DispatchStatus, TaskState

    agent_session = _proxy_session()
    cm_session = _proxy_session()
    task = MagicMock()
    task.id = uuid.uuid4()
    task.title = "Test task"
    task.description = "desc"
    task.state = TaskState.IN_PROGRESS
    task.dispatch_status = DispatchStatus.READY
    owner_id = uuid.uuid4()

    agent_row = MagicMock()
    agent_row.owner_id = owner_id
    agent_row.nickname = "alpha"
    agent_row.tools_allowed = None
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = agent_row
    result.scalars.return_value.all.return_value = []
    result.first.return_value = None
    result.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    sets = {
        agent_session.agent_id: (
            [{"name": "mcp__exec__ping"}], [{"type": "function", "function": {"name": "mcp__exec__ping"}}],
            {"mcp__exec__ping": _binding("exec")}, [],
        ),
        cm_session.agent_id: (
            [{"name": "mcp__cm__ping"}], [{"type": "function", "function": {"name": "mcp__cm__ping"}}],
            {"mcp__cm__ping": _binding("cm")},
            [McpCapabilityFailure(installed_item_id=uuid.uuid4(), display_name="cm-broken",
                                  code="mcp_unreachable", message="m", remedy="r")],
        ),
    }

    async def fake_caps(room_id, user_id, _db, *, agent_id):
        assert user_id == owner_id
        return sets[agent_id]

    calls: list[dict] = []

    def fake_llm(*args, **kwargs):
        calls.append({"sys": args[1], **kwargs})
        return MagicMock()

    llm_result = ("Agent reply", None, 50, 100, "m", 50, 50)
    cm_close = (None, MagicMock(), 40, 80, "m", 40, 40)
    with (
        patch("app.core.mcp_gateway.room_mcp_capabilities", side_effect=fake_caps) as caps,
        patch("app.core.proxy_session._call_llm_with_tools", new=MagicMock(side_effect=fake_llm)),
        patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock,
              return_value={"provider": "anthropic", "model": "m", "api_key": "x"}),
        patch("app.core.vision_dispatch.agent_can_see_images", new_callable=AsyncMock, return_value=True),
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.proxy_session._call_with_heartbeat", new_callable=AsyncMock,
              side_effect=[llm_result, cm_close]),
        patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.ProxyBroker") as broker_cls,
        patch("app.core.proxy_session._close_side", new_callable=AsyncMock,
              return_value=MagicMock(summary="done", files_changed=[], output=None)),
        patch("app.core.proxy_session._emit_closed", new_callable=AsyncMock),
        patch("app.core.proxy_session._render_close_note", return_value="[done]"),
        patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
    ):
        broker_cls.return_value = AsyncMock()
        await _run_proxy_loop(
            agent_session=agent_session, cm_session=cm_session, task=task,
            task_data_text="Do something", room_id=uuid.uuid4(), db=db,
        )

    assert {c.kwargs["agent_id"] for c in caps.await_args_list} == {agent_session.agent_id, cm_session.agent_id}
    by_actor = {}
    for c in calls:
        names = set(c.get("mcp_bindings") or {})
        by_actor.setdefault(c.get("session_id"), names)
    assert by_actor[agent_session.id] == {"mcp__exec__ping"}
    assert by_actor[cm_session.id] == {"mcp__cm__ping"}
    # ChatMaster's own failure note goes to ChatMaster, not to the executor.
    agent_sys = next(c["sys"] for c in calls if c.get("session_id") == agent_session.id)
    cm_sys = next(c["sys"] for c in calls if c.get("session_id") == cm_session.id)
    assert "cm-broken" not in agent_sys
    assert "cm-broken" in cm_sys


# ---------------------------------------------------------------------------
# E4 — ChatMaster's agent summary on the real schema
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.asyncio
async def test_briefs_and_agents_list_follow_scope(sessions):
    from app.core.mcp_gateway import room_mcp_briefs
    from app.core.memory_layer import _agents_list_short, _attach_agents_mcp, _render_agents_list

    async with sessions() as db:
        user, room, (a, b), items = await _world(db, servers=("srv-a", "srv-b", "srv-c"))
        await _select(db, room, items["srv-a"], [a.id])
        await _select(db, room, items["srv-c"], [])
        items["srv-b"].status = "disabled"
        db.add(items["srv-b"])
        await db.commit()

        briefs = {br.name: br for br in await room_mcp_briefs(room.id, db)}
        assert set(briefs) == {"srv-a", "srv-c"}  # inactive item is not listed
        assert briefs["srv-a"].available_to(a.id) and not briefs["srv-a"].available_to(b.id)

        agents = await _agents_list_short(room.id, db)
        unassigned = await _attach_agents_mcp(room.id, agents, db)
        by_nick = {ag["nickname"]: ag for ag in agents}
        assert by_nick["alpha"]["mcp"] == ["srv-a"]
        assert by_nick["beta"]["mcp"] == []
        assert unassigned == ["srv-c"]
        rendered = _render_agents_list(agents, unassigned)
        assert "MCP connected but assigned to no agent: srv-c" in rendered
