"""Room agent membership after creation (core/room_members.py).

Covers:
- remove_agent_from_room refuses: not a member, the room's ChatMaster, an
  agent with unfinished tasks (409 body lists them), an agent mid-reply
- remove_agent_from_room on success: closes open direct dialogues without a
  memory flush, unassigns unstarted plan steps, deletes memory and the link,
  announces the closed dialogues only after every change
- add_agent_to_room: only the user's own agents, never twice, fresh memory
- prepare_task_dispatch refuses an assignee that left the room
- the HTTP mapping of refusals
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import room_members as rm
from app.models.agent_memory import AgentMemory
from app.models.room import RoomAgentLink, RoomType
from app.models.task import TaskState


def _room(*, master_id=None, has_coordinator=True, room_type=RoomType.group):
    return SimpleNamespace(
        id=uuid.uuid4(),
        room_type=room_type,
        has_coordinator=has_coordinator,
        coordinator_config={"master_agent_id": str(master_id)} if master_id else {},
    )


def _agent(agent_id=None, owner_id=None, nickname="coder"):
    return SimpleNamespace(id=agent_id or uuid.uuid4(), owner_id=owner_id or uuid.uuid4(), nickname=nickname)


def _result(all_rows=None, rowcount=0):
    m = MagicMock()
    m.scalars.return_value.all.return_value = all_rows or []
    m.rowcount = rowcount
    return m


def _db(agent=None, execute_results=None):
    db = AsyncMock()
    db.add = MagicMock()
    db.get = AsyncMock(return_value=agent)
    db.execute = AsyncMock(side_effect=execute_results or [])
    return db


class TestRemoveRefusals:
    @pytest.mark.asyncio
    async def test_not_a_member(self):
        db = _db()
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=False)):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.remove_agent_from_room(_room(), uuid.uuid4(), db)
        assert exc.value.code == "not_member"
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chat_master_stays(self):
        agent = _agent()
        db = _db(agent)
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.remove_agent_from_room(_room(master_id=agent.id), agent.id, db)
        assert exc.value.code == "is_chat_master"
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_master_key_without_coordinator_does_not_block(self):
        """has_coordinator=False: a stale master id in the config is not a ChatMaster."""
        agent = _agent()
        db = _db(agent, [_result(), _result(), _result(), _result()])
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)), \
             patch.object(rm, "active_tasks_of_agent", AsyncMock(return_value=[])):
            await rm.remove_agent_from_room(
                _room(master_id=agent.id, has_coordinator=False), agent.id, db
            )

    @pytest.mark.asyncio
    async def test_unfinished_tasks_block_and_are_listed(self):
        agent = _agent(nickname="designer")
        task = SimpleNamespace(id=uuid.uuid4(), title="Landing hero", state=TaskState.IN_PROGRESS)
        db = _db(agent)
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)), \
             patch.object(rm, "active_tasks_of_agent", AsyncMock(return_value=[task])):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.remove_agent_from_room(_room(), agent.id, db)
        assert exc.value.code == "has_active_tasks"
        assert "@designer" in str(exc.value)
        assert exc.value.details["tasks"] == [
            {"id": str(task.id), "title": "Landing hero", "state": "in_progress"}
        ]
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_agent_mid_reply_blocks(self):
        agent = _agent()
        dialogue = SimpleNamespace(id=uuid.uuid4(), closed_at=None, close_reason=None)
        db = _db(agent, [_result(all_rows=[dialogue])])
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)), \
             patch.object(rm, "active_tasks_of_agent", AsyncMock(return_value=[])), \
             patch("app.core.user_sessions.turn_in_progress", return_value=True):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.remove_agent_from_room(_room(), agent.id, db)
        assert exc.value.code == "agent_busy"
        assert dialogue.closed_at is None

    @pytest.mark.asyncio
    async def test_direct_room_keeps_its_only_agent(self):
        agent = _agent()
        db = _db(agent)
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)), \
             patch.object(rm, "room_agent_ids", AsyncMock(return_value={agent.id})):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.remove_agent_from_room(_room(room_type=RoomType.direct), agent.id, db)
        assert exc.value.code == "direct_room"
        db.execute.assert_not_awaited()


class TestRemoveSuccess:
    @pytest.mark.asyncio
    async def test_closes_dialogues_unassigns_steps_drops_memory_and_link(self):
        agent = _agent()
        room = _room()
        dialogue = SimpleNamespace(id=uuid.uuid4(), closed_at=None, close_reason=None)
        db = _db(agent, [
            _result(all_rows=[dialogue]),   # open direct dialogues
            _result(rowcount=2),            # unassign unstarted plan steps
            _result(),                      # delete AgentMemory
            _result(),                      # delete RoomAgentLink
        ])
        order: list[str] = []
        db.flush = AsyncMock(side_effect=lambda: order.append("flush"))
        emit = AsyncMock(side_effect=lambda *a, **k: order.append("emit"))
        flush_memory = AsyncMock()
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)), \
             patch.object(rm, "active_tasks_of_agent", AsyncMock(return_value=[])), \
             patch("app.core.user_sessions.turn_in_progress", return_value=False), \
             patch("app.core.watchdog._emit_session_closed", emit), \
             patch("app.core.session_close.flush_session_memory", flush_memory):
            result = await rm.remove_agent_from_room(room, agent.id, db)

        assert result.closed_sessions == 1
        assert result.unassigned_plan_steps == 2
        assert dialogue.closed_at is not None
        assert dialogue.close_reason == "cancelled"
        emit.assert_awaited_once_with(dialogue, "cancelled", db)
        # emit_event commits — it must come after every membership change.
        assert order == ["flush", "emit"]
        flush_memory.assert_not_awaited()

        statements = [str(c.args[0]) for c in db.execute.await_args_list]
        assert "UPDATE goal_plan_step" in statements[1]
        assert statements[2].startswith("DELETE FROM agentmemory")
        assert statements[3].startswith("DELETE FROM roomagentlink")


class TestAdd:
    @pytest.mark.asyncio
    async def test_direct_room_refuses_a_second_agent(self):
        owner = uuid.uuid4()
        db = _db(_agent(owner_id=owner))
        with pytest.raises(rm.RoomAgentMembershipError) as exc:
            await rm.add_agent_to_room(_room(room_type=RoomType.direct), uuid.uuid4(), owner, db)
        assert exc.value.code == "direct_room"
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_foreign_agent_is_not_found(self):
        db = _db(_agent(owner_id=uuid.uuid4()))
        with pytest.raises(rm.RoomAgentMembershipError) as exc:
            await rm.add_agent_to_room(_room(), uuid.uuid4(), uuid.uuid4(), db)
        assert exc.value.code == "agent_not_found"
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_member(self):
        owner = uuid.uuid4()
        agent = _agent(owner_id=owner)
        db = _db(agent)
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=True)):
            with pytest.raises(rm.RoomAgentMembershipError) as exc:
                await rm.add_agent_to_room(_room(), agent.id, owner, db)
        assert exc.value.code == "already_member"
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_links_with_fresh_memory(self):
        owner = uuid.uuid4()
        agent = _agent(owner_id=owner)
        room = _room()
        db = _db(agent, [_result()])  # clear a leftover memory row
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=False)):
            out = await rm.add_agent_to_room(room, agent.id, owner, db)
        assert out is agent
        added = [c.args[0] for c in db.add.call_args_list]
        assert any(isinstance(o, RoomAgentLink) and o.agent_id == agent.id for o in added)
        assert any(isinstance(o, AgentMemory) and o.agent_id == agent.id for o in added)


class TestDispatchPreflight:
    @pytest.mark.asyncio
    async def test_assignee_outside_the_room_is_refused(self):
        from app.core.proxy_session import TaskDispatchPreflightError, prepare_task_dispatch
        from app.models.task import DispatchStatus

        agent = _agent(nickname="gone")
        task = SimpleNamespace(
            id=uuid.uuid4(), room_id=uuid.uuid4(), goal_id=None,
            assigned_agent_id=agent.id, state=TaskState.ASSIGNED,
            dispatch_status=DispatchStatus.READY,
        )
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = agent
        db = AsyncMock()
        db.execute = AsyncMock(return_value=agent_result)
        with patch.object(rm, "agent_in_room", AsyncMock(return_value=False)):
            with pytest.raises(TaskDispatchPreflightError) as exc:
                await prepare_task_dispatch(task, db)
        assert exc.value.code == "assigned_agent_not_in_room"
        assert "@gone" in str(exc.value)


class TestHttpMapping:
    def test_codes(self):
        from app.api.v1.rooms import _membership_http_error

        busy = _membership_http_error(
            rm.RoomAgentMembershipError("has_active_tasks", "busy", details={"tasks": []})
        )
        assert busy.status_code == 409
        assert busy.detail == {"code": "has_active_tasks", "message": "busy", "tasks": []}
        assert _membership_http_error(rm.RoomAgentMembershipError("not_member", "x")).status_code == 404
        assert _membership_http_error(rm.RoomAgentMembershipError("is_chat_master", "x")).status_code == 409
