"""Room agent membership after the room exists: add, remove, former members.

Removing an agent from a room is not deleting the agent. The Agent row stays,
so every task, plan step and message the agent ever touched keeps a valid
reference and its nickname. Only the membership (RoomAgentLink) and the agent's
memory in this room go away — memory is not archived, a returning agent starts
this room from scratch.

An agent with unfinished work in the room cannot be removed: its tasks would
be left with an executor the router and dispatcher no longer see. The user
finishes, cancels or reassigns them first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.router_agent import is_router
from app.models.agent import Agent
from app.models.agent_memory import AgentMemory
from app.models.goal import RoomGoal
from app.models.goal_plan import GoalPlan, GoalPlanStep, PlanStepStatus
from app.models.room import Room, RoomAgentLink, RoomType
from app.models.task import Task, TaskState

# A task in one of these states is finished for membership purposes: nothing
# will dispatch it again without an explicit user action, and that action
# (retry/approve) passes prepare_task_dispatch, which re-checks membership.
# REJECTED is here because it can be neither cancelled nor kept waiting — it is
# retried (with a new assignee if needed) or deleted.
_SETTLED_TASK_STATES: frozenset[TaskState] = frozenset({
    TaskState.COMPLETED,
    TaskState.CANCELLED,
    TaskState.CONTINUED,
    TaskState.REJECTED,
})

# Plan steps nobody started yet: on removal they lose the assignee and
# cm_self_plan picks a room agent for them, as for any unassigned step.
_UNSTARTED_STEP_STATUSES = (PlanStepStatus.PENDING, PlanStepStatus.BLOCKED)


class RoomAgentMembershipError(ValueError):
    """The membership change is refused; *code* is stable for the UI."""

    def __init__(self, code: str, message: str, *, details: dict | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


@dataclass
class RemovalResult:
    closed_sessions: int = 0
    unassigned_plan_steps: int = 0
    notes: list[str] = field(default_factory=list)


def room_chat_master_id(room: Room) -> UUID | None:
    """The ChatMaster of *room*, under any of the keys coordinator_config uses."""
    cfg = room.coordinator_config or {}
    raw = cfg.get("agent_id") or cfg.get("chat_master_agent_id") or cfg.get("master_agent_id")
    if not raw:
        return None
    try:
        return UUID(str(raw))
    except (ValueError, AttributeError):
        return None


async def room_agent_ids(room_id: UUID, db: AsyncSession) -> set[UUID]:
    rows = await db.execute(select(RoomAgentLink.agent_id).where(RoomAgentLink.room_id == room_id))
    return set(rows.scalars().all())


async def agent_in_room(room_id: UUID, agent_id: UUID, db: AsyncSession) -> bool:
    row = await db.execute(
        select(RoomAgentLink.agent_id).where(
            RoomAgentLink.room_id == room_id, RoomAgentLink.agent_id == agent_id
        )
    )
    return row.first() is not None


async def active_tasks_of_agent(room_id: UUID, agent_id: UUID, db: AsyncSession) -> list[Task]:
    rows = await db.execute(
        select(Task).where(
            Task.room_id == room_id,
            Task.assigned_agent_id == agent_id,
            Task.state.notin_(_SETTLED_TASK_STATES),
        ).order_by(Task.created_at)
    )
    return list(rows.scalars().all())


async def former_agents(room_id: UUID, db: AsyncSession) -> list[Agent]:
    """Agents referenced by this room's tasks or plan steps but no longer members."""
    task_refs = select(Task.assigned_agent_id).where(Task.room_id == room_id)
    creator_refs = select(Task.created_by_agent_id).where(Task.room_id == room_id)
    step_refs = (
        select(GoalPlanStep.assigned_agent_id)
        .join(GoalPlan, GoalPlan.id == GoalPlanStep.plan_id)
        .join(RoomGoal, RoomGoal.id == GoalPlan.goal_id)
        .where(RoomGoal.room_id == room_id)
    )
    referenced: set[UUID] = set()
    for query in (task_refs, creator_refs, step_refs):
        referenced.update(a for a in (await db.execute(query)).scalars().all() if a)
    gone = referenced - await room_agent_ids(room_id, db)
    if not gone:
        return []
    rows = await db.execute(select(Agent).where(Agent.id.in_(gone)).order_by(Agent.nickname))
    return list(rows.scalars().all())


async def add_agent_to_room(room: Room, agent_id: UUID, owner_id: UUID, db: AsyncSession) -> Agent:
    """Link one of *owner_id*'s agents to *room* with a fresh memory row."""
    if room.room_type == RoomType.direct:
        raise RoomAgentMembershipError(
            "direct_room",
            "A direct room has exactly one agent. Create a group room to work with several agents.",
        )
    agent = await db.get(Agent, agent_id)
    if agent is None or agent.owner_id != owner_id:
        raise RoomAgentMembershipError("agent_not_found", "agent not found")
    if is_router(agent):
        raise RoomAgentMembershipError(
            "system_agent", "Router is a system agent, it does not take part in rooms"
        )
    if await agent_in_room(room.id, agent_id, db):
        raise RoomAgentMembershipError("already_member", f"@{agent.nickname} is already in the room")

    db.add(RoomAgentLink(room_id=room.id, agent_id=agent_id))
    # A leftover row can exist only if an older code path skipped the cleanup;
    # the agent joins with empty memory either way.
    await db.execute(
        delete(AgentMemory).where(AgentMemory.room_id == room.id, AgentMemory.agent_id == agent_id)
    )
    db.add(AgentMemory(room_id=room.id, agent_id=agent_id))
    await db.flush()
    return agent


async def remove_agent_from_room(room: Room, agent_id: UUID, db: AsyncSession) -> RemovalResult:
    """Unlink *agent_id* from *room* — refused while it has work there.

    The caller commits. Nothing is changed when this raises.
    """
    if not await agent_in_room(room.id, agent_id, db):
        raise RoomAgentMembershipError("not_member", "the agent is not in this room")
    agent = await db.get(Agent, agent_id)
    nick = f"@{agent.nickname}" if agent else "the agent"

    # A direct room is a dialogue with its one agent. Only an extra agent that
    # got in before adding was refused can be taken out.
    if room.room_type == RoomType.direct and len(await room_agent_ids(room.id, db)) <= 1:
        raise RoomAgentMembershipError(
            "direct_room", f"{nick} is the only agent of this direct room and cannot be removed."
        )

    if room.has_coordinator and room_chat_master_id(room) == agent_id:
        raise RoomAgentMembershipError(
            "is_chat_master",
            f"{nick} is this room's ChatMaster. Assign another coordinator first.",
        )

    active = await active_tasks_of_agent(room.id, agent_id, db)
    if active:
        raise RoomAgentMembershipError(
            "has_active_tasks",
            f"{nick} has unfinished tasks in this room ({len(active)}). "
            "Finish, cancel or reassign them, then remove the agent.",
            details={
                "tasks": [
                    {"id": str(t.id), "title": t.title, "state": t.state.value}
                    for t in active
                ],
            },
        )

    from app.models.session.user_agent import UserAgentSession

    open_dialogues = list(
        (
            await db.execute(
                select(UserAgentSession).where(
                    UserAgentSession.room_id == room.id,
                    UserAgentSession.agent_id == agent_id,
                    UserAgentSession.closed_at.is_(None),
                )
            )
        ).scalars().all()
    )
    from app.core.user_sessions import turn_in_progress

    if any(turn_in_progress(s.id) for s in open_dialogues):
        raise RoomAgentMembershipError(
            "agent_busy",
            f"{nick} is replying in a dialogue right now. Wait for the reply to end and try again.",
        )

    result = RemovalResult()

    # The dialogue transcript stays with its session; no memory flush and no
    # Gate 1 finalize — the memory is deleted below, and a dialogue that did
    # work already has a task (which would have blocked the removal above).
    now = datetime.now(timezone.utc)
    for session in open_dialogues:
        session.closed_at = now
        session.close_reason = "cancelled"
        db.add(session)
    result.closed_sessions = len(open_dialogues)

    room_plan_ids = (
        select(GoalPlan.id)
        .join(RoomGoal, RoomGoal.id == GoalPlan.goal_id)
        .where(RoomGoal.room_id == room.id)
    )
    unassigned = await db.execute(
        update(GoalPlanStep)
        .where(
            GoalPlanStep.plan_id.in_(room_plan_ids),
            GoalPlanStep.assigned_agent_id == agent_id,
            GoalPlanStep.status.in_(_UNSTARTED_STEP_STATUSES),
        )
        .values(assigned_agent_id=None, updated_at=datetime.utcnow())
        .execution_options(synchronize_session=False)
    )
    result.unassigned_plan_steps = unassigned.rowcount or 0

    await db.execute(
        delete(AgentMemory).where(AgentMemory.room_id == room.id, AgentMemory.agent_id == agent_id)
    )
    await db.execute(
        delete(RoomAgentLink).where(
            RoomAgentLink.room_id == room.id, RoomAgentLink.agent_id == agent_id
        )
    )
    await db.flush()

    # emit_event commits, so the announcements go last, after every change.
    if open_dialogues:
        from app.core.watchdog import _emit_session_closed

        for session in open_dialogues:
            await _emit_session_closed(session, "cancelled", db)
    return result
