from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.events import emit_event
from app.core.plan_session import dispatch_plan_event, render_plan_text_with_agents
from app.core.room_members import agent_in_room
from app.db.session import get_session
from app.models.agent import Agent
from app.models.goal import RoomGoal
from app.models.goal_plan import GoalPlan, GoalPlanStep, GoalPlanStatus, PlanStepStatus
from app.models.room import Room, RoomUserLink
from app.models.task import Task
from app.models.user import User
from app.schemas.goal_plan import (
    GoalPlanCreate, GoalPlanOut, GoalPlanUpdate,
    GoalPlanStepCreate, GoalPlanStepOut, GoalPlanStepUpdate,
)

router = APIRouter()


async def _verify_membership(room_id: UUID, user: User, db: AsyncSession):
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _get_goal(goal_id: UUID, room_id: UUID, db: AsyncSession) -> RoomGoal:
    res = await db.execute(
        select(RoomGoal).where(RoomGoal.id == goal_id, RoomGoal.room_id == room_id)
    )
    goal = res.scalars().first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


async def _require_room_agent(room_id: UUID, agent_id: Optional[UUID], db: AsyncSession) -> None:
    """A plan step may be given only to an agent that is in the room right now."""
    if agent_id is None:
        return
    if not await agent_in_room(room_id, agent_id, db):
        raise HTTPException(status_code=422, detail="The agent is not a member of this room")


async def _get_plan(goal_id: UUID, db: AsyncSession) -> Optional[GoalPlan]:
    res = await db.execute(select(GoalPlan).where(GoalPlan.goal_id == goal_id))
    return res.scalars().first()


async def _agent_name(agent_id: Optional[UUID], db: AsyncSession) -> Optional[str]:
    if not agent_id:
        return None
    res = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = res.scalars().first()
    if not agent:
        return None
    return agent.nickname or agent.agent_uid


async def _step_out(step: GoalPlanStep, db: AsyncSession, room_id: UUID) -> GoalPlanStepOut:
    return GoalPlanStepOut(
        id=step.id,
        plan_id=step.plan_id,
        title=step.title,
        description=step.description,
        status=step.status,
        assigned_agent_id=step.assigned_agent_id,
        assigned_agent_name=await _agent_name(step.assigned_agent_id, db),
        assigned_agent_in_room=(
            step.assigned_agent_id is None
            or await agent_in_room(room_id, step.assigned_agent_id, db)
        ),
        position=step.position,
        is_parallel=step.is_parallel,
        created_at=step.created_at,
        updated_at=step.updated_at,
        completed_at=step.completed_at,
    )


async def _goal_tasks_count(goal_id: UUID, db: AsyncSession) -> int:
    """How many Tasks already hang off this goal.

    The single source of truth for "this plan has been decomposed already":
    the decomposition session (``cm_self_plan``) is what creates them, so a
    non-zero count means the plan has been through the pipeline whatever its
    stored ``status`` says.
    """
    res = await db.execute(
        select(func.count()).select_from(Task).where(Task.goal_id == goal_id)
    )
    return int(res.scalar() or 0)


async def _plan_out(plan: GoalPlan, db: AsyncSession, room_id: UUID) -> GoalPlanOut:
    res = await db.execute(
        select(GoalPlanStep)
        .where(GoalPlanStep.plan_id == plan.id)
        .order_by(GoalPlanStep.position.asc(), GoalPlanStep.created_at.asc())
    )
    steps = [await _step_out(s, db, room_id) for s in res.scalars().all()]
    return GoalPlanOut(
        tasks_count=await _goal_tasks_count(plan.goal_id, db),
        id=plan.id,
        goal_id=plan.goal_id,
        status=plan.status,
        steps=steps,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        approved_at=plan.approved_at,
    )


# ── GET plan ──────────────────────────────────────────────────────────────────

@router.get("/{room_id}/goals/{goal_id}/plan", response_model=GoalPlanOut)
async def get_plan(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for this goal yet")
    return await _plan_out(plan, db, room_id)


# ── CREATE / REPLACE plan ─────────────────────────────────────────────────────

@router.post("/{room_id}/goals/{goal_id}/plan", response_model=GoalPlanOut, status_code=201)
async def create_or_replace_plan(
    room_id: UUID,
    goal_id: UUID,
    body: GoalPlanCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    for sd in body.steps:
        await _require_room_agent(room_id, sd.assigned_agent_id, db)

    existing = await _get_plan(goal_id, db)
    if existing:
        res = await db.execute(select(GoalPlanStep).where(GoalPlanStep.plan_id == existing.id))
        for step in res.scalars().all():
            await db.delete(step)
        plan = existing
        plan.status = GoalPlanStatus.DRAFT
        plan.updated_at = datetime.utcnow()
        plan.approved_at = None
    else:
        plan = GoalPlan(goal_id=goal_id)
        db.add(plan)

    await db.flush()

    for i, sd in enumerate(body.steps):
        position = sd.position if sd.position is not None else i
        db.add(GoalPlanStep(
            plan_id=plan.id,
            title=sd.title,
            description=sd.description,
            position=position,
            is_parallel=sd.is_parallel,
            assigned_agent_id=sd.assigned_agent_id,
        ))

    await db.commit()
    await db.refresh(plan)

    # Emit PLAN_CREATED and schedule the cm_self_plan decomposition session
    steps_res = await db.execute(
        select(GoalPlanStep)
        .where(GoalPlanStep.plan_id == plan.id)
        .order_by(GoalPlanStep.position.asc())
    )
    steps = steps_res.scalars().all()
    plan_text = await render_plan_text_with_agents(plan, list(steps), db)

    room_res = await db.execute(select(Room).where(Room.id == room_id))
    room = room_res.scalars().first()

    await emit_event(
        room_id=room_id,
        event_type="PLAN_CREATED",
        content=f"Plan {plan.id} created for goal {goal_id}.",
        db=db,
        metadata={
            "goal_id": str(goal_id),
            "plan_id": str(plan.id),
            "full_plan_text": plan_text,
        },
    )

    if room and room.has_coordinator:
        # Decomposition starts right here, so the plan is past the approval
        # gate — record that, or the Goals panel would keep offering an
        # Approve button that re-runs the very dispatch below.
        plan.status = GoalPlanStatus.APPROVED
        plan.approved_at = datetime.utcnow()
        db.add(plan)
        await db.commit()
        await db.refresh(plan)
        dispatch_plan_event(room_id, goal_id, plan.id, plan_text)

    return await _plan_out(plan, db, room_id)


# ── UPDATE plan (approve / change status) ─────────────────────────────────────

@router.patch("/{room_id}/goals/{goal_id}/plan", response_model=GoalPlanOut)
async def update_plan(
    room_id: UUID,
    goal_id: UUID,
    body: GoalPlanUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for this goal yet")

    # A plan is decomposed once. ``approved_at`` alone is not enough to know
    # that: plans born from an accepted propose_plan (``_trigger_plan_pipeline``)
    # used to be stored as drafts even though their tasks were already on the
    # board, so approving one from the Goals panel dispatched a *second*
    # decomposition and duplicated every task. Existing tasks veto the dispatch.
    already_decomposed = await _goal_tasks_count(goal_id, db) > 0
    first_approval = (
        body.status == GoalPlanStatus.APPROVED
        and plan.approved_at is None
        and not already_decomposed
    )

    if body.status is not None:
        plan.status = body.status
        if body.status == GoalPlanStatus.APPROVED and not plan.approved_at:
            plan.approved_at = datetime.utcnow()
        elif body.status == GoalPlanStatus.DRAFT:
            plan.approved_at = None

    plan.updated_at = datetime.utcnow()
    db.add(plan)
    await db.commit()
    await db.refresh(plan)

    if first_approval:
        steps_res = await db.execute(
            select(GoalPlanStep)
            .where(GoalPlanStep.plan_id == plan.id)
            .order_by(GoalPlanStep.position.asc())
        )
        steps = steps_res.scalars().all()
        plan_text = await render_plan_text_with_agents(plan, list(steps), db)

        room_res = await db.execute(select(Room).where(Room.id == room_id))
        room = room_res.scalars().first()

        await emit_event(
            room_id=room_id,
            event_type="PLAN_UPDATED",
            content=f"Plan {plan.id} approved for goal {goal_id}.",
            db=db,
            metadata={
                "goal_id": str(goal_id),
                "plan_id": str(plan.id),
                "full_plan_text": plan_text,
            },
        )

        if room and room.has_coordinator:
            dispatch_plan_event(room_id, goal_id, plan.id, plan_text)

    return await _plan_out(plan, db, room_id)


# ── DELETE plan ───────────────────────────────────────────────────────────────

@router.delete("/{room_id}/goals/{goal_id}/plan", status_code=204)
async def delete_plan(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for this goal yet")

    res = await db.execute(select(GoalPlanStep).where(GoalPlanStep.plan_id == plan.id))
    for step in res.scalars().all():
        await db.delete(step)
    await db.delete(plan)
    await db.commit()


# ── ADD step ──────────────────────────────────────────────────────────────────

@router.post("/{room_id}/goals/{goal_id}/plan/steps", response_model=GoalPlanStepOut, status_code=201)
async def add_step(
    room_id: UUID,
    goal_id: UUID,
    body: GoalPlanStepCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        # Auto-create empty plan on first step
        plan = GoalPlan(goal_id=goal_id)
        db.add(plan)
        await db.flush()

    await _require_room_agent(room_id, body.assigned_agent_id, db)
    if body.position is None:
        res = await db.execute(
            select(func.max(GoalPlanStep.position)).where(GoalPlanStep.plan_id == plan.id)
        )
        max_pos = res.scalar()
        position = (max_pos + 1) if max_pos is not None else 0
    else:
        position = body.position

    step = GoalPlanStep(
        plan_id=plan.id,
        title=body.title,
        description=body.description,
        position=position,
        is_parallel=body.is_parallel,
        assigned_agent_id=body.assigned_agent_id,
    )
    db.add(step)
    plan.updated_at = datetime.utcnow()
    db.add(plan)
    await db.commit()
    await db.refresh(step)
    return await _step_out(step, db, room_id)


# ── UPDATE step ───────────────────────────────────────────────────────────────

@router.patch("/{room_id}/goals/{goal_id}/plan/steps/{step_id}", response_model=GoalPlanStepOut)
async def update_step(
    room_id: UUID,
    goal_id: UUID,
    step_id: UUID,
    body: GoalPlanStepUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for this goal yet")

    res = await db.execute(
        select(GoalPlanStep).where(GoalPlanStep.id == step_id, GoalPlanStep.plan_id == plan.id)
    )
    step = res.scalars().first()
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    if body.title is not None:
        step.title = body.title
    if body.description is not None:
        step.description = body.description
    if body.status is not None:
        step.status = body.status
        if body.status == PlanStepStatus.COMPLETED and not step.completed_at:
            step.completed_at = datetime.utcnow()
        elif body.status != PlanStepStatus.COMPLETED:
            step.completed_at = None
    if body.position is not None:
        step.position = body.position
    if body.is_parallel is not None:
        step.is_parallel = body.is_parallel
    # Support explicit null to clear agent assignment
    if "assigned_agent_id" in body.model_fields_set:
        await _require_room_agent(room_id, body.assigned_agent_id, db)
        step.assigned_agent_id = body.assigned_agent_id

    step.updated_at = datetime.utcnow()
    plan.updated_at = datetime.utcnow()
    db.add(step)
    db.add(plan)
    await db.commit()
    await db.refresh(step)
    return await _step_out(step, db, room_id)


# ── DELETE step ───────────────────────────────────────────────────────────────

@router.delete("/{room_id}/goals/{goal_id}/plan/steps/{step_id}", status_code=204)
async def delete_step(
    room_id: UUID,
    goal_id: UUID,
    step_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)
    plan = await _get_plan(goal_id, db)
    if not plan:
        raise HTTPException(status_code=404, detail="No plan for this goal yet")

    res = await db.execute(
        select(GoalPlanStep).where(GoalPlanStep.id == step_id, GoalPlanStep.plan_id == plan.id)
    )
    step = res.scalars().first()
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    await db.delete(step)
    plan.updated_at = datetime.utcnow()
    db.add(plan)
    await db.commit()
