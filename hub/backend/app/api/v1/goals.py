from datetime import datetime
from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.db.session import get_session
from app.models.goal import RoomGoal, GoalStatus
from app.models.goal_plan import GoalPlan, GoalPlanStep
from app.models.room import Room, RoomUserLink
from app.models.user import User
from app.schemas.goal import GoalCreate, GoalOut, GoalReviewResolve, GoalUpdate, RequirementsGapResolve

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


async def _get_room(room_id: UUID, db: AsyncSession) -> Room:
    res = await db.execute(select(Room).where(Room.id == room_id))
    room = res.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


async def _get_goal(goal_id: UUID, room_id: UUID, db: AsyncSession) -> RoomGoal:
    res = await db.execute(
        select(RoomGoal).where(RoomGoal.id == goal_id, RoomGoal.room_id == room_id)
    )
    goal = res.scalars().first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.get("/{room_id}/goals", response_model=List[GoalOut])
async def list_goals(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    res = await db.execute(
        select(RoomGoal)
        .where(RoomGoal.room_id == room_id)
        .order_by(RoomGoal.position.asc(), RoomGoal.created_at.asc())
    )
    return res.scalars().all()


@router.post("/{room_id}/goals", response_model=GoalOut, status_code=201)
async def create_goal(
    room_id: UUID,
    body: GoalCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_room(room_id, db)

    if body.position is None:
        res = await db.execute(
            select(func.max(RoomGoal.position)).where(RoomGoal.room_id == room_id)
        )
        max_pos = res.scalar()
        position = (max_pos + 1) if max_pos is not None else 0
    else:
        position = body.position

    goal = RoomGoal(
        room_id=room_id,
        title=body.title,
        description=body.description,
        position=position,
    )
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal


@router.patch("/{room_id}/goals/{goal_id}", response_model=GoalOut)
async def update_goal(
    room_id: UUID,
    goal_id: UUID,
    body: GoalUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)

    if body.title is not None:
        goal.title = body.title
    if body.description is not None:
        goal.description = body.description
    if body.position is not None:
        goal.position = body.position
    if body.status is not None:
        goal.status = body.status
        if body.status == GoalStatus.COMPLETED and not goal.completed_at:
            goal.completed_at = datetime.utcnow()
        elif body.status != GoalStatus.COMPLETED:
            goal.completed_at = None

    goal.updated_at = datetime.utcnow()
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal


@router.post("/{room_id}/goals/{goal_id}/review/resolve", response_model=GoalOut)
async def resolve_goal_review_endpoint(
    room_id: UUID,
    goal_id: UUID,
    body: GoalReviewResolve,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 5 — approve or reject a goal's
    ``goal/<id> → dev`` gate.

    Approve does a clean ``merge --no-ff`` of the goal branch into ``dev`` and
    completes the goal; it is refused with 409 when the full CI suite on the
    goal branch is not green, or when a MUST requirement of the goal is still
    ``uncovered`` (internal design notes Stage 7) — either refusal
    is bypassed with ``force``. A merge conflict is escalated as
    ``GIT_CONFLICT`` and returns 409 — ``dev`` is left untouched. Reject
    leaves the goal ``ACTIVE`` for the agents to iterate.
    """
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)

    if body.decision not in {"approve", "reject"}:
        raise HTTPException(status_code=422, detail="decision must be 'approve' or 'reject'")
    if goal.review_state != "waiting_review":
        raise HTTPException(
            status_code=422,
            detail=f"Goal review is {goal.review_state!r}, not 'waiting_review'",
        )

    from app.core.room_git_flow import resolve_goal_review

    result = await resolve_goal_review(
        str(room_id), goal, db,
        decision=body.decision,
        note=body.note,
        force=body.force,
        resolved_by=current_user.id,
    )
    await db.commit()

    if result.get("blocked") and result.get("reason") == "uncovered_must":
        codes = ", ".join(r.get("code", "?") for r in result.get("uncovered_must") or [])
        raise HTTPException(
            status_code=409,
            detail=f"Required requirements not covered: {codes} — pass force=true to merge anyway",
        )
    if result.get("blocked"):
        raise HTTPException(
            status_code=409,
            detail=f"CI on the goal branch is {result.get('ci_status')!r} — pass force=true to merge anyway",
        )
    if result.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="goal branch → dev did not merge cleanly — resolve the conflict manually (GIT_CONFLICT emitted)",
        )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=f"Goal review failed: {result.get('error')}")

    await db.refresh(goal)
    return goal


@router.post("/{room_id}/goals/{goal_id}/requirements-gap-resolve")
async def resolve_requirements_gap_endpoint(
    room_id: UUID,
    goal_id: UUID,
    body: RequirementsGapResolve,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 6 — resolve the soft coverage-gate
    card. ``accept`` just dismisses it; ``rework`` reopens ``user_cm_p2`` with
    the uncovered/unverified/open-question list as the first message.
    """
    await _verify_membership(room_id, current_user, db)
    room = await _get_room(room_id, db)
    goal = await _get_goal(goal_id, room_id, db)

    if body.decision not in {"accept", "rework"}:
        raise HTTPException(status_code=422, detail="decision must be 'accept' or 'rework'")

    from app.core.requirements import resolve_requirements_gap

    try:
        gap = await resolve_requirements_gap(room, goal, current_user, body.decision, body.note, db)
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))

    await db.commit()
    return {"ok": True, "decision": body.decision, "gap": gap}


@router.get("/{room_id}/goals/{goal_id}/requirements")
async def list_goal_requirements_endpoint(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 9 — the goal panel's
    requirements list: every ``R-NNN``/``NG-NNN``/``Q-NNN`` of the goal with
    its current status and, for requirements/non-goals, the tasks that claim
    to cover it. Empty list for a goal that never went through
    ``propose_plan`` with requirements (single-line goals, `direct_session`).
    """
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)

    from app.core.requirements import list_requirements_view

    return await list_requirements_view(goal_id, db)


@router.post("/{room_id}/goals/{goal_id}/merge-branch", response_model=GoalOut)
async def merge_goal_branch_endpoint(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes follow-up — fold a finished goal's still-unmerged
    ``goal/<id>`` branch into ``dev`` directly, for a goal whose Stage 5 gate
    never opened (goal finished before the gate existed, or a fail-open
    completion left ``review_state=None``).

    Allowed only when:
    - goal state is ``COMPLETED``;
    - goal has a ``branch_name``;
    - ``review_state`` is not pending (``testing`` / ``waiting_review`` → use
      the review gate instead);
    - the branch exists in workspaced and is NOT already merged into ``dev``.

    Same clean ``merge --no-ff`` as an approved gate. A conflict → 409,
    ``GIT_CONFLICT`` emitted, ``dev`` untouched.
    """
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)

    if goal.status != GoalStatus.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Goal is {goal.status.value!r} — only COMPLETED goals can be merged this way",
        )
    if not goal.branch_name:
        raise HTTPException(status_code=422, detail="Goal has no branch to merge")
    if goal.review_state in {"testing", "waiting_review"}:
        raise HTTPException(
            status_code=422,
            detail=f"Goal review is {goal.review_state!r} — resolve it through the gate",
        )

    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import get_flow_config, merge_goal_branch_direct

    room = await _get_room(room_id, db)
    cfg = get_flow_config(room)
    branches = await wsd.list_workspace_branches(
        str(room_id), {goal.branch_name: cfg.working_branch}
    )
    info = next((b for b in branches if b["name"] == goal.branch_name), None)
    if info is None:
        raise HTTPException(status_code=422, detail="Branch not found in workspace")
    if info.get("merged") is True:
        raise HTTPException(status_code=422, detail="Branch is already merged")

    result = await merge_goal_branch_direct(str(room_id), goal, db)
    await db.commit()

    if result.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="goal branch → dev did not merge cleanly — resolve the conflict manually (GIT_CONFLICT emitted)",
        )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=f"Goal merge failed: {result.get('error')}")

    await db.refresh(goal)
    return goal


@router.post("/{room_id}/goals/{goal_id}/review/publish", response_model=GoalOut)
async def publish_goal_review_branch(
    room_id: UUID,
    goal_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 4 — push the goal's ``goal/<id>``
    branch to the bound remote so the forge CI (GitHub Actions / GitLab CI) can
    run on it.

    Valid only while the goal gate is open (``review_state="waiting_review"``).
    Publishing is a plain ``git push`` — it never changes ``review_state`` and
    never completes the goal; the user still approves → merges into ``dev``
    afterwards via ``.../review/resolve``. A non-fast-forward push returns 409.
    """
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)
    room = await _get_room(room_id, db)

    if goal.review_state != "waiting_review":
        raise HTTPException(
            status_code=422,
            detail=f"Goal review is {goal.review_state!r}, not 'waiting_review'",
        )
    if not goal.branch_name:
        raise HTTPException(status_code=422, detail="Goal has no branch to publish")
    if not (room.git_config or {}).get("repo_url"):
        raise HTTPException(status_code=422, detail="Room has no bound repository")

    # Same token resolution as POST /rooms/{id}/git/push (secret-store binding
    # or legacy inline access_token).
    from app.api.v1.git import _resolve_git_access_token
    from app.core.secrets.service import SecretResourceAccessDenied
    from app.core.secrets.store import SecretAccessDenied, SecretUnavailable

    try:
        token_value = await _resolve_git_access_token(room, current_user, request, db)
    except (SecretAccessDenied, SecretResourceAccessDenied, SecretUnavailable):
        raise HTTPException(status_code=403, detail="Git credential is unavailable") from None
    token_str = token_value.reveal() if token_value else None

    from app.core.room_git_flow import publish_goal_branch

    try:
        result = await publish_goal_branch(
            str(room_id), goal, db, access_token=token_str
        )
    finally:
        token_value = None
    await db.commit()

    if not result.get("ok"):
        if result.get("hint") == "sync_then_promote":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The goal branch has diverged from remote — pull the changes first "
                    "(Pull / Sync), then publish again."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"Could not publish the branch: {result.get('error')}",
        )

    await db.refresh(goal)
    return goal


@router.get("/{room_id}/goals/{goal_id}/dialogue")
async def get_goal_dialogue(
    room_id: UUID,
    goal_id: UUID,
    view: bool = False,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 7 — metadata (and, with ``view=true``,
    the raw markdown) of the ``.legion/plans/goal-{id}.md`` task-formation
    dialogue archive behind the card chip."""
    await _verify_membership(room_id, current_user, db)
    await _get_goal(goal_id, room_id, db)

    from app.core import plan_dialogue as pd

    meta = await pd.archive_meta(room_id, goal_id=goal_id)
    if view and meta["bound"]:
        meta["content"] = await pd.read_archive(room_id, meta["path"])
    return meta


@router.post("/{room_id}/goals/{goal_id}/dialogue/unlink")
async def unlink_goal_dialogue(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Detach the dialogue archive from a goal — delete the file and clear
    ``origin_session_id`` so the chip goes away."""
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)

    from app.core import plan_dialogue as pd

    ok = await pd.unlink_archive(room_id, goal_id=goal_id)
    goal.origin_session_id = None
    goal.updated_at = datetime.utcnow()
    db.add(goal)
    await db.commit()
    return {"ok": ok}


@router.delete("/{room_id}/goals/{goal_id}", status_code=204)
async def delete_goal(
    room_id: UUID,
    goal_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    await _verify_membership(room_id, current_user, db)
    goal = await _get_goal(goal_id, room_id, db)

    plan_res = await db.execute(select(GoalPlan).where(GoalPlan.goal_id == goal_id))
    plan = plan_res.scalars().first()
    if plan:
        steps_res = await db.execute(select(GoalPlanStep).where(GoalPlanStep.plan_id == plan.id))
        for step in steps_res.scalars().all():
            await db.delete(step)
        await db.delete(plan)

    await db.delete(goal)
    await db.commit()

    # Stage 8: a deleted Goal takes its task-formation dialogue archive with it
    # (the …-attachments/ files are cleared by the retention sweep).
    try:
        from app.core import plan_dialogue as pd

        await pd.unlink_archive(room_id, goal_id=goal_id)
    except Exception:  # best-effort — the Goal is already gone
        pass
