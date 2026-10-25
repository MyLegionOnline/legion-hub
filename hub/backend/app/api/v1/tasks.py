from datetime import datetime, timezone
import json
import logging
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.context_waterfall import context_waterfall
from app.core.events import emit_event
from app.core.proxy_session import (
    INFRASTRUCTURE_INTERRUPTION_REASONS,
    TaskDispatchPreflightError,
    load_paused_history,
    prepare_task_dispatch,
    request_task_dispatch,
)
from app.core.task_deps import (
    get_blocked_dependents,
    publish_unblocked_tasks,
    try_unblock_tasks,
)
from app.core.task_runtime import (
    RUNTIME_RUNNING,
    RUNTIME_STOP_REQUESTED,
    fence_task_runtime,
    request_task_stop,
)
from app.core.watchdog import close_open_task_sessions
from app.db.session import get_session
from app.models.agent import Agent
from app.models.agent_session import AgentSession
from app.models.context_artifact import ContextArtifact
from app.models.message import (
    CHECKPOINT_RESOLVED,
    Message,
    TASK_DEPENDENCY_FAILED,
    TASK_DISPATCH_FAILED,
    TASK_DISPATCH_STARTED,
    TASK_HANDOFF_CREATED,
    TASK_RESUMED,
    TASK_RETRIED,
    TASK_STALLED,
    TASK_STATE_CHANGED,
    TOOL_LIMIT_REACHED,
)
from app.models.room import Room, RoomUserLink
from app.models.session import CmAgentProxySession, Session
from app.models.task import (
    DispatchStatus, Task, TaskDependency, TaskHandoff, TaskState, TASK_TRANSITIONS,
    is_checkpoint_approval,
)
from app.models.user import User
from app.schemas.task import (
    CheckpointResolve, DivergenceResolveRequest, TaskCreate, TaskOut,
    TaskStateChange, TaskRetry,
    PromotionOut, PromotionResolve, PromotionPreview,
    SessionCreate, SessionComplete, SessionOut, ProxySessionOut, RoomSessionsOut,
)

_logger = logging.getLogger(__name__)

_DEPRECATION_DATE = "2026-09-01"
_DEPRECATION_LINK = "https://github.com/MyLegionOnline/legion-hub/blob/main/hub/CHATFLOW_REFERENCE.md"

router = APIRouter()


# ── helpers ─────────────────────────────────────────────────────────────────

async def _verify_membership(room_id: UUID, user: User, db: AsyncSession):
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _require_room_agent(room_id: UUID, agent_id: UUID | None, db: AsyncSession) -> None:
    """A task may be given only to an agent that is in the room right now."""
    if agent_id is None:
        return
    from app.core.room_members import agent_in_room

    if not await agent_in_room(room_id, agent_id, db):
        raise HTTPException(status_code=422, detail="The agent is not a member of this room")


async def _get_room(room_id: UUID, db: AsyncSession) -> Room:
    res = await db.execute(select(Room).where(Room.id == room_id))
    room = res.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


async def _get_task(task_id: UUID, room_id: UUID, db: AsyncSession) -> Task:
    res = await db.execute(
        select(Task).where(Task.id == task_id, Task.room_id == room_id)
    )
    task = res.scalars().first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _checkpoint_content(value: Any, limit: int = 12_000) -> str:
    """Render saved provider-specific history as compact handoff text.

    Extracts human-readable text from Anthropic/OpenAI message content blocks
    and head-truncates to *limit* chars so the agent sees the beginning
    (headers, schema definitions) rather than the tail.
    """
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or {}
                arg_str = ", ".join(f"{k}={v!r}" for k, v in inp.items())
                parts.append(f"[{name}({arg_str})]")
            elif btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                parts.append(str(content))
            else:
                try:
                    parts.append(json.dumps(block, ensure_ascii=False, default=str))
                except Exception:
                    parts.append(str(block))
        text = "\n".join(parts)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    # Head-truncate: the beginning of file reads is the most informative part
    return text if len(text) <= limit else text[:limit] + f"\n…[+{len(text) - limit} more chars]"


async def _handoff_read_manifest(
    source_session: CmAgentProxySession,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    """Build the durable fingerprint/range/ref manifest for a continuation.

    ``context_state`` is the source of truth for ranges; artifacts are joined
    only to retain refs for exact recall.  Legacy sessions may have no ledger,
    so callers can still keep the old ``files_read`` list as a fallback.
    """
    state = source_session.context_state or {}
    ledger = state.get("read_ledger") if isinstance(state, dict) else {}
    ledger = ledger if isinstance(ledger, dict) else {}
    artifacts = (
        await db.execute(
            select(ContextArtifact)
            .where(ContextArtifact.session_id == source_session.id)
            .order_by(ContextArtifact.created_at.asc())
        )
    ).scalars().all()
    by_path: dict[str, list[ContextArtifact]] = {}
    for artifact in artifacts:
        metadata = artifact.artifact_metadata or {}
        path = metadata.get("path") if isinstance(metadata, dict) else None
        if path:
            by_path.setdefault(str(path), []).append(artifact)

    manifest: list[dict[str, Any]] = []
    for path in sorted(ledger):
        entry = ledger.get(path) or {}
        if not isinstance(entry, dict):
            continue
        refs = [str(ref) for ref in (entry.get("observation_refs") or []) if ref]
        # Keep refs from artifacts when older ledger writes omitted them.
        if not refs:
            refs = [
                str(artifact.id)
                for artifact in by_path.get(path, [])
                if not getattr(artifact, "stale", False)
            ][-50:]
        manifest.append(
            {
                "path": path,
                "fingerprint": entry.get("fingerprint"),
                "ranges": entry.get("ranges") or [],
                "observation_refs": refs,
                "stale": False,
            }
        )
    return manifest


def _merge_handoff_manifests(
    current: list[dict[str, Any]],
    ancestors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge manifests by path/fingerprint while preserving non-overlapping ranges."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in [*ancestors, *current]:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        key = (str(item["path"]), str(item.get("fingerprint") or ""))
        target = merged.setdefault(
            key,
            {
                "path": str(item["path"]),
                "fingerprint": item.get("fingerprint"),
                "ranges": [],
                "observation_refs": [],
                "stale": bool(item.get("stale", False)),
            },
        )
        target["ranges"] = sorted(
            {
                (int(pair[0]), int(pair[1]))
                for pair in [*(target["ranges"] or []), *(item.get("ranges") or [])]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            }
        )
        target["ranges"] = [list(pair) for pair in target["ranges"]]
        target["observation_refs"] = list(
            dict.fromkeys(
                [*(target["observation_refs"] or []), *(item.get("observation_refs") or [])]
            )
        )[-50:]
        target["stale"] = bool(target["stale"] or item.get("stale", False))
    return [merged[key] for key in sorted(merged)]


_TERMINAL_TASK_STATES = frozenset({TaskState.COMPLETED, TaskState.CANCELLED, TaskState.CONTINUED})

# Fed to the respawned agent when a person clicks "Back to Work" on a task
# waiting for review, so it knows the submission was not accepted instead of
# silently re-reading its old transcript.
_REVIEW_REOPEN_FEEDBACK = (
    "Work not accepted: the task was returned to work with a review. "
    "Re-read the task requirements, finish what is missing and close the session again "
    "with an updated result."
)

# Human-only "mark as done" override.  TASK_TRANSITIONS deliberately routes
# completion through WAITING_REVIEW so an agent can never declare its own work
# finished (the same table gates the `task_update` tool).  A *person* looking
# at the board has knowledge the workflow does not: a task can sit BLOCKED or
# stuck IN_PROGRESS while the work is in fact done (e.g. the reviewer simply
# could not see the produced files).  Before this override the only ways out
# were Cancel — which marks the goal abandoned and fails dependants — or
# completing the parent goal by hand and leaving the task hanging.  So the
# REST endpoint (and only it, never the agent tool) accepts COMPLETED from any
# non-terminal state where work has plausibly happened.
_MANUAL_COMPLETE_STATES = frozenset({
    TaskState.ASSIGNED,
    TaskState.IN_PROGRESS,
    TaskState.PAUSING,
    TaskState.PAUSED,
    TaskState.RESUMING,
    TaskState.BLOCKED,
    # A handed-off task the user considers finished anyway: without this the
    # CONTINUED head (and the successor still grinding away on it) had no exit
    # at all — no Cancel, no Mark Done, nothing to click.
    TaskState.CONTINUED,
})


async def _continuation_chain(task_id: UUID, db: AsyncSession) -> List[Task]:
    """Every successor task reachable from *task_id* via continuation handoffs.

    A handoff leaves the source in CONTINUED and moves the live work to a
    successor (``continuation_of_task_id``), which may itself have been handed
    off again. Stopping the head therefore has to reach the whole chain — that
    is where the agents actually are.
    """
    chain: List[Task] = []
    seen: set[UUID] = {task_id}
    frontier = [task_id]
    # Bounded by MAX_CONTINUATION_CHAIN_DEPTH in practice; the seen-set also
    # makes a corrupt cycle harmless.
    while frontier:
        rows = (
            await db.execute(
                select(Task).where(Task.continuation_of_task_id.in_(frontier))
            )
        ).scalars().all()
        frontier = []
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            chain.append(row)
            frontier.append(row.id)
    return chain


async def _compute_task_runtime(
    task: Task, sessions: List[CmAgentProxySession], db: AsyncSession
) -> dict[str, Any]:
    """Derive TaskOut's runtime-visibility fields from already-fetched rows.

    *sessions* is the same CmAgentProxySession list _enrich_task fetches for
    session_count — reused here instead of a second near-identical query.
    No new persisted counters are needed beyond TASK_DISPATCH_STARTED (stage
    4): Task.dispatch_status/run_id/state plus those rows already carry
    enough information to answer "why isn't this executing / is it executing
    right now" without depending on ephemeral WS statuses that don't survive
    a reload.
    """
    open_sessions = [s for s in sessions if s.closed_at is None]
    active_session_id = next(
        (s.id for s in open_sessions if s.side == "agent"),
        open_sessions[0].id if open_sessions else None,
    )

    # The current run_id's own TASK_DISPATCH_STARTED row is always the most
    # recent one for this task (a new run_id is only minted at a fresh claim,
    # strictly after any earlier one) — order-by-desc + limit(1) finds it
    # without a dialect-specific JSON predicate; the run_id equality check
    # below is just a safety net against a missed/failed emit.
    dispatch_started_at: Optional[datetime] = None
    if task.run_id is not None:
        started_row = (
            await db.execute(
                select(Message.timestamp, Message.event_metadata)
                .where(Message.task_id == task.id, Message.event_type == TASK_DISPATCH_STARTED)
                .order_by(Message.timestamp.desc())
                .limit(1)
            )
        ).first()
        if started_row is not None and (started_row[1] or {}).get("run_id") == task.run_id:
            dispatch_started_at = started_row[0]

    # A dispatch failure is only worth surfacing while it's still the last
    # word on this task — once a newer attempt has actually started
    # (dispatch_started_at moved past it), it's stale history, not the
    # current blocker.
    dispatch_reason: Optional[str] = None
    failure_row = (
        await db.execute(
            select(Message.timestamp, Message.event_metadata)
            .where(Message.task_id == task.id, Message.event_type == TASK_DISPATCH_FAILED)
            .order_by(Message.timestamp.desc())
            .limit(1)
        )
    ).first()
    if failure_row is not None:
        failure_ts, failure_meta = failure_row
        if dispatch_started_at is None or failure_ts > dispatch_started_at:
            dispatch_reason = (failure_meta or {}).get("reason")

    # Why the last execution attempt *stopped* — separate axis from
    # dispatch_reason above (that's why an attempt never *started*).  Kept
    # even after a newer dispatch supersedes it (marked via "superseded"
    # rather than dropped) so a reload doesn't lose the explanation while a
    # Retry/Resume is in flight (internal design notes, part 4, stage 2).
    last_interruption: Optional[dict[str, Any]] = None
    interruption_row = (
        await db.execute(
            select(Message.timestamp, Message.event_metadata)
            .where(
                Message.task_id == task.id,
                or_(
                    Message.event_type == TOOL_LIMIT_REACHED,
                    # Infrastructure failures (provider down, request timeout)
                    # ride on TASK_STALLED; the watchdog's own stall reasons
                    # share that event type and must not read as interruptions.
                    and_(
                        Message.event_type == TASK_STALLED,
                        Message.event_metadata["reason"].as_string().in_(
                            INFRASTRUCTURE_INTERRUPTION_REASONS
                        ),
                    ),
                ),
            )
            .order_by(Message.timestamp.desc())
            .limit(1)
        )
    ).first()
    if interruption_row is not None:
        interruption_ts, interruption_meta = interruption_row
        meta = interruption_meta or {}
        last_interruption = {
            "reason": meta.get("reason"),
            "recommended_action": meta.get("recommended_action"),
            "tool_calls_used": meta.get("tool_calls_used"),
            "tool_calls_limit": meta.get("tool_calls_limit"),
            "tokens_used": meta.get("tokens_used"),
            "tokens_limit": meta.get("tokens_limit"),
            "trigger_tool": meta.get("trigger_tool"),
            "repeat_count": meta.get("repeat_count"),
            "safe_error": meta.get("safe_error"),
            "occurred_at": interruption_ts,
            "superseded": dispatch_started_at is not None and dispatch_started_at > interruption_ts,
        }

    if task.state in _TERMINAL_TASK_STATES:
        runtime_status = "terminal"
    elif task.state == TaskState.WAITING_REVIEW:
        runtime_status = "waiting_review"
    elif task.state in (TaskState.PAUSED, TaskState.PAUSING, TaskState.RESUMING):
        runtime_status = "paused"
    elif task.state == TaskState.REJECTED:
        runtime_status = "rejected"
    elif task.state == TaskState.BLOCKED:
        # Workflow-level BLOCKED (agent reported blocked, or a manual
        # Block/Cancel/watchdog reset) — an authoritative "not executing,
        # needs Retry" signal that must win even if dispatch_status is
        # stale-DISPATCHED (change_task_state's Block/Cancel path doesn't
        # reset it; only watchdog/_emit_dispatch_failed do).
        runtime_status = "needs_retry"
    elif active_session_id is not None:
        runtime_status = "running"
    elif task.state == TaskState.IN_PROGRESS and task.run_id is None:
        # No CmAgentProxySession and no run_id fence: the legacy
        # approve_and_start pipeline (chat_master.py) drives this task
        # directly and never goes through the proxy dispatcher.
        runtime_status = "running"
    elif (
        task.state == TaskState.IN_PROGRESS
        and task.run_id is not None
        and getattr(task, "runtime_state", None) in {"paused", "stopped", "failed"}
    ):
        # Shows IN_PROGRESS but its generation is over and no session is open:
        # a pause that never resumed, or a dispatch that died right after
        # claiming (leaving dispatch_status=DISPATCHED). Surface it as
        # actionable rather than a perpetual "dispatching…" spinner — Retry,
        # or a Start click, runs the explicit unwedge path.
        runtime_status = "needs_retry"
    elif task.dispatch_status == DispatchStatus.DISPATCHED:
        runtime_status = "dispatching"
    elif task.dispatch_status == DispatchStatus.BLOCKED:
        runtime_status = "blocked_by_dependencies"
    elif dispatch_reason is not None:
        runtime_status = "dispatch_failed"
    elif task.state in (TaskState.DRAFT, TaskState.WAITING_APPROVAL):
        runtime_status = task.state.value
    else:
        runtime_status = "queued"

    return {
        "runtime_status": runtime_status,
        "active_session_id": active_session_id,
        "dispatch_attempt_id": task.run_id,
        "dispatch_started_at": dispatch_started_at,
        "dispatch_reason": dispatch_reason,
        "last_interruption": last_interruption,
    }


async def _enrich_task(task: Task, db: AsyncSession) -> TaskOut:
    nickname = None
    if task.assigned_agent_id:
        res = await db.execute(select(Agent).where(Agent.id == task.assigned_agent_id))
        agent = res.scalars().first()
        nickname = agent.nickname if agent else None

    res = await db.execute(
        select(CmAgentProxySession).where(CmAgentProxySession.task_id == task.id)
    )
    sessions = res.scalars().all()

    out = TaskOut.model_validate(task)
    out.assigned_agent_nickname = nickname
    if task.assigned_agent_id:
        from app.core.room_members import agent_in_room

        out.assigned_agent_in_room = await agent_in_room(task.room_id, task.assigned_agent_id, db)
    out.session_count = len(sessions)
    for key, value in (await _compute_task_runtime(task, sessions, db)).items():
        setattr(out, key, value)

    if task.goal_id:
        try:
            from app.core.requirements import task_requirements_payload

            out.covered_requirements = await task_requirements_payload(task, db)
        except Exception as exc:
            _logger.warning("_enrich_task: requirements payload failed for task %s: %s", task.id, exc)

    return out


async def _enrich_proxy_session(session: CmAgentProxySession, db: AsyncSession) -> ProxySessionOut:
    nickname = None
    if session.agent_id:
        agent = await db.get(Agent, session.agent_id)
        nickname = agent.nickname if agent else None
    out = ProxySessionOut.model_validate(session)
    out.agent_nickname = nickname
    return out


async def _enrich_session(session: AgentSession, db: AsyncSession, task_title: str | None = None) -> SessionOut:
    nickname = None
    if session.agent_id:
        res = await db.execute(select(Agent).where(Agent.id == session.agent_id))
        agent = res.scalars().first()
        if agent:
            nickname = agent.nickname
    out = SessionOut.model_validate(session)
    out.agent_nickname = nickname
    out.task_title = task_title
    core_session = await db.get(Session, session.id)
    snapshot = (
        core_session.context_snapshot
        if core_session is not None
        else session.context_snapshot
    )
    out.context_waterfall = context_waterfall(snapshot)
    return out


# ── Room-level sessions endpoint ────────────────────────────────────────────

@router.get("/{room_id}/legacy-sessions", response_model=RoomSessionsOut)
async def list_room_legacy_sessions(
    room_id: UUID,
    response: Response,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return `agentsession` rows for a room — the pre-`session`-table format.

    Deprecated. This used to be registered as GET /{room_id}/sessions and, because
    tasks.router is included before sessions.router, it shadowed the real
    implementation in api/v1/sessions.py; the Sessions tab therefore read the
    empty legacy table and showed zeros for rooms with hundreds of sessions
    (internal design notes stage 5.1). Kept on an explicit
    path for one release in case an external consumer still reads this shape.
    """
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f'</api/v1/rooms/{room_id}/sessions>; rel="successor-version"'
    await _verify_membership(room_id, current_user, db)
    await _get_room(room_id, db)

    # Fetch all tasks for this room
    tasks_res = await db.execute(select(Task).where(Task.room_id == room_id))
    tasks = tasks_res.scalars().all()
    task_map = {t.id: t.title for t in tasks}
    task_ids = [t.id for t in tasks]

    from sqlalchemy import or_
    sessions_res = await db.execute(
        select(AgentSession)
        .where(
            or_(
                AgentSession.room_id == room_id,
                AgentSession.task_id.in_(task_ids) if task_ids else False,
            )
        )
        .order_by(AgentSession.started_at.desc())
    )
    sessions = sessions_res.scalars().all()

    enriched = [await _enrich_session(s, db, task_title=task_map.get(s.task_id)) for s in sessions]

    total_input = sum(s.input_tokens for s in sessions)
    total_output = sum(s.output_tokens for s in sessions)
    total_cost = round(sum(s.cost_usd for s in sessions), 6)

    return RoomSessionsOut(
        sessions=enriched,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_usd=total_cost,
    )


# ── Task endpoints ───────────────────────────────────────────────────────────

@router.post("/{room_id}/tasks", response_model=TaskOut)
async def create_task(
    room_id: UUID,
    body: TaskCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_room(room_id, db)
    await _require_room_agent(room_id, body.assigned_agent_id, db)

    task = Task(
        room_id=room_id,
        goal_id=body.goal_id,
        title=body.title,
        description=body.description,
        assigned_agent_id=body.assigned_agent_id,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    await emit_event(
        room_id=room_id,
        event_type="TASK_CREATED",
        content=f"Task created: {task.title}",
        db=db,
        task_id=task.id,
        metadata={"task_title": task.title, "task_state": task.state.value},
    )
    return await _enrich_task(task, db)


@router.get("/{room_id}/tasks", response_model=List[TaskOut])
async def list_tasks(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    res = await db.execute(
        select(Task).where(Task.room_id == room_id).order_by(Task.created_at.asc())
    )
    tasks = res.scalars().all()
    return [await _enrich_task(t, db) for t in tasks]


@router.get("/{room_id}/tasks/{task_id}", response_model=TaskOut)
async def get_task(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)
    return await _enrich_task(task, db)


@router.get("/{room_id}/tasks/{task_id}/dialogue")
async def get_task_dialogue(
    room_id: UUID,
    task_id: UUID,
    view: bool = False,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 7 — metadata (and, with ``view=true``,
    the raw markdown) of the task-formation dialogue archive behind the card
    chip. Falls back to the goal archive when the task has no archive of its own.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core import plan_dialogue as pd

    meta = await pd.archive_meta(room_id, task_id=task_id, goal_id=task.goal_id)
    if view and meta["bound"]:
        meta["content"] = await pd.read_archive(room_id, meta["path"])
    return meta


@router.post("/{room_id}/tasks/{task_id}/dialogue/unlink")
async def unlink_task_dialogue(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Detach the dialogue archive from a task — delete the task-scoped file and
    clear ``origin_session_id``. The goal archive (shared) is never touched."""
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core import plan_dialogue as pd

    ok = await pd.unlink_archive(room_id, task_id=task_id)
    task.origin_session_id = None
    db.add(task)
    await db.commit()
    return {"ok": ok}


@router.patch("/{room_id}/tasks/{task_id}/state", response_model=TaskOut)
async def change_task_state(
    room_id: UUID,
    task_id: UUID,
    body: TaskStateChange,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    # Idempotent Start (part 3, stage 4): TASK_TRANSITIONS has no
    # in_progress→in_progress entry — states normally only *change* — so
    # without this a same-state Start request (frontend retry, or a click
    # racing an already-completed auto-claim) would get a spurious "transition
    # not allowed" 422 instead of the no-op request_task_dispatch already
    # guarantees for a genuinely active task (stage 2). Only IN_PROGRESS gets
    # this treatment; other repeated PATCHes keep failing loudly.
    if body.state == TaskState.IN_PROGRESS and task.state == TaskState.IN_PROGRESS:
        # run_id is only ever set by the modern proxy dispatcher
        # (handle_task_created). A task stuck at IN_PROGRESS with run_id=None
        # belongs to the legacy approve_and_start pipeline
        # (TASK_LIFECYCLE_REFERENCE.md), which this endpoint cannot restart —
        # echo current state without calling the modern dispatcher, or a
        # second execution would start alongside the still-running legacy loop.
        if task.run_id is not None:
            # Only unwedge when the generation is *provably* over: an explicit
            # dead runtime_state and no session still open. An unknown/missing
            # runtime_state is treated as still-alive so a genuinely running
            # task is never torn down here.
            runtime_dead = getattr(task, "runtime_state", None) in {
                "paused", "stopped", "failed"
            }
            open_session = None
            if runtime_dead:
                open_session = (
                    await db.execute(
                        select(CmAgentProxySession.id).where(
                            CmAgentProxySession.task_id == task.id,
                            CmAgentProxySession.closed_at.is_(None),
                        ).limit(1)
                    )
                ).first()
            if runtime_dead and open_session is None:
                # Wedged: shows IN_PROGRESS but its generation is paused/dead
                # (a pause that never got resumed, or a dispatch that died right
                # after claiming the task and left dispatch_status=DISPATCHED).
                # request_task_dispatch would just report "already active", so
                # do the explicit restart /retry and /resume perform.
                from app.models.task import DispatchStatus

                await close_open_task_sessions(task.id, "retry", db)
                task.dispatch_status = DispatchStatus.READY
                task.run_id = None
                task.runtime_state = "idle"
                task.stop_requested_at = None
                task.stop_reason = None
                task.updated_at = datetime.utcnow()
                db.add(task)
                await db.commit()
                await db.refresh(task)
                outcome = await request_task_dispatch(
                    task, room_id, db, resume=True, source="manual_start_unwedge",
                )
                _logger.info(
                    "tasks: unwedged stuck in_progress task %s outcome=%s", task.id, outcome,
                )
            else:
                outcome = await request_task_dispatch(
                    task, room_id, db, source="manual_start_idempotent",
                )
                _logger.info(
                    "tasks: idempotent Start for already in_progress task %s outcome=%s",
                    task.id, outcome,
                )
        return await _enrich_task(task, db)

    # Approve on a checkpoint task from the TaskPanel/board finishes the gate,
    # same as the chat checkpoint card (see is_checkpoint_approval).
    checkpoint_approval = is_checkpoint_approval(task, body.state)
    if checkpoint_approval:
        body.state = TaskState.COMPLETED
        task.checkpoint_choice = task.checkpoint_choice or {
            "approved": True,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

    allowed = TASK_TRANSITIONS.get(task.state, [])
    if checkpoint_approval:
        allowed = [*allowed, TaskState.COMPLETED]
    # Manual "mark as done" (see _MANUAL_COMPLETE_STATES): a user-initiated
    # COMPLETED is accepted from a few states the workflow table does not
    # allow, because the person, unlike the reviewing agent, may know the work
    # is actually finished. Everything below treats it like any other manual
    # completion — with the extra runtime fencing the workflow path never
    # needs (a forced completion can land on a task that is still executing).
    manual_complete = (
        body.state == TaskState.COMPLETED
        and body.state not in allowed
        and task.state in _MANUAL_COMPLETE_STATES
    )
    if body.state not in allowed and not manual_complete:
        raise HTTPException(
            status_code=422,
            detail=f"Transition {task.state} → {body.state} is not allowed. "
                   f"Allowed: {[s.value for s in allowed]}",
        )

    dispatch = None
    await _require_room_agent(room_id, body.assigned_agent_id, db)
    original_assignee = task.assigned_agent_id
    if body.assigned_agent_id is not None:
        # Let Start+Assign use the new assignee for preflight without
        # persisting anything if dispatch is impossible.
        task.assigned_agent_id = body.assigned_agent_id
    if body.state == TaskState.IN_PROGRESS:
        from app.core.task_deps import refresh_blocked_dispatch_status
        from app.models.task import DispatchStatus

        if task.dispatch_status == DispatchStatus.BLOCKED:
            # Predecessors that finished before the edge existed never fire the
            # unblocking transition — check their states instead of trusting the
            # stored flag, or the Start button stays dead forever.
            await refresh_blocked_dispatch_status(task, db)
        if task.dispatch_status == DispatchStatus.BLOCKED:
            task.assigned_agent_id = original_assignee
            from app.core.task_deps import blocking_predecessors

            blockers = await blocking_predecessors(task.id, db)
            names = ", ".join(f"“{b.title}” ({b.state.value})" for b in blockers)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Task is blocked by unfinished dependencies and cannot be "
                    f"started yet: {names}" if names else
                    "Task is blocked by unfinished dependencies and cannot be started yet"
                ),
            )
        try:
            # A manual Start may restart a stale DISPATCHED task, so readiness
            # is reset below after old sessions are closed. Configuration must
            # still be valid before the visible state can change.
            # Preflight the *target* state: the task is still sitting in its
            # previous one here, and WAITING_REVIEW is non-dispatchable — so
            # without target_state "Back to Work" refused itself with
            # invalid_task_state on every click.
            dispatch = await prepare_task_dispatch(
                task, db, require_ready=False, target_state=TaskState.IN_PROGRESS,
            )
        except TaskDispatchPreflightError as exc:
            task.assigned_agent_id = original_assignee
            raise HTTPException(
                status_code=422,
                detail=f"Task cannot be started: {exc.code} ({exc})",
            ) from None

    prev_state = task.state
    resuming_from_pause = body.state in {
        TaskState.IN_PROGRESS, TaskState.RESUMING
    } and prev_state in {TaskState.PAUSED, TaskState.PAUSING, TaskState.RESUMING}
    # "Back to Work" out of review. Like Resume — and unlike an ordinary Start —
    # this is an explicit restart: the agent already submitted, so its sessions
    # are closed and dispatch_status is still DISPATCHED from the finished run.
    # Without the same close+reset dance below, request_task_dispatch would
    # report "already_active" and the task would sit in IN_PROGRESS with
    # nothing executing.
    returning_from_review = (
        body.state == TaskState.IN_PROGRESS and prev_state == TaskState.WAITING_REVIEW
    )
    task.state = body.state
    if body.state in {TaskState.PAUSED, TaskState.PAUSING}:
        task.pause_reason = task.pause_reason or "user"
        task.paused_at = datetime.utcnow()
    elif resuming_from_pause:
        task.pause_reason = None
        task.paused_at = None
    task.updated_at = datetime.utcnow()

    db.add(task)
    await db.commit()
    await db.refresh(task)

    await emit_event(
        room_id=room_id,
        event_type="TASK_STATE_CHANGED",
        content=f"Task \"{task.title}\": {prev_state.value} → {task.state.value}",
        db=db,
        task_id=task.id,
        metadata={
            "task_title": task.title,
            "prev_state": prev_state.value,
            "task_state": task.state.value,
            # Audit marker: this completion bypassed the review gate because a
            # human forced it, not because the workflow reached it.
            "manual_complete": manual_complete,
        },
    )

    # A lifecycle change must not merely repaint the task card.  Blocking or
    # cancelling fences the currently executing proxy generation and closes
    # its durable session rows; cancellation also interrupts a same-worker
    # coroutine immediately where possible.
    # A forced completion goes down the same road: without it an agent still
    # executing the task would keep writing into a session whose task is
    # already terminal.  It differs in the close reason — the work is done,
    # not failed or abandoned — and in fencing only a live runtime (below).
    # A user's "Send for Review" on a task that is still executing is the same
    # kind of stop: the person decided the work is ready to be judged. Before
    # this the click only repainted the card — the agent kept thinking and
    # ChatMaster kept waiting on it, inside a task that was already in review.
    manual_review = body.state == TaskState.WAITING_REVIEW and prev_state in {
        TaskState.IN_PROGRESS, TaskState.ASSIGNED,
    }
    manual_stop = manual_complete or manual_review
    stop_runtime = body.state in {TaskState.BLOCKED, TaskState.CANCELLED} or manual_stop
    if stop_runtime:
        interrupt_local = body.state == TaskState.CANCELLED or manual_stop
        if body.state == TaskState.CANCELLED:
            close_reason = "cancelled"
        elif manual_stop:
            close_reason = "task_done"
        else:
            close_reason = "error"
        # The stop fence stays visible on the task ("Stopping…") until a worker
        # acknowledges it, so a forced completion writes it only when something
        # is really executing — otherwise an already finished task would wear a
        # stop marker nobody is left to clear.  Sessions are closed either way:
        # a stale open session must not outlive its now-terminal task.
        was_running = task.runtime_state in {RUNTIME_RUNNING, RUNTIME_STOP_REQUESTED}
        if was_running or not manual_stop:
            await fence_task_runtime(
                task,
                body.state.value,
                db,
                cancel_local=interrupt_local,
            )
        await db.commit()
        sessions_closed = await close_open_task_sessions(
            task.id,
            close_reason,
            db,
            cancel_local=interrupt_local,
            # Block/Cancel/forced completion are lifecycle stops: a direct
            # dialogue about this task must end with it, not keep driving the
            # workspace after the card says the task is over.
            include_dialogue=True,
        )
        # For Block/Cancel the event is the user-visible record of the stop and
        # is always emitted.  A manual completion usually lands on a task that
        # is already idle (blocked, never started) — announcing a "stop" there
        # would be noise, so it is only recorded when something really stopped.
        if manual_review:
            # Whatever the agent wrote through bash before the stop is still
            # loose in the working tree; put it on the task branch so the
            # reviewer sees it (Complete would commit it too, but only then).
            try:
                from app.core.room_git_flow import commit_working_tree

                await commit_working_tree(
                    str(room_id), task, db, reason="manual send for review",
                )
            except Exception as exc:
                _logger.warning(
                    "tasks: commit before manual review failed for task %s: %s",
                    task.id, exc,
                )
        if not manual_stop or was_running or sessions_closed:
            await emit_event(
                room_id=room_id,
                event_type="TASK_RUNTIME_STOP_REQUESTED",
                content=(
                    f"Stopping execution of task '{task.title}' was requested "
                    f"(reason: {body.state.value})."
                ),
                db=db,
                task_id=task.id,
                metadata={
                    "task_id": str(task.id),
                    "reason": body.state.value,
                    "sessions_closed": sessions_closed,
                },
            )

        # Stopping a handed-off task means stopping the chain it handed the work
        # to: the CONTINUED card the user clicked has no session of its own, the
        # successor does. Without this the Cancel looked like it worked while the
        # agent kept writing into the successor of a task the user just killed.
        if prev_state == TaskState.CONTINUED:
            for successor in await _continuation_chain(task.id, db):
                if successor.state in {
                    TaskState.COMPLETED, TaskState.CANCELLED, TaskState.CONTINUED
                }:
                    continue
                successor_prev = successor.state
                successor.state = TaskState.CANCELLED
                successor.updated_at = datetime.utcnow()
                db.add(successor)
                await db.commit()
                await fence_task_runtime(
                    successor, TaskState.CANCELLED.value, db, cancel_local=True,
                )
                await db.commit()
                closed = await close_open_task_sessions(
                    successor.id, "cancelled", db,
                    cancel_local=True, include_dialogue=True,
                )
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_STATE_CHANGED",
                    content=(
                        f"Task \"{successor.title}\": {successor_prev.value} → "
                        f"{TaskState.CANCELLED.value} "
                        f"(continuation of the stopped task '{task.title}')"
                    ),
                    db=db,
                    task_id=successor.id,
                    metadata={
                        "task_title": successor.title,
                        "prev_state": successor_prev.value,
                        "task_state": TaskState.CANCELLED.value,
                        "continuation_of_task_id": str(task.id),
                        "sessions_closed": closed,
                    },
                )

    # User pressed Pause on a running task. Fence the proxy generation (no local
    # cancel — the loop must reach its next boundary so it can flush the
    # in-flight transcript for Resume) and close the session rows with a
    # resumable reason. If nothing is executing, settle straight to PAUSED.
    if body.state == TaskState.PAUSING:
        was_running = await request_task_stop(task.id, "user", db)
        await db.commit()
        await db.refresh(task)
        sessions_closed = await close_open_task_sessions(task.id, "workspace_paused", db)
        if not was_running and task.state == TaskState.PAUSING:
            from app.models.task import DispatchStatus

            task.state = TaskState.PAUSED
            task.runtime_state = "paused"
            if task.dispatch_status == DispatchStatus.DISPATCHED:
                task.dispatch_status = DispatchStatus.READY
            task.updated_at = datetime.utcnow()
            db.add(task)
            await db.commit()
            await db.refresh(task)
        await emit_event(
            room_id=room_id,
            event_type="TASK_RUNTIME_STOP_REQUESTED",
            content=f"Pause of task '{task.title}' requested by the user.",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "reason": "user_pause",
                "sessions_closed": sessions_closed,
            },
        )

    # When a task transitions to IN_PROGRESS (user pressed Start/Unblock/Rework),
    # kick off the proxy session (CM ↔ Agent) so the agent actually executes.
    # Ordinary Start is idempotent, not a forced restart: if the task is
    # already dispatched or has an open session, request_task_dispatch is a
    # no-op — it must NOT close sessions or reset dispatch_status here, or a
    # click racing a just-started claim would tear down a live session pair.
    # An actual restart of a stuck/failed task is the job of /retry, which
    # performs that close+reset dance explicitly before dispatching.
    #
    # Resuming a paused task IS an explicit restart, though: its previous
    # generation is fenced and its sessions are closed (workspace pause, user
    # pause, or an interrupted worker), yet dispatch_status can still read
    # DISPATCHED. Do the same close+reset dance /retry and /resume do — but with
    # resume=True so the agent continues from the saved transcript instead of
    # starting the task over.
    #
    # "Back to Work" out of WAITING_REVIEW is the same kind of explicit restart:
    # the submitted generation is over and its sessions are closed, so the
    # idempotent path would find a stale DISPATCHED task and do nothing at all.
    if body.state == TaskState.IN_PROGRESS:
        assert dispatch is not None
        restart = resuming_from_pause or returning_from_review
        if restart:
            from app.models.task import DispatchStatus

            await close_open_task_sessions(task.id, "retry", db)
            task.dispatch_status = DispatchStatus.READY
            task.run_id = None
            task.runtime_state = "idle"
            task.stop_requested_at = None
            task.stop_reason = None
            task.updated_at = datetime.utcnow()
            db.add(task)
            await db.commit()
            await db.refresh(task)
        outcome = await request_task_dispatch(
            task,
            room_id,
            db,
            resume=restart,
            # A human verdict is fed back the same way ChatMaster's is: with
            # rework_feedback set, the agent is respawned on its own transcript
            # from the submitted run instead of starting the task from scratch.
            # rework_count is deliberately left alone — MAX_TASK_REWORKS caps
            # the agent's review loop, and a person's decision to reopen a task
            # should not spend that budget.
            rework_feedback=_REVIEW_REOPEN_FEEDBACK if returning_from_review else None,
            source="manual_rework" if returning_from_review else "manual_start",
        )
        _logger.info(
            "tasks: manual Start for task %s (prev_state=%s, outcome=%s, resume=%s)",
            task.id,
            prev_state,
            outcome,
            restart,
        )

    # internal design notes Stage 3 — Gate 1 human approval: a task that was
    # held in WAITING_REVIEW (risk normal/risky) has its branch folded into
    # `dev` (or its goal branch) only now, on the human's click. `trivial`
    # tasks were already auto-merged and reach COMPLETED without passing
    # through WAITING_REVIEW here.
    if prev_state == TaskState.WAITING_REVIEW and body.state == TaskState.COMPLETED:
        if body.requirement_decisions:
            try:
                from app.core.requirements import apply_gate1_requirement_decisions

                await apply_gate1_requirement_decisions(task, body.requirement_decisions, db)
            except Exception as exc:
                _logger.warning(
                    "tasks: requirement decisions failed for task %s: %s", task.id, exc,
                )
        try:
            from app.core.room_git_flow import merge_task_branch

            merge_result = await merge_task_branch(str(room_id), task, db)
            if merge_result.get("ok"):
                task.merge_commit = merge_result.get("sha")
                db.add(task)
                await db.commit()
                await db.refresh(task)
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_MERGED",
                    content=f"Task '{task.title}' merged into {merge_result.get('target')} after review.",
                    db=db,
                    task_id=task.id,
                    metadata={
                        "task_id": str(task.id),
                        "branch": merge_result.get("branch"),
                        "target": merge_result.get("target"),
                        "merge_commit": task.merge_commit,
                    },
                )
            elif merge_result.get("conflict"):
                # The task branch diverged from its base (the pre-merge refresh
                # conflicted). Keep the task in review rather than reporting it
                # done; GIT_CONFLICT is already emitted by merge_task_branch.
                task.state = TaskState.WAITING_REVIEW
                task.merge_commit = None
                db.add(task)
                await db.commit()
                await db.refresh(task)
                return await _enrich_task(task, db)
        except Exception as exc:
            _logger.warning("tasks: task branch merge failed for %s: %s", task.id, exc)

    # When a task is manually completed or approved, unblock dependent tasks.
    # APPROVED matters here because a checkpoint task (generation checkpoint
    # plan, stage 3) can also be approved from the TaskPanel through this
    # endpoint — its dependents must be unblocked even though the chat
    # checkpoint/resolve endpoint is the primary surface. Checkpoint tasks
    # among the unblocked ones are published as CHECKPOINT_REQUEST instead of
    # being dispatched.
    # (Unblocking is not conditional on task.goal_id: dependencies are recorded
    # per task, and proxy_session's completion path unblocks regardless of
    # whether the finished task belonged to a goal.)
    if body.state in {TaskState.APPROVED, TaskState.COMPLETED}:
        newly_ready = await try_unblock_tasks(task.id, db)
        await publish_unblocked_tasks(
            newly_ready, room_id, db,
            source="unblock_after_state_change",
            unblocked_by=task.id,
            predecessor_title=task.title,
        )

    # A task reaching a terminal state may be the last one its goal was
    # waiting for. Every *agent-driven* terminal transition already runs this
    # check (proxy_session on CM approval, session_close on close actions);
    # this endpoint did not, so a goal whose last task was finished by hand
    # stayed ACTIVE and had to be closed manually too.
    if task.goal_id and body.state in {TaskState.COMPLETED, TaskState.CANCELLED}:
        from app.core.goal_progress import maybe_complete_goal

        await maybe_complete_goal(task.goal_id, room_id, db)

    # internal design notes Stage 4 — Gate 2 `promote_policy` hook: under
    # `on_each_task` a completed task opens a `dev → main` promotion for review.
    # A no-op for `manual` / `scheduled` and whenever there is nothing to
    # publish; never raises.
    if body.state == TaskState.COMPLETED:
        from app.core.room_git_flow import maybe_open_promotion

        await maybe_open_promotion(str(room_id), db, initiated_by=current_user.id)

    return await _enrich_task(task, db)


# ── Generation checkpoint resolve (internal design notes, stage 3) ──


def _format_checkpoint_choice_note(task: Task) -> str:
    """Deterministic text appended to dependent tasks' descriptions.

    The next agent receives its task description verbatim in the first turn
    (proxy_session.render_task_data), so the human's choice reaches it
    deterministically through its own task data — not only via chat history.
    """
    choice = task.checkpoint_choice or {}
    lines: List[str] = [
        f"[CHECKPOINT CHOICE] Checkpoint '{task.title}' resolved by the user."
    ]
    if task.checkpoint_kind == "select_variant":
        variant = choice.get("variant")
        variant = variant if isinstance(variant, dict) else {}
        label = variant.get("label") or choice.get("choice_id")
        lines.append(f"Selected variant: {label} (id={choice.get('choice_id')})")
        paths = variant.get("paths")
        if isinstance(paths, list) and paths:
            lines.append("Variant files: " + ", ".join(str(p) for p in paths))
        if variant.get("note"):
            lines.append(f"Variant note: {variant['note']}")
        lines.append("Use ONLY this variant in your work; ignore the other variants.")
    else:
        lines.append("The user confirmed to continue.")
    if choice.get("note"):
        lines.append(f"User note: {choice['note']}")
    return "\n".join(lines)


@router.post("/{room_id}/tasks/{task_id}/checkpoint/resolve", response_model=TaskOut)
async def resolve_task_checkpoint(
    room_id: UUID,
    task_id: UUID,
    body: CheckpointResolve,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Resolve a generation checkpoint task parked in WAITING_APPROVAL.

    Approve (with an explicit variant choice for ``select_variant``) or
    reject. On approve the choice is written to ``checkpoint_choice``,
    appended to the descriptions of direct dependent tasks, and dependents
    are unblocked/dispatched — mirroring confirm_file_read/reject_file_read
    as the human-decision counterpart on the task surface.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state != TaskState.WAITING_APPROVAL:
        raise HTTPException(
            status_code=422,
            detail=(
                "Checkpoint can only be resolved from waiting_approval, "
                f"current state: {task.state.value}"
            ),
        )
    if not task.checkpoint_kind:
        raise HTTPException(status_code=422, detail="Task is not a checkpoint task")

    payload = task.checkpoint_payload or {}
    variants = payload.get("variants")
    variants = variants if isinstance(variants, list) else []

    prev_state = task.state
    if body.approve:
        choice: dict[str, Any] = {"approved": True}
        if task.checkpoint_kind == "select_variant":
            if not body.choice_id:
                raise HTTPException(
                    status_code=422,
                    detail="choice_id is required to approve a select_variant checkpoint",
                )
            variant = next(
                (
                    v for v in variants
                    if isinstance(v, dict) and str(v.get("id")) == str(body.choice_id)
                ),
                None,
            )
            if variants and variant is None:
                known = [str(v.get("id")) for v in variants if isinstance(v, dict)]
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown choice_id {body.choice_id!r}; known variants: {known}",
                )
            choice["choice_id"] = str(body.choice_id)
            if variant is not None:
                choice["variant"] = variant
        if body.note:
            choice["note"] = body.note
        choice["resolved_at"] = datetime.now(timezone.utc).isoformat()

        task.checkpoint_choice = choice
        # Approving a checkpoint finishes it (see is_checkpoint_approval):
        # COMPLETED unblocks dependents like APPROVED did, and also lets the
        # goal complete and moves the gate out of the active task lists.
        task.state = TaskState.COMPLETED
        task.updated_at = datetime.utcnow()
        db.add(task)

        dep_rows = await db.execute(
            select(TaskDependency.task_id).where(
                TaskDependency.depends_on_id == task.id
            )
        )
        dep_ids = [row[0] for row in dep_rows.fetchall()]
        if dep_ids:
            note = _format_checkpoint_choice_note(task)
            dep_tasks = (
                await db.execute(select(Task).where(Task.id.in_(dep_ids)))
            ).scalars().all()
            for dep in dep_tasks:
                if dep.state in {
                    TaskState.COMPLETED,
                    TaskState.CANCELLED,
                    TaskState.CONTINUED,
                    TaskState.REJECTED,
                }:
                    continue
                sep = "\n\n" if dep.description else ""
                dep.description = f"{dep.description}{sep}{note}"
                dep.updated_at = datetime.utcnow()
                db.add(dep)

        await db.commit()
        await db.refresh(task)

        newly_ready = await try_unblock_tasks(task.id, db)
        await publish_unblocked_tasks(
            newly_ready, room_id, db,
            source="checkpoint_resolve",
            unblocked_by=task.id,
            predecessor_title=task.title,
        )

        from app.core.session_close import maybe_refresh_room_memory

        await maybe_refresh_room_memory(room_id, db, trigger="checkpoint_resolved")

        # The checkpoint may be the goal's last open task (a final human
        # acceptance step) — without this the goal stayed ACTIVE.
        if task.goal_id:
            from app.core.goal_progress import maybe_complete_goal

            await maybe_complete_goal(task.goal_id, room_id, db)

        await emit_event(
            room_id=room_id,
            event_type=CHECKPOINT_RESOLVED,
            content=f"Checkpoint '{task.title}' approved by user.",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "task_title": task.title,
                "approved": True,
                "checkpoint_kind": task.checkpoint_kind,
                "choice_id": choice.get("choice_id"),
                "note": choice.get("note"),
            },
        )
    else:
        choice = {"approved": False}
        if body.note:
            choice["note"] = body.note
        choice["resolved_at"] = datetime.now(timezone.utc).isoformat()

        task.checkpoint_choice = choice
        task.state = TaskState.REJECTED
        task.updated_at = datetime.utcnow()
        db.add(task)
        await db.commit()
        await db.refresh(task)

        # Dependents will never run now — same contract as any rejected
        # predecessor: TASK_DEPENDENCY_FAILED, they stay blocked.
        blocked = await get_blocked_dependents(task.id, db)
        for dep in blocked:
            from app.core.metrics import record_task_dependency_failure

            record_task_dependency_failure()
            await emit_event(
                room_id=room_id,
                event_type=TASK_DEPENDENCY_FAILED,
                content=(
                    f"Task '{dep.title}' is blocked: checkpoint "
                    f"'{task.title}' was rejected by the user."
                ),
                db=db,
                task_id=dep.id,
                metadata={
                    "task_id": str(dep.id),
                    "blocked_because": str(task.id),
                    "failed_status": "rejected",
                },
            )

        await emit_event(
            room_id=room_id,
            event_type=CHECKPOINT_RESOLVED,
            content=f"Checkpoint '{task.title}' rejected by user.",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "task_title": task.title,
                "approved": False,
                "checkpoint_kind": task.checkpoint_kind,
                "note": choice.get("note"),
            },
        )

    await emit_event(
        room_id=room_id,
        event_type=TASK_STATE_CHANGED,
        content=f"Task \"{task.title}\": {prev_state.value} → {task.state.value}",
        db=db,
        task_id=task.id,
        metadata={
            "task_title": task.title,
            "prev_state": prev_state.value,
            "task_state": task.state.value,
        },
    )

    return await _enrich_task(task, db)


MAX_TASK_RETRIES = 3
MAX_CONTINUATION_CHAIN_DEPTH = 3
# A proxy-side failure can close both sessions before the dispatcher has moved
# the task beyond ASSIGNED.  Keeping ASSIGNED retryable prevents that task from
# becoming impossible to recover through the normal retry endpoint.
RETRYABLE_STATES = (
    TaskState.ASSIGNED, TaskState.IN_PROGRESS, TaskState.BLOCKED,
    # PAUSING normally settles to PAUSED when the owning proxy run observes
    # its stop fence.  It can remain there after an interrupted worker,
    # though, and must not become a lifecycle dead end: Retry closes that
    # generation and starts a fresh one.
    TaskState.PAUSING, TaskState.PAUSED, TaskState.REJECTED,
)


@router.post("/{room_id}/tasks/{task_id}/retry", response_model=TaskOut)
async def retry_task(
    room_id: UUID,
    task_id: UUID,
    body: TaskRetry,
    force: bool = False,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Restart a stuck or rejected task (chat upgrade stage 5).

    Closes any still-open proxy sessions with reason='retry', resets the task
    to ASSIGNED + READY and redispatches it. After MAX_TASK_RETRIES automatic
    retries the endpoint refuses unless ?force=true is passed.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state not in RETRYABLE_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Only {', '.join(s.value for s in RETRYABLE_STATES)} tasks can be retried",
        )
    if task.retry_count >= MAX_TASK_RETRIES and not force:
        raise HTTPException(
            status_code=422,
            detail=f"Retry limit ({MAX_TASK_RETRIES}) reached — pass ?force=true to retry anyway",
        )

    await _require_room_agent(room_id, body.assigned_agent_id, db)
    original_assignee = task.assigned_agent_id
    if body.assigned_agent_id is not None:
        task.assigned_agent_id = body.assigned_agent_id

    try:
        # Validate before mutating anything — configuration must be valid
        # before the visible state changes. request_task_dispatch resolves
        # the dispatch context again below, after the reset lands.
        await prepare_task_dispatch(task, db, require_ready=False)
    except TaskDispatchPreflightError as exc:
        task.assigned_agent_id = original_assignee
        raise HTTPException(
            status_code=422,
            detail=f"Task cannot be retried: {exc.code} ({exc})",
        ) from None

    sessions_closed = await close_open_task_sessions(task.id, "retry", db)

    from app.models.task import DispatchStatus
    prev_state = task.state
    task.state = TaskState.ASSIGNED
    task.dispatch_status = DispatchStatus.READY
    task.pause_reason = None
    task.paused_at = None
    task.retry_count += 1
    task.updated_at = datetime.utcnow()

    db.add(task)
    await db.commit()
    await db.refresh(task)

    await emit_event(
        room_id=room_id,
        event_type=TASK_RETRIED,
        content=f"Task \"{task.title}\" re-queued for retry (attempt {task.retry_count}).",
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "task_title": task.title,
            "prev_state": prev_state.value,
            "retry_count": task.retry_count,
            "sessions_closed": sessions_closed,
        },
    )

    # Retry deliberately starts a new agent conversation.  In contrast, the
    # /resume endpoint restores the paused transcript, so it is important not
    # to inherit it here: the fresh task-agent system prompt must be applied.
    outcome = await request_task_dispatch(task, room_id, db, resume=False, source="retry")
    _logger.info("tasks: retry dispatch for task %s outcome=%s", task.id, outcome)

    return await _enrich_task(task, db)


@router.post("/{room_id}/tasks/{task_id}/revert-merge", response_model=TaskOut)
async def revert_task_merge_endpoint(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 3 — one-click undo of a task's merge into
    ``dev`` (or its goal branch). Reverts ``task.merge_commit`` with a fresh
    revert commit; a conflicting revert is escalated as ``GIT_CONFLICT``.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if not task.merge_commit:
        raise HTTPException(
            status_code=422,
            detail="This task has no recorded merge to revert",
        )

    from app.core.room_git_flow import revert_task_merge

    result = await revert_task_merge(str(room_id), task, db)
    await db.commit()

    if result.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="Revert did not apply cleanly — resolve the conflict manually (GIT_CONFLICT emitted)",
        )
    if not result.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"Revert failed: {result.get('error') or result.get('stderr') or 'unknown error'}",
        )

    await db.refresh(task)
    return await _enrich_task(task, db)


@router.post("/{room_id}/tasks/{task_id}/return-to-review", response_model=TaskOut)
async def return_task_to_review(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 6 — return a terminal unmerged task branch back
    into the normal Gate 1 flow without raw workspace/merge calls.

    Allowed only when:
    - task has a ``branch_name``;
    - the branch exists and is NOT yet merged into its base;
    - task state is terminal (COMPLETED / CANCELLED / CONTINUED).

    After success the task is ``WAITING_REVIEW`` with ``ci_status=None`` and a
    fresh Gate 1 run launched in the background.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state not in _TERMINAL_TASK_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Task is {task.state!r} — only terminal tasks can be returned to review",
        )
    if not task.branch_name:
        raise HTTPException(status_code=422, detail="Task has no branch to return")

    # Verify the branch is NOT already merged via the workspaced git state
    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import _task_base_branch, run_task_checks

    base = await _task_base_branch(room_id, task, db)
    branches = await wsd.list_workspace_branches(str(room_id), {task.branch_name: base})
    branch_info = next((b for b in branches if b["name"] == task.branch_name), None)

    if branch_info is None:
        raise HTTPException(status_code=422, detail="Branch not found in workspace")
    if branch_info.get("merged") is True:
        raise HTTPException(
            status_code=422,
            detail="Branch is already merged — nothing to return to review",
        )

    prev_state = task.state
    task.state = TaskState.WAITING_REVIEW
    task.merge_commit = None
    task.ci_status = None
    db.add(task)

    await emit_event(
        room_id=room_id,
        event_type="TASK_RETURNED_TO_REVIEW",
        content=f"Task '{task.title}' returned to review (was {prev_state}). Starting Gate 1 checks…",
        db=db,
        task_id=task.id,
        metadata={"prev_state": str(prev_state), "branch": task.branch_name},
    )
    await db.commit()
    await db.refresh(task)

    # Restart Gate 1 CI in background — stale result from before the divergence
    async def _rerun_ci() -> None:
        from sqlalchemy.orm import sessionmaker

        from app.db.session import engine

        maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as bg_db:
            bg_task = await bg_db.get(type(task), task.id)
            if bg_task and bg_task.state == TaskState.WAITING_REVIEW:
                await run_task_checks(room_id, bg_task, bg_db)
                await bg_db.commit()

    import asyncio
    asyncio.create_task(_rerun_ci())

    return await _enrich_task(task, db)


@router.post("/{room_id}/tasks/{task_id}/merge-branch", response_model=TaskOut)
async def merge_reviewed_task_branch(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes — fold a COMPLETED task's still-unmerged branch into
    its base (``dev`` or the goal branch) directly, without bouncing it back
    through the Gate 1 review flow.

    This is the "review already happened, just merge it" shortcut for the
    branch tree: a task that reached COMPLETED but whose branch never landed
    (an old completion, a merge that silently failed, a manual finish).

    Allowed only when:
    - task state is ``COMPLETED``;
    - task has a ``branch_name``;
    - the branch exists and is NOT already merged into its base.

    Uses the same clean path as Gate 1 approval (``merge_task_branch``): the
    task branch is first refreshed against its moved base with a plain 3-way
    merge, so a real overlap surfaces as ``GIT_CONFLICT`` (409 here) instead of
    being dropped. On success ``task.merge_commit`` is stamped and
    ``TASK_MERGED`` is emitted; the task stays COMPLETED.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state != TaskState.COMPLETED:
        raise HTTPException(
            status_code=422,
            detail=f"Task is {task.state!r} — only COMPLETED tasks can be merged this way",
        )
    if not task.branch_name:
        raise HTTPException(status_code=422, detail="Task has no branch to merge")

    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import _task_base_branch, merge_task_branch

    base = await _task_base_branch(room_id, task, db)
    branches = await wsd.list_workspace_branches(str(room_id), {task.branch_name: base})
    branch_info = next((b for b in branches if b["name"] == task.branch_name), None)

    if branch_info is None:
        raise HTTPException(status_code=422, detail="Branch not found in workspace")
    if branch_info.get("merged") is True:
        raise HTTPException(status_code=422, detail="Branch is already merged")

    merge_result = await merge_task_branch(str(room_id), task, db)

    if merge_result.get("conflict"):
        # GIT_CONFLICT already emitted by merge_task_branch. Leave the task
        # COMPLETED and let the user resolve the conflict or return it to review.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Branch {merge_result.get('branch')} has diverged from {merge_result.get('target')} — "
                "resolve the conflict manually or return the task to review"
            ),
        )
    if not merge_result.get("ok"):
        raise HTTPException(
            status_code=422,
            detail=merge_result.get("stderr") or "merge failed",
        )

    task.merge_commit = merge_result.get("sha")
    db.add(task)
    await emit_event(
        room_id=room_id,
        event_type="TASK_MERGED",
        content=f"Task '{task.title}' merged into {merge_result.get('target')}.",
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "branch": merge_result.get("branch"),
            "target": merge_result.get("target"),
            "merge_commit": task.merge_commit,
        },
    )
    await db.commit()
    await db.refresh(task)
    return await _enrich_task(task, db)


# ── internal design notes — "Divergence review" ─────────────────────────

# Tasks whose forced review turn is running. Process-local: it guards a
# background coroutine that lives in this process (mirrors chat._plan_requests).
_divergence_reviews_in_flight: set[UUID] = set()


@router.get("/{room_id}/tasks/{task_id}/divergence-review")
async def get_divergence_review(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """State of the "Divergence review" flow for this task, plus the last
    analysis + report if one has been produced."""
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    ev = (await db.execute(
        select(Message)
        .where(Message.task_id == task_id, Message.event_type == "DIVERGENCE_REVIEW")
        .order_by(Message.timestamp.desc())
        .limit(1)
    )).scalars().first()

    state = task.divergence_state
    if task_id in _divergence_reviews_in_flight:
        state = "reviewing"
    return {
        "task_id": str(task_id),
        "divergence_state": state,
        "branch": task.branch_name,
        "analysis": (ev.event_metadata or {}).get("analysis") if ev else None,
        "report": (ev.event_metadata or {}).get("report") if ev else None,
        "reviewed_at": ev.timestamp.isoformat() if ev else None,
    }


@router.post("/{room_id}/tasks/{task_id}/divergence-review", status_code=202)
async def request_divergence_review(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Run the ChatMaster analysis of this task branch's divergence.

    Router-free, one forced JSON turn on the ChatMaster's model (not a
    session). 202 + a background coroutine; the ``DIVERGENCE_REVIEW`` event
    carries the result.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if not task.branch_name:
        raise HTTPException(status_code=422, detail="The task has no branch")
    if task.divergence_state == "resolved":
        raise HTTPException(status_code=409, detail="The divergence is already resolved")
    if task_id in _divergence_reviews_in_flight:
        raise HTTPException(status_code=409, detail="The analysis is already running")

    _divergence_reviews_in_flight.add(task_id)

    # Immediate visible acknowledgement — the analysis turn can take tens of
    # seconds and the user must not be left wondering whether anything started.
    await emit_event(
        room_id=room_id,
        event_type="DIVERGENCE_REVIEW_STARTED",
        content=f"ChatMaster is analyzing the divergence of branch {task.branch_name}…",
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "branch": task.branch_name},
    )
    await db.commit()

    async def _bg() -> None:
        _logger = logging.getLogger(__name__)
        try:
            from sqlalchemy.orm import sessionmaker

            from app.core.divergence_review import run_divergence_review
            from app.db.session import engine

            maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with maker() as bg_db:
                await run_divergence_review(room_id, task_id, bg_db)
        except Exception:
            _logger.exception("divergence review failed for task %s", task_id)
        finally:
            _divergence_reviews_in_flight.discard(task_id)

    import asyncio
    asyncio.create_task(_bg())
    return {"status": "reviewing", "task_id": str(task_id)}


@router.post("/{room_id}/tasks/{task_id}/divergence-resolve", response_model=TaskOut)
async def resolve_divergence_review(
    room_id: UUID,
    task_id: UUID,
    body: DivergenceResolveRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Execute the user's choice: take_branch / rework / delete_branch."""
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core.divergence_review import DECISIONS, resolve_divergence

    if body.decision not in DECISIONS:
        raise HTTPException(status_code=422, detail=f"Unknown option: {body.decision}")
    if not task.branch_name:
        raise HTTPException(status_code=422, detail="The task has no branch")
    if task.divergence_state == "resolved":
        raise HTTPException(status_code=409, detail="The divergence is already resolved")

    result = await resolve_divergence(room_id, task, db, body.decision, body.note)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("detail") or "could not complete")

    await db.refresh(task)
    return await _enrich_task(task, db)


@router.get("/{room_id}/tasks/{task_id}/ci-log")
async def get_task_ci_log(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 8 — a presigned link to the Gate 1 check
    run archived for this task (``task.ci_log_key``). 404 when no run was
    archived or the log backend has no presigned URLs (``NoopLogArchive``)."""
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if not task.ci_log_key:
        raise HTTPException(status_code=404, detail="No CI log recorded for this task")

    from app.core.log_storage import get_log_archive

    url = await get_log_archive().get_download_url(task.ci_log_key, expires_in=900)
    if not url:
        raise HTTPException(status_code=404, detail="Log backend does not support presigned URLs")
    return {"url": url}


# ── Gate 2 — dev → main promotions (internal design notes Stage 4) ────────────


async def _enrich_promotion(promo: "Promotion", db: AsyncSession) -> PromotionOut:
    from app.core.room_git_flow import _task_titles

    out = PromotionOut.model_validate(promo)
    out.task_titles = await _task_titles(list(promo.task_ids or []), db)
    return out


@router.get("/{room_id}/promotions", response_model=List[PromotionOut])
async def list_promotions(
    room_id: UUID,
    state: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    from app.models.promotion import Promotion

    stmt = select(Promotion).where(Promotion.room_id == room_id)
    if state:
        stmt = stmt.where(Promotion.state == state)
    stmt = stmt.order_by(Promotion.created_at.desc()).limit(min(limit, 100))
    rows = (await db.execute(stmt)).scalars().all()
    return [await _enrich_promotion(p, db) for p in rows]


@router.get("/{room_id}/promotions/preview", response_model=PromotionPreview)
async def promotion_preview(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 8 — everything the room panel needs to decide
    whether to offer the "Publish to main" button and what to say next to it."""
    await _verify_membership(room_id, current_user, db)
    room = await _get_room(room_id, db)

    from app.core.room_git_flow import (
        get_flow_config,
        _open_promotion_row,
        _promotable_task_ids,
        _task_titles,
    )

    cfg = get_flow_config(room)
    open_row = await _open_promotion_row(room_id, db)
    task_ids = await _promotable_task_ids(room_id, db)
    titles = await _task_titles(task_ids, db)

    diff_stat = ""
    try:
        from app.core import workspaced_client as wsd

        resp = await wsd.git_diff_workspace(
            str(room_id), from_ref=cfg.release_branch, to_ref=cfg.working_branch, stat=True
        )
        if isinstance(resp, dict):
            diff_stat = str(resp.get("diff", "") or "")
    except Exception:  # workspaced offline — button still renders from task count
        diff_stat = ""

    return PromotionPreview(
        promote_policy=cfg.promote_policy,
        working_branch=cfg.working_branch,
        release_branch=cfg.release_branch,
        ci_provider=cfg.ci_provider,
        diff_stat=diff_stat[:4000],
        pending_task_count=len(task_ids),
        pending_task_titles=titles,
        open_promotion_id=open_row.id if open_row is not None else None,
    )


@router.post("/{room_id}/promotions", response_model=PromotionOut)
async def open_promotion_endpoint(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Open a Gate 2 promotion (``dev → main``) for human review.

    409 when a promotion is already awaiting review or ``dev`` has nothing
    ``main`` does not already have.
    """
    await _verify_membership(room_id, current_user, db)
    await _get_room(room_id, db)

    from app.core.room_git_flow import (
        PromotionAlreadyOpen,
        PromotionEmpty,
        PromotionError,
        open_promotion,
    )

    try:
        promo = await open_promotion(str(room_id), db, initiated_by=current_user.id)
    except (PromotionAlreadyOpen, PromotionEmpty) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PromotionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

    await db.commit()
    await db.refresh(promo)
    return await _enrich_promotion(promo, db)


@router.post("/{room_id}/promotions/{promotion_id}/resolve", response_model=PromotionOut)
async def resolve_promotion_endpoint(
    room_id: UUID,
    promotion_id: UUID,
    body: PromotionResolve,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Approve (merge ``dev`` into ``main``) or reject a Gate 2 promotion.

    Approve is refused with 409 when the full CI suite on ``dev`` is not green,
    unless ``force`` is set. A merge conflict is escalated as ``GIT_CONFLICT``
    and returns 409 — ``main`` is left untouched.
    """
    await _verify_membership(room_id, current_user, db)
    from app.models.promotion import Promotion

    promo = (
        await db.execute(
            select(Promotion).where(
                Promotion.id == promotion_id, Promotion.room_id == room_id
            )
        )
    ).scalar_one_or_none()
    if promo is None:
        raise HTTPException(status_code=404, detail="Promotion not found")

    if body.decision not in {"approve", "reject"}:
        raise HTTPException(status_code=422, detail="decision must be 'approve' or 'reject'")

    # Stage 7: a PR-backed promotion is merged through the forge API on approve.
    git_token: Optional[str] = None
    if body.decision == "approve" and promo.pr_url:
        room = await _get_room(room_id, db)
        from app.core.git_forge import resolve_forge_token

        git_token = await resolve_forge_token(db, room, request=request)

    from app.core.room_git_flow import resolve_promotion

    result = await resolve_promotion(
        str(room_id), promo, db,
        decision=body.decision,
        note=body.note,
        force=body.force,
        resolved_by=current_user.id,
        git_token=git_token,
    )
    await db.commit()

    if result.get("blocked"):
        raise HTTPException(
            status_code=409,
            detail=f"CI on dev is {result.get('ci_status')!r} — pass force=true to promote anyway",
        )
    if result.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail="dev → main did not merge cleanly — resolve the conflict manually (GIT_CONFLICT emitted)",
        )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=f"Promotion failed: {result.get('error')}")

    await db.refresh(promo)
    return await _enrich_promotion(promo, db)


@router.post("/{room_id}/tasks/{task_id}/resume", response_model=TaskOut)
async def resume_task(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Resume a task paused on a tool-call or token-budget limit.

    Unlike /retry, the new agent session continues from the saved conversation
    history of the paused session instead of starting the task over. Does not
    consume a retry attempt.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state not in (TaskState.BLOCKED, TaskState.IN_PROGRESS, TaskState.PAUSED):
        raise HTTPException(
            status_code=422,
            detail="Only paused, blocked or in_progress tasks can be resumed",
        )
    if await load_paused_history(task_id, db) is None:
        raise HTTPException(
            status_code=422,
            detail="No paused limit session found for this task — use retry instead",
        )
    try:
        # Validate before mutating anything, same as /retry above.
        await prepare_task_dispatch(task, db, require_ready=False)
    except TaskDispatchPreflightError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Task cannot be resumed: {exc.code} ({exc})",
        ) from None

    sessions_closed = await close_open_task_sessions(task.id, "retry", db)

    from app.models.task import DispatchStatus
    prev_state = task.state
    task.state = TaskState.ASSIGNED
    task.dispatch_status = DispatchStatus.READY
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.commit()
    await db.refresh(task)

    await emit_event(
        room_id=room_id,
        event_type=TASK_RESUMED,
        content=f"Task \"{task.title}\" resumed from where it stopped (the new session's counters were reset).",
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "task_title": task.title,
            "prev_state": prev_state.value,
            "sessions_closed": sessions_closed,
        },
    )

    outcome = await request_task_dispatch(task, room_id, db, resume=True, source="resume")
    _logger.info("tasks: resume dispatch for task %s outcome=%s", task.id, outcome)

    return await _enrich_task(task, db)


@router.post("/{room_id}/tasks/{task_id}/handoff", response_model=TaskOut)
async def create_task_handoff(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Create a compact continuation task after a proxy token-budget stop.

    Unlike ``resume``, the successor starts with a fresh conversation and gets
    only a task-scoped checkpoint through the memory layer.  The endpoint is
    idempotent: repeated calls return the already-created successor.
    """
    await _verify_membership(room_id, current_user, db)

    locked_task = (
        await db.execute(
            select(Task)
            .where(Task.id == task_id, Task.room_id == room_id)
            .with_for_update()
        )
    ).scalars().first()
    if locked_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    existing_handoff = (
        await db.execute(
            select(TaskHandoff).where(TaskHandoff.source_task_id == locked_task.id)
        )
    ).scalars().first()
    if existing_handoff and existing_handoff.target_task_id:
        existing_target = (
            await db.execute(select(Task).where(Task.id == existing_handoff.target_task_id))
        ).scalars().first()
        if existing_target:
            return await _enrich_task(existing_target, db)

    source_session = (
        await db.execute(
            select(CmAgentProxySession)
            .where(
                CmAgentProxySession.task_id == locked_task.id,
                CmAgentProxySession.side == "agent",
                CmAgentProxySession.close_reason.in_(["token_limit", "context_limit"]),
            )
            .order_by(CmAgentProxySession.started_at.desc())
            .limit(1)
        )
    ).scalars().first()
    if source_session is None:
        raise HTTPException(
            status_code=422,
            detail="A continuation task can be created only after a token-budget or context-window limit",
        )
    if not locked_task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Task has no assigned agent")

    tool_events = (
        await db.execute(
            select(Message)
            .where(
                Message.task_id == locked_task.id,
                Message.event_type == "AGENT_TOOL_CALL",
            )
            .order_by(Message.timestamp.desc())
            .limit(120)
        )
    ).scalars().all()
    tool_events = list(reversed(tool_events))

    files_read: list[str] = []
    read_manifest = await _handoff_read_manifest(source_session, db)
    files_read.extend(str(item["path"]) for item in read_manifest if item.get("path"))
    artifacts: list[str] = []
    tests_and_results: list[str] = []
    activity: list[str] = []
    for event in tool_events:
        meta = event.event_metadata or {}
        tool = meta.get("tool")
        target = meta.get("target")
        activity.append(event.content)
        if tool in ("read_file", "grep", "glob") and target:
            files_read.append(str(target))
        if tool in ("write_file", "edit_file") and target:
            artifacts.append(str(target))
        if tool == "bash" and target:
            command = str(target)
            if any(marker in command.lower() for marker in ("test", "pytest", "vitest", "jest", "lint", "typecheck")):
                tests_and_results.append(command)

    # Caps mirror _HANDOFF_* limits in memory_layer._render_task_handoff so the
    # DB snapshot itself is compact — the render is a second safety net, not the
    # only line of defense.  Oversized snapshots used to make continuation tasks
    # unable to start (preflight rejected the system prompt with prompt_cap).
    saved_messages = (source_session.context_snapshot or {}).get("messages") or []
    recent_context = [
        {
            "role": msg.get("role"),
            "content": _checkpoint_content(msg.get("content"), 800),
        }
        for msg in saved_messages[-5:]
        if isinstance(msg, dict)
    ]
    summary = (
        f"Continuation of task '{locked_task.title}' after the token budget was reached. "
        f"The original session used {source_session.tokens_in + source_session.tokens_out:,} tokens. "
        f"Actions recorded: {len(activity)}, paths read: {len(set(files_read))}, "
        f"artifacts created or changed: {len(set(artifacts))}."
    )
    checkpoint = {
        "source_task_id": str(locked_task.id),
        "source_session_id": str(source_session.id),
        "completed_work": activity[-15:],
        "artifacts": sorted(set(artifacts)),
        "files_read": sorted(set(files_read)),
        "read_manifest": read_manifest,
        "observation_refs": sorted(
            {
                ref
                for item in read_manifest
                for ref in (item.get("observation_refs") or [])
            }
        ),
        "decisions": (source_session.summary or "")[:2_000],
        "tests_and_results": tests_and_results[-10:],
        "remaining_work": (
            "Continue the original task from the last saved step; "
            "first check the unfinished items without repeating the listed actions."
        ),
        "recent_context": recent_context,
        "token_usage": {
            "tokens_in": source_session.tokens_in,
            "tokens_out": source_session.tokens_out,
            "limit": 100_000,
        },
    }

    # 8.3 — walk the continuation chain and accumulate artifacts/files_read
    # from ancestor handoffs so the Nth successor knows what the 1st and 2nd
    # runs already touched (max 5 levels to avoid runaway DB queries).
    ancestor_artifacts: set[str] = set()
    ancestor_files_read: set[str] = set()
    ancestor_manifest: list[dict[str, Any]] = []
    chain_depth = 1
    _cur = locked_task
    for _ in range(5):
        parent_id = _cur.continuation_of_task_id
        if not parent_id:
            break
        parent_handoff_row = (
            await db.execute(
                select(TaskHandoff).where(TaskHandoff.target_task_id == _cur.id)
            )
        ).scalars().first()
        if parent_handoff_row and parent_handoff_row.context_snapshot:
            snap = parent_handoff_row.context_snapshot
            ancestor_artifacts.update(snap.get("artifacts") or [])
            ancestor_files_read.update(snap.get("files_read") or [])
            ancestor_manifest.extend(snap.get("read_manifest") or [])
        parent_task = await db.get(Task, parent_id)
        if parent_task is None:
            break
        _cur = parent_task
        chain_depth += 1

    if ancestor_artifacts or ancestor_files_read:
        checkpoint["artifacts"] = sorted(set(checkpoint.get("artifacts") or []) | ancestor_artifacts)
        checkpoint["files_read"] = sorted(set(checkpoint.get("files_read") or []) | ancestor_files_read)
    if ancestor_manifest:
        checkpoint["read_manifest"] = _merge_handoff_manifests(
            checkpoint.get("read_manifest") or [],
            ancestor_manifest,
        )
        checkpoint["observation_refs"] = sorted(
            {
                ref
                for item in checkpoint["read_manifest"]
                for ref in (item.get("observation_refs") or [])
            }
        )
    checkpoint["chain_depth"] = chain_depth
    if chain_depth >= MAX_CONTINUATION_CHAIN_DEPTH:
        raise HTTPException(
            status_code=422,
            detail=(
                "Continuation limit reached. The same task has exhausted "
                f"{MAX_CONTINUATION_CHAIN_DEPTH} provider sessions; review the "
                "saved diagnostic checkpoint before starting another task."
            ),
        )

    successor = Task(
        room_id=locked_task.room_id,
        goal_id=locked_task.goal_id,
        title=f"{locked_task.title} — continuation",
        description=(
            locked_task.description
            + "\n\nContinue the work per the task handoff; completed actions and artifacts "
              "will be passed to the start of the new session."
        ).strip(),
        assigned_agent_id=locked_task.assigned_agent_id,
        created_by_agent_id=locked_task.created_by_agent_id,
        continuation_of_task_id=locked_task.id,
        state=TaskState.ASSIGNED,
    )
    db.add(successor)
    await db.flush()

    handoff = existing_handoff or TaskHandoff(
        source_task_id=locked_task.id,
        source_session_id=source_session.id,
    )
    handoff.target_task_id = successor.id
    handoff.summary = summary
    handoff.context_snapshot = checkpoint
    db.add(handoff)

    # Downstream tasks must wait for the successor instead of treating the
    # partially completed source task as finished.
    outgoing_edges = (
        await db.execute(
            select(TaskDependency).where(TaskDependency.depends_on_id == locked_task.id)
        )
    ).scalars().all()
    for edge in outgoing_edges:
        db.add(TaskDependency(task_id=edge.task_id, depends_on_id=successor.id))
        await db.delete(edge)

    locked_task.state = TaskState.CONTINUED
    locked_task.updated_at = datetime.utcnow()
    db.add(locked_task)
    await db.commit()
    await db.refresh(successor)

    await emit_event(
        room_id=room_id,
        event_type=TASK_HANDOFF_CREATED,
        content=(
            f"An interim result was prepared and a continuation task was created: "
            f"'{successor.title}'. The new session will start with a zero token counter."
        ),
        db=db,
        task_id=successor.id,
        metadata={
            "source_task_id": str(locked_task.id),
            "task_id": str(successor.id),
            "task_title": successor.title,
            "handoff_id": str(handoff.id),
        },
    )

    try:
        outcome = await request_task_dispatch(successor, room_id, db, source="handoff")
    except TaskDispatchPreflightError as exc:
        _logger.info("continuation task %s not dispatched: %s", successor.id, exc.code)
    else:
        _logger.info("continuation task %s dispatch outcome=%s", successor.id, outcome)
    return await _enrich_task(successor, db)


@router.delete("/{room_id}/tasks/{task_id}", status_code=204)
async def delete_task(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if task.state not in (TaskState.DRAFT, TaskState.REJECTED, TaskState.CANCELLED):
        raise HTTPException(
            status_code=422,
            detail="Only DRAFT, REJECTED or CANCELLED tasks can be deleted",
        )

    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(AgentSession).where(AgentSession.task_id == task_id))
    await db.delete(task)
    await db.commit()

    # Stage 8: a deleted Task takes its task-formation dialogue archive with it
    # (the …-attachments/ files are cleared by the retention sweep).
    try:
        from app.core import plan_dialogue as pd

        await pd.unlink_archive(room_id, task_id=task_id)
    except Exception:  # best-effort — the Task is already gone
        pass


# ── Session endpoints ────────────────────────────────────────────────────────

@router.get("/{room_id}/tasks/{task_id}/sessions", response_model=List[ProxySessionOut])
async def list_sessions(
    room_id: UUID,
    task_id: UUID,
    response: Response,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_task(task_id, room_id, db)

    res = await db.execute(
        select(CmAgentProxySession)
        .where(CmAgentProxySession.task_id == task_id)
        .order_by(CmAgentProxySession.started_at.asc())
    )
    sessions = res.scalars().all()
    return [await _enrich_proxy_session(s, db) for s in sessions]


@router.get("/{room_id}/tasks/{task_id}/runtime-status")
async def get_task_runtime_status(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 1 — "Executor status": what the running
    task is doing (model request / tool / idle, attempt, turn, deadline) and a
    verdict. Answered from runtime data, no LLM — works when the provider hangs.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core import task_status_probe

    return await task_status_probe.collect(task, db)


class TaskInterjectIn(BaseModel):
    text: str


@router.post("/{room_id}/tasks/{task_id}/interject")
async def interject_task(
    room_id: UUID,
    task_id: UUID,
    body: TaskInterjectIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 2 — "Message the agent": queue a message into
    the running task. The executor gets it before its next model request and
    answers with ``reply_to_user``; the response carries the status card
    snapshot, which answers "is it stuck?" right away. 409 ``{code, message}``
    when the task cannot take a message (paused, not running, waiting for you).
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core import task_interject, task_status_probe

    try:
        row = await task_interject.create(task, user=current_user, text=body.text, db=db)
    except task_interject.InterjectRefused as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": exc.message})
    return {
        "interjection": {
            "id": str(row.id),
            "message_id": str(row.message_id) if row.message_id else None,
            "created_at": row.created_at,
        },
        "status": await task_status_probe.collect(task, db),
    }


@router.post("/{room_id}/tasks/{task_id}/llm-call/restart")
async def restart_task_llm_call(
    room_id: UUID,
    task_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 3 — "Abort the request and retry": drop the
    model request in flight and send the same request again (the retry attempt
    is not burned, the turn is kept). ``delivered``: ``now`` — the loop is in
    this process and the request was cut; ``queued`` — the loop's heartbeat
    picks it up within LLM_HEARTBEAT_SECONDS. 409 ``{code, message}``:
    not_running / not_in_llm / limit / no_session.
    """
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    from app.core import llm_call_restart, task_status_probe

    try:
        delivered = await llm_call_restart.request(task, db)
    except llm_call_restart.RestartRefused as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": exc.message})
    return {"delivered": delivered, "status": await task_status_probe.collect(task, db)}


@router.post("/{room_id}/tasks/{task_id}/sessions", response_model=SessionOut)
async def start_session(
    room_id: UUID,
    task_id: UUID,
    body: SessionCreate,
    response: Response,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    _logger.warning(
        "DEPRECATED: POST /rooms/%s/tasks/%s/sessions — use the new chat flow "
        "(see %s). This endpoint will be removed on %s.",
        room_id, task_id, _DEPRECATION_LINK, _DEPRECATION_DATE,
    )
    response.headers["Deprecation"] = f'@{_DEPRECATION_DATE}'
    response.headers["Sunset"] = _DEPRECATION_DATE
    response.headers["Link"] = f'<{_DEPRECATION_LINK}>; rel="deprecation"'
    await _verify_membership(room_id, current_user, db)
    task = await _get_task(task_id, room_id, db)

    if not task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Task has no assigned agent")
    if task.state not in (TaskState.ASSIGNED, TaskState.IN_PROGRESS):
        raise HTTPException(
            status_code=422,
            detail="Session can only be started for ASSIGNED or IN_PROGRESS tasks",
        )

    session = AgentSession(
        task_id=task_id,
        agent_id=task.assigned_agent_id,
        context_snapshot=body.context_snapshot,
    )
    db.add(session)

    if task.state == TaskState.ASSIGNED:
        task.state = TaskState.IN_PROGRESS
        task.updated_at = datetime.utcnow()
        db.add(task)

    await db.commit()
    await db.refresh(session)

    enriched = await _enrich_session(session, db)
    await emit_event(
        room_id=room_id,
        event_type="SESSION_STARTED",
        content=f"Agent {enriched.agent_nickname or session.agent_id} started session for task \"{task.title}\"",
        db=db,
        task_id=task_id,
        metadata={"task_title": task.title, "agent_nickname": enriched.agent_nickname},
    )
    return enriched


@router.patch(
    "/{room_id}/tasks/{task_id}/sessions/{session_id}/complete",
    response_model=SessionOut,
)
async def complete_session(
    room_id: UUID,
    task_id: UUID,
    session_id: UUID,
    body: SessionComplete,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_task(task_id, room_id, db)

    res = await db.execute(
        select(AgentSession).where(
            AgentSession.id == session_id,
            AgentSession.task_id == task_id,
        )
    )
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at:
        raise HTTPException(status_code=422, detail="Session already completed")

    task = await _get_task(task_id, room_id, db)
    session.summary = body.summary
    session.token_count = body.token_count
    session.input_tokens = body.input_tokens
    session.output_tokens = body.output_tokens
    session.cost_usd = body.cost_usd
    if body.model_id:
        session.model_id = body.model_id
    session.ended_at = datetime.utcnow()

    db.add(session)
    await db.commit()
    await db.refresh(session)

    enriched = await _enrich_session(session, db)
    await emit_event(
        room_id=room_id,
        event_type="SESSION_COMPLETED",
        content=f"Agent {enriched.agent_nickname or session.agent_id} completed session for task \"{task.title}\"",
        db=db,
        task_id=task_id,
        metadata={
            "task_title": task.title,
            "agent_nickname": enriched.agent_nickname,
            "token_count": body.token_count,
        },
    )
    return enriched
