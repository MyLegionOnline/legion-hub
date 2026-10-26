"""Dispatcher branching for connector-executed tasks.

``_run_proxy_loop`` is a *local* executor: the LLM + tool loop runs inside our
own sandbox. ``_run_connector_loop`` is the *remote* executor for the exact
same session pair, task and ``run_id`` — it only sends the task to a
connector over the durable outbox (``core/connector_broker.py``) and keeps
the run's cross-worker fence (``core/task_runtime.py``) alive while the
remote side works. Everything that happens before/after the call in
``proxy_session.handle_task_created`` (claim, session-pair creation,
``finalize_run``, rework redispatch) is shared, untouched code.

Once the dispatch frame is acked, this loop's only job is to wait on the
``asyncio.Future`` ``core/connector_reverse.py`` resolves when the
connector's ``result``/``error`` frame settles the generation — the actual
``status``/``reasoning``/``ask_cm``/``ask_user`` backchannel handling lives
there, keyed off the same fenced inbound frames ``connector_ws.py`` accepts.
Before that ack, this loop's job is narrower but still load-bearing: get the
``dispatch`` frame to the connector exactly once reliably, detect a
connector that never shows up (``CONNECTOR_DISPATCH_TIMEOUT``) and fail that
generation the same retryable way a room/ChatMaster misconfiguration does,
and stay cancellable so Stop/Pause (which cancel whatever coroutine
``core/task_runtime.register_local_proxy_task`` tracks for this task — this
loop included, since ``dispatch_task_event`` wraps ``handle_task_created``
itself, not ``_run_proxy_loop`` specifically) settle correctly without any
executor-specific code in the stop/pause path itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import connector_broker, connector_compat, connector_reverse
from app.core.events import emit_event
from app.core.task_runtime import is_run_current
from app.models.agent import Agent
from app.models.connector import Connector, ConnectorOutbox
from app.models.room_variable import RoomVariable
from app.models.session.base import Session
from app.models.task import Task
from app.websockets.manager import manager

logger = logging.getLogger(__name__)


def _minutes_env(name: str, default: float) -> timedelta:
    try:
        return timedelta(minutes=float(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("connector_session: invalid %s — using default %s min", name, default)
        return timedelta(minutes=default)


def _dispatch_timeout() -> timedelta:
    """How long a dispatch frame may sit unacked before the task fails retryably.

    A connector that is offline at the moment of dispatch is not considered stalled until
    CONNECTOR_DISPATCH_TIMEOUT (15 minutes by default). This is deliberately independent of the
    watchdog's SESSION_STALL_MINUTES/SESSION_KILL_MINUTES — a connector
    session is excluded from that generic check (core/watchdog.py) precisely
    because *this* is its own timeout.
    """
    return _minutes_env("CONNECTOR_DISPATCH_TIMEOUT_MINUTES", 15)


def _poll_interval_seconds() -> float:
    try:
        return float(os.getenv("CONNECTOR_DISPATCH_POLL_SECONDS", "5"))
    except ValueError:
        return 5.0


async def resolve_executor(agent: Optional[Agent], db: AsyncSession) -> str:
    """Which loop should run this task's session: "local" or "connector".

    An agent without a linked connector stays "local" even if ``executor``
    was somehow set to "connector" — the grant/offline checks downstream only
    make sense once there is an actual connector to dispatch to.
    """
    if agent is not None and agent.executor == "connector" and agent.connector_id is not None:
        return "connector"
    return "local"


@dataclass(frozen=True)
class DispatchContext:
    """Everything a connector needs to start one task generation."""

    task_id: UUID
    room_id: UUID
    goal_id: Optional[UUID]
    plan_id: Optional[UUID]
    mode: str
    task_data_text: str
    history: list[dict[str, Any]]
    room_variables: dict[str, str]
    # mode=worker only — the connector has no git access of its own
    # (nothing but an outbound WS to us), so this is the tar
    # snapshot of the task branch's tip it works against locally, base64'd to
    # travel through the same durable outbox as everything else. None for
    # mode=operator, and for mode=worker only when the archive could not be
    # built (no repo, workspaced down) — ``_run_connector_loop`` fails the
    # dispatch rather than send a worker task nothing to work from.
    task_branch: Optional[str] = None
    workspace_archive_b64: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": str(self.task_id),
            "room_id": str(self.room_id),
            "goal_id": str(self.goal_id) if self.goal_id else None,
            "plan_id": str(self.plan_id) if self.plan_id else None,
            "mode": self.mode,
            "task_data_text": self.task_data_text,
            "history": self.history,
            "room_variables": self.room_variables,
        }
        if self.mode == "worker":
            payload["task_branch"] = self.task_branch
            payload["workspace_archive_b64"] = self.workspace_archive_b64
        return payload


async def _room_variables(room_id: UUID, db: AsyncSession) -> dict[str, str]:
    """Non-secret Room variables (ROOM_SECRETS_REFERENCE.md §1) — always sent.

    Only the shared (environment-agnostic) values: mode=operator never
    selects a Room environment, so a variable scoped to one has nothing to
    resolve against here.
    """
    rows = (
        await db.execute(
            select(RoomVariable).where(
                RoomVariable.room_id == room_id,
                RoomVariable.materialize_to_env.is_(True),
                RoomVariable.environment.is_(None),
            )
        )
    ).scalars().all()
    return {row.name: row.value for row in rows}


_OPERATOR_PREAMBLE = (
    "[Mode: operator. You are working on YOUR OWN machine through your own "
    "CLI agent — you have no access to this Room's repository and do not need to commit "
    "to it (and there is no point — it simply is not there). The task describes "
    "a change on your server: deploy it, restart the needed service and "
    "check the result.\n"
    "When done, return a final report to the human/ChatMaster that must "
    "include:\n"
    "1) what was changed (files/configs/version),\n"
    "2) what was restarted (service/process),\n"
    "3) the healthcheck result (response code, status, or an explicit error if "
    "the check did not pass).\n"
    "If something failed — do not invent success: describe what broke and "
    "finish with success=false.]"
)


_DIRECT_PREAMBLE = (
    "[Mode: operator, direct chat. The owner of this machine is talking to you "
    "from the Legion chat; there is no ChatMaster and no review. Answer the "
    "message below; if it asks for work, do it here on your own machine. Your "
    "final summary is shown in the chat as your reply, so write it for the "
    "owner. If something failed, say so plainly and finish with success=false.]"
)


def _worker_archive_max_bytes() -> int:
    try:
        return int(os.getenv("CONNECTOR_WORKER_ARCHIVE_MAX_BYTES", str(5_000_000)))
    except ValueError:
        return 5_000_000


_WORKER_PREAMBLE = (
    "[Mode: worker. You are working in a local copy of the Room's working tree "
    "(without git history — just files) on your own machine. When you finish, your "
    "edits will be captured as a diff and applied to the real task branch in "
    "the Room, then the usual Gate 1 (checks, risk classification, "
    "review), as for local tasks. Edit only the files relevant to the task "
    "in the current directory; you do not need to commit yourself.]"
)


async def _build_dispatch_context(
    task: Task,
    agent: Agent,
    agent_session: Session,
    room_id: UUID,
    task_data_text: str,
    agent_history_seed: Optional[list[dict[str, Any]]],
    db: AsyncSession,
) -> DispatchContext:
    mode = str((agent.llm_config or {}).get("connector_mode") or "operator")
    room_variables = await _room_variables(room_id, db)
    wrapped_task_data_text = task_data_text
    task_branch: Optional[str] = None
    workspace_archive_b64: Optional[str] = None
    from app.models.task import TASK_ORIGIN_DIRECT_CONNECTOR

    if mode == "operator" and getattr(task, "origin", None) == TASK_ORIGIN_DIRECT_CONNECTOR:
        # The connector reads only task_data_text, so the earlier chat travels
        # inside it (core/connector_direct.py).
        from app.core.connector_direct import conversation_context

        earlier = await conversation_context(task, db)
        parts = [_DIRECT_PREAMBLE]
        if earlier:
            parts.append(f"Earlier in this chat:\n\n{earlier}")
        parts.append(f"New message from the owner:\n\n{task.description or task_data_text}")
        wrapped_task_data_text = "\n\n".join(parts)
    elif mode == "operator":
        # The remote CLI agent has no other way to
        # learn it must not touch the Room's repository and must end with a
        # structured report — the local loop's equivalent framing lives in
        # its system prompt, which connector-executed sessions don't get
        # (connector_reverse.py's CM review has no workspace/MCP tools
        # either, by the same non-goal). This is the one place to say it.
        wrapped_task_data_text = f"{_OPERATOR_PREAMBLE}\n\n{task_data_text}"
    elif mode == "worker":
        wrapped_task_data_text = f"{_WORKER_PREAMBLE}\n\n{task_data_text}"
        task_branch = getattr(task, "branch_name", None)
        if task_branch:
            from app.core import workspaced_client as wsd

            archive_bytes = await wsd.archive_workspace_branch(str(room_id), task_branch)
            if archive_bytes is not None and len(archive_bytes) <= _worker_archive_max_bytes():
                import base64

                workspace_archive_b64 = base64.b64encode(archive_bytes).decode("ascii")
    return DispatchContext(
        task_id=task.id,
        room_id=room_id,
        goal_id=getattr(agent_session, "goal_id", None),
        plan_id=getattr(agent_session, "plan_id", None),
        mode=mode,
        task_data_text=wrapped_task_data_text,
        history=agent_history_seed or [],
        room_variables=room_variables,
        task_branch=task_branch,
        workspace_archive_b64=workspace_archive_b64,
    )


async def _dispatch_frame_acked(outbox_id: UUID, db: AsyncSession) -> bool:
    acked_at = (
        await db.execute(select(ConnectorOutbox.acked_at).where(ConnectorOutbox.id == outbox_id))
    ).scalar_one_or_none()
    return acked_at is not None


async def _send_stop_frame(
    *, connector_id: UUID, session_id: UUID, run_id: Optional[str], reason: str, db: AsyncSession
) -> None:
    try:
        await connector_broker.enqueue(
            db,
            connector_id=connector_id,
            kind="stop",
            payload={"reason": reason},
            session_id=session_id,
            run_id=run_id,
        )
        await db.commit()
        await manager.send_to_connector(
            {
                "type": "stop",
                "run_id": run_id,
                "session_id": str(session_id),
                "payload": {"reason": reason},
            },
            connector_id,
        )
    except Exception:
        logger.warning(
            "connector_session: failed to enqueue stop frame for connector=%s session=%s",
            connector_id, session_id, exc_info=True,
        )


async def connector_block_reason(db: AsyncSession, agent: Agent) -> Optional[str]:
    """``connector_incompatible`` / ``connector_mode_unsupported`` if the agent's
    connector cannot take this agent's tasks, else ``None``."""
    if agent.connector_id is None:
        return None
    connector = (
        await db.execute(select(Connector).where(Connector.id == agent.connector_id))
    ).scalar_one_or_none()
    if connector is None:
        return None
    return connector_compat.dispatch_block_reason(
        connector.version,
        connector.capabilities,
        connector_compat.agent_mode(agent.llm_config),
    )


async def _run_connector_loop(
    agent_session: Session,
    cm_session: Optional[Session],
    task: Task,
    task_data_text: str,
    room_id: UUID,
    db: AsyncSession,
    agent_history_seed: Optional[list[dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    pending_dispatch: Optional[dict[str, Any]] = None,
) -> Optional[bool]:
    """Remote executor counterpart of ``proxy_session._run_proxy_loop``.

    Same call signature on purpose: the caller settles the run via the
    shared ``finalize_run``/rework-redispatch code after this returns or is
    cancelled exactly as it would after the local loop. ``cm_session`` is
    None for a direct room's message (core/connector_direct.py).
    """
    agent = (
        await db.execute(select(Agent).where(Agent.id == task.assigned_agent_id))
    ).scalars().first()
    if agent is None or agent.connector_id is None:
        # handle_task_created's connector_not_granted preflight
        # already gates this before the session pair is even created — this
        # is only a fail-safe against the agent's connector being unlinked
        # between that check and here.
        from app.core.proxy_session import _emit_dispatch_failed

        await _emit_dispatch_failed(task, room_id, "connector_not_granted", db, run_id=run_id)
        return True

    connector_id = agent.connector_id

    agent_session.executor = "connector"
    agent_session.connector_id = connector_id
    agent_session.billing = "external"
    db.add(agent_session)
    if cm_session is not None:
        cm_session.executor = "connector"
        cm_session.connector_id = connector_id
        cm_session.billing = "external"
        db.add(cm_session)
    await db.flush()

    block_reason = await connector_block_reason(db, agent)
    if block_reason is not None:
        # Fail-safe: handle_task_created already refuses this before the
        # session pair exists; this covers the connector reconnecting with a
        # different version between that check and here. Same shape as the
        # connector_offline timeout below — fail retryably, close both rows.
        from app.core.proxy_session import _emit_dispatch_failed
        from app.core.watchdog import close_open_task_sessions

        await _emit_dispatch_failed(task, room_id, block_reason, db, run_id=run_id)
        await close_open_task_sessions(task.id, block_reason, db)
        await db.commit()
        return True

    context = await _build_dispatch_context(
        task, agent, agent_session, room_id, task_data_text, agent_history_seed, db,
    )

    if context.mode == "worker" and not context.workspace_archive_b64:
        # No task branch, no repo, an oversized archive, or workspaced down —
        # a worker dispatch with nothing to work from is worse than failing
        # it retryably. Both session rows were already marked
        # executor=connector/billing=external above, so this closes them the
        # same way the "connector offline" timeout further below does,
        # rather than leaving them open with nothing ever dispatched.
        from app.core.proxy_session import _emit_dispatch_failed
        from app.core.watchdog import close_open_task_sessions

        await _emit_dispatch_failed(task, room_id, "worker_no_archive", db, run_id=run_id)
        await close_open_task_sessions(task.id, "worker_no_archive", db)
        await db.commit()
        return True

    payload = context.to_payload()

    outbox_row = await connector_broker.enqueue(
        db,
        connector_id=connector_id,
        kind="dispatch",
        payload=payload,
        session_id=agent_session.id,
        run_id=run_id,
    )
    await db.commit()

    try:
        await manager.send_to_connector(
            {
                "type": "dispatch",
                "seq": outbox_row.seq,
                "run_id": run_id,
                "session_id": str(agent_session.id),
                "payload": payload,
            },
            connector_id,
        )
    except Exception:
        logger.info(
            "connector_session: live send failed for connector=%s — outbox will "
            "redeliver on reconnect", connector_id,
        )

    logger.info(
        "connector_session: dispatched task %s to connector %s (run_id=%s, seq=%s)",
        task.id, connector_id, run_id, outbox_row.seq,
    )

    # Room-visible audit trail: which connector ran this
    # task, distinct from the local proxy loop's own dispatch, which has no
    # equivalent event today. Audit: Room events carry connector_id.
    await emit_event(
        room_id,
        "CONNECTOR_DISPATCHED",
        f"Task '{task.title}' dispatched to connector (mode={context.mode}).",
        db,
        task_id=task.id,
        metadata={
            "connector_id": str(connector_id),
            "task_id": str(task.id),
            "mode": context.mode,
            "run_id": run_id,
        },
    )

    deadline: Optional[datetime] = datetime.now(timezone.utc) + _dispatch_timeout()
    run_future = connector_reverse.register_run(
        agent_session.id, task, room_id, pending_dispatch if pending_dispatch is not None else {}
    )
    try:
        while True:
            done, _pending = await asyncio.wait({run_future}, timeout=_poll_interval_seconds())
            if run_future in done:
                # core/connector_reverse.py resolved this on `result`/`error` —
                # the generation is over; finalize_run/rework redispatch in the
                # shared caller code take it from here, same as the local loop
                # returning after its own verdict branch.
                return not run_future.result().get("failed", False)

            if not await is_run_current(task.id, run_id, db):
                # Superseded (retry/pause/another generation) — the caller that
                # flipped the fence already closed the sessions, exactly like
                # the local loop's _pair_closed_externally check. A local loop
                # simply stops iterating; the remote agent does not, so tell the
                # connector to stop (Pause fences the run without cancelling
                # this coroutine, so the CancelledError branch below never runs).
                stop_reason = (
                    await db.execute(select(Task.stop_reason).where(Task.id == task.id))
                ).scalar_one_or_none()
                await _send_stop_frame(
                    connector_id=connector_id,
                    session_id=agent_session.id,
                    run_id=run_id,
                    reason=stop_reason or "stop",
                    db=db,
                )
                # finalize_run(task=task) in the caller decides "paused vs
                # stopped" from this object; it still holds the state from
                # before the fence, and would leave a paused task in PAUSING.
                await db.refresh(task)
                return True

            if deadline is not None and await _dispatch_frame_acked(outbox_row.id, db):
                # The connector picked up the dispatch — from here on this
                # loop only waits on run_future (the result/ask_cm/ask_user
                # backchannel, core/connector_reverse.py) without re-timing
                # out a run that is genuinely in progress.
                deadline = None

            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                from app.core.proxy_session import _emit_dispatch_failed
                from app.core.watchdog import close_open_task_sessions

                await _emit_dispatch_failed(task, room_id, "connector_offline", db, run_id=run_id)
                await close_open_task_sessions(task.id, "connector_offline", db)
                await db.commit()
                return True
    except asyncio.CancelledError:
        # Stop/Pause already wrote the durable fence and closed the session
        # rows from the caller's DB session before cancelling this coroutine
        # (core/watchdog.close_open_task_sessions + task_runtime.fence_task_
        # runtime) — the only thing still missing is telling the *remote*
        # side to actually stop, which is connector-specific.
        await _send_stop_frame(
            connector_id=connector_id,
            session_id=agent_session.id,
            run_id=run_id,
            reason=task.stop_reason or "stop",
            db=db,
        )
        raise
    finally:
        connector_reverse.discard_run(agent_session.id)
