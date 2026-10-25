"""Connector reverse channel.

``connector_ws.py`` fences every inbound ``status``/``reasoning``/``ask_cm``/
``ask_user``/``result``/``error`` frame through
``connector_broker.accept_inbound_frame`` but does nothing with an
accepted frame beyond logging it. This module is the missing "what a frame
means" layer, mapping each inbound kind onto the exact same room-visible
primitives a *local* proxy-loop turn already uses (``core/events.py``'s
``emit_status``/``emit_reasoning_event``, ``ProxyBroker.publish``,
``ask_user_registry``, ``core/session_close.py``) so a connector-run agent is
indistinguishable in the chat from a local one: the connector is a transport, not a channel.

``_run_connector_loop`` (``core/connector_session.py``) has no way to learn a
connector-run generation is finished; that is this module's other job.
:func:`register_run` hands it an ``asyncio.Future`` keyed by the *agent-side*
session id (stable for the lifetime of one generation, unlike ``run_id``,
which local test callers are allowed to leave ``None``); :func:`_handle_result`
and :func:`_handle_error` resolve it once a verdict is reached, the same way
``ask_user_registry``'s ``asyncio.Event`` bridges a REST answer back into a
suspended local coroutine.

CM's turn is invoked with the exact same building blocks
``proxy_session._run_proxy_loop`` uses per-turn (``build_memory_layer`` +
``build_system_prompt("cm_agent_proxy_cm")`` + ``_call_llm_with_tools`` +
``_call_with_agent_secret``) rather than a re-implementation, so it shares one
prompt/verdict contract (``close_session``'s ``verdict`` field, with the
``TASK_REJECTED``/``CHANGES_REQUESTED`` markers in ``summary`` as the fallback —
see ``proxy_session.classify_review_verdict``) with the local loop. It deliberately does not pass the
local loop's workspace/MCP tool sets: mode=operator (the only mode this stage
supports) never touches the Room's own repository, so
the CM has nothing there to inspect.

Secret masking is wired at every point connector-supplied text reaches
chat via ``room_masking.mask_room_secrets`` — but today's frame protocol
gives the connector no way to *request* a Room secret by name, so there
is nothing yet to resolve into the mask map. Proactively resolving every Room
secret just in case would trip environment-lock gates and audit logs for
secrets nobody asked for, which is worse than not masking. The call sites are
in place; wiring in a real name→value map is a one-line change once a
secret-request frame kind exists.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import ask_user_registry
from app.core import connector_broker
from app.core.events import AgentStatus, emit_event, emit_reasoning_event, emit_status
from app.core.proxy_broker import ProxyBroker, ProxyBrokerError
from app.core.secrets.room_masking import mask_room_secrets
from app.models.agent import Agent
from app.models.session.cm_agent_proxy import CmAgentProxySession
from app.models.task import DispatchStatus, Task, TaskState
from app.schemas.proxy import ProxyMessageCreate
from app.websockets.manager import manager

logger = logging.getLogger(__name__)

_ASK_USER_TIMEOUT_SECONDS = int(os.getenv("CONNECTOR_ASK_USER_TIMEOUT_SECONDS", "600"))
_CM_ANSWER_MAX_TOOL_CALLS = int(os.getenv("PROXY_CM_ANSWER_MAX_TOOL_CALLS", "20"))
_CM_REVIEW_MAX_TOOL_CALLS = int(os.getenv("CONNECTOR_CM_REVIEW_MAX_TOOL_CALLS", "3"))


def _mask(text: Optional[str]) -> str:
    """Best-effort mask of any Room secret whose value is already known.

    ``secrets={}`` today (see module docstring) — kept as the single call site
    every connector-supplied string flows through so plugging in a real
    name→value map later touches one function, not every handler.
    """
    return mask_room_secrets(text or "", {}) or ""


# ---------------------------------------------------------------------------
# Run-completion bridge between connector_session._run_connector_loop and the
# result/error handlers below — the loop has nothing else to wait on.
# ---------------------------------------------------------------------------


@dataclass
class _RunEntry:
    task: Task
    room_id: UUID
    pending_dispatch: dict[str, Any]
    future: "asyncio.Future[dict[str, Any]]"


_RUNS: dict[UUID, _RunEntry] = {}


def register_run(
    agent_session_id: UUID, task: Task, room_id: UUID, pending_dispatch: dict[str, Any]
) -> "asyncio.Future[dict[str, Any]]":
    """Called once by ``_run_connector_loop`` right after the dispatch frame is sent."""
    future: "asyncio.Future[dict[str, Any]]" = asyncio.get_running_loop().create_future()
    _RUNS[agent_session_id] = _RunEntry(
        task=task, room_id=room_id, pending_dispatch=pending_dispatch, future=future
    )
    return future


def discard_run(agent_session_id: UUID) -> None:
    """Called by ``_run_connector_loop`` in a ``finally`` on every exit path."""
    _RUNS.pop(agent_session_id, None)


async def _send_down(
    connector_id: UUID, *, kind: str, session_id: UUID, run_id: Optional[str], content: str, db: AsyncSession
) -> None:
    """Durable-enqueue + best-effort live push, mirroring connector_session._send_stop_frame."""
    try:
        row = await connector_broker.enqueue(
            db,
            connector_id=connector_id,
            kind=kind,
            payload={"content": content},
            session_id=session_id,
            run_id=run_id,
        )
        await db.commit()
        await manager.send_to_connector(
            {
                "type": kind,
                "seq": row.seq,
                "run_id": run_id,
                "session_id": str(session_id),
                "payload": {"content": content},
            },
            connector_id,
        )
    except Exception:
        logger.warning(
            "connector_reverse: failed to send %s frame down to connector=%s",
            kind, connector_id, exc_info=True,
        )


async def _cm_turn(
    cm_session: CmAgentProxySession,
    task: Task,
    room_id: UUID,
    history: list[dict[str, Any]],
    db: AsyncSession,
    *,
    on_exhaust: str,
    max_tool_calls: int,
    run_id: Optional[str],
) -> tuple[Optional[str], Optional[dict[str, Any]], int, int, str, int, int]:
    """One CM LLM turn, reusing exactly the building blocks the local loop calls per-turn.

    No workspace/MCP tools are offered (see module docstring) — the only tool
    available to the model is the always-present ``close_session``.
    """
    from app.core.memory_layer import build_memory_layer, build_system_prompt
    from app.core.proxy_session import _call_llm_with_tools, _call_with_agent_secret, _get_agent_llm_config

    cm_llm_config = await _get_agent_llm_config(cm_session.agent_id, db)
    cm_memory = await build_memory_layer(cm_session, db)
    cm_sys = build_system_prompt(cm_memory, "cm_agent_proxy_cm")
    cm_row = (await db.execute(select(Agent).where(Agent.id == cm_session.agent_id))).scalars().first()
    cm_label = cm_row.nickname if cm_row else "ChatMaster"

    async def _factory(secret_value):
        return await _call_llm_with_tools(
            cm_llm_config,
            cm_sys,
            history,
            room_id,
            cm_session.id,
            task.id,
            db,
            max_tool_calls=max_tool_calls,
            actor_agent_id=cm_session.agent_id,
            actor_label=cm_label,
            on_exhaust=on_exhaust,
            context_role="chat_master",
            secret_value=secret_value,
            run_id=run_id,
        )

    return await _call_with_agent_secret(
        owner_id=cm_row.owner_id if cm_row else None,
        agent_id=cm_session.agent_id,
        db=db,
        request_id=task.id,
        config=cm_llm_config,
        call_factory=_factory,
    )


# ---------------------------------------------------------------------------
# Entry point called from connector_ws.py after fencing accepted the frame.
# ---------------------------------------------------------------------------


async def handle_inbound_frame(
    db: AsyncSession,
    *,
    frame_type: str,
    payload: dict[str, Any],
    session_id: Optional[UUID],
    run_id: Optional[str],
    task_id: Optional[UUID],
    connector_id: UUID,
) -> None:
    if session_id is None:
        return
    agent_session = await db.get(CmAgentProxySession, session_id)
    if agent_session is None:
        logger.warning(
            "connector_reverse: unknown session_id=%s for frame_type=%s", session_id, frame_type
        )
        return

    cm_session: Optional[CmAgentProxySession] = None
    if agent_session.peer_session_id is not None:
        cm_session = await db.get(CmAgentProxySession, agent_session.peer_session_id)

    task: Optional[Task] = None
    effective_task_id = task_id or getattr(agent_session, "task_id", None)
    if effective_task_id is not None:
        task = await db.get(Task, effective_task_id)

    room_id = agent_session.room_id

    if frame_type == "status":
        await _handle_status(agent_session, payload)
    elif frame_type == "reasoning":
        await _handle_reasoning(db, agent_session, payload)
    elif frame_type == "ask_cm":
        await _handle_ask_cm(db, agent_session, cm_session, task, room_id, connector_id, run_id, payload)
    elif frame_type == "ask_user":
        await _handle_ask_user(agent_session, room_id, connector_id, run_id, payload)
    elif frame_type == "result":
        await _handle_result(db, agent_session, cm_session, task, room_id, run_id, payload, connector_id)
    elif frame_type == "error":
        await _handle_error(db, agent_session, task, room_id, run_id, payload, connector_id)
    else:
        logger.debug("connector_reverse: no handler for frame_type=%s — ignoring", frame_type)


# ---------------------------------------------------------------------------
# status / reasoning — pure passthrough onto the existing live-status channel.
# ---------------------------------------------------------------------------


async def _handle_status(agent_session: CmAgentProxySession, payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if not status:
        return
    detail = payload.get("detail")
    extra: Optional[dict[str, Any]] = None
    if isinstance(detail, dict):
        extra = {k: (_mask(v) if isinstance(v, str) else v) for k, v in detail.items()}
    await emit_status(
        agent_session.room_id,
        str(status),
        agent_id=agent_session.agent_id,
        session_id=agent_session.id,
        task_id=getattr(agent_session, "task_id", None),
        extra={**extra, "side": "agent"} if extra else {"side": "agent"},
    )


async def _handle_reasoning(db: AsyncSession, agent_session: CmAgentProxySession, payload: dict[str, Any]) -> None:
    content = _mask(payload.get("content"))
    if not content.strip():
        return
    agent_row = (
        await db.execute(select(Agent).where(Agent.id == agent_session.agent_id))
    ).scalars().first()
    await emit_reasoning_event(
        room_id=agent_session.room_id,
        session_id=agent_session.id,
        agent_id=agent_session.agent_id,
        actor_label=agent_row.nickname if agent_row else None,
        task_id=getattr(agent_session, "task_id", None),
        thinking=content,
        reasoning_tokens=0,
        duration_ms=0,
        db=db,
    )
    await db.commit()


# ---------------------------------------------------------------------------
# ask_cm — publish visibly (PROXY_MESSAGE, watchdog-visible) + a real CM turn.
# ---------------------------------------------------------------------------


async def _handle_ask_cm(
    db: AsyncSession,
    agent_session: CmAgentProxySession,
    cm_session: Optional[CmAgentProxySession],
    task: Optional[Task],
    room_id: UUID,
    connector_id: UUID,
    run_id: Optional[str],
    payload: dict[str, Any],
) -> None:
    if cm_session is None or task is None:
        logger.warning(
            "connector_reverse: ask_cm with no CM peer/task for session=%s", agent_session.id
        )
        return
    content = _mask(payload.get("content")).strip()
    if not content:
        return

    from app.core.proxy_session import _MID_TASK_NOTE, _get_history, _save_history

    broker = ProxyBroker()
    try:
        await broker.publish(
            ProxyMessageCreate(
                room_id=room_id,
                sender_type="agent",
                sender_id=agent_session.agent_id,
                content=content,
                task_id=task.id,
                session_id=agent_session.id,
                peer_session_id=cm_session.id,
                kind="question",
            ),
            db,
        )
    except ProxyBrokerError as exc:
        logger.warning("connector_reverse: publish ask_cm failed: %s", exc)
    await db.commit()

    if not cm_session.is_open:
        # CM already closed (e.g. a stale question after the task ended) —
        # visible in chat via the publish above, nothing more to do.
        return

    cm_history = _get_history(cm_session)
    cm_history.append({"role": "user", "content": _MID_TASK_NOTE + content})

    await emit_status(
        room_id, AgentStatus.LLM_REQUEST_SENT,
        agent_id=cm_session.agent_id, session_id=cm_session.id,
        task_id=task.id, extra={"side": "chat_master"},
    )
    cm_text: Optional[str] = None
    cm_tool: Optional[dict[str, Any]] = None
    try:
        cm_text, cm_tool, *_rest = await _cm_turn(
            cm_session, task, room_id, cm_history, db,
            on_exhaust="answer", max_tool_calls=_CM_ANSWER_MAX_TOOL_CALLS, run_id=run_id,
        )
    except Exception:
        logger.exception("connector_reverse: CM answer turn failed for task %s", task.id)
    await emit_status(
        room_id, AgentStatus.LLM_RESPONSE_RECEIVED,
        agent_id=cm_session.agent_id, session_id=cm_session.id,
        task_id=task.id, extra={"side": "chat_master"},
    )

    # A close_session call here would end a session that is still working —
    # same rule the local loop applies (_CM_CLOSE_CONVERTED_NOTE): treat it as
    # the answer text, never as a verdict.
    reply = ((cm_tool or {}).get("summary") or cm_text or "").strip()
    if not reply:
        reply = "ChatMaster could not formulate a reply. Act at your own discretion."
    cm_history.append({"role": "assistant", "content": reply})
    _save_history(cm_session, cm_history)
    db.add(cm_session)
    await db.flush()

    reply_masked = _mask(reply)
    try:
        await broker.publish(
            ProxyMessageCreate(
                room_id=room_id,
                sender_type="chat_master",
                sender_id=cm_session.agent_id,
                content=reply_masked,
                task_id=task.id,
                session_id=cm_session.id,
                peer_session_id=agent_session.id,
                kind="answer",
            ),
            db,
        )
    except ProxyBrokerError as exc:
        logger.warning("connector_reverse: publish CM answer failed: %s", exc)
    await db.commit()

    await _send_down(
        connector_id, kind="proxy_message", session_id=agent_session.id,
        run_id=run_id, content=reply_masked, db=db,
    )


# ---------------------------------------------------------------------------
# ask_user — same registry/event a local ask_user tool call uses; the answer
# has no local coroutine to wake, so a background task relays it downward.
# ---------------------------------------------------------------------------


async def _handle_ask_user(
    agent_session: CmAgentProxySession,
    room_id: UUID,
    connector_id: UUID,
    run_id: Optional[str],
    payload: dict[str, Any],
) -> None:
    question_id = str(payload.get("question_id") or uuid4())
    content = _mask(payload.get("content")).strip() or "The agent is asking a question"
    raw_options = payload.get("options") or []
    options: list[dict[str, str]] = []
    for opt in raw_options:
        if isinstance(opt, dict):
            label = str(opt.get("label") or opt.get("id") or "").strip()
        else:
            label = str(opt).strip()
        if label:
            options.append({"label": _mask(label), "description": ""})
    while len(options) < 2:
        # AGENT_QUESTION's card renders 2-4 options; pad rather than reject a
        # connector's single-option/free-text-only question.
        options.append({"label": "Continue", "description": "Free-form answer / agreement"})
    questions = [{"question": content, "header": "Question", "options": options}]

    from app.db.session import engine as _engine
    from sqlalchemy.orm import sessionmaker as _sessionmaker

    session_factory = _sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        try:
            await emit_event(
                room_id=room_id,
                event_type="AGENT_QUESTION",
                content=content,
                db=db,
                task_id=getattr(agent_session, "task_id", None),
                metadata={
                    "question_id": question_id,
                    "session_id": str(agent_session.id),
                    "questions": questions,
                    "timeout_seconds": _ASK_USER_TIMEOUT_SECONDS,
                },
            )
        except Exception:
            logger.exception("connector_reverse: failed to emit AGENT_QUESTION")
            return

    pq = ask_user_registry.register(
        question_id, agent_session.id, room_id, {"questions": questions}
    )
    # Captured now, not read off `agent_session` after the wait below: by then
    # the ORM object's owning AsyncSession (from the outer handle_inbound_frame
    # call) is long closed, and a lazy attribute touch outside that session's
    # greenlet context raises `MissingGreenlet` — confirmed live, which silently dropped every `answer` frame (the task in
    # `asyncio.create_task` below fails with an unawaited exception, so
    # nothing ever reached the connector and the ask_user card just sat there
    # answered-but-not-relayed).
    agent_session_id = agent_session.id

    async def _await_and_relay() -> None:
        try:
            await asyncio.wait_for(pq.event.wait(), timeout=float(_ASK_USER_TIMEOUT_SECONDS))
        except asyncio.TimeoutError:
            ask_user_registry.expire(question_id)
            content_down = "The user did not respond in time. Act at your own discretion."
        else:
            answer = pq.answer or {}
            selected = [str(s) for s in (answer.get("selected_options") or [])]
            free_text = answer.get("free_text")
            parts = list(selected)
            if free_text:
                parts.append(str(free_text))
            content_down = "; ".join(parts) or "The user confirmed without clarifications."

        async with session_factory() as relay_db:
            await _send_down(
                connector_id, kind="answer", session_id=agent_session_id,
                run_id=run_id, content=content_down, db=relay_db,
            )

    asyncio.create_task(_await_and_relay())


# ---------------------------------------------------------------------------
# result / error — end the generation. See connector_session._run_connector_
# loop, which is awaiting the future this resolves.
# ---------------------------------------------------------------------------


async def _handle_result(
    db: AsyncSession,
    agent_session: CmAgentProxySession,
    cm_session: Optional[CmAgentProxySession],
    task: Optional[Task],
    room_id: UUID,
    run_id: Optional[str],
    payload: dict[str, Any],
    connector_id: UUID,
) -> None:
    if task is None or cm_session is None:
        logger.warning(
            "connector_reverse: result with no task/CM peer for session=%s", agent_session.id
        )
        return

    # Room-visible audit trail: the result frame arrived, tagged
    # with which connector sent it, independent of the CM's later verdict
    # events (TASK_APPROVED/REJECTED/CHANGES_REQUESTED) below.
    await emit_event(
        room_id,
        "CONNECTOR_RESULT",
        f"Connector reported result for task '{task.title}'.",
        db,
        task_id=task.id,
        metadata={
            "connector_id": str(connector_id),
            "task_id": str(task.id),
            "success": payload.get("success"),
            "run_id": run_id,
        },
    )

    from app.core.proxy_session import (
        MAX_TASK_REWORKS,
        VERDICT_CHANGES,
        VERDICT_REJECT,
        classify_review_verdict,
        _close_side,
        _emit_closed,
        _render_close_note,
        _get_history,
    )
    from app.core.session_close import SessionCloseActions

    run = _RUNS.get(agent_session.id)

    summary_raw = str(payload.get("summary") or payload.get("report") or "").strip()
    summary = _mask(summary_raw) or "[The operator finished the work without details]"
    success = payload.get("success")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}

    # mode=worker's own half of the "patch through the WS
    # transport" design — the
    # connector has no git access of its own, so it diffed its local copy
    # of the task branch's archived tree (core/connector_session.py's
    # `_build_dispatch_context`) and sent that diff back here. Applying it
    # now, before the CM even looks at the result, means everything after
    # this block — CM review, approve/reject/changes, merge — is exactly
    # the same code a local task's verdict already runs.
    mode = "operator"
    agent_row_for_mode = (
        await db.execute(select(Agent).where(Agent.id == task.assigned_agent_id))
    ).scalars().first()
    if agent_row_for_mode is not None:
        mode = str((agent_row_for_mode.llm_config or {}).get("connector_mode") or "operator")

    worker_conflict = False
    if mode == "worker":
        patch_text = str(payload.get("patch") or "").strip()
        if patch_text and task.branch_name:
            from app.core import workspaced_client as wsd

            apply_result = await wsd.apply_patch_workspace(
                str(room_id), task.branch_name, patch_text, message=f"connector: {task.title}",
            )
            if not apply_result.get("ok"):
                worker_conflict = True
                await emit_event(
                    room_id,
                    "GIT_CONFLICT",
                    f"Could not apply the connector's changes for task '{task.title}' "
                    "— the task branch diverged from what the connector worked on.",
                    db,
                    task_id=task.id,
                    metadata={
                        "task_id": str(task.id),
                        "branch": task.branch_name,
                        "stderr": str(apply_result.get("stderr") or "")[:2000],
                    },
                )

    broker = ProxyBroker()
    if agent_session.is_open:
        # A connector agent has no close_session tool call to parse — feed the
        # connector's own report through the same tool_input shape so
        # flush_session_memory/_render_close_note see a real summary.
        agent_actions = await _close_side(agent_session, {"summary": summary}, "task_done", db)
        await _emit_closed(agent_session, "task_done", agent_actions, db)
        try:
            await broker.publish(
                ProxyMessageCreate(
                    room_id=room_id,
                    sender_type="agent",
                    sender_id=agent_session.agent_id,
                    content=_render_close_note(agent_actions),
                    task_id=task.id,
                    session_id=agent_session.id,
                    peer_session_id=cm_session.id,
                ),
                db,
            )
        except ProxyBrokerError as exc:
            logger.warning("connector_reverse: publish connector result failed: %s", exc)
    await db.commit()

    if worker_conflict:
        # The diff no longer applies cleanly — nothing coherent for the CM to
        # review. Leave the branch alone and the task in WAITING_REVIEW, same
        # outcome as a local task's merge conflict (proxy_session's verdict
        # path) or a divergent branch elsewhere: a human resolves it through
        # the existing "Divergence review" flow rather than either side
        # guessing at a resolution.
        task.state = TaskState.WAITING_REVIEW
        db.add(task)
        if cm_session.is_open:
            cm_actions = SessionCloseActions(
                summary="The connector sent changes, but they did not apply to the task branch — manual review is needed."
            )
            await _close_side(cm_session, None, "error", db)
            await _emit_closed(cm_session, "error", cm_actions, db)
        await db.commit()
        if run is not None:
            _RUNS.pop(agent_session.id, None)
            run.future.set_result({"failed": True})
        return

    if not cm_session.is_open:
        if run is not None:
            _RUNS.pop(agent_session.id, None)
            run.future.set_result({"failed": False})
        return

    notification_parts = [
        f"[External connector (mode={mode}) finished the work and sent a report.]",
        summary,
    ]
    healthcheck = artifacts.get("healthcheck")
    if healthcheck is not None:
        notification_parts.append(f"Healthcheck: {_mask(str(healthcheck))}")
    if success is False:
        notification_parts.append(
            "[SYSTEM: the connector reported success=false — take this into account when giving the verdict.]"
        )
    if mode == "worker":
        # The CM turn here gets no workspace/MCP tools (module docstring) —
        # unlike a local task's review, it cannot `git diff` the branch
        # itself even though the connector's patch is already committed onto
        # it by this point. The diff text is the only grounding it gets.
        patch_preview = _mask(str(payload.get("patch") or "")).strip()
        if patch_preview:
            notification_parts.append(
                "Diff, applied onto the task branch:\n" + patch_preview[:8000]
            )
        notification_parts.append(
            "[Review the result and call close_session to approve the task, request "
            "changes by including CHANGES_REQUESTED in the summary with concrete "
            "feedback, or reject it by including TASK_REJECTED in the summary. The "
            "connector already committed this diff onto the task branch — approving "
            "folds it into the Room's repository like any local task's work "
            "(mode=worker).]"
        )
    else:
        notification_parts.append(
            "[Review the result and call close_session to approve the task, request "
            "changes by including CHANGES_REQUESTED in the summary with concrete "
            "feedback, or reject it by including TASK_REJECTED in the summary. There "
            "is no Room repository to inspect for this task — it ran on the "
            "connector's own machine (mode=operator).]"
        )
    cm_history = _get_history(cm_session)
    cm_history.append({"role": "user", "content": "\n\n".join(notification_parts)})

    cm_text: Optional[str] = None
    cm_tool: Optional[dict[str, Any]] = None
    try:
        cm_text, cm_tool, *_rest = await _cm_turn(
            cm_session, task, room_id, cm_history, db,
            on_exhaust="force_close", max_tool_calls=_CM_REVIEW_MAX_TOOL_CALLS, run_id=run_id,
        )
    except Exception:
        logger.exception("connector_reverse: CM review turn failed for task %s", task.id)

    if cm_tool is None:
        # Even a forced close produced nothing usable — fail safe to a human
        # rather than silently guessing a verdict.
        cm_actions = SessionCloseActions(summary=cm_text or "ChatMaster could not reach a decision.")
        await _close_side(cm_session, None, "error", db)
        await _emit_closed(cm_session, "error", cm_actions, db)
        task.state = TaskState.WAITING_REVIEW
        db.add(task)
        await db.commit()
        if run is not None:
            _RUNS.pop(agent_session.id, None)
            run.future.set_result({"failed": True})
        return

    verdict = classify_review_verdict(cm_tool)
    is_rejected = verdict == VERDICT_REJECT
    is_changes = verdict == VERDICT_CHANGES
    if is_changes and task.rework_count >= MAX_TASK_REWORKS:
        is_changes = False
        is_rejected = True

    cm_close_reason = "changes_requested" if is_changes else "task_done"
    cm_actions = await _close_side(cm_session, cm_tool, cm_close_reason, db)
    await _emit_closed(cm_session, cm_close_reason, cm_actions, db)

    try:
        await broker.publish(
            ProxyMessageCreate(
                room_id=room_id,
                sender_type="chat_master",
                sender_id=cm_session.agent_id,
                content=_render_close_note(cm_actions),
                task_id=task.id,
                session_id=cm_session.id,
                peer_session_id=agent_session.id,
                kind="reject" if is_rejected else ("request_changes" if is_changes else "approve"),
            ),
            db,
        )
    except ProxyBrokerError as exc:
        logger.warning("connector_reverse: publish CM verdict failed: %s", exc)

    if is_changes:
        task.rework_count += 1
        task.state = TaskState.ASSIGNED
        task.dispatch_status = DispatchStatus.READY
        db.add(task)
        await db.flush()
        await emit_event(
            room_id=room_id,
            event_type="TASK_CHANGES_REQUESTED",
            content=(
                f"Task '{task.title}' returned for rework "
                f"(attempt {task.rework_count} of {MAX_TASK_REWORKS})."
            ),
            db=db,
            metadata={
                "task_id": str(task.id),
                "task_title": task.title,
                "cm_session_id": str(cm_session.id),
                "rework_count": task.rework_count,
                "max_reworks": MAX_TASK_REWORKS,
                "feedback": cm_actions.summary,
            },
        )
        if run is not None:
            run.pending_dispatch.update(
                {"resume": True, "rework_feedback": cm_actions.summary or "", "source": "rework"}
            )
    elif is_rejected:
        task.state = TaskState.REJECTED
        db.add(task)
        await db.flush()
        await emit_event(
            room_id=room_id,
            event_type="TASK_REJECTED",
            content=f"Task '{task.title}' rejected by ChatMaster. Reason: {cm_actions.summary}",
            db=db,
            metadata={
                "task_id": str(task.id),
                "cm_session_id": str(cm_session.id),
                "reason": cm_actions.summary,
            },
        )
        from app.core.task_deps import get_blocked_dependents

        for blocked_task in await get_blocked_dependents(task.id, db):
            await emit_event(
                room_id=room_id,
                event_type="TASK_DEPENDENCY_FAILED",
                content=(
                    f"Task '{blocked_task.title}' is blocked: "
                    f"predecessor '{task.title}' was rejected."
                ),
                db=db,
                metadata={
                    "task_id": str(blocked_task.id),
                    "blocked_because": str(task.id),
                    "failed_status": "rejected",
                },
            )
    else:
        # Approve. mode=operator never touches the Room's own repository
        # — no merge_task_branch there.
        # mode=worker already committed the connector's diff onto the task
        # branch above (the apply_patch_workspace call) — from here it is
        # exactly proxy_session's local verdict path: fold the branch into
        # its base, same conflict handling (a failed merge here is a race
        # against something else touching the base between the apply and
        # now, not the same conflict as `worker_conflict` above).
        if mode == "worker":
            from app.core.room_git_flow import merge_task_branch

            try:
                merge_result = await merge_task_branch(str(room_id), task, db)
            except Exception:
                logger.exception("connector_reverse: task branch merge failed for task %s", task.id)
                merge_result = {"ok": False}
            if not merge_result.get("ok"):
                task.state = TaskState.WAITING_REVIEW
                db.add(task)
                await db.commit()
                if run is not None:
                    _RUNS.pop(agent_session.id, None)
                    run.future.set_result({"failed": True})
                return
            task.merge_commit = merge_result.get("sha")

        task.state = TaskState.COMPLETED
        db.add(task)
        await db.flush()
        await emit_event(
            room_id=room_id,
            event_type="TASK_APPROVED",
            content=f"Task '{task.title}' approved by ChatMaster.",
            db=db,
            metadata={"task_id": str(task.id), "cm_session_id": str(cm_session.id)},
        )
        if task.goal_id:
            from app.core.goal_progress import maybe_complete_goal
            from app.core.session_close import maybe_refresh_room_memory

            await maybe_complete_goal(task.goal_id, room_id, db)
            await maybe_refresh_room_memory(room_id, db, trigger="task_approved")
        from app.core.task_deps import publish_unblocked_tasks, try_unblock_tasks

        newly_ready = await try_unblock_tasks(task.id, db)
        await publish_unblocked_tasks(
            newly_ready, room_id, db,
            source="connector_task_unblock",
            unblocked_by=task.id,
            predecessor_title=task.title,
        )

    await db.commit()

    if run is not None:
        _RUNS.pop(agent_session.id, None)
        run.future.set_result({"failed": is_rejected})


async def _handle_error(
    db: AsyncSession,
    agent_session: CmAgentProxySession,
    task: Optional[Task],
    room_id: UUID,
    run_id: Optional[str],
    payload: dict[str, Any],
    connector_id: UUID,
) -> None:
    code = payload.get("code") or "connector_error"
    message = _mask(str(payload.get("message") or "The connector reported an error."))
    logger.warning(
        "connector_reverse: error frame for task=%s: %s — %s",
        task.id if task else None, code, message,
    )
    if task is not None:
        from app.core.proxy_session import _emit_dispatch_failed
        from app.core.watchdog import close_open_task_sessions

        await emit_event(
            room_id,
            "CONNECTOR_RESULT",
            f"Connector reported an error for task '{task.title}': {message}",
            db,
            task_id=task.id,
            metadata={"connector_id": str(connector_id), "task_id": str(task.id), "code": code, "run_id": run_id},
        )
        await _emit_dispatch_failed(task, room_id, "connector_error", db, run_id=run_id)
        await close_open_task_sessions(task.id, "connector_error", db)
        await db.commit()

    run = _RUNS.pop(agent_session.id, None)
    if run is not None:
        run.future.set_result({"failed": True})


__all__ = [
    "handle_inbound_frame",
    "register_run",
    "discard_run",
]
