from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
import json
import asyncio
import logging

from app.db.session import get_session, engine
from app.db.release import release_db
from app.models.agent import Agent
from app.models.user import User
from app.models.room import Room, RoomUserLink, RoomAgentLink, RoomType
from app.models.message import Message
from app.websockets.manager import manager
from app.api import deps
from jose import jwt, JWTError
from app.core import legion_auth, security, chat_master
from app.core.room_coordinator import get_cm_agent_id as _get_cm_agent_id
from app.core.secrets.runtime import resolve_agent_secret
from app.core.router import RouteDecision, SplitPart, parse_mentions, route_user_message
from app.core.user_sessions import start_user_session, process_user_turn, turn_in_progress
from app.core.path_contract import USERFILES_DIR, normalize_workspace_path, WorkspacePathError
from app.schemas.task_change import TaskChangeActionRequest, TaskChangeProposal

router = APIRouter()

def _make_session():
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


CHAT_MAX_ATTACHMENTS = 20


class InvalidChatAttachments(ValueError):
    """The client supplied an attachment reference that cannot be trusted."""


def _validate_chat_metadata(raw_metadata) -> dict:
    """Return the small, server-owned metadata envelope accepted for a chat.

    Uploads happen through the authenticated Files API first.  The subsequent
    WebSocket message may therefore carry references, but the browser is not
    allowed to choose an arbitrary path or storage scope.  Keep this helper
    deliberately synchronous: it is also used by unit tests and all
    authorization has already happened when the socket is accepted.
    """

    if not isinstance(raw_metadata, dict):
        return {}
    raw_attachments = raw_metadata.get("attachments")
    if raw_attachments is None:
        return {}
    if not isinstance(raw_attachments, list) or not raw_attachments:
        raise InvalidChatAttachments("attachments must be a non-empty list")
    if len(raw_attachments) > CHAT_MAX_ATTACHMENTS:
        raise InvalidChatAttachments(
            f"a message may contain at most {CHAT_MAX_ATTACHMENTS} attachments"
        )

    attachments = []
    private_prefix = f"{USERFILES_DIR}/"
    for item in raw_attachments:
        if not isinstance(item, dict):
            raise InvalidChatAttachments("attachment metadata must be an object")
        try:
            normalized = normalize_workspace_path(item.get("path", ""))
        except (WorkspacePathError, TypeError) as exc:
            raise InvalidChatAttachments(f"invalid attachment path: {exc}") from exc
        if not normalized.startswith(private_prefix):
            raise InvalidChatAttachments(
                "chat attachments must be stored under .userfiles"
            )

        try:
            size = int(item.get("size", 0))
        except (TypeError, ValueError) as exc:
            raise InvalidChatAttachments("attachment size must be an integer") from exc
        if size < 0:
            raise InvalidChatAttachments("attachment size cannot be negative")

        # Derive the display name from the trusted normalized path.  The
        # browser-provided name is intentionally ignored, as is its claimed
        # storage scope.  media_type is presentation metadata only.
        name = normalized.rsplit("/", 1)[-1] or "unnamed"
        media_type = str(item.get("media_type") or "application/octet-stream")[:160]
        attachments.append(
            {
                "path": normalized,
                "name": name,
                "size": size,
                "media_type": media_type,
                "storage_scope": "room_private",
            }
        )
    return {"attachments": attachments}


async def _verify_session_room_membership(
    session,
    user: User,
    db: AsyncSession,
) -> None:
    """Require the authenticated user to belong to the session's room.

    Session IDs are not authorization boundaries by themselves.  Keep this
    check next to the session-action endpoints so confirmation/rejection
    cannot be used to mutate a room by guessing a session ID.
    """

    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == session.room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _verify_room_membership(
    room_id: UUID,
    user: User,
    db: AsyncSession,
) -> None:
    """Require the authenticated user to belong to the room they act on."""

    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _get_session_for_action(
    session_id: UUID,
    db: AsyncSession,
    current_user: User,
    *,
    require_open: bool = True,
):
    """Load and authorize a user session used by a button action."""

    from app.models.session.base import Session

    res = await db.execute(
        select(Session).where(Session.id == session_id).with_for_update()
    )
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await _verify_session_room_membership(session, current_user, db)
    if require_open and not session.is_open:
        raise HTTPException(status_code=400, detail="Session already closed")
    return session


def _task_change_conflict_detail(exc, proposal_id: UUID, current_task=None) -> dict:
    """Return a stable, frontend-readable optimistic-lock conflict payload."""

    actual_state = exc.actual_state
    if actual_state is not None:
        actual_state = getattr(actual_state, "value", str(actual_state))
    actual_updated_at = exc.actual_updated_at
    current = None
    if current_task is not None:
        current = {
            "task_id": str(current_task.id),
            "title": current_task.title,
            "state": getattr(current_task.state, "value", str(current_task.state)),
            "updated_at": current_task.updated_at.isoformat()
            if current_task.updated_at is not None
            else None,
        }
    elif exc.task_id is not None:
        current = {
            "task_id": str(exc.task_id),
            "state": actual_state,
            "updated_at": actual_updated_at.isoformat()
            if actual_updated_at is not None
            else None,
        }
    return {
        "code": "task_change_conflict",
        "message": str(exc),
        "proposal_id": str(proposal_id),
        "task": current,
    }

async def get_client_id(token: str, db: AsyncSession):
    # Check if it's a User token
    try:
        payload = security.decode_access_token(token)
        user_id = payload.get("sub")
        if user_id:
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalars().first()
            if (user and user.is_active
                    and not await legion_auth.session_revoked(db, user.legion_account_id, payload)):
                return UUID(user_id), False
    except (JWTError, Exception):
        pass
    
    # Check if it's an Agent token
    result = await db.execute(select(Agent).where(Agent.token == token))
    agent = result.scalars().first()
    if agent:
        return agent.id, True
    
    return None, None

@router.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: UUID,
    token: str = Query(...),
    db: AsyncSession = Depends(get_session)
):
    await websocket.accept()

    client_id, is_agent = await get_client_id(token, db)
    if not client_id:
        await websocket.close(code=1008)
        return

    # Check if client is member of room
    if is_agent:
        membership = await db.execute(select(RoomAgentLink).where(RoomAgentLink.room_id == room_id, RoomAgentLink.agent_id == client_id))
    else:
        membership = await db.execute(select(RoomUserLink).where(RoomUserLink.room_id == room_id, RoomUserLink.user_id == client_id))

    if not membership.scalars().first():
        await websocket.close(code=1008)
        return

    # internal design notes T8: an open room is billed, so out of hours it
    # does not open (core/room_close.py, core/work_gates.py).
    from datetime import datetime, timezone

    opened_at = datetime.now(timezone.utc)
    if not is_agent:
        from app.core import work_gates
        from app.core.workspace_lifecycle import room_closed_frame

        open_refusal = await work_gates.open_refusal(room_id, db)
        if open_refusal is not None:
            await websocket.send_json(
                room_closed_frame(room_id, open_refusal.reason, open_refusal.text)
            )
            await websocket.close(code=4402)
            return

    # Release the pooled DB connection while the socket idles: without this
    # every open socket pins one connection in "idle in transaction" state,
    # and a reconnect storm (e.g. after a backend restart) exhausts the pool —
    # QueuePool limit reached → 500 on every HTTP endpoint.
    await db.rollback()

    await manager.connect(websocket, room_id, client_id, is_agent)

    # A member having the room open is billed room time, container or not
    # (core/billing_interval.py). The beat lives exactly as long as the socket.
    presence_task = None
    if not is_agent:
        from app.core.billing_interval import presence_beat

        presence_task = asyncio.create_task(presence_beat(room_id, websocket, opened_at))

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            content = message_data.get("content")
            try:
                chat_metadata = _validate_chat_metadata(message_data.get("metadata"))
            except InvalidChatAttachments as exc:
                # Keep the socket usable so a malformed client-side reference
                # does not disconnect the user from the room.  The
                # client_message_id lets the composer mark the retained files
                # as saved instead of deleting them on an uncertain send.
                await websocket.send_json(
                    {
                        "type": "error",
                        "error_code": "invalid_attachment_metadata",
                        "message": str(exc),
                        "client_message_id": message_data.get("client_message_id"),
                    }
                )
                continue

            attachments = chat_metadata.get("attachments", [])
            if attachments and not str(content or "").strip():
                content = "Attached files:\n" + "\n".join(
                    f"- /.{item['path']}" for item in attachments
                )

            if content:
                # Save message to DB
                new_msg = Message(
                    room_id=room_id,
                    content=content,
                    sender_user_id=client_id if not is_agent else None,
                    sender_agent_id=client_id if is_agent else None,
                    event_metadata=chat_metadata,
                )
                db.add(new_msg)
                await db.commit()
                
                # Broadcast message
                broadcast_msg = {
                    "type": "message",
                    "event_type": "MESSAGE",
                    "id": str(new_msg.id),
                    "room_id": str(room_id),
                    "sender_id": str(client_id),
                    "sender_type": "agent" if is_agent else "user",
                    "content": content,
                    "timestamp": new_msg.timestamp.isoformat(),
                    "status": new_msg.status,
                    "task_id": None,
                    "metadata": chat_metadata,
                }
                await manager.broadcast(broadcast_msg, room_id)
                
                # Route message based on room type:
                # - direct rooms: same session-flow as group rooms, minus the
                #   router LLM call (the single agent is already known) — see
                #   _bg_direct_agent. This is what gives direct rooms
                #   workspace/MCP tool support (internal design notes §9.3).
                # - group rooms: all user messages go through the LLM router (§7.10)
                room_res = await db.execute(select(Room).where(Room.id == room_id))
                room = room_res.scalars().first()
                if not is_agent and room and room.room_type == RoomType.direct:
                    asyncio.create_task(_bg_direct_agent(content, room_id, client_id))
                elif not is_agent and room:
                    asyncio.create_task(_bg_group_flow(room_id, client_id, content, message_id=new_msg.id))
            
            elif message_data.get("type") == "status_update":
                # Handle status updates for existing messages
                msg_id = message_data.get("message_id")
                new_status = message_data.get("status")
                if msg_id and new_status:
                    result = await db.execute(select(Message).where(Message.id == msg_id))
                    msg = result.scalars().first()
                    if msg:
                        msg.status = new_status
                        db.add(msg)
                        await db.commit()
                        
                        # Broadcast status update
                        update_msg = {
                            "type": "status_update",
                            "message_id": str(msg_id),
                            "room_id": str(room_id),
                            "status": new_status
                        }
                        await manager.broadcast(update_msg, room_id)

            # Drop any transaction opened while handling this frame so the
            # pooled connection is released between messages.
            await db.rollback()

    except WebSocketDisconnect:
        pass
    finally:
        # Every way out deregisters this socket (a dropped connection or a
        # handler error is not always a WebSocketDisconnect); a dead one left
        # behind only costs a failed send later, but there is no reason to keep it.
        manager.disconnect(room_id, client_id, is_agent, websocket)
        if presence_task is not None:
            presence_task.cancel()


@router.post("/sessions/{session_id}/confirm_plan", status_code=200)
async def confirm_plan_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Accept a plan proposed by ChatMaster via the propose_plan tool.

    Bypasses the LLM entirely — reads the stored plan from session.context_snapshot
    and applies close_session actions directly (goal creation + pipeline trigger),
    then closes the session: the conversation that produced the plan is over, and
    apply_close_actions never closes a session by contract
    (internal design notes, stage 1.1).
    """
    from datetime import datetime, timezone

    from app.core.events import emit_event
    from app.core.session_close import SessionCloseActions, GoalCreate, apply_close_actions
    from app.core.user_sessions import (
        _plan_steps_from_proposal,
        _requirements_from_proposal,
        _trigger_plan_pipeline,
    )
    from app.models.session.base import Session

    res = await db.execute(select(Session).where(Session.id == session_id))
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await _verify_session_room_membership(session, current_user, db)
    if not session.is_open:
        raise HTTPException(status_code=400, detail="Session already closed")

    snap = dict(session.context_snapshot or {})
    pending = snap.get("pending_plan")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending plan")

    goal_title = pending.get("goal_title", "")
    goal_description = pending.get("goal_description", "")
    if not goal_title:
        raise HTTPException(status_code=422, detail="Plan has no goal_title")

    # One dialogue, one kind of work (core/dialogue_scope). The card may have
    # been proposed before the quick tasks were started, so the invariant is
    # re-checked at the moment the goal would actually be created.
    from app.core.dialogue_scope import (
        quick_tasks_block_plan_message,
        quick_tasks_for_session,
    )

    quick_tasks = await quick_tasks_for_session(session.id, db)
    if quick_tasks:
        raise HTTPException(
            status_code=409,
            detail=quick_tasks_block_plan_message(quick_tasks),
        )

    actions = SessionCloseActions(
        summary=f"Plan accepted by {current_user.full_name or current_user.email}",
        goals_create=[GoalCreate(title=goal_title, description=goal_description)],
    )
    plan_steps = _plan_steps_from_proposal(pending)
    requirements_payload = _requirements_from_proposal(pending)
    await apply_close_actions(session, actions, db)
    await _trigger_plan_pipeline(
        session.room_id,
        actions,
        db,
        plan_steps=plan_steps,
        requirements_payload=requirements_payload,
        session=session,
    )

    snapshot = dict(session.context_snapshot or {})
    snapshot.pop("pending_plan", None)
    session.context_snapshot = snapshot
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = "task_done"
    if not session.summary:
        session.summary = f"Plan accepted: {goal_title}"
    db.add(session)
    await db.flush()

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=f"Session {session.id} closed (kind={session.kind}, reason=task_done)",
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "close_reason": "task_done",
            "summary": session.summary,
            "goal_title": goal_title,
        },
    )

    return {"status": "confirmed", "goal_title": goal_title}


# Sessions whose forced-plan turn is still running.  The button is one click
# away from a second identical LLM turn, and two of them racing would write the
# same history twice.  Process-local on purpose: it guards a background task
# that lives in this process.
_plan_requests_in_flight: set[UUID] = set()


async def _plan_request_state(room_id: UUID, user: User, db: AsyncSession) -> dict:
    """Whether "Form a plan" can act right now, and why not if it cannot."""
    from app.repositories.session import SessionRepository

    session = await SessionRepository(db).find_open_user_session(room_id, user.id)
    if session is None:
        return {
            "available": False,
            "session_id": None,
            "reason": "no_session",
            "detail": "No open dialogue with ChatMaster — write a message in the chat.",
        }
    if session.kind != "user_cm_p2":
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "wrong_kind",
            "detail": "A dialogue of a different type is open — a plan is formed only in a new-topic discussion with ChatMaster.",
        }
    # One dialogue, one kind of work (core/dialogue_scope): this dialogue has
    # already put agents to work without a goal, so pressing the button would
    # only produce a plan that diverges from what is already running.
    from app.core.dialogue_scope import quick_tasks_for_session

    quick_tasks = await quick_tasks_for_session(session.id, db)
    if quick_tasks:
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "quick_tasks",
            "detail": (
                "This discussion already has quick tasks running without a goal — "
                "a goal is not formed from it. Wait for or stop those "
                "tasks and start the goal discussion again."
            ),
        }
    if (session.context_snapshot or {}).get("pending_plan"):
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "plan_pending",
            "detail": "The plan has already been proposed — accept it or send it back for rework.",
        }
    # The button is ChatMaster's to raise (the offer_plan tool): until it says
    # the requirements are enough, there is nothing for the user to confirm.
    if not (session.context_snapshot or {}).get("plan_offered"):
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "not_offered",
            "detail": "ChatMaster has not offered a plan yet — keep discussing the task.",
        }
    if session.id in _plan_requests_in_flight:
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "in_flight",
            "detail": "ChatMaster is already forming the plan.",
        }
    if turn_in_progress(session.id):
        # Two turns on one session race on its history: the plan this one would
        # store could be overwritten by the reply the other one is still writing.
        return {
            "available": False,
            "session_id": str(session.id),
            "reason": "busy",
            "detail": "ChatMaster is replying right now — wait for the reply and press again.",
        }
    return {"available": True, "session_id": str(session.id), "reason": None, "detail": None}


@router.post("/rooms/{room_id}/stop", status_code=200)
async def stop_dialogue(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """The Stop button: end the agent's running turn in this room's dialogue.

    A direct dialogue has no task while the agent works, so there is no task
    card to cancel — this is the only handle on it.  The turn winds down at its
    next boundary (including a ``run_subtask`` child), the files written so far
    are kept and sent through Gate 1, and the conversation stays open.
    """
    from app.core.user_sessions import stop_dialogue_turn

    await _verify_room_membership(room_id, current_user, db)
    return await stop_dialogue_turn(room_id, current_user.id, db)


@router.get("/rooms/{room_id}/plan_request", status_code=200)
async def plan_request_state(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """State of the composer's "Form a plan" button for this room."""
    await _verify_room_membership(room_id, current_user, db)
    return await _plan_request_state(room_id, current_user, db)


@router.post("/rooms/{room_id}/request_plan", status_code=202)
async def request_plan(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """The user pressed "Form a plan" — make the open CM dialogue propose one.

    Deliberately router-free: the whole point of the button is that it is not
    another message about the spec for the router to read as "continue the
    discussion".  It runs one forced turn on the already-open user_cm_p2 session
    with ``propose_plan`` as the only allowed answer, so requirement gathering
    ends on a click instead of on the model deciding it has read enough.
    """
    from app.core.events import emit_event
    from app.models.message import PLAN_REQUESTED

    await _verify_room_membership(room_id, current_user, db)

    from app.core import work_gates

    gate_refusal = await work_gates.work_refusal(room_id, db)
    if gate_refusal is not None:
        raise HTTPException(status_code=402, detail=gate_refusal.text)
    from app.core.room_storage import room_storage_refusal

    storage_refusal = await room_storage_refusal(room_id, db)
    if storage_refusal is not None:
        raise HTTPException(status_code=402, detail=storage_refusal)

    state = await _plan_request_state(room_id, current_user, db)
    if not state["available"]:
        raise HTTPException(status_code=409, detail=state["detail"])

    session_id = UUID(state["session_id"])
    _plan_requests_in_flight.add(session_id)
    try:
        await emit_event(
            room_id=room_id,
            event_type=PLAN_REQUESTED,
            content="Plan formation requested",
            db=db,
            metadata={"session_id": str(session_id), "user_id": str(current_user.id)},
        )
        await db.commit()
    except Exception:
        _plan_requests_in_flight.discard(session_id)
        raise

    asyncio.create_task(_bg_request_plan(session_id))
    return {"status": "requested", "session_id": str(session_id)}


async def _bg_request_plan(session_id: UUID) -> None:
    """Background: run the forced propose_plan turn for a button press."""
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as db:
            from sqlalchemy.orm import selectin_polymorphic

            from app.core.user_sessions import FORCE_PLAN_USER_MESSAGE
            from app.models.session.base import Session
            from app.models.session import SESSION_SUBTYPES

            # Polymorphic load: the forced turn reads chat_master_agent_id and
            # the other user_cm columns, and a lazy load of those would raise
            # MissingGreenlet on this async path.
            res = await db.execute(
                select(Session)
                .where(Session.id == session_id)
                .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
            )
            session = res.scalars().first()
            if not session or not session.is_open:
                return
            await process_user_turn(
                session, FORCE_PLAN_USER_MESSAGE, db, force_plan=True
            )
            await db.commit()
    except Exception as exc:
        _logger.exception("_bg_request_plan failed for session %s: %s", session_id, exc)
    finally:
        _plan_requests_in_flight.discard(session_id)


@router.post("/sessions/{session_id}/reject_plan", status_code=200)
async def reject_plan_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Reject a plan proposed by ChatMaster — clears pending_plan, session stays open."""
    from app.models.session.base import Session

    res = await db.execute(select(Session).where(Session.id == session_id))
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await _verify_session_room_membership(session, current_user, db)

    snap = dict(session.context_snapshot or {})
    if not snap.pop("pending_plan", None):
        raise HTTPException(status_code=400, detail="No pending plan")

    session.context_snapshot = snap
    db.add(session)
    await db.commit()

    return {"status": "rejected"}


@router.post("/sessions/{session_id}/confirm_task_change", status_code=200)
async def confirm_task_change_session(
    session_id: UUID,
    body: TaskChangeActionRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Apply a stored task-change proposal without invoking ChatMaster."""

    from app.core.events import emit_event
    from app.core.task_change import (
        TaskChangeConflictError,
        apply_task_change_proposal,
    )
    from app.models.message import PersistedEventType
    from app.models.task import Task

    session = await _get_session_for_action(session_id, db, current_user)
    if getattr(session, "kind", "user_cm_p1") != "user_cm_p1":
        raise HTTPException(
            status_code=400,
            detail="Task changes are supported only for user_cm_p1 sessions",
        )
    snap = dict(session.context_snapshot or {})
    pending = snap.get("pending_task_change")
    if not pending:
        applied = snap.get("last_applied_task_change")
        if (
            isinstance(applied, dict)
            and str(applied.get("proposal_id")) == str(body.proposal_id)
        ):
            # The first request already committed the board change.  Returning
            # the durable result makes a retried/double-clicked confirm safe
            # without re-emitting events or dispatching replacements again.
            from app.core.metrics import (
                log_task_change_event,
                record_task_change_confirmation,
                record_task_change_idempotent_retry,
            )

            record_task_change_confirmation()
            record_task_change_idempotent_retry()
            log_task_change_event(
                "task_change_confirmed_idempotent",
                room_id=session.room_id,
                session_id=session.id,
                proposal_id=body.proposal_id,
            )
            return {"status": "confirmed", **applied}
        raise HTTPException(status_code=400, detail="No pending task change")

    try:
        proposal = TaskChangeProposal.model_validate(pending)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "invalid_pending_task_change",
                "message": "The stored task-change proposal is no longer valid",
            },
        ) from exc

    if proposal.proposal_id != body.proposal_id:
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail={
                "code": "proposal_id_mismatch",
                "message": "The requested proposal does not match the pending proposal",
                "pending_proposal_id": str(proposal.proposal_id),
            },
        )

    try:
        result = await apply_task_change_proposal(session, proposal, db)
    except TaskChangeConflictError as exc:
        # The service rolls back before raising.  Include the current row when
        # it is still present so the card can offer a fresh proposal.
        current_task = None
        if exc.task_id is not None:
            current_task = (
                await db.execute(select(Task).where(Task.id == exc.task_id))
            ).scalars().first()
        detail = _task_change_conflict_detail(exc, proposal.proposal_id, current_task)
        from app.core.metrics import log_task_change_event, record_task_change_conflict

        record_task_change_conflict()
        log_task_change_event(
            "task_change_conflict",
            room_id=session.room_id,
            session_id=session.id,
            proposal_id=proposal.proposal_id,
            task_id=exc.task_id,
        )
        try:
            await emit_event(
                room_id=session.room_id,
                event_type=PersistedEventType.TASK_CHANGE_CONFLICT,
                content="The task replacement proposal is stale and was not applied.",
                db=db,
                metadata={
                    "session_id": str(session.id),
                    **detail,
                },
            )
        except Exception:
            await db.rollback()
            logging.getLogger(__name__).exception(
                "failed to emit TASK_CHANGE_CONFLICT for proposal %s",
                proposal.proposal_id,
            )
        raise HTTPException(status_code=409, detail=detail) from exc
    except Exception as exc:
        from app.schemas.task_change import TaskChangeValidationError

        if isinstance(exc, TaskChangeValidationError):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": getattr(exc, "code", "invalid_task_change_proposal"),
                    "message": str(exc),
                },
            ) from exc
        raise

    from app.core.metrics import log_task_change_event, record_task_change_confirmation

    record_task_change_confirmation()
    log_task_change_event(
        "task_change_confirmed",
        room_id=session.room_id,
        session_id=session.id,
        proposal_id=proposal.proposal_id,
        created_count=len(result.get("created_tasks", [])),
    )
    return {"status": "confirmed", **result}


@router.post("/sessions/{session_id}/reject_task_change", status_code=200)
async def reject_task_change_session(
    session_id: UUID,
    body: TaskChangeActionRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Discard only the stored task-change proposal; leave the board intact."""

    from app.core.events import emit_event
    from app.models.message import PersistedEventType

    session = await _get_session_for_action(session_id, db, current_user)
    if getattr(session, "kind", "user_cm_p1") != "user_cm_p1":
        raise HTTPException(
            status_code=400,
            detail="Task changes are supported only for user_cm_p1 sessions",
        )
    snap = dict(session.context_snapshot or {})
    pending = snap.get("pending_task_change")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending task change")

    try:
        proposal = TaskChangeProposal.model_validate(pending)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "invalid_pending_task_change",
                "message": "The stored task-change proposal is no longer valid",
            },
        ) from exc

    if proposal.proposal_id != body.proposal_id:
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail={
                "code": "proposal_id_mismatch",
                "message": "The requested proposal does not match the pending proposal",
                "pending_proposal_id": str(proposal.proposal_id),
            },
        )

    # Do not rewrite any other context state: reject is intentionally a
    # proposal-only operation and must not touch Task or dependency rows.
    snap.pop("pending_task_change", None)
    session.context_snapshot = snap
    db.add(session)
    await db.flush()

    await emit_event(
        room_id=session.room_id,
        event_type=PersistedEventType.TASK_CHANGE_REJECTED,
        content="The task replacement proposal was rejected.",
        db=db,
        metadata={
            "session_id": str(session.id),
            "proposal_id": str(proposal.proposal_id),
            "rationale": proposal.rationale,
        },
    )
    await db.commit()

    from app.core.metrics import log_task_change_event, record_task_change_rejection

    record_task_change_rejection()
    log_task_change_event(
        "task_change_rejected",
        room_id=session.room_id,
        session_id=session.id,
        proposal_id=proposal.proposal_id,
    )

    return {"status": "rejected", "proposal_id": str(proposal.proposal_id)}


async def _bg_resume_session(session_id: UUID) -> None:
    """Background: resume an open CM session from saved history (no new user message injected)."""
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as db:
            from app.models.session.base import Session
            res = await db.execute(select(Session).where(Session.id == session_id))
            session = res.scalars().first()
            if not session or not session.is_open:
                return
            from app.core.work_gates import refuse_room_work

            if session.room_id is not None and await refuse_room_work(session.room_id, db):
                return
            await process_user_turn(session, "", db, inject_user_message=False)
            await db.commit()
    except Exception as exc:
        _logger.exception("_bg_resume_session failed for session %s: %s", session_id, exc)


@router.post("/sessions/{session_id}/confirm_file_read", status_code=200)
async def confirm_file_read_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Approve CM's file-read request — reads the file and resumes the CM turn."""
    from app.models.session.base import Session
    from app.core.user_sessions import _get_history
    from app.core.tools import REGISTRY as _ws_registry, ToolContext

    res = await db.execute(select(Session).where(Session.id == session_id))
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_open:
        raise HTTPException(status_code=400, detail="Session already closed")

    snap = dict(session.context_snapshot or {})
    pending = snap.get("pending_file_read")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending file read request")

    path = pending.get("path", "")

    content = "[file could not be read]"
    try:
        ctx = ToolContext(room_id=session.room_id, session_id=session_id, task_id=None, db=db)
        content = await _ws_registry.get("read_file").execute(ctx, {"path": path})
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "confirm_file_read: read_file failed for %r: %s", path, exc
        )
        content = f"Error reading file {path!r}: {exc}"

    history = _get_history(session)
    history.append({
        "role": "user",
        "content": f"[The user allowed reading the file `{path}`. Contents:]\n\n{content}",
    })
    snap.pop("pending_file_read", None)
    snap["messages"] = history
    session.context_snapshot = snap
    db.add(session)
    await db.flush()
    await db.commit()

    asyncio.create_task(_bg_resume_session(session_id))
    return {"status": "approved", "path": path}


@router.post("/sessions/{session_id}/reject_file_read", status_code=200)
async def reject_file_read_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Reject CM's file-read request — injects a refusal note and resumes the CM turn."""
    from app.models.session.base import Session
    from app.core.user_sessions import _get_history

    res = await db.execute(select(Session).where(Session.id == session_id))
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_open:
        raise HTTPException(status_code=400, detail="Session already closed")

    snap = dict(session.context_snapshot or {})
    pending = snap.pop("pending_file_read", None)
    if not pending:
        raise HTTPException(status_code=400, detail="No pending file read request")

    path = pending.get("path", "")
    history = _get_history(session)
    history.append({
        "role": "user",
        "content": (
            f"[The user denied reading the file `{path}`. "
            "Continue without it — use only the information already available to you.]"
        ),
    })
    snap["messages"] = history
    session.context_snapshot = snap
    db.add(session)
    await db.flush()
    await db.commit()

    asyncio.create_task(_bg_resume_session(session_id))
    return {"status": "rejected", "path": path}


@router.delete("/sessions/{session_id}", status_code=200)
async def force_close_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """Force-close an open user session (admin escape hatch)."""
    from app.models.session.base import Session
    from datetime import datetime, timezone

    res = await db.execute(select(Session).where(Session.id == session_id))
    session = res.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_open:
        return {"status": "already_closed"}

    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = "cancelled"
    db.add(session)
    await db.commit()
    return {"status": "closed", "session_id": str(session_id)}


async def handle_mentions(content, room_id, db: AsyncSession):
    import re
    import logging
    import json as _json
    from datetime import datetime
    from app.models.message import Message
    from app.models.task import Task, TaskState
    from app.models.agent_session import AgentSession
    from app.core.chat_master import (
        get_llm_completion, invoke_agent_with_context, get_repo_snapshot,
        _extract_files_from_reply, _save_agent_files, _compute_cost,
        get_project_memory, update_project_memory, MEMORY_AGENT_INSTRUCTION,
    )
    from app.core.events import emit_event
    from app.core.skills import resolve_agent_skills_text

    logger = logging.getLogger(__name__)
    mentions = re.findall(r"@([\w\-\+]+)", content)
    handled = False
    for mention in mentions:
        # Look the nickname up among this room's agents only: nicknames are not
        # unique across accounts (every account has a "Router"), so a global
        # .first() could pick another account's agent and skip the member.
        result = await db.execute(
            select(Agent)
            .join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
            .where(
                RoomAgentLink.room_id == room_id,
                (Agent.agent_uid == mention) | (Agent.nickname == mention),
            )
        )
        agent = result.scalars().first()
        if not agent:
            continue

        membership = await db.execute(
            select(RoomAgentLink).where(RoomAgentLink.room_id == room_id, RoomAgentLink.agent_id == agent.id)
        )
        if not membership.scalars().first():
            continue

        if agent.agent_type == "internal" and agent.llm_config and agent.llm_config.get("provider"):
            try:
                # Check for an active task assigned to this agent in this room
                active_task_result = await db.execute(
                    select(Task).where(
                        Task.room_id == room_id,
                        Task.assigned_agent_id == agent.id,
                        Task.state.in_([TaskState.ASSIGNED, TaskState.IN_PROGRESS]),
                    ).limit(1)
                )
                active_task = active_task_result.scalars().first()

                if active_task:
                    # Use isolated context: task + memory + repo + previous session summary
                    reply = await invoke_agent_with_context(agent, active_task.id, content, db)
                    task_id_str = str(active_task.id)
                else:
                    # No active task — fall back to last 20 room messages + repo snapshot
                    hist_result = await db.execute(
                        select(Message).where(Message.room_id == room_id)
                        .order_by(Message.timestamp.desc()).limit(20)
                    )
                    history = list(reversed(hist_result.scalars().all()))

                    cfg = agent.llm_config
                    repo_snapshot = await get_repo_snapshot(room_id, db)
                    proj_memory = await get_project_memory(room_id, db)
                    skills_text = await resolve_agent_skills_text(agent, db)

                    system_parts = []
                    if cfg.get("system_prompt"):
                        system_parts.append(cfg["system_prompt"])
                    if skills_text:
                        system_parts.append(f"Skills:\n{skills_text}")
                    memory_text = proj_memory.content if proj_memory else "(empty)"
                    system_parts.append(
                        f"## Project Memory\n{memory_text}\n\n{MEMORY_AGENT_INSTRUCTION}"
                    )
                    if repo_snapshot:
                        repo_text = _json.dumps(repo_snapshot, ensure_ascii=False, indent=2)
                        system_parts.append(f"## Repository Snapshot\n{repo_text}")
                    system_parts.append(
                        "## Saving Files\n"
                        "When your response includes content that should be saved as a file (code, documents, "
                        "configs, specs), use this exact format so the system can persist it automatically:\n\n"
                        "```<ext> /path/to/filename.ext\n"
                        "file content here\n"
                        "```\n\n"
                        "Example:\n"
                        "```py /src/main.py\n"
                        "print('hello')\n"
                        "```\n\n"
                        "The path MUST start with /. Use meaningful paths. "
                        "Files are saved to the room repository automatically."
                    )
                    system_text = "\n\n".join(system_parts) if system_parts else f"You are {agent.nickname}, an AI assistant."

                    messages = [{"role": "system", "content": system_text}]
                    for m in history:
                        role = "assistant" if m.sender_agent_id == agent.id else "user"
                        messages.append({"role": role, "content": m.content})

                    from app.core.log_storage import get_log_buffer, get_log_archive
                    from app.core.session_logger import SessionLogger
                    from app.models.room import Room as _Room

                    room_res2 = await db.execute(select(_Room).where(_Room.id == room_id))
                    _room = room_res2.scalars().first()
                    creator_id = _room.creator_id if _room else None

                    # Create session upfront so it's visible immediately
                    started_at = datetime.utcnow()
                    session_rec = AgentSession(
                        room_id=room_id,
                        task_id=None,
                        agent_id=agent.id,
                        session_title=content[:80] if content else "Chat",
                        initiator_type="user",
                        model_id=cfg.get("model"),
                        started_at=started_at,
                        log_status="recording",
                        context_snapshot={},
                    )
                    db.add(session_rec)
                    await db.flush()

                    sl = SessionLogger(
                        session_id=session_rec.id,
                        user_id=creator_id or session_rec.id,
                        buffer=get_log_buffer(),
                    )

                    secret_value = await resolve_agent_secret(
                        db=db,
                        owner_id=agent.owner_id,
                        agent_id=agent.id,
                        config=cfg,
                        request_id=session_rec.id,
                    )
                    try:
                        await release_db(db, reason="chat/mentions")
                        llm_result = await get_llm_completion(
                            cfg, messages, secret_value=secret_value
                        )
                    finally:
                        secret_value = None
                    reply = llm_result.text
                    task_id_str = None

                    await sl.log_turn(
                        turn=0,
                        messages_sent=messages,
                        response=reply,
                        tool_calls=[],
                        tool_results=[],
                        usage={"input": llm_result.input_tokens, "output": llm_result.output_tokens},
                        model=cfg.get("model", ""),
                    )
                    log_key, log_bytes = await sl.flush_to_s3(get_log_archive())

                    cost = _compute_cost(cfg.get("model", ""), llm_result.input_tokens, llm_result.output_tokens, cfg)
                    session_rec.summary = reply
                    session_rec.context_snapshot = {"messages_sent": len(messages)}
                    session_rec.ended_at = datetime.utcnow()
                    session_rec.input_tokens = llm_result.input_tokens
                    session_rec.output_tokens = llm_result.output_tokens
                    session_rec.token_count = llm_result.input_tokens + llm_result.output_tokens
                    session_rec.cost_usd = cost
                    session_rec.log_key = log_key or None
                    session_rec.log_size_bytes = log_bytes
                    session_rec.log_status = "flushed" if log_key else "recording"
                    db.add(session_rec)

                # Save any files embedded in the reply to the room repository
                extracted_files = _extract_files_from_reply(reply)
                if extracted_files:
                    saved_paths = await _save_agent_files(room_id, agent.id, extracted_files, db)
                    for path in saved_paths:
                        await emit_event(
                            room_id=room_id,
                            event_type="FILE_CHANGED",
                            content=f"Agent @{agent.nickname} saved file: {path}",
                            db=db,
                            task_id=active_task.id if active_task else None,
                            metadata={"path": path, "agent_nickname": agent.nickname, "source": "agent"},
                        )
                    # Git push is manual-only (Push to Git button) — no auto-push here.

                new_msg = Message(room_id=room_id, content=reply, sender_agent_id=agent.id)
                db.add(new_msg)
                await db.commit()

                # Update project memory after session (only for no-task path; task path handled by ChatMaster)
                if not active_task:
                    try:
                        await update_project_memory(room_id, reply, cfg, db)
                    except Exception as mem_exc:
                        logger.warning("Memory update failed for room %s: %s", room_id, mem_exc)

                broadcast_msg = {
                    "type": "message",
                    "event_type": "MESSAGE",
                    "id": str(new_msg.id),
                    "room_id": str(room_id),
                    "sender_id": str(agent.id),
                    "sender_type": "agent",
                    "nickname": agent.nickname,
                    "content": reply,
                    "timestamp": new_msg.timestamp.isoformat(),
                    "status": "sent",
                    "task_id": task_id_str,
                    "metadata": {},
                }
                await manager.broadcast(broadcast_msg, room_id)
                handled = True
            except Exception as e:
                logger.error(f"Internal agent {agent.agent_uid} failed to respond: {e}")
        else:
            # External agent: send mention notification via WebSocket
            notification = {
                "type": "mention",
                "room_id": str(room_id),
                "content": content,
            }
            await manager.send_to_client(notification, agent.id, is_agent=True)
            handled = True

    return handled


async def handle_internal_agents(content: str, room_id, db: AsyncSession):
    """Auto-respond with all internal agents in the room when no @mention is used."""
    import logging
    from datetime import datetime
    from app.models.message import Message
    from app.models.agent_session import AgentSession
    from app.core.chat_master import (
        get_llm_completion, _compute_cost,
        get_project_memory, update_project_memory, MEMORY_AGENT_INSTRUCTION,
    )
    from app.core.skills import resolve_agent_skills_text

    logger = logging.getLogger(__name__)

    result = await db.execute(
        select(Agent)
        .join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
        .where(RoomAgentLink.room_id == room_id, Agent.agent_type == "internal")
    )
    internal_agents = result.scalars().all()

    for agent in internal_agents:
        cfg = agent.llm_config or {}
        if not cfg.get("provider"):
            continue
        try:
            hist_result = await db.execute(
                select(Message).where(Message.room_id == room_id)
                .order_by(Message.timestamp.desc()).limit(20)
            )
            history = list(reversed(hist_result.scalars().all()))

            proj_memory = await get_project_memory(room_id, db)
            memory_text = proj_memory.content if proj_memory else "(empty)"
            skills_text = await resolve_agent_skills_text(agent, db)

            system_parts = []
            if cfg.get("system_prompt"):
                system_parts.append(cfg["system_prompt"])
            if skills_text:
                system_parts.append(f"Skills:\n{skills_text}")
            system_parts.append(
                f"## Project Memory\n{memory_text}\n\n{MEMORY_AGENT_INSTRUCTION}"
            )
            system_text = "\n\n".join(system_parts)

            messages = [{"role": "system", "content": system_text}]
            for m in history:
                role = "assistant" if m.sender_agent_id == agent.id else "user"
                messages.append({"role": role, "content": m.content})

            from app.core.log_storage import get_log_buffer, get_log_archive
            from app.core.session_logger import SessionLogger
            from app.models.room import Room as _Room2

            _room2_res = await db.execute(select(_Room2).where(_Room2.id == room_id))
            _room2 = _room2_res.scalars().first()
            _creator_id = _room2.creator_id if _room2 else None

            started_at = datetime.utcnow()
            session_rec = AgentSession(
                room_id=room_id,
                task_id=None,
                agent_id=agent.id,
                session_title=content[:80] if content else "Chat",
                initiator_type="user",
                model_id=cfg.get("model"),
                started_at=started_at,
                log_status="recording",
                context_snapshot={},
            )
            db.add(session_rec)
            await db.flush()

            sl = SessionLogger(
                session_id=session_rec.id,
                user_id=_creator_id or session_rec.id,
                buffer=get_log_buffer(),
            )

            secret_value = await resolve_agent_secret(
                db=db,
                owner_id=agent.owner_id,
                agent_id=agent.id,
                config=cfg,
                request_id=session_rec.id,
            )
            try:
                await release_db(db, reason="chat/internal_agents")
                llm_result = await get_llm_completion(
                    cfg, messages, secret_value=secret_value
                )
            finally:
                secret_value = None
            reply = llm_result.text

            await sl.log_turn(
                turn=0,
                messages_sent=messages,
                response=reply,
                tool_calls=[],
                tool_results=[],
                usage={"input": llm_result.input_tokens, "output": llm_result.output_tokens},
                model=cfg.get("model", ""),
            )
            log_key, log_bytes = await sl.flush_to_s3(get_log_archive())

            cost = _compute_cost(cfg.get("model", ""), llm_result.input_tokens, llm_result.output_tokens, cfg)
            session_rec.summary = reply
            session_rec.context_snapshot = {"messages_sent": len(messages)}
            session_rec.ended_at = datetime.utcnow()
            session_rec.input_tokens = llm_result.input_tokens
            session_rec.output_tokens = llm_result.output_tokens
            session_rec.token_count = llm_result.input_tokens + llm_result.output_tokens
            session_rec.cost_usd = cost
            session_rec.log_key = log_key or None
            session_rec.log_size_bytes = log_bytes
            session_rec.log_status = "flushed" if log_key else "recording"
            db.add(session_rec)
            await db.flush()

            new_msg = Message(room_id=room_id, content=reply, sender_agent_id=agent.id)
            db.add(new_msg)
            await db.commit()

            try:
                await update_project_memory(room_id, reply, cfg, db)
            except Exception as mem_exc:
                logger.warning("Memory update failed for room %s: %s", room_id, mem_exc)

            broadcast_msg = {
                "type": "message",
                "event_type": "MESSAGE",
                "id": str(new_msg.id),
                "room_id": str(room_id),
                "sender_id": str(agent.id),
                "sender_type": "agent",
                "nickname": agent.nickname,
                "content": reply,
                "timestamp": new_msg.timestamp.isoformat(),
                "status": "sent",
                "task_id": None,
                "metadata": {},
            }
            await manager.broadcast(broadcast_msg, room_id)
        except Exception as e:
            logger.error(f"Internal agent {agent.agent_uid} failed to respond: {e}")
            err_msg = {
                "type": "message",
                "id": f"err-{agent.id}",
                "room_id": str(room_id),
                "sender_id": str(agent.id),
                "sender_type": "agent",
                "nickname": agent.nickname,
                "content": f"⚠️ Error: {e}",
                "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
                "status": "error",
            }
            await manager.broadcast(err_msg, room_id)


async def _bg_chat_master(room_id, content):
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as session:
            await chat_master.chat_master_process(room_id, content, session)
    except Exception as exc:
        _logger.exception("_bg_chat_master failed for room %s: %s", room_id, exc)

async def _bg_direct_agent(content, room_id, user_id):
    """Deliver a message to the single agent in a direct room via the same
    session-flow group rooms use (process_user_turn/_call_with_ws_loop) —
    the router LLM call is skipped since the target agent is already known,
    but everything downstream (workspace tools, MCP tools, memory, session
    continuity) is the same code path. Direct rooms used to bypass all of
    that via handle_mentions(); that path had no tool-calling at all, so
    e.g. a Market MCP server attached to a direct room was silently unusable
    (internal design notes §9.3 follow-up).
    """
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as session:
            from app.models.task import Task, TaskState
            from app.repositories.session import SessionRepository

            room_res = await session.execute(select(Room).where(Room.id == room_id))
            room = room_res.scalars().first()
            user_res = await session.execute(select(User).where(User.id == user_id))
            user = user_res.scalars().first()
            if not room or not user:
                return
            # internal design notes T8: out of hours, the message stays in
            # the history but no agent works on it.
            from app.core.work_gates import refuse_room_work

            if await refuse_room_work(room_id, session):
                return

            agent_res = await session.execute(
                select(Agent)
                .join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
                .where(RoomAgentLink.room_id == room_id)
                .limit(1)
            )
            agent = agent_res.scalars().first()
            if not agent:
                return

            active_task_res = await session.execute(
                select(Task).where(
                    Task.room_id == room_id,
                    Task.assigned_agent_id == agent.id,
                    Task.state.in_([TaskState.ASSIGNED, TaskState.IN_PROGRESS]),
                ).limit(1)
            )
            active_task = active_task_res.scalars().first()

            # Direct rooms have no router LLM call — the agent is already known —
            # but the "continue or start fresh" decision is still made explicitly
            # (stage 2.1): continue only a session that is still young AND still
            # bound to the task the room is working on. Anything else is closed
            # gracefully so its memory survives.
            repo = SessionRepository(session)
            existing_session = await repo.find_open_user_session(room.id, user.id)
            if existing_session is not None:
                from app.core.user_sessions import (
                    close_user_session_gracefully,
                    open_session_context,
                )

                open_ctx = await open_session_context(existing_session, session)
                task_unchanged = getattr(existing_session, "task_id", None) == (
                    active_task.id if active_task else None
                )
                if open_ctx is not None and task_unchanged:
                    await process_user_turn(existing_session, content, session)
                    await session.commit()
                    return
                await close_user_session_gracefully(
                    existing_session,
                    session,
                    rotation_reason="task_changed" if open_ctx else "not_continuable",
                )

            if active_task:
                decision = RouteDecision(
                    route="single_agent_p1",
                    mentioned_agent_ids=[agent.id],
                    matched_task_id=active_task.id,
                    matched_goal_id=active_task.goal_id,
                    confidence=1.0,
                    reasoning="direct room: single agent with an active task, router bypassed",
                )
            else:
                decision = RouteDecision(
                    route="single_agent_p2",
                    mentioned_agent_ids=[agent.id],
                    confidence=1.0,
                    reasoning="direct room: single agent, router bypassed",
                )

            new_session = await start_user_session(room, user, decision, session)
            await process_user_turn(new_session, content, session)
            await session.commit()
    except Exception as exc:
        _logger.exception("_bg_direct_agent failed for room %s: %s", room_id, exc)

async def _bg_internal_agents(content, room_id):
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as session:
            await handle_internal_agents(content, room_id, session)
    except Exception as exc:
        _logger.exception("_bg_internal_agents failed for room %s: %s", room_id, exc)

async def _bg_mentions(content, room_id):
    _logger = logging.getLogger(__name__)
    try:
        async with _make_session()() as session:
            handled = await handle_mentions(content, room_id, session)
            if not handled:
                await chat_master.chat_master_process(room_id, content, session)
    except Exception as exc:
        _logger.exception("_bg_mentions failed for room %s: %s", room_id, exc)


# ---------------------------------------------------------------------------
# New group-room flow (§7.10): all user messages go through the LLM router
# ---------------------------------------------------------------------------

# ``_get_cm_agent_id`` is imported from ``app.core.room_coordinator`` (see the
# note there): the triple of coordinator_config keys it reads is shared with
# the router and the ChatMaster, and used to be duplicated in four files.


async def _handle_direct_agent_session(
    room: Room,
    user: User,
    content: str,
    decision: RouteDecision,
    db: AsyncSession,
) -> None:
    """Handle single_agent_p1/p2 — one session between user and a specific agent."""
    session = await start_user_session(room, user, decision, db)
    await process_user_turn(session, content, db)


async def _handle_chatmaster_session(
    room: Room,
    user: User,
    content: str,
    decision: RouteDecision,
    db: AsyncSession,
) -> None:
    """Handle chatmaster_p1/p2 — one session between user and ChatMaster."""
    _logger = logging.getLogger(__name__)
    cm_agent_id = _get_cm_agent_id(room)
    if cm_agent_id is None:
        if room.has_coordinator:
            await chat_master.chat_master_process(room.id, content, db)
            return
        _logger.warning(
            "Group room %s has no coordinator config — falling back to first internal agent",
            room.id,
        )
        agent_res = await db.execute(
            select(Agent)
            .join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
            .where(RoomAgentLink.room_id == room.id, Agent.agent_type == "internal")
            .limit(1)
        )
        agent = agent_res.scalars().first()
        if agent:
            await handle_mentions(f"@{agent.agent_uid} {content}", room.id, db)
        else:
            _logger.warning("No internal agents in room %s — cannot respond", room.id)
        return
    session = await start_user_session(room, user, decision, db, cm_agent_id=cm_agent_id)
    await process_user_turn(session, content, db)


async def _handle_split(
    room: Room,
    user: User,
    content: str,
    decision: RouteDecision,
    db: AsyncSession,
) -> None:
    """Handle multi_agent_split — create one session per split part (N ≤ 3).

    If the router returned more than 3 parts, force to multi_agent_ask_detail (§7.9).
    """
    split_msgs = decision.split_messages or []

    if len(split_msgs) > 3:
        forced = decision.model_copy(
            update={"route": "multi_agent_ask_detail", "can_split": False}
        )
        await _handle_ask_detail(room, user, content, forced, db)
        return

    for part in split_msgs:
        sub_decision = RouteDecision(
            route="single_agent_p2",
            mentioned_agent_ids=[part.agent_id],
            confidence=decision.confidence,
            reasoning=f"split part for agent {part.agent_id}",
        )
        session = await start_user_session(room, user, sub_decision, db)
        await process_user_turn(session, part.text, db)


async def _handle_ask_detail(
    room: Room,
    user: User,
    content: str,
    decision: RouteDecision,
    db: AsyncSession,
) -> None:
    """Handle multi_agent_ask_detail — route to ChatMaster to ask for clarification."""
    cm_agent_id = _get_cm_agent_id(room)
    if cm_agent_id is None:
        await chat_master.chat_master_process(room.id, content, db)
        return
    cm_decision = RouteDecision(
        route="chatmaster_p2",
        confidence=decision.confidence,
        reasoning="multi_agent_ask_detail: routing to ChatMaster for clarification",
    )
    session = await start_user_session(room, user, cm_decision, db, cm_agent_id=cm_agent_id)
    await process_user_turn(session, content, db)


async def _handle_interject(
    room: Room,
    user: User,
    content: str,
    decision: RouteDecision,
    db: AsyncSession,
    message_id: UUID | None,
) -> bool:
    """internal design notes Stage 4: put the message into the running task.

    The user's chat message is already saved — it is tagged with the task and
    queued for the executor's next model-request boundary.  The Hub answers at
    once, without a model, with the status verdict (Stage 1).  False when the
    task cannot take it any more (finished meanwhile, waiting for the user…):
    the caller routes the message the old way.
    """
    from app.core import task_interject, task_status_probe
    from app.core.events import emit_event
    from app.models.task import Task

    task = (
        await db.execute(
            select(Task).where(Task.id == decision.matched_task_id, Task.room_id == room.id)
        )
    ).scalars().first()
    if task is None:
        return False
    try:
        row = await task_interject.create(
            task, user=user, text=content, db=db, existing_message_id=message_id,
        )
    except task_interject.InterjectRefused as exc:
        logging.getLogger(__name__).info(
            "interject_task refused for task %s (%s) — routing the old way", task.id, exc.code,
        )
        return False

    try:
        status = await task_status_probe.collect(task, db)
        verdict = status.get("verdict") or {}
    except Exception:  # noqa: BLE001 — the message is queued either way
        logging.getLogger(__name__).warning("interject_task: status probe failed", exc_info=True)
        verdict = {}
    agent = await db.get(Agent, task.assigned_agent_id) if task.assigned_agent_id else None
    who = f"@{agent.nickname}" if agent is not None else "the assignee"
    text = f"Message passed to {who} for task '{task.title}' — they will get it before the next step."
    if verdict.get("text"):
        text += f" {verdict['text']}"
    await emit_event(
        room_id=room.id,
        event_type="TASK_INTERJECTION_ROUTED",
        content=text,
        db=db,
        task_id=task.id,
        metadata={
            "interjection_id": str(row.id),
            "message_id": str(row.message_id) if row.message_id else None,
            "agent_id": str(task.assigned_agent_id) if task.assigned_agent_id else None,
            "verdict": verdict.get("level"),
        },
    )
    return True


async def _dispatch_group_route(
    room: Room,
    user: User,
    content: str,
    db: AsyncSession,
    *,
    message_id: UUID | None = None,
) -> None:
    """Route a user message through the LLM router and dispatch to the right handler.

    Called for every user message in a group room (§1.1, §7.9, §7.10).
    New messages during an active proxy session always start their own router (§7.3).

    An open user session is not a reason to skip the router (defect C1 — that
    bypass is what produced one 185-message session per room). It is passed to
    the router as context instead, so "continue this session" becomes a routing
    decision the model makes deliberately; on any other route the open session
    is closed gracefully, its memory saved (stage 2.1/2.2).
    """
    _logger = logging.getLogger(__name__)

    from app.core.user_sessions import close_user_session_gracefully, open_session_context
    from app.repositories.session import SessionRepository
    repo = SessionRepository(db)
    existing_session = await repo.find_open_user_session(room.id, user.id)
    open_ctx = (
        await open_session_context(existing_session, db)
        if existing_session is not None
        else None
    )

    # waiting_reply_safeguard S1: a short, mention-free reply ("finish", "ok",
    # "wrap it up") to a task parked waiting for the user is acted on directly —
    # approve-and-merge, or re-open the task — instead of being cold-routed to a
    # fresh ChatMaster session. Only when there is no live session to carry it:
    # a young open session continues normally (the router's continue_session).
    if open_ctx is None and not parse_mentions(content):
        from app.core.waiting_reply import (
            classify_reply,
            find_waiting_task,
            handle_waiting_reply,
        )

        if classify_reply(content) is not None:
            _wait_task = await find_waiting_task(
                room.id, db,
                prefer_session_id=existing_session.id if existing_session else None,
            )
            if _wait_task is not None and await handle_waiting_reply(
                room, user, content, db,
                prefer_session_id=(
                    existing_session.id if existing_session else None
                ),
                task=_wait_task,
            ):
                if existing_session is not None:
                    # The reply landed on the task; the stale dialogue it came
                    # from is over — close it so its memory is saved and the
                    # next message does not resume it.
                    await close_user_session_gracefully(
                        existing_session, db, rotation_reason="not_continuable",
                    )
                return

    decision = await route_user_message(room, content, db, open_session=open_ctx)
    _logger.info(
        "Router decision for room %s: route=%s confidence=%.2f reasoning=%s",
        room.id, decision.route, decision.confidence, decision.reasoning,
    )

    if decision.route == "continue_session":
        if (
            existing_session is not None
            and decision.continue_session_id == existing_session.id
        ):
            _logger.info(
                "Continuing open session %s (kind=%s) for user %s in room %s",
                existing_session.id, existing_session.kind, user.id, room.id,
            )
            await process_user_turn(existing_session, content, db)
            return
        # The session named by the decision is not the one we offered (or there
        # is none). Answering in the wrong session is worse than starting a new
        # one, and leaving the turn unanswered is worse than both.
        _logger.warning(
            "Router asked to continue session %s which is not the open session "
            "in room %s — starting a new ChatMaster session instead",
            decision.continue_session_id, room.id,
        )
        decision = decision.model_copy(
            update={"route": "chatmaster_p2", "continue_session_id": None}
        )

    if decision.route == "interject_task":
        # Stage 4: a word into a running task is not a new conversation — the
        # open session (if any) stays as it is.
        if decision.matched_task_id is not None and await _handle_interject(
            room, user, content, decision, db, message_id,
        ):
            return
        decision = decision.model_copy(update={
            "route": "single_agent_p1" if decision.matched_task_id else "single_agent_p2",
        })

    if existing_session is not None:
        # The router chose a different route: this conversation is over. Close it
        # here rather than leaving a second open session behind (which is what
        # find_open_user_session would hand the next message).
        await close_user_session_gracefully(
            existing_session,
            db,
            rotation_reason="topic_change" if open_ctx else "not_continuable",
        )

    if decision.route == "multi_agent_split":
        await _handle_split(room, user, content, decision, db)
    elif decision.route == "multi_agent_ask_detail":
        await _handle_ask_detail(room, user, content, decision, db)
    elif decision.route in ("single_agent_p1", "single_agent_p2"):
        if not decision.mentioned_agent_ids:
            # The model picked a single-agent route but the deterministic
            # @-parse resolved no room agent (a mention of someone not in the
            # room, or none at all). start_user_session would IndexError on
            # mentioned_agent_ids[0] and the turn would vanish — route to
            # ChatMaster instead so the user still gets an answer.
            _logger.warning(
                "Router chose %s for room %s with no resolved @mention — "
                "routing to ChatMaster instead", decision.route, room.id,
            )
            forced = decision.model_copy(update={"route": "chatmaster_p2"})
            await _handle_chatmaster_session(room, user, content, forced, db)
        else:
            await _handle_direct_agent_session(room, user, content, decision, db)
    elif decision.route == "resume_waiting_task":
        # waiting_reply_safeguard S2: the router's fuzzy backstop for a reply the
        # deterministic S1 hook did not catch. Falls through to ChatMaster if the
        # task is gone or the handler declines.
        from app.core.waiting_reply import handle_waiting_reply
        from app.models.task import Task

        handled = False
        if decision.matched_task_id is not None:
            task = (
                await db.execute(
                    select(Task).where(
                        Task.id == decision.matched_task_id, Task.room_id == room.id
                    )
                )
            ).scalars().first()
            if task is not None:
                handled = await handle_waiting_reply(
                    room, user, content, db, task=task
                )
        if not handled:
            forced = decision.model_copy(update={"route": "chatmaster_p2"})
            await _handle_chatmaster_session(room, user, content, forced, db)
    elif decision.route in ("chatmaster_p1", "chatmaster_p2"):
        await _handle_chatmaster_session(room, user, content, decision, db)


async def _bg_group_flow(
    room_id: UUID, user_id: UUID, content: str, *, message_id: UUID | None = None,
) -> None:
    """Background task: run the new group-room chat flow for a user message."""
    _logger = logging.getLogger(__name__)
    _logger.info("_bg_group_flow started for room %s, user %s", room_id, user_id)
    try:
        async with _make_session()() as db:
            room_res = await db.execute(select(Room).where(Room.id == room_id))
            room = room_res.scalars().first()
            if not room:
                _logger.warning("_bg_group_flow: room %s not found", room_id)
                return
            user_res = await db.execute(select(User).where(User.id == user_id))
            user = user_res.scalars().first()
            if not user:
                _logger.warning("_bg_group_flow: user %s not found", user_id)
                return
            # internal design notes T8: out of hours, the message stays in
            # the history but neither the router nor an agent works on it.
            from app.core.work_gates import refuse_room_work

            if await refuse_room_work(room_id, db):
                return
            await _dispatch_group_route(room, user, content, db, message_id=message_id)
            await db.commit()
    except Exception as exc:
        _logger.exception("_bg_group_flow failed for room %s: %s", room_id, exc)
