from datetime import datetime, timezone
import logging
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func
from sqlalchemy.orm.attributes import flag_modified
from uuid import UUID

from app.api import deps
from app.db.session import get_session
from app.models.room import Room, RoomUserLink, RoomAgentLink, ROOM_TRANSITIONS
from app.models.message import Message
from app.models.agent import Agent
from app.models.agent_memory import AgentMemory
from app.models.user import User
from app.models.market import RoomMcpServer, UserInstalledItem
from app.models.workspace import Workspace, WorkspaceUsage
from app.models.personal_mcp import PersonalMcpRuntime
from app.api.v1.user_installed_items import InstalledItemOut, _as_out
from app.core.mcp_discovery import is_http_link, mcp_connection_status
from app.models.mcp_connection import McpConnectionSnapshot
from app.schemas.mcp_connection import McpRoomBindingOut
from app.schemas.room import RoomCreate, RoomOut, RoomUpdate, RoomStateUpdate
from app.core.room_members import (
    RoomAgentMembershipError,
    add_agent_to_room,
    former_agents,
    remove_agent_from_room,
    room_agent_ids,
)


class QuestionAnswerIn(BaseModel):
    selected_options: list[str] = []
    free_text: Optional[str] = None


class RoomSwitchIn(BaseModel):
    to_room_id: UUID
    request_id: UUID
    confirm_stop: bool = False
    force_stop: bool = False

router = APIRouter()
logger = logging.getLogger(__name__)


async def _require_room_member(room_id: UUID, user_id: UUID, db: AsyncSession) -> Room:
    """Return a room only when it is available to the authenticated user."""
    room = (
        await db.execute(
            select(Room)
            .join(RoomUserLink)
            .where(Room.id == room_id, RoomUserLink.user_id == user_id)
        )
    ).scalars().first()
    if room is None:
        raise HTTPException(status_code=403, detail="Not a member of this room")
    return room


async def _require_owned_active_mcp(
    installed_item_id: UUID, user_id: UUID, db: AsyncSession, *, require_active: bool = True
) -> UserInstalledItem:
    item = await db.get(UserInstalledItem, installed_item_id)
    if item is None or item.user_id != user_id:
        raise HTTPException(status_code=403, detail="Installed item does not belong to you")
    if item.type != "mcp":
        raise HTTPException(status_code=422, detail="Only MCP items can be assigned to a room")
    if require_active and item.status != "active":
        raise HTTPException(status_code=422, detail="Only active installed items can be assigned")
    return item


@router.get("/{room_id}/mcp-servers", response_model=list[InstalledItemOut])
async def list_room_mcp_servers(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[InstalledItemOut]:
    """List the caller's active MCP assignments for one accessible room."""
    await _require_room_member(room_id, current_user.id, db)
    result = await db.execute(
        select(UserInstalledItem)
        .join(RoomMcpServer, RoomMcpServer.user_installed_item_id == UserInstalledItem.id)
        .where(
            RoomMcpServer.room_id == room_id,
            UserInstalledItem.user_id == current_user.id,
            UserInstalledItem.type == "mcp",
        )
        .order_by(RoomMcpServer.created_at)
    )
    items = list(result.scalars().all())
    snapshots: dict[UUID, McpConnectionSnapshot] = {}
    link_items = [item for item in items if is_http_link(item)]
    if link_items:
        snapshot_result = await db.execute(
            select(McpConnectionSnapshot).where(
                McpConnectionSnapshot.installed_item_id.in_([item.id for item in link_items])
            )
        )
        snapshots = {
            snapshot.installed_item_id: snapshot
            for snapshot in snapshot_result.scalars().all()
        }
    output: list[InstalledItemOut] = []
    for item in items:
        connection = await mcp_connection_status(
            db,
            item,
            snapshot=snapshots.get(item.id),
        )
        available = connection.available if connection is not None else True
        output.append(
            _as_out(
                item,
                mcp_status=connection,
                room_bindings=[
                    McpRoomBindingOut(
                        room_id=room_id,
                        available=available,
                        status="available" if available else "unavailable",
                    )
                ],
            )
        )
    return output


@router.put("/{room_id}/mcp-servers/{installed_item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def assign_room_mcp_server(
    room_id: UUID,
    installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Idempotently make one of the caller's MCP servers available in a room."""
    room = await _require_room_member(room_id, current_user.id, db)
    item = await _require_owned_active_mcp(installed_item_id, current_user.id, db)
    if item.is_custom and (item.custom_config or {}).get("mode") in {"hosted", "managed_stdio"} and room.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="A personal MCP can only be attached to its owner's room")
    if item.is_custom and (item.custom_config or {}).get("mode") == "hosted":
        from app.core.workspace_quota import get_effective_limits
        limits = await get_effective_limits(current_user.id, db)
        max_rooms = limits.get("mcp_personal_max_rooms")
        if isinstance(max_rooms, int):
            assigned = await db.execute(select(func.count(RoomMcpServer.room_id)).where(RoomMcpServer.user_installed_item_id == item.id))
            existing_rooms = int(assigned.scalar_one() or 0)
            if existing_rooms >= max_rooms and await db.get(RoomMcpServer, (room_id, installed_item_id)) is None:
                raise HTTPException(status_code=429, detail=f"Room limit for the personal MCP reached: {existing_rooms}/{max_rooms}.")
    if item.is_custom and (item.custom_config or {}).get("mode") == "managed_stdio":
        runtime = await db.get(PersonalMcpRuntime, item.id)
        if runtime is None or runtime.actual_status != "verified":
            raise HTTPException(status_code=409, detail="Personal MCP must be verified before room assignment")
        from app.core.workspace_quota import get_effective_limits
        limits = await get_effective_limits(current_user.id, db)
        from app.api.v1.personal_mcps import _personal_limits
        max_bindings = _personal_limits(limits).get("mcp_personal_max_room_bindings_per_server")
        existing = await db.get(RoomMcpServer, (room_id, installed_item_id))
        if existing is None and isinstance(max_bindings, int):
            assigned = await db.execute(
                select(func.count(RoomMcpServer.room_id)).where(
                    RoomMcpServer.user_installed_item_id == item.id
                )
            )
            if int(assigned.scalar_one() or 0) >= max_bindings:
                raise HTTPException(status_code=429, detail="Personal MCP room binding limit reached")
    existing = await db.get(RoomMcpServer, (room_id, installed_item_id))
    if existing is None:
        db.add(
            RoomMcpServer(
                room_id=room_id,
                user_installed_item_id=installed_item_id,
                added_by=current_user.id,
            )
        )
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{room_id}/mcp-servers/{installed_item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unassign_room_mcp_server(
    room_id: UUID,
    installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Remove an MCP assignment without allowing cross-user unlinking."""
    await _require_room_member(room_id, current_user.id, db)
    await _require_owned_active_mcp(installed_item_id, current_user.id, db, require_active=False)
    await db.execute(
        delete(RoomMcpServer).where(
            RoomMcpServer.room_id == room_id,
            RoomMcpServer.user_installed_item_id == installed_item_id,
        )
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/", response_model=RoomOut)
async def create_room(
    room_in: RoomCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    db_obj = Room(
        name=room_in.name,
        type=room_in.type,
        room_type=room_in.room_type,
        has_coordinator=room_in.has_coordinator,
        coordinator_config=room_in.coordinator_config,
        creator_id=current_user.id,
    )
    db.add(db_obj)
    await db.flush() # Get the ID
    
    # Add creator as member
    db.add(RoomUserLink(room_id=db_obj.id, user_id=current_user.id))
    
    # Add other members
    if room_in.user_ids:
        for u_id in room_in.user_ids:
            if u_id != current_user.id:
                db.add(RoomUserLink(room_id=db_obj.id, user_id=u_id))
    
    if room_in.agent_ids:
        # internal design notes Stage 1: the system Router never joins a room.
        routers = (await db.execute(
            select(Agent.id).where(
                Agent.id.in_(room_in.agent_ids), Agent.system_role == "router"
            )
        )).scalars().all()
        if routers:
            await db.rollback()
            raise HTTPException(
                status_code=409,
                detail={"code": "system_agent",
                        "message": "Router is a system agent, it does not take part in rooms"},
            )
        for a_id in room_in.agent_ids:
            db.add(RoomAgentLink(room_id=db_obj.id, agent_id=a_id))
            db.add(AgentMemory(room_id=db_obj.id, agent_id=a_id))

    # T9: `workspaced` must create the root and attach its XFS project before
    # this room is visible.  A file upload may be the first operation — it
    # must not gain an unbounded directory merely because no container has
    # been started yet.  Production enables this path; the public self-hosted
    # Compose keeps level-1 accounting unless an operator opts in.
    from app.core.storage_projects import hard_quota_sync_enabled, sync_owner_project

    if hard_quota_sync_enabled():
        try:
            await sync_owner_project(current_user.id, db)
        except Exception as exc:
            await db.rollback()
            logger.exception("room creation: could not prepare hard storage quota")
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "storage_quota_unavailable",
                    "message": "Workspace storage is temporarily unavailable. Try again.",
                },
            ) from exc

    await db.commit()
    await db.refresh(db_obj)
    return db_obj

@router.get("/", response_model=List[RoomOut])
async def read_rooms(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    # Get rooms where user is a member, most recently active first (last
    # message timestamp, falling back to room creation time for rooms with
    # no messages yet) — see internal design notes Part 2, Stage 1.
    last_message_subq = (
        select(
            Message.room_id.label("room_id"),
            func.max(Message.timestamp).label("last_message_at"),
        )
        .group_by(Message.room_id)
        .subquery()
    )
    # Runtime state comes along in the same query — the sidebar dot must not
    # cost one request per room. One outer join, one row per room: Workspace
    # is 1:1 with Room (unique room_id), so this cannot fan out.
    # The dot's green is billing, not the container: an open usage interval.
    billing_active = (
        select(WorkspaceUsage.id)
        .where(
            WorkspaceUsage.workspace_id == Workspace.id,
            WorkspaceUsage.stopped_at.is_(None),
        )
        .exists()
    )
    result = await db.execute(
        select(Room, last_message_subq.c.last_message_at, Workspace.status, billing_active)
        .join(RoomUserLink, RoomUserLink.room_id == Room.id)
        .outerjoin(last_message_subq, last_message_subq.c.room_id == Room.id)
        .outerjoin(Workspace, Workspace.room_id == Room.id)
        .where(RoomUserLink.user_id == current_user.id)
        # Rooms with a message sort by its timestamp; rooms without one (both
        # sides NULL) fall back to creation time. Two order_by clauses instead
        # of COALESCE — Message.timestamp is a naive column while Room.created_at
        # is timezone-aware, and Postgres has no implicit cast between them.
        .order_by(last_message_subq.c.last_message_at.desc().nullslast(), Room.created_at.desc())
    )
    rooms_out: List[RoomOut] = []
    for room, last_message_at, workspace_status, is_billing in result.all():
        room_out = RoomOut.model_validate(room)
        room_out.last_message_at = last_message_at
        room_out.runtime_status = workspace_status
        room_out.billing_active = bool(is_billing)
        rooms_out.append(room_out)
    return rooms_out


@router.post("/{room_id}/open")
async def open_room(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Open a room and idempotently ensure its workspace is starting.

    The room itself is returned immediately; workspace startup runs in the
    background and lifecycle status changes are delivered over the room
    WebSocket.  This keeps room navigation responsive and makes retries safe.
    """
    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == current_user.id,
        )
    )
    if not membership.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a member of this room")

    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.deletion_started_at is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "room_deleting",
                "message": "Room deletion is in progress; workspace start is disabled.",
            },
        )

    # internal design notes T8: an open room is billed, so out of hours it
    # does not open (core/room_close.py, core/work_gates.py).
    from app.core import work_gates

    open_refusal = await work_gates.open_refusal(room_id, db)
    if open_refusal is not None:
        raise HTTPException(
            status_code=402,
            detail={"error_code": open_refusal.reason, "message": open_refusal.text},
        )

    from app.core.workspace_lifecycle import (
        WorkspaceStartError,
        _get_or_create_workspace,
        _workspace_payload,
        ensure_workspace_started,
    )

    # TR4, §5.5 (T11): an expired plan's rooms (archived after 30 days) and
    # rooms frozen after the plan shrank open for reading only — no start.
    # A renewed plan's archived room gets its files back first. Both answers
    # come from the plugin's gates (core/work_gates.py); none without one.
    read_only = await work_gates.room_read_only(room_id, db)
    if read_only is None:
        read_only = await work_gates.restore_on_open(room_id, db)
    if read_only is not None:
        ws = await _get_or_create_workspace(room_id, db)
        await db.commit()
        return {
            "room": RoomOut.model_validate(room),
            "workspace": _workspace_payload(ws),
            "read_only": read_only,
        }

    try:
        workspace = await ensure_workspace_started(
            room_id,
            db,
            quota_owner_id=current_user.id,
        )
    except WorkspaceStartError as exc:
        from app.core.disk_watch import PAUSE_REASON as DISK_EMERGENCY
        from app.core.room_storage import PAUSE_REASON as STORAGE_FULL

        if exc.error_code == STORAGE_FULL:
            # §5.3 (T3): the room opens without its workspace, so its files
            # can still be read, deleted and cleaned of dependencies.
            ws = await _get_or_create_workspace(room_id, db)
            await db.commit()
            return {
                "room": RoomOut.model_validate(room),
                "workspace": _workspace_payload(ws),
                "storage_full": {"reason": STORAGE_FULL, "message": exc.message},
            }
        if exc.error_code == DISK_EMERGENCY:
            # T0.16: stopped by the disk watchdog — the room opens without its
            # workspace, so files can be deleted and dependencies cleaned.
            ws = await _get_or_create_workspace(room_id, db)
            await db.commit()
            return {
                "room": RoomOut.model_validate(room),
                "workspace": _workspace_payload(ws),
                "disk_stopped": {"reason": DISK_EMERGENCY, "message": exc.message},
            }
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc

    # This endpoint intentionally has no response_model because workspace is
    # a lifecycle envelope.  Still serialize the room through the same
    # redacting schema as list/detail endpoints so legacy git secrets cannot
    # escape from this response.
    return {"room": RoomOut.model_validate(room), "workspace": workspace}


@router.post("/{room_id}/close")
async def close_room_endpoint(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """The stop button next to the room's dot: close the room.

    Agents paused, container stopped, the hour meter closed and the room
    dropped from every open view (core/room_close.py). This is the only way
    to stop the meter while the user is around: a room left open is in use
    and billed, even with no agent working.
    """
    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == current_user.id,
        )
    )
    if not membership.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a member of this room")
    if await db.get(Room, room_id) is None:
        raise HTTPException(status_code=404, detail="Room not found")

    from app.core.room_close import close_room

    result = await close_room(room_id, db)
    return {
        "ok": True,
        "paused_task_ids": [str(task_id) for task_id in result["paused_task_ids"]],
        "container_stopped": result["container_stopped"],
        "billing_closed": result["billing_closed"],
    }


@router.post("/{room_id}/switch")
async def switch_room(
    room_id: UUID,
    body: RoomSwitchIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Request an idempotent room switch with explicit quota consent."""
    if body.to_room_id == room_id:
        raise HTTPException(status_code=400, detail="Target room must differ")
    for candidate in (room_id, body.to_room_id):
        membership = await db.execute(
            select(RoomUserLink).where(
                RoomUserLink.room_id == candidate,
                RoomUserLink.user_id == current_user.id,
            )
        )
        if membership.scalar_one_or_none() is None:
            raise HTTPException(status_code=403, detail="Not a member of this room")

    deleting_rooms = (
        await db.execute(
            select(Room.id).where(
                Room.id.in_([room_id, body.to_room_id]),
                Room.deletion_started_at.is_not(None),
            )
        )
    ).scalars().all()
    if deleting_rooms:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "room_deleting",
                "message": "Room switch is unavailable while a room is being deleted.",
            },
        )

    from app.core.workspace_lifecycle import (
        WorkspaceStartError,
        request_switch,
    )

    try:
        return await request_switch(
            room_id,
            body.to_room_id,
            body.request_id,
            db,
            quota_owner_id=current_user.id,
            confirm_stop=body.confirm_stop,
            force_stop=body.force_stop,
        )
    except WorkspaceStartError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc

@router.get("/{room_id}/messages")
async def get_room_messages(
    room_id: UUID,
    limit: int = Query(300, le=1000),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    # Verify membership
    membership = await db.execute(
        select(RoomUserLink).where(RoomUserLink.room_id == room_id, RoomUserLink.user_id == current_user.id)
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")

    # Latest N messages, returned in chronological order. Ordering by
    # timestamp.asc() + limit would return the OLDEST N and silently drop
    # everything after the room outgrows the limit.
    result = await db.execute(
        select(Message).where(Message.room_id == room_id)
        .order_by(Message.timestamp.desc(), Message.id.desc()).limit(limit)
    )
    messages = list(reversed(result.scalars().all()))

    # Enrich with nicknames for agent messages
    out = []
    for m in messages:
        meta = m.event_metadata or {}
        if m.sender_user_id:
            sender_type = "user"
        elif m.sender_agent_id:
            # agent messages: check metadata override (master agents emit with sender_type=master)
            sender_type = meta.get("sender_type", "agent")
        else:
            # system/master events: use metadata if set, else infer from event type
            sender_type = meta.get("sender_type", "master")
        entry = {
            "id": str(m.id),
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
            "status": m.status,
            "sender_type": sender_type,
            "sender_id": str(m.sender_user_id or m.sender_agent_id or "hub-chat-master"),
            "nickname": meta.get("nickname"),
            "event_type": m.event_type or "MESSAGE",
            "task_id": str(m.task_id) if m.task_id else None,
            "metadata": meta,
        }
        if m.sender_agent_id:
            agent_res = await db.execute(select(Agent).where(Agent.id == m.sender_agent_id))
            agent = agent_res.scalars().first()
            if agent:
                entry["nickname"] = agent.nickname
        out.append(entry)
    return out


@router.get("/{room_id}/members")
async def get_room_members(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    # Verify membership
    membership = await db.execute(
        select(RoomUserLink).where(RoomUserLink.room_id == room_id, RoomUserLink.user_id == current_user.id)
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")

    room_res = await db.execute(select(Room).where(Room.id == room_id))
    room = room_res.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    users_res = await db.execute(
        select(User).join(RoomUserLink, RoomUserLink.user_id == User.id).where(RoomUserLink.room_id == room_id)
    )
    users = users_res.scalars().all()

    agents_res = await db.execute(
        select(Agent).join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id).where(RoomAgentLink.room_id == room_id)
    )
    agents = agents_res.scalars().all()

    master_agent_id = str((room.coordinator_config or {}).get("master_agent_id", ""))

    return {
        "has_coordinator": room.has_coordinator,
        "coordinator_config": room.coordinator_config or {},
        "users": [{"id": str(u.id), "name": u.full_name or u.email, "type": "user"} for u in users],
        "agents": [
            {
                "id": str(a.id),
                "name": a.nickname,
                "agent_uid": a.agent_uid,
                "type": "agent",
                "agent_type": a.agent_type,
                "short_description": a.short_description or "",
                "provider": (a.llm_config or {}).get("provider"),
                "is_master": room.has_coordinator and str(a.id) == master_agent_id,
            }
            for a in agents
        ],
        # Agents that worked here and were removed: tasks and plan steps still
        # name them, so the UI labels them "not in the room" instead of a bare id.
        "former_agents": [
            {
                "id": str(a.id),
                "name": a.nickname,
                "agent_uid": a.agent_uid,
                "type": "agent",
                "in_room": False,
            }
            for a in await former_agents(room_id, db)
        ],
    }


class RoomAgentAddIn(BaseModel):
    agent_id: UUID


def _membership_http_error(exc: RoomAgentMembershipError) -> HTTPException:
    status_code = {
        "agent_not_found": 404,
        "not_member": 404,
    }.get(exc.code, 409)
    return HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": str(exc), **exc.details},
    )


@router.post("/{room_id}/agents", status_code=201)
async def add_room_agent(
    room_id: UUID,
    body: RoomAgentAddIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Bring one of the user's agents into an existing room."""
    from app.core.events import emit_event

    room = await _require_room_member(room_id, current_user.id, db)
    try:
        agent = await add_agent_to_room(room, body.agent_id, current_user.id, db)
    except RoomAgentMembershipError as exc:
        await db.rollback()
        raise _membership_http_error(exc) from None
    await db.commit()
    await emit_event(
        room_id=room_id,
        event_type="ROOM_AGENTS_CHANGED",
        content=f"@{agent.nickname} added to the room",
        db=db,
        metadata={"action": "added", "agent_id": str(agent.id), "nickname": agent.nickname},
    )
    return {"agent_id": str(agent.id), "nickname": agent.nickname}


@router.delete("/{room_id}/agents/{agent_id}")
async def remove_room_agent(
    room_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Take an agent out of the room; its past work keeps pointing at it.

    409 while the agent has unfinished tasks here, is the room's ChatMaster or
    is mid-reply in a dialogue — the body names the blocking tasks.
    """
    from app.core.events import emit_event

    room = await _require_room_member(room_id, current_user.id, db)
    agent = await db.get(Agent, agent_id)
    try:
        result = await remove_agent_from_room(room, agent_id, db)
    except RoomAgentMembershipError as exc:
        await db.rollback()
        raise _membership_http_error(exc) from None
    await db.commit()
    nickname = agent.nickname if agent else str(agent_id)[:8]
    await emit_event(
        room_id=room_id,
        event_type="ROOM_AGENTS_CHANGED",
        content=f"@{nickname} removed from the room",
        db=db,
        metadata={
            "action": "removed",
            "agent_id": str(agent_id),
            "nickname": nickname,
            "closed_sessions": result.closed_sessions,
            "unassigned_plan_steps": result.unassigned_plan_steps,
        },
    )
    return {
        "agent_id": str(agent_id),
        "closed_sessions": result.closed_sessions,
        "unassigned_plan_steps": result.unassigned_plan_steps,
    }


@router.put("/{room_id}", response_model=RoomOut)
async def update_room(
    room_id: UUID,
    room_in: RoomUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    # Check if user is creator or member
    result = await db.execute(
        select(Room).join(RoomUserLink).where(Room.id == room_id, RoomUserLink.user_id == current_user.id)
    )
    db_obj = result.scalars().first()
    if not db_obj:
        raise HTTPException(status_code=404, detail="Room not found or access denied")
    
    if room_in.name is not None:
        db_obj.name = room_in.name
    if room_in.type is not None:
        db_obj.type = room_in.type
    if room_in.has_coordinator is not None:
        db_obj.has_coordinator = room_in.has_coordinator
    if room_in.coordinator_config is not None:
        db_obj.coordinator_config = room_in.coordinator_config
        flag_modified(db_obj, 'coordinator_config')
    if room_in.git_config is not None:
        db_obj.git_config = room_in.git_config
        flag_modified(db_obj, 'git_config')
    if room_in.vps_access_mode is not None:
        db_obj.vps_access_mode = room_in.vps_access_mode

    # Update memberships if provided
    if room_in.user_ids is not None:
        # Clear and re-add (simple approach)
        await db.execute(delete(RoomUserLink).where(RoomUserLink.room_id == room_id))
        db.add(RoomUserLink(room_id=room_id, user_id=current_user.id))
        for u_id in room_in.user_ids:
            if u_id != current_user.id:
                db.add(RoomUserLink(room_id=room_id, user_id=u_id))

    if room_in.agent_ids is not None:
        # Same rules as the per-agent endpoints: only own agents join, and an
        # agent with unfinished work (or the ChatMaster) cannot leave.
        old_agent_ids = await room_agent_ids(room_id, db)
        new_agent_ids = set(room_in.agent_ids)
        try:
            for a_id in old_agent_ids - new_agent_ids:
                await remove_agent_from_room(db_obj, a_id, db)
            for a_id in new_agent_ids - old_agent_ids:
                await add_agent_to_room(db_obj, a_id, current_user.id, db)
        except RoomAgentMembershipError as exc:
            await db.rollback()
            raise _membership_http_error(exc) from None

    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


@router.patch("/{room_id}/state", response_model=RoomOut)
async def update_room_state(
    room_id: UUID,
    body: RoomStateUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(
        select(Room).join(RoomUserLink).where(Room.id == room_id, RoomUserLink.user_id == current_user.id)
    )
    room = result.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found or access denied")

    allowed = ROOM_TRANSITIONS.get(room.state, [])
    if body.state not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot transition from '{room.state}' to '{body.state}'. Allowed: {[s.value for s in allowed]}",
        )

    room.state = body.state
    db.add(room)
    await db.commit()
    await db.refresh(room)
    return room


@router.delete("/{room_id}", status_code=204)
async def delete_room(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    result = await db.execute(
        select(Room)
        .where(Room.id == room_id, Room.creator_id == current_user.id)
        .with_for_update()
    )
    room = result.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found or access denied")

    # Mark first and commit separately from the destructive external call. A
    # backend restart can then be reconciled safely, and concurrent open/start
    # requests see the marker before they can create a replacement container.
    if room.deletion_started_at is None:
        room.deletion_started_at = datetime.now(timezone.utc)
        db.add(room)
        await db.commit()

    from app.core import workspaced_client as wsd

    try:
        # The workspaced operation is idempotent: it removes the container
        # even when no Workspace row exists and deletes only this room's
        # private volume subtree. If it fails, keep the room and its marker so
        # a retry/reconciler can finish instead of losing metadata silently.
        await wsd.destroy_workspace(str(room_id))
        # §5.5 (T11): a room archived after its plan ended takes its archive along.
        archived_at = (
            await db.execute(select(Workspace.archived_at).where(Workspace.room_id == room_id))
        ).scalar_one_or_none()
        if isinstance(archived_at, datetime):
            await wsd.delete_workspace_archive(str(room_id))
    except (httpx.HTTPError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "workspace_cleanup_failed",
                "message": "Workspace cleanup failed; the room was not deleted. Please retry.",
                "reason": str(exc),
            },
        ) from exc
    except Exception as exc:
        # Keep the marker durable for unexpected workspaced client errors as
        # well (for example a malformed internal response). The reconcile loop
        # can retry the external cleanup after a backend restart.
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "workspace_cleanup_failed",
                "message": "Workspace cleanup failed; the room was not deleted. Please retry.",
                "reason": str(exc),
            },
        ) from exc

    from app.core.room_delete import delete_room_rows

    await delete_room_rows(room, db)


@router.post("/{room_id}/plan/confirm", status_code=200)
async def confirm_plan(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Room).where(Room.id == room_id))
    room = result.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    coord_cfg = dict(room.coordinator_config or {})
    pending_plan = coord_cfg.get("pending_plan")
    if not pending_plan:
        raise HTTPException(status_code=400, detail="No pending plan")

    from app.core.events import emit_event
    from app.core.chat_master import approve_and_start

    # Capture config before popping pending_plan
    config = {k: v for k, v in coord_cfg.items() if k not in ("pending_plan",)}
    coord_cfg.pop("pending_plan", None)
    room.coordinator_config = coord_cfg
    db.add(room)
    await db.commit()

    # Build status message
    tasks = pending_plan.get("tasks", [])
    task_summaries = ", ".join(
        f'“{t["title"]}” → @{t["agent_nickname"]}' for t in tasks
    ) if tasks else "the plan"
    status_msg = f"{current_user.full_name or current_user.email} approved {task_summaries}"

    await emit_event(
        room_id=room_id,
        event_type="ROOM_STATE_CHANGED",
        content=status_msg,
        db=db,
        metadata={"new_state": "execution"},
    )

    result2 = await db.execute(
        select(Agent).join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
        .where(RoomAgentLink.room_id == room_id)
    )
    room_agents = result2.scalars().all()

    await approve_and_start(room, pending_plan, room_agents, config, db)
    return {"status": "confirmed", "message": status_msg}


@router.post("/{room_id}/plan/reject", status_code=200)
async def reject_plan(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(select(Room).where(Room.id == room_id))
    room = result.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    coord_cfg = dict(room.coordinator_config or {})
    if not coord_cfg.get("pending_plan"):
        raise HTTPException(status_code=400, detail="No pending plan")

    from app.models.room import RoomState, ROOM_TRANSITIONS
    from app.core.events import emit_event

    coord_cfg.pop("pending_plan", None)
    room.coordinator_config = coord_cfg
    if RoomState.PLANNING in ROOM_TRANSITIONS.get(room.state, []):
        room.state = RoomState.PLANNING
    db.add(room)
    await db.commit()

    status_msg = f"{current_user.full_name or current_user.email} rejected the plan"
    await emit_event(
        room_id=room_id,
        event_type="ROOM_STATE_CHANGED",
        content=status_msg,
        db=db,
        metadata={"new_state": "planning"},
    )
    return {"status": "rejected", "message": status_msg}


@router.post("/{room_id}/questions/{question_id}/answer", status_code=200)
async def answer_agent_question(
    room_id: UUID,
    question_id: str,
    answer_in: QuestionAnswerIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Resolve an agent's pending ask_user question with the user's selected options."""
    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == current_user.id,
        )
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")

    from app.core import ask_user_registry as _reg

    resolved = _reg.resolve(
        question_id,
        {
            "selected_options": answer_in.selected_options,
            "free_text": answer_in.free_text,
        },
    )
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="Question not found or already answered (may have timed out)",
        )

    return {"status": "ok", "question_id": question_id}
