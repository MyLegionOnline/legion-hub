from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.db.release import release_db
from app.core.events import emit_event
from app.core.secrets.runtime import resolve_agent_secret
from app.db.session import get_session
from app.models.project_memory import ProjectMemory, DEFAULT_TEMPLATE
from app.models.room import Room, RoomUserLink
from app.models.agent import Agent
from app.models.user import User
from app.schemas.memory import ProjectMemoryOut, ProjectMemoryUpdate, ProjectMemoryMerge

router = APIRouter()

MAX_CHARS = 8000  # ~2000 tokens


def _count_tokens(text: str) -> int:
    return len(text) // 4


async def _get_or_create_memory(room_id: UUID, db: AsyncSession) -> ProjectMemory:
    result = await db.execute(
        select(ProjectMemory).where(ProjectMemory.room_id == room_id)
    )
    memory = result.scalars().first()
    if not memory:
        memory = ProjectMemory(room_id=room_id)
        memory.token_count = _count_tokens(memory.content)
        db.add(memory)
        await db.commit()
        await db.refresh(memory)
    return memory


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


@router.get("/{room_id}/memory", response_model=ProjectMemoryOut)
async def get_memory(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    return await _get_or_create_memory(room_id, db)


@router.put("/{room_id}/memory", response_model=ProjectMemoryOut)
async def update_memory(
    room_id: UUID,
    body: ProjectMemoryUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    memory = await _get_or_create_memory(room_id, db)

    content = body.content[:MAX_CHARS]
    memory.content = content
    memory.token_count = _count_tokens(content)
    memory.updated_at = datetime.utcnow()

    db.add(memory)
    await db.commit()
    await db.refresh(memory)
    return memory


@router.post("/{room_id}/memory/merge", response_model=ProjectMemoryOut)
async def merge_memory(
    room_id: UUID,
    body: ProjectMemoryMerge,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    room = await _get_room(room_id, db)
    memory = await _get_or_create_memory(room_id, db)

    coordinator = await _get_coordinator_llm_config(room, db)
    llm_config = coordinator.llm_config if coordinator else None

    if llm_config:
        merged = await _merge_via_llm(
            memory.content, body.session_summary, llm_config, db=db, agent=coordinator
        )
    else:
        merged = _merge_simple(memory.content, body.session_summary)

    merged = merged[:MAX_CHARS]
    memory.content = merged
    memory.token_count = _count_tokens(merged)
    memory.updated_at = datetime.utcnow()

    db.add(memory)
    await db.commit()
    await db.refresh(memory)

    await emit_event(
        room_id=room_id,
        event_type="MEMORY_UPDATED",
        content="Project memory updated",
        db=db,
        metadata={"token_count": memory.token_count},
    )
    return memory


async def _get_coordinator_llm_config(room: Room, db: AsyncSession):
    config = room.coordinator_config or {}
    master_id = config.get("master_agent_id")
    if not master_id or not room.has_coordinator:
        return None
    try:
        agent_id = UUID(str(master_id))
    except (ValueError, AttributeError):
        return None
    res = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = res.scalars().first()
    return agent


async def _merge_via_llm(
    current: str,
    summary: str,
    config: dict,
    *,
    db: AsyncSession,
    agent: Agent,
) -> str:
    from app.core.chat_master import get_llm_completion

    messages = [
        {
            "role": "system",
            "content": (
                "You are a project memory manager. "
                "Merge the session summary into the existing project memory. "
                "Keep the markdown structure. Remove outdated info. "
                "Be concise — the result must fit in 2000 tokens."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Current Project Memory\n\n{current}\n\n"
                f"## Session Summary to Merge\n\n{summary}\n\n"
                "Return only the updated project memory markdown."
            ),
        },
    ]
    try:
        secret_value = await resolve_agent_secret(
            db=db,
            owner_id=agent.owner_id,
            agent_id=agent.id,
            config=config,
            request_id=agent.id,
        )
        try:
            await release_db(db, reason="memory/merge")
            return (await get_llm_completion(config, messages, secret_value=secret_value)).text
        finally:
            secret_value = None
    except Exception:
        return _merge_simple(current, summary)


def _merge_simple(current: str, summary: str) -> str:
    separator = "\n\n---\n\n## Session Update\n\n"
    merged = current.rstrip() + separator + summary.strip()
    if len(merged) > MAX_CHARS:
        overflow = len(merged) - MAX_CHARS
        merged = merged[overflow:]
    return merged
