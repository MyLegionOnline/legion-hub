from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.events import emit_event
from app.db.session import get_session
from app.models.agent_memory import AgentMemory
from app.models.room import Room, RoomUserLink, RoomAgentLink
from app.models.agent import Agent
from app.models.user import User
from app.schemas.agent_memory import AgentMemoryOut, AgentMemoryUpdate, AgentMemoryMerge

router = APIRouter()

MAX_CHARS = 8000  # ~2000 tokens


def _count_tokens(text: str) -> int:
    return len(text) // 4


async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _verify_agent_in_room(room_id: UUID, agent_id: UUID, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomAgentLink).where(
            RoomAgentLink.room_id == room_id,
            RoomAgentLink.agent_id == agent_id,
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=404, detail="Agent is not a member of this room")


async def _get_memory_or_404(room_id: UUID, agent_id: UUID, db: AsyncSession) -> AgentMemory:
    result = await db.execute(
        select(AgentMemory).where(
            AgentMemory.room_id == room_id,
            AgentMemory.agent_id == agent_id,
        )
    )
    memory = result.scalars().first()
    if not memory:
        # Agent memory is created when the agent joins the room — missing here means
        # the link predates this feature. Self-heal instead of erroring the UI.
        await _verify_agent_in_room(room_id, agent_id, db)
        memory = AgentMemory(room_id=room_id, agent_id=agent_id)
        memory.token_count = _count_tokens(memory.content)
        db.add(memory)
        await db.commit()
        await db.refresh(memory)
    return memory


async def _get_room(room_id: UUID, db: AsyncSession) -> Room:
    res = await db.execute(select(Room).where(Room.id == room_id))
    room = res.scalars().first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


@router.get("/{room_id}/agents/{agent_id}/memory", response_model=AgentMemoryOut)
async def get_agent_memory(
    room_id: UUID,
    agent_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    return await _get_memory_or_404(room_id, agent_id, db)


@router.put("/{room_id}/agents/{agent_id}/memory", response_model=AgentMemoryOut)
async def update_agent_memory_endpoint(
    room_id: UUID,
    agent_id: UUID,
    body: AgentMemoryUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    memory = await _get_memory_or_404(room_id, agent_id, db)

    content = body.content[:MAX_CHARS]
    memory.content = content
    memory.token_count = _count_tokens(content)
    memory.updated_at = datetime.utcnow()

    db.add(memory)
    await db.commit()
    await db.refresh(memory)
    return memory


@router.post("/{room_id}/agents/{agent_id}/memory/merge", response_model=AgentMemoryOut)
async def merge_agent_memory_endpoint(
    room_id: UUID,
    agent_id: UUID,
    body: AgentMemoryMerge,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    await _get_room(room_id, db)
    memory = await _get_memory_or_404(room_id, agent_id, db)

    res = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = res.scalars().first()
    llm_config = (agent.llm_config or None) if agent else None

    from app.core.chat_master import _merge_agent_memory_content

    merged = await _merge_agent_memory_content(memory.content, body.session_summary, llm_config, agent)
    merged = merged[:MAX_CHARS]
    memory.content = merged
    memory.token_count = _count_tokens(merged)
    memory.updated_at = datetime.utcnow()

    db.add(memory)
    await db.commit()
    await db.refresh(memory)

    await emit_event(
        room_id=room_id,
        event_type="AGENT_MEMORY_UPDATED",
        content="Agent memory updated",
        db=db,
        metadata={"agent_id": str(agent_id), "token_count": memory.token_count},
    )
    return memory
