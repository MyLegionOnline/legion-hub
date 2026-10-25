"""``POST /api/v1/router/test`` — "Test a route" (internal design notes §5).

Routes a sample message in one of the user's rooms exactly as a real message
would be routed right now — the room's Router, its open session for this user,
its goals and waiting tasks — but as a dry run: no events, no metrics, no
router_decision_log row and no action on the decision.  The Jev / LLM calls
are real, so the latency and cost shown are real too.
"""

from __future__ import annotations

import time
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.api.v1.rooms import _require_room_member
from app.db.session import get_session
from app.models.user import User

router = APIRouter()


class RouterTestIn(BaseModel):
    room_id: UUID
    text: str = Field(min_length=1, max_length=4000)
    # Off: route as if the user had no open conversation in the room.
    use_open_session: bool = True


class RouterTestOut(BaseModel):
    route: str
    confidence: float
    reasoning: str
    engine: str
    escalation_reason: Optional[str] = None
    model: Optional[str] = None
    latency_ms: int
    cost_usd: float
    jev_route: Optional[str] = None
    jev_confidence: Optional[float] = None
    jev_latency_ms: Optional[int] = None
    jev_probabilities: Optional[Any] = None
    jev_error: Optional[str] = None
    fail_reason: Optional[str] = None
    matched_goal_id: Optional[str] = None
    matched_task_id: Optional[str] = None
    open_session_id: Optional[str] = None
    # The Router of the room's owner routes the room — not necessarily yours.
    own_router: bool


@router.post("/test", response_model=RouterTestOut)
async def test_route(
    body: RouterTestIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    from app.core.router import route_user_message

    room = await _require_room_member(body.room_id, current_user.id, db)

    open_ctx = None
    if body.use_open_session:
        from app.core.user_sessions import open_session_context
        from app.repositories.session import SessionRepository

        existing = await SessionRepository(db).find_open_user_session(
            room.id, current_user.id
        )
        if existing is not None:
            open_ctx = await open_session_context(existing, db)

    report: dict[str, Any] = {}
    t0 = time.monotonic()
    await route_user_message(
        room, body.text, db, open_session=open_ctx, dry_run=True, report=report
    )
    report.setdefault("latency_ms", round((time.monotonic() - t0) * 1000))
    return RouterTestOut(**report, own_router=room.creator_id == current_user.id)
