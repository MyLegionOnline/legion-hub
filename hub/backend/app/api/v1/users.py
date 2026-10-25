from datetime import datetime
import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core import legion_auth, legion_wallet, mcp_credits
from app.core import security
from app.db.session import get_session
from app.models.user import User
from app.schemas.user import UserOut, UserUpdate
from app.core.workspace_quota import get_usage_summary
from app.websockets.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


async def _user_id_from_token(token: str, db: AsyncSession) -> Optional[UUID]:
    """Same JWT check as ``deps.get_current_user``, minus the HTTP errors.

    Users only: agent tokens are rejected. An agent has no room list to keep
    fresh, and this channel is keyed by user, not by room membership of the
    socket owner.
    """
    try:
        payload = security.decode_access_token(token)
        user_id = UUID(str(payload.get("sub")))
    except (JWTError, ValueError, TypeError):
        return None
    user = (await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))).scalars().first()
    if user is None or await legion_auth.session_revoked(db, user.legion_account_id, payload):
        return None
    return user.id


@router.websocket("/me/events")
async def user_events_ws(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_session),
):
    """Per-user event channel (internal design notes).

    One socket per logged-in tab, carrying events of every room the user is a
    member of — today only the sidebar's ``room_runtime`` frames. Server →
    client only; whatever the client sends is ignored and serves just to
    notice the disconnect.
    """
    await websocket.accept()

    user_id = await _user_id_from_token(token, db)
    if not user_id:
        await websocket.close(code=1008)
        return

    # Release the pooled DB connection before idling on the socket — same as
    # the room socket in chat.py. Otherwise every open tab pins a connection
    # "idle in transaction" and a reconnect storm after a backend restart
    # exhausts the pool (internal design notes).
    await db.rollback()

    manager.connect_user_events(websocket, user_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # an odd close must not leak a registry entry
        logger.debug("user events socket for %s ended: %s", user_id, exc)
    finally:
        manager.disconnect_user_events(websocket, user_id)


@router.get("/me", response_model=UserOut)
async def read_user_me(
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    return current_user

@router.get("/me/usage")
async def get_my_usage(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return current user's workspace quota and usage for this calendar month.

    ``mcp_credits`` is fetched through its own short-lived session (see
    ``app/core/mcp_credits.py``) rather than the request's ``db`` — the wallet
    roll it may perform on read needs to commit independently of whatever
    else this request's session is doing.
    """
    summary = await get_usage_summary(current_user, db)
    summary["mcp_credits"] = await mcp_credits.wallet_summary(current_user.id)
    # "legion": Market MCP is paid from the Legion account's central wallet
    # (payments plan §3.10), not from the wallet above.
    summary["mcp_credits_wallet"] = "legion" if legion_wallet.central() else "hub"
    return summary


@router.get("/me/mcp-credits/ledger")
async def get_my_mcp_credits_ledger(
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """This user's own MCP credit transactions for the current weekly window —
    the per-call breakdown behind ``spent_this_period`` on ``/me/usage``, so
    the Settings "Plan and limits" tab can show what the spend was for.

    Scoped to entries since the wallet's current ``period_started_at``; older
    history is the Admin Center's concern, not the end-user view.
    """
    wallet = await mcp_credits.wallet_summary(current_user.id)
    period_started_at = datetime.fromisoformat(wallet["period_started_at"])
    entries, total = await mcp_credits.list_ledger(
        current_user.id, limit=limit, offset=offset, since=period_started_at
    )
    for entry in entries:
        entry["actor"] = mcp_credits.user_facing_actor(entry.get("actor"))
    return {
        "period_started_at": wallet["period_started_at"],
        "period_ends_at": wallet["period_ends_at"],
        "ledger": entries,
        "ledger_total": total,
        "limit": limit,
        "offset": offset,
    }


@router.put("/me", response_model=UserOut)
async def update_user_me(
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    if (user_in.email or user_in.password) and current_user.legion_account_id and legion_auth.legion_enabled():
        # The email and password of a Legion user live in the Legion account;
        # a change here would only fork the fallback copy.
        raise HTTPException(status_code=409, detail="Change your email and password in your Legion account")
    if user_in.email:
        current_user.email = user_in.email
    if user_in.full_name:
        current_user.full_name = user_in.full_name
    if user_in.password:
        current_user.hashed_password = security.get_password_hash(user_in.password)
    if user_in.two_factor_enabled is not None:
        current_user.two_factor_enabled = user_in.two_factor_enabled
    
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user
