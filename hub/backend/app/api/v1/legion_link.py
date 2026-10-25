"""Settings → Legion account on a self-hosted Hub (internal design notes §2.3, stage 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.core import legion_auth, legion_link, legion_wallet
from app.core.legion_link import LegionLinkError
from app.db.session import get_session
from app.models.legion_link import LegionLink
from app.models.user import User

router = APIRouter()


class LinkRequest(BaseModel):
    key: str = Field(min_length=1, max_length=200)


def _refuse(exc: LegionLinkError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail={"code": exc.code, "message": str(exc)})


def _require_available() -> None:
    if not legion_link.available():
        raise HTTPException(status_code=409, detail={
            "code": "legion_sign_in",
            "message": "This Hub signs in with Legion accounts; there is nothing to link.",
        })


def _state(row, *, check: str | None = None, wallet: dict | None = None) -> dict:
    return {
        "available": legion_link.available(),
        "profile_url": legion_link.profile_url(),
        "legal_urls": legion_auth.legal_urls(),
        "link": legion_link.link_out(row),
        # None: Market answered this request (or was not asked); else why not.
        "check_error": check,
        # Market MCP on this Hub is paid from the account's Own Hubs wallet
        # (payments plan §3.10, MCP_CREDITS_WALLET=legion).
        "pays_market_mcp": legion_wallet.central(),
        "wallet": wallet,
    }


async def _own_hubs_wallet(row) -> dict | None:
    """The Own Hubs wallet of the linked account, read from Market by the key;
    ``{"error": code}`` when Market cannot say."""
    if row is None or row.status != "linked":
        return None
    try:
        body = await legion_wallet.fetch_wallet(legion_link.stored_key(row).reveal())
    except LegionLinkError as exc:
        return {"error": exc.code}
    wallet = body["wallet"]
    legion_wallet.forget_balance(row.user_id)
    return {
        "balance": wallet.get("balance"),
        "purchased_total": wallet.get("purchased_total"),
        "spent_total": wallet.get("spent_total"),
        "ledger": (wallet.get("ledger") or [])[:10],
    }


@router.get("")
async def get_link(
    refresh: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict:
    if not legion_link.available():
        return _state(None)
    row = await db.get(LegionLink, current_user.id)
    check = None
    if row is not None and refresh:
        try:
            check = await legion_link.refresh(db, row)
        except LegionLinkError as exc:
            check = exc.code
    return _state(row, check=check, wallet=await _own_hubs_wallet(row) if refresh else None)


@router.put("")
async def put_link(
    body: LinkRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict:
    _require_available()
    try:
        row = await legion_link.link(db, current_user.id, body.key)
    except LegionLinkError as exc:
        raise _refuse(exc) from None
    return _state(row, wallet=await _own_hubs_wallet(row))


@router.delete("")
async def delete_link(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict:
    removed = await legion_link.unlink(db, current_user.id)
    return {**_state(None), "removed": removed}
