"""The owner's invitations to a self-hosted Hub (internal design notes stage 4.3).

Under ``registration.mode = "invite"`` a person registers with the token the
owner hands them. The token is shown once, when it is made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.core import registration
from app.db.session import get_session
from app.models.registration_invite import RegistrationInvite
from app.models.user import User

router = APIRouter()


class InviteCreate(BaseModel):
    email: Optional[EmailStr] = None
    ttl_days: int = Field(default=registration.INVITE_DEFAULT_TTL_DAYS, ge=1, le=registration.INVITE_MAX_TTL_DAYS)


class InviteOut(BaseModel):
    id: UUID
    email: Optional[str] = None
    status: str
    created_at: datetime
    expires_at: datetime
    used_at: Optional[datetime] = None


class InviteCreated(InviteOut):
    token: str


def _out(invite: RegistrationInvite) -> dict[str, Any]:
    return {
        "id": invite.id, "email": invite.email, "status": registration.invite_status(invite),
        "created_at": invite.created_at, "expires_at": invite.expires_at, "used_at": invite.used_at,
    }


@router.get("/invites", response_model=list[InviteOut])
async def list_invites(
    db: AsyncSession = Depends(get_session), owner: User = Depends(deps.get_current_owner),
) -> Any:
    rows = (await db.execute(
        select(RegistrationInvite).order_by(RegistrationInvite.created_at.desc()).limit(200)
    )).scalars().all()
    return [_out(row) for row in rows]


@router.post("/invites", response_model=InviteCreated, status_code=201)
async def create_invite(
    body: InviteCreate, db: AsyncSession = Depends(get_session), owner: User = Depends(deps.get_current_owner),
) -> Any:
    """Make an invitation. ``token`` appears in this answer only."""
    if registration.get_mode() != "invite":
        raise HTTPException(
            status_code=409,
            detail={"code": "invites_off",
                    "message": 'Invitations are only used when registration is set to "invite" '
                               "([registration] mode in legion.toml or LEGION_REGISTRATION)."},
        )
    invite, token = await registration.create_invite(
        db, owner_id=owner.id, email=body.email, ttl_days=body.ttl_days,
    )
    return {**_out(invite), "token": token}


@router.delete("/invites/{invite_id}", status_code=204)
async def revoke_invite(
    invite_id: UUID, db: AsyncSession = Depends(get_session), owner: User = Depends(deps.get_current_owner),
) -> None:
    invite = await db.get(RegistrationInvite, invite_id)
    if invite is None:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if registration.invite_status(invite) != "pending":
        raise HTTPException(status_code=409, detail="Only a pending invitation can be revoked")
    invite.revoked_at = datetime.now(timezone.utc)
    await db.commit()
