"""An invitation to register on a self-hosted Hub (internal design notes stage 4.3).

The owner creates it, the invitee presents the token at registration. Only the
SHA-256 of the token is stored, so a database leak does not hand out working
invitations. One invitation makes one user.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RegistrationInvite(SQLModel, table=True):
    __tablename__ = "registration_invite"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    token_hash: str = Field(unique=True, index=True)
    # Lower-cased. When set, only this address can use the invitation.
    email: Optional[str] = None
    created_by: UUID
    created_at: datetime = Field(default_factory=_now, sa_column=Column(DateTime(timezone=True), nullable=False))
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    used_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    used_by: Optional[UUID] = None
    revoked_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
