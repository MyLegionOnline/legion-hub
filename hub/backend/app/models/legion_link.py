"""A self-hosted Hub user's link to a Legion account (internal design notes §2.3, stage 6).

The user pastes an API key issued in their Legion profile; the Hub checks it
against Market (``GET /api/account/me``) and keeps it encrypted with the same
keyring and AEAD as the secret store. Market's payment gateway will take the
same key later (payments plan, stage 3b). One link per user: on a Hub with
several local users each one spends from their own account.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LegionLink(SQLModel, table=True):
    __tablename__ = "legion_link"

    user_id: UUID = Field(
        sa_column=sa.Column(sa.Uuid, sa.ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    )
    # The key itself: AEAD ciphertext, never stored or logged in clear.
    ciphertext: bytes = Field(sa_column=sa.Column(sa.LargeBinary, nullable=False))
    key_id: str = Field(max_length=64)
    aad_version: int = 1
    key_prefix: str = Field(max_length=32)  # "lgn_xxxxxxxx", what Market shows next to the key
    # What Market answered for the key at the last check.
    account_id: UUID
    email: str
    display_name: Optional[str] = None
    # linked: Market accepted the key at the last check; rejected: Market said
    # it is revoked or invalid (or the account is blocked).
    status: str = Field(default="linked", max_length=16)
    checked_at: datetime = Field(
        default_factory=_utcnow, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )
