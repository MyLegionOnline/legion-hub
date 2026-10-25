"""Our Hub's copy of a Legion account's credentials (internal design notes §2.2).

Market pushes it (PUT /internal/accounts/{id}/credentials) on registration,
password change or reset, sign-out everywhere, block and unblock. The Hub
uses it for the fallback password form while Market is down and to refuse a
blocked account or a stale id_token. Keyed by the account, not the user: a
person who registered in Legion but has not opened the Hub yet still gets in
through the fallback form, and gets their user then.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class LegionCredential(SQLModel, table=True):
    __tablename__ = "legion_credential"

    account_id: UUID = Field(primary_key=True)
    email: str = Field(index=True)  # lower-cased, as Market keeps it
    password_hash: Optional[str] = None  # NULL: an account without a password (Google/GitHub only)
    token_version: int = 0
    status: str = "active"  # active | blocked
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
