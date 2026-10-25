"""Hub-owned registry records for Legion Market items.

The Market service owns its catalogue.  Hub owns which catalogue snapshots a
user has installed and the room/agent assignments, so these models deliberately
do not have a database foreign key to the Market database.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel, UniqueConstraint


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserInstalledItem(SQLModel, table=True):
    """A Market or custom item activated in one user's global registry."""

    __tablename__ = "user_installed_item"
    __table_args__ = (
        UniqueConstraint("user_id", "market_item_id", name="uq_user_installed_market_item"),
        sa.CheckConstraint("type IN ('skill', 'mcp')", name="ck_user_installed_item_type"),
        sa.CheckConstraint("status IN ('active', 'disabled')", name="ck_user_installed_item_status"),
        sa.CheckConstraint(
            "isolation IS NULL OR isolation IN ('stateless', 'tenant_aware')",
            name="ck_user_installed_item_isolation",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    type: str
    is_custom: bool = Field(default=False)
    market_item_id: Optional[UUID] = Field(default=None, index=True)
    market_slug: Optional[str] = Field(default=None, max_length=100)
    name: str = Field(max_length=255)
    isolation: Optional[str] = Field(default=None)
    custom_config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    skill_id: Optional[UUID] = Field(default=None, foreign_key="skill.id", index=True)
    status: str = Field(default="active")
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


class UserDefaultProvision(SQLModel, table=True):
    """Ledger of default Market items already offered to a user.

    Prevents a default from reappearing after the user deletes it, and lets a
    newly-flagged `is_default` market item be provisioned exactly once per
    user (internal design notes stage 3 and vision plan stage
    2). Deliberately never
    deleted when the user removes the resulting UserInstalledItem — "offered
    once, user decided" — and has no FK to market_items since that table
    lives in a different database.
    """

    __tablename__ = "user_default_provision"

    user_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    market_item_id: UUID = Field(sa_column=sa.Column(sa.Uuid, primary_key=True, nullable=False))
    market_slug: Optional[str] = Field(default=None, max_length=100)
    provisioned_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


class RoomMcpServer(SQLModel, table=True):
    """MCP servers available to every agent executing in a room."""

    __tablename__ = "room_mcp_server"

    room_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("room.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    user_installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    added_by: UUID = Field(foreign_key="user.id")
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


class AgentSkill(SQLModel, table=True):
    """Skills explicitly assigned to an agent."""

    __tablename__ = "agent_skill"

    agent_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("agent.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    user_installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
