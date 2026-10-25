from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import SQLModel, Field


class AgentAvatar(SQLModel, table=True):
    """Custom per-agent avatar image (binary content stored out-of-band of
    the hot `agent` row, by design — see internal design notes §3 R1).

    `agent_id` is the primary key: each agent has at most one avatar, so no
    surrogate id is needed. `ON DELETE CASCADE` is declared in the SQL
    migration (stage 1) rather than via SQLModel, because SQLAlchemy only
    emits `ondelete` on `create_all` (clean DB) — for already-deployed
    databases the cascade must come from the migration's CREATE TABLE.

    `content` holds base64-encoded image bytes in the default `db` storage
    backend; it is an unused placeholder when `AVATAR_STORAGE_BACKEND=s3`
    (bytes then live under `avatars/{owner_id}/{agent_id}` in S3, see stage 2).
    """

    __tablename__ = "agentavatar"

    agent_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("agent.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        )
    )
    mime: str
    content: str
    size_bytes: int
    updated_at: datetime = Field(default_factory=datetime.utcnow)
