"""Durable OAuth attempts; renewable credentials live in Secrets Vault."""
from datetime import datetime, timezone
from uuid import UUID, uuid4
import sqlalchemy as sa
from sqlmodel import Field, SQLModel

class ChatGPTHost(SQLModel, table=True):
    __tablename__ = "chatgpt_host"
    id: int = Field(default=1, primary_key=True)
    host_id: str = Field(default_factory=lambda: f"urn:uuid:{uuid4()}")

class ChatGPTAuthorization(SQLModel, table=True):
    __tablename__ = "chatgpt_authorization"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    owner_id: UUID = Field(sa_column=sa.Column(sa.Uuid, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True))
    connection_id: UUID = Field(sa_column=sa.Column(sa.Uuid, sa.ForeignKey("provider_connection.id", ondelete="CASCADE"), nullable=False))
    ticket_hash: str = Field(index=True, unique=True)
    expires_at: datetime = Field(sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False))
    ciphertext: bytes = Field(sa_column=sa.Column(sa.LargeBinary, nullable=False))
    key_id: str
    aad_version: int = 1
    redirect_uri: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False))
