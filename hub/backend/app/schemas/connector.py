from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, computed_field

from app.core import connector_compat


class ConnectorCreate(BaseModel):
    label: str = ""
    # Optional: link the new connector to an already-existing external Agent
    # in one step, so the "Connect an external agent" button on the agent
    # card only needs one request .
    agent_id: Optional[UUID] = None


class ConnectorOut(BaseModel):
    id: UUID
    owner_id: UUID
    label: str
    status: str
    version: Optional[str] = None
    capabilities: Dict[str, Any] = {}
    last_seen_at: Optional[datetime] = None
    host_fingerprint: Optional[str] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def protocol_version(self) -> Optional[str]:
        """Protocol the connector announced in `hello`; `None` before the first one."""
        value = (self.capabilities or {}).get("protocol_version")
        return None if value is None else str(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compatibility(self) -> str:
        """`ok` / `protocol_too_old` / `protocol_too_new` / `unknown` (no hello yet)."""
        return connector_compat.compatibility(self.version, self.capabilities)


class ConnectorPairingOut(BaseModel):
    connector: ConnectorOut
    pairing_code: str
    expires_at: datetime
    install_command: str


class ConnectorPairRequest(BaseModel):
    pairing_code: str


class ConnectorPairResponse(BaseModel):
    connector_id: UUID
    refresh_token: str


class ConnectorTokenRequest(BaseModel):
    refresh_token: str


class ConnectorTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ConnectorRoomGrantOut(BaseModel):
    id: UUID
    connector_id: UUID
    room_id: UUID
    granted_by: UUID
    granted_at: datetime
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConnectorRoomAccessOut(BaseModel):
    """A Room the connector currently has access to (the agent card's list)."""

    room_id: UUID
    room_name: str
    room_type: str
    granted_at: datetime
