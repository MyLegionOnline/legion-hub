from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, model_validator, field_validator
from app.models.room import RoomState, RoomType, VpsAccessMode
from app.core.secrets.redaction import assert_no_secret_fields, sanitize_response


def _normalize_git_config(value):
    """Clean a browser-pasted ``repo_url`` down to a git-usable clone URL.

    Users routinely paste ``https://github.com/owner/repo/tree/main`` from the
    address bar; stored as-is it later reaches ``git fetch`` and fails with
    ``repository '.../tree/main/' not found`` (adopt/pull/push). Normalizing at
    the write boundary keeps the stored config — and the "Connected" UI label —
    clean. ``git_sync._clone_url`` stays the last line of defence for configs
    written by other paths.
    """
    if isinstance(value, dict) and value.get("repo_url"):
        from app.core.git_sync import _clone_url

        cleaned = _clone_url(str(value["repo_url"]))
        if cleaned and cleaned != value.get("repo_url"):
            value = {**value, "repo_url": cleaned}
    return value


class RoomBase(BaseModel):
    name: str
    type: str = "group"
    room_type: RoomType = RoomType.group
    has_coordinator: bool = False
    coordinator_config: Dict[str, Any] = {}
    git_config: Dict[str, Any] = {}
    vps_access_mode: VpsAccessMode = VpsAccessMode.readonly

    @field_validator("coordinator_config", "git_config")
    @classmethod
    def reject_plaintext_secret_fields(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        assert_no_secret_fields(value, path="room_config")
        return value

    @field_validator("git_config")
    @classmethod
    def normalize_repo_url(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_git_config(value)


class RoomCreate(RoomBase):
    user_ids: List[UUID] = []
    agent_ids: List[UUID] = []

    @model_validator(mode="after")
    def validate_direct_room(self) -> "RoomCreate":
        if self.room_type == RoomType.direct:
            if len(self.agent_ids) != 1:
                raise ValueError("A direct room must have exactly one agent.")
        return self

    @model_validator(mode="after")
    def validate_group_room_has_chatmaster(self) -> "RoomCreate":
        # A group room routes through the ChatMaster's model (router stage 1),
        # so the room contract requires one at creation time.  Without this
        # check 39 of 41 group rooms had no master and the router fell back to
        # a random agent's model — see internal design notes §0.2.
        if self.room_type == RoomType.group:
            cfg = self.coordinator_config or {}
            master = cfg.get("master_agent_id") or cfg.get("agent_id") or cfg.get("chat_master_agent_id")
            if not master:
                raise ValueError("A group room must have a ChatMaster assigned.")
            if UUID(str(master)) not in set(self.agent_ids):
                raise ValueError("The ChatMaster must be one of the room's agents.")
        return self


class RoomUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    has_coordinator: Optional[bool] = None
    coordinator_config: Optional[Dict[str, Any]] = None
    git_config: Optional[Dict[str, Any]] = None
    vps_access_mode: Optional[VpsAccessMode] = None
    user_ids: Optional[List[UUID]] = None
    agent_ids: Optional[List[UUID]] = None

    @field_validator("coordinator_config", "git_config")
    @classmethod
    def reject_plaintext_secret_fields(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None:
            assert_no_secret_fields(value, path="room_config")
        return value

    @field_validator("git_config")
    @classmethod
    def normalize_repo_url(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return _normalize_git_config(value) if value is not None else value


class RoomStateUpdate(BaseModel):
    state: RoomState


class RoomOut(RoomBase):
    id: UUID
    creator_id: UUID
    state: RoomState = RoomState.PLANNING
    room_type: RoomType = RoomType.group
    created_at: Optional[datetime] = None
    # Populated by GET /rooms/ from a MAX(Message.timestamp) aggregate; None
    # for rooms without messages yet (sidebar falls back to created_at).
    last_message_at: Optional[datetime] = None
    # Lifecycle state of the room's runtime (its container): "running",
    # "starting", "stopping", "draining", "stopped", "error" — or None for a
    # room that never had one. Since 2026-09-25 this is NOT what billing is
    # metered on — see billing_active below; the sidebar uses it only for
    # transitional/error states.
    #
    # Named `runtime_status`, not `workspace_status`, on purpose: the frontend
    # already keeps a client-side `workspace_status` on the selected room, and
    # that one holds the full status OBJECT assembled from lifecycle events.
    # Two different shapes under one name on the same object is a trap.
    runtime_status: Optional[str] = None
    # True while an open `workspace_usage` interval exists — "minutes are
    # being drawn from my hours right now". That is the room being used
    # (open in a browser, a task, a working session), container or not
    # (core/billing_interval.py), and it is what the sidebar dot's green means.
    billing_active: bool = False

    @field_validator("coordinator_config", "git_config", mode="before")
    @classmethod
    def sanitize_legacy_config(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return sanitize_response(value or {})

    class Config:
        from_attributes = True
