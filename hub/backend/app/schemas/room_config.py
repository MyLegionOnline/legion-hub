"""Schemas for the Room "Secrets & variables" tab — variables (Stage 1).

internal design notes Stage 1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RoomVariableCreate(BaseModel):
    name: str
    value: str
    description: Optional[str] = None
    environment: Optional[str] = None  # None / "" → shared value
    materialize_to_env: bool = True


class RoomVariableUpdate(BaseModel):
    # Every field optional; only the ones actually sent are applied
    # (``model_dump(exclude_unset=True)``).
    name: Optional[str] = None
    value: Optional[str] = None
    description: Optional[str] = None
    environment: Optional[str] = None
    materialize_to_env: Optional[bool] = None


class RoomVariableOut(BaseModel):
    id: UUID
    room_id: UUID
    name: str
    value: str
    description: Optional[str] = None
    environment: Optional[str] = None
    materialize_to_env: bool
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Room secrets (Stage 2) ─────────────────────────────────────────────────
# Values only ever travel inbound.  Nothing here returns a secret value.


class RoomSecretCreate(BaseModel):
    env_name: str
    value: str = Field(min_length=1)
    label: Optional[str] = None
    scope: Optional[List[str]] = None
    environment: Optional[str] = None
    materialize_to_env: bool = False
    # "room_env" (default) | "deploy_ssh_key" | "room_mcp_env" | "room_mcp_auth_header"
    purpose: Optional[str] = None
    # Required for "room_mcp_env" / "room_mcp_auth_header" (Stage 7): the
    # caller's own personal MCP install this secret feeds.
    mcp_item_id: Optional[UUID] = None


class RoomSecretUpdate(BaseModel):
    # Replace the value and/or adjust the binding config.  env_name and
    # environment are the binding identity and are not editable here.
    value: Optional[str] = Field(default=None, min_length=1)
    scope: Optional[List[str]] = None
    materialize_to_env: Optional[bool] = None


class RoomSecretOut(BaseModel):
    binding_id: UUID
    env_name: Optional[str] = None
    purpose: str
    scope: List[str] = []
    environment: Optional[str] = None
    materialize_to_env: bool = False
    # Deploy keys only (Stage 5) — never the private key itself.
    public_key: Optional[str] = None
    fingerprint: Optional[str] = None
    # room_mcp_env / room_mcp_auth_header only (Stage 7).
    mcp_item_id: Optional[UUID] = None
    label: Optional[str] = None
    secret_kind: Optional[str] = None
    state: str
    is_own: bool
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None


# ── `.env` import (Stage 10) ────────────────────────────────────────────────
# §3.6: preview and the real write are the same request, `dry_run` toggles it.


class RoomEnvironmentOut(BaseModel):
    environment: str
    protected: bool
    unlocked_until: Optional[datetime] = None


class RoomEnvironmentUpdate(BaseModel):
    protected: bool


class RoomEnvImportRequest(BaseModel):
    text: str = Field(min_length=1)
    environment: Optional[str] = None
    dry_run: bool = False
    overwrite: bool = False
    # name -> "variable" | "secret"; anything absent or unrecognized stays
    # the default "secret" (plan §3.6 — never auto-downgraded).
    marks: Dict[str, str] = {}


# ── VPS server inventory (internal design notes Stage 3) ──────────────────────────
# Values (root password/key, everyday key/password) never travel outbound —
# only presence flags. The whole thing is a naming convention over Room
# variables + `purpose=vps_access` secrets (core/vps_inventory.py).


class RoomVpsServerCreate(BaseModel):
    name: str
    host: str
    port: int = 22
    user: str = "root"
    root_password: Optional[str] = None
    root_key: Optional[str] = None
    environment: Optional[str] = None


class RoomVpsPrepareRequest(BaseModel):
    """Wizard checkboxes for one preparation run (internal design notes §3.6)."""

    docker: bool = False
    nginx: bool = False
    certbot: bool = False
    git: bool = True
    #: The one step that can lock a careless operator out of their own server,
    #: so it is off unless explicitly asked for and the UI spells out why.
    disable_root_password_login: bool = False
    environment: Optional[str] = None


class RoomVpsFingerprintDecision(BaseModel):
    accept: bool
    environment: Optional[str] = None


class RoomVpsPrepareStepOut(BaseModel):
    id: str
    title: str
    status: str
    detail: str = ""
    error_code: Optional[str] = None


class RoomVpsPrepareReportOut(BaseModel):
    server: str
    environment: Optional[str] = None
    actor: str
    status: str
    profile: dict
    steps: List[RoomVpsPrepareStepOut]
    inventory: dict = {}
    suggestions: List[str] = []
    pending_fingerprint: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: str
    finished_at: Optional[str] = None


class RoomVpsServerOut(BaseModel):
    name: str
    host: Optional[str] = None
    port: int
    user: str
    fingerprint: Optional[str] = None
    prepared_at: Optional[str] = None
    has_key: bool
    has_password: bool
    has_root_key: bool
    has_root_password: bool
    # internal design notes Stage 13 — which repository credential this server
    # holds. Metadata only: `deploy_key_id` is the forge's public id for the
    # key and `deploy_remote` the clone URL the server is meant to use.
    deploy_credential: str = "none"
    deploy_key_id: Optional[str] = None
    deploy_remote: Optional[str] = None


class RoomVpsDeployKeyRequest(BaseModel):
    """Issue a repository credential for one prepared server (Stage 13).

    ``https_token`` is the fallback for a forge where the SSH deploy-key path
    is unavailable: a **read-only** repository token the person creates and
    pastes. Omitted, the Hub generates an SSH key pair instead and the Room's
    own Git token never leaves the Hub either way.
    """

    environment: Optional[str] = None
    https_token: Optional[str] = None


class RoomVpsDeployKeyOut(BaseModel):
    status: str
    credential: Optional[str] = None
    #: Present only when the forge would not take the key: the person adds it
    #: by hand as a read-only deploy key, then presses "Check access".
    public_key: Optional[str] = None
    remote_url: Optional[str] = None
    key_id: Optional[str] = None
    repo: dict = {}
    detail: str = ""


class RoomVpsDeployKeyRevokedOut(BaseModel):
    status: str
    forge_key_removed: bool = False
    remote_url: Optional[str] = None
    notes: List[str] = []
