"""Connector identity, pairing, and per-Room grants.

Two routers:

- ``router`` — owner-scoped connector lifecycle, mounted at ``/api/v1/connectors``.
  ``pair``/``token`` authenticate with the pairing code / refresh token
  themselves (no user JWT — the code/token *is* the credential).
- ``room_grants_router`` — per-Room admission, mounted at ``/api/v1/rooms``
  alongside the other room-nested resources (``room_config``, ``sessions``, …).
"""

from __future__ import annotations

import os
import shlex
from typing import Any, List, Optional
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core import connector_pairing
from app.db.session import get_session
from app.models.connector import Connector
from app.models.room import RoomUserLink
from app.models.user import User
from app.websockets.manager import manager
from app.schemas.connector import (
    ConnectorCreate,
    ConnectorOut,
    ConnectorPairingOut,
    ConnectorPairRequest,
    ConnectorPairResponse,
    ConnectorRoomGrantOut,
    ConnectorTokenRequest,
    ConnectorTokenResponse,
)

router = APIRouter()
room_grants_router = APIRouter()


DEFAULT_CONNECTOR_REPO_URL = "https://github.com/MyLegionOnline/legion-connector.git"
CONNECTOR_WS_PATH = "/api/v1/connector/ws"


def _connector_repo_url() -> str:
    """LEGION_CONNECTOR_REPO_URL when it is an http(s) address, else the public repository."""

    configured = (os.getenv("LEGION_CONNECTOR_REPO_URL") or "").strip()
    if configured.startswith(("https://", "http://")):
        return configured
    return DEFAULT_CONNECTOR_REPO_URL


def _hub_public_origin(request: Request) -> str:
    """The origin a connector on another machine should dial.

    ``HUB_PUBLIC_URL`` wins. Otherwise the browser's own ``Origin`` header (the
    pairing UI posts from the page the user is on, so it is the address that
    works for them), then the request's host with the forwarded scheme. The
    value only ends up in a command shown to the user who asked for it.
    """

    public_url = (os.getenv("HUB_PUBLIC_URL") or "").strip()
    candidates = [public_url, request.headers.get("origin", "")]
    for candidate in candidates:
        parts = urlsplit(candidate)
        if parts.scheme in ("http", "https") and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    if scheme not in ("http", "https"):
        scheme = "http"
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


def _connector_ws_url(request: Request) -> str:
    origin = _hub_public_origin(request)
    ws_origin = "wss://" + origin[len("https://"):] if origin.startswith("https://") else "ws://" + origin[len("http://"):]
    return ws_origin + CONNECTOR_WS_PATH


def _install_command(pairing_code: str, request: Request) -> str:
    """One line for the pairing dialog.

    The connector is installed from its public repository (``install.sh`` builds
    it and installs the systemd unit). The Hub address has to be part of the
    command: without it the connector would dial its own default (localhost).
    """

    return (
        f"git clone {shlex.quote(_connector_repo_url())} legion-connector"
        f" && cd legion-connector"
        f" && ./install.sh {shlex.quote(pairing_code)} {shlex.quote(_connector_ws_url(request))}"
    )


async def _load_owned_connector(
    connector_id: UUID, user_id: UUID, db: AsyncSession
) -> Optional[Connector]:
    result = await db.execute(
        select(Connector).where(Connector.id == connector_id, Connector.owner_id == user_id)
    )
    return result.scalars().first()


async def _get_owned_connector(connector_id: UUID, user_id: UUID, db: AsyncSession) -> Connector:
    connector = await _load_owned_connector(connector_id, user_id, db)
    if connector is None:
        raise HTTPException(status_code=404, detail="Connector not found")
    return connector


async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id, RoomUserLink.user_id == user.id
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


async def _load_owned_agent(agent_id: UUID, user_id: UUID, db: AsyncSession):
    from app.models.agent import Agent

    result = await db.execute(
        select(Agent).where(Agent.id == agent_id, Agent.owner_id == user_id)
    )
    return result.scalars().first()


@router.post("", response_model=ConnectorPairingOut, status_code=201)
async def create_connector(
    body: ConnectorCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    if body.agent_id is not None:
        agent = await _load_owned_agent(body.agent_id, current_user.id, db)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
    else:
        agent = None

    issued = await connector_pairing.create_connector(
        db, owner_id=current_user.id, label=body.label
    )
    if agent is not None:
        agent.executor = "connector"
        agent.connector_id = issued.connector_id
        db.add(agent)
    await db.commit()

    connector = await _get_owned_connector(issued.connector_id, current_user.id, db)
    return ConnectorPairingOut(
        connector=ConnectorOut.model_validate(connector),
        pairing_code=issued.pairing_code,
        expires_at=issued.expires_at,
        install_command=_install_command(issued.pairing_code, request),
    )


@router.post("/pair", response_model=ConnectorPairResponse)
async def pair_connector(
    body: ConnectorPairRequest,
    db: AsyncSession = Depends(get_session),
) -> Any:
    try:
        paired = await connector_pairing.pair(db, pairing_code=body.pairing_code)
        await db.commit()
    except connector_pairing.ConnectorPairingError as exc:
        await db.rollback()
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from None
    return ConnectorPairResponse(
        connector_id=paired.connector_id, refresh_token=paired.refresh_token
    )


@router.post("/token", response_model=ConnectorTokenResponse)
async def issue_connector_token(
    body: ConnectorTokenRequest,
    db: AsyncSession = Depends(get_session),
) -> Any:
    try:
        access_token = await connector_pairing.issue_access_token(
            db, refresh_token=body.refresh_token
        )
        await db.commit()
    except connector_pairing.ConnectorPairingError as exc:
        await db.rollback()
        status_code = 401 if isinstance(
            exc, connector_pairing.ConnectorPairingAlreadyUsed
        ) else exc.http_status
        raise HTTPException(status_code=status_code, detail=str(exc)) from None
    return ConnectorTokenResponse(
        access_token=access_token,
        expires_in=connector_pairing.CONNECTOR_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("", response_model=List[ConnectorOut])
async def list_connectors(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    result = await db.execute(
        select(Connector)
        .where(Connector.owner_id == current_user.id)
        .order_by(Connector.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/{connector_id}", status_code=204)
async def revoke_connector(
    connector_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    connector = await _get_owned_connector(connector_id, current_user.id, db)
    await connector_pairing.revoke_connector(db, connector=connector)
    await db.commit()
    await manager.close_connector(connector.id)
    return Response(status_code=204)


# ── Per-Room grants ─────────────────────────────────────────────────────────


@room_grants_router.put(
    "/{room_id}/connectors/{connector_id}",
    response_model=ConnectorRoomGrantOut,
    status_code=201,
)
async def grant_connector_room(
    room_id: UUID,
    connector_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    connector = await _get_owned_connector(connector_id, current_user.id, db)
    grant = await connector_pairing.grant_room(
        db, connector_id=connector.id, room_id=room_id, granted_by=current_user.id
    )
    await db.commit()
    return grant


@room_grants_router.delete(
    "/{room_id}/connectors/{connector_id}", status_code=204
)
async def revoke_connector_room(
    room_id: UUID,
    connector_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    await _verify_membership(room_id, current_user, db)
    connector = await _get_owned_connector(connector_id, current_user.id, db)
    await connector_pairing.revoke_room_grant(
        db, connector_id=connector.id, room_id=room_id
    )
    await db.commit()
    return Response(status_code=204)
