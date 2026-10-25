"""The authenticated user's Hub-owned Market registry.

Custom MCP credentials are intentionally accepted only on creation and never
serialized back to the browser.  The Gateway will consume them server-side in
a later stage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import urlparse, urlsplit, urlunsplit
import os
import httpx
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.core import mcp_oauth
from app.core.secret_config import configured_secret
from app.core.secrets.redaction import assert_no_secret_fields
from app.db.session import get_session
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot
from app.models.mcp_oauth import McpOAuthFlow
from app.models.secret import SecretBinding, SecretPurpose, SecretResourceType
from app.models.skill import Skill
from app.models.user import User
from app.core.mcp_discovery import (
    is_http_link,
    mcp_connection_status,
    verify_mcp_connection,
)
from app.schemas.mcp_connection import McpConnectionStatusOut, McpRoomBindingOut

logger = logging.getLogger(__name__)

router = APIRouter()


class CustomInstalledItemIn(BaseModel):
    """A manually connected MCP endpoint or one of the user's uploaded skills."""

    type: Literal["skill", "mcp"]
    is_custom: Literal[True] = True
    name: str = Field(min_length=1, max_length=255)
    isolation: Literal["stateless", "tenant_aware"] | None = None
    custom_config: dict[str, Any] | None = None
    skill_id: UUID | None = None


class InstalledItemUpdateIn(BaseModel):
    """Safe mutable fields for a user-owned installed item."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    status: Literal["active", "disabled"] | None = None
    custom_config: dict[str, Any] | None = None


class InstalledItemOut(BaseModel):
    id: UUID
    type: Literal["skill", "mcp"]
    is_custom: bool
    market_item_id: UUID | None
    market_slug: str | None
    name: str
    isolation: Literal["stateless", "tenant_aware"] | None
    skill_id: UUID | None
    status: Literal["active", "disabled"]
    has_custom_config: bool
    hosted: bool = False
    # A user-facing provenance label.  It is intentionally derived from the
    # Hub-owned installation snapshot rather than from custom_config itself,
    # which must never be returned to the browser.
    origin: Literal["legion", "link", "personal_image", "personal_stdio"]
    personal_status: str | None = None
    # internal design notes Stage 7 — the auth mode and whether a fresh login is
    # needed. Neither field leaks a secret: "oauth"/"bearer" is a shape
    # already public via origin="link", and needs_reauth is a boolean flag
    # mirrored from custom_config.auth (never the tokens themselves).
    auth_mode: Literal["bearer", "oauth"] | None = None
    needs_reauth: bool = False
    # Populated only for origin="legion", from the cached mcp_price_snapshot —
    # personal MCPs are never billed (internal design notes, stage 6).
    call_price: float | None = None
    tool_prices: dict[str, float] | None = None
    mcp_status: McpConnectionStatusOut | None = None
    room_bindings: list[McpRoomBindingOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


def _as_out(
    item: UserInstalledItem,
    price_snapshot: Any | None = None,
    *,
    mcp_status: McpConnectionStatusOut | None = None,
    room_bindings: list[McpRoomBindingOut] | None = None,
) -> InstalledItemOut:
    """Keep custom_config private: it can contain bearer tokens or API keys."""
    config = item.custom_config or {}
    hosted = config.get("mode") == "hosted"
    managed_stdio = config.get("mode") == "managed_stdio"
    origin: Literal["legion", "link", "personal_image", "personal_stdio"] = (
        "legion"
        if item.market_item_id is not None
        else ("personal_image" if hosted else ("personal_stdio" if managed_stdio else "link"))
    )
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    auth = config.get("auth") if isinstance(config.get("auth"), dict) else None
    auth_mode = auth.get("mode") if auth else None
    needs_reauth = bool(auth.get("needs_reauth")) if auth else False
    call_price: float | None = None
    tool_prices: dict[str, float] | None = None
    if origin == "legion" and price_snapshot is not None:
        from app.core import mcp_credits

        call_price = mcp_credits.mcr_to_credits(price_snapshot.call_price_mcr)
        if price_snapshot.tool_price_overrides:
            tool_prices = {
                name: mcp_credits.mcr_to_credits(price)
                for name, price in price_snapshot.tool_price_overrides.items()
            }
    return InstalledItemOut(
        id=item.id,
        type=item.type,
        is_custom=item.is_custom,
        market_item_id=item.market_item_id,
        market_slug=item.market_slug,
        name=item.name,
        isolation=item.isolation,
        skill_id=item.skill_id,
        status=item.status,
        has_custom_config=bool(item.custom_config),
        hosted=hosted,
        origin=origin,
        personal_status=runtime.get("status") if hosted else None,
        auth_mode=auth_mode if auth_mode in {"bearer", "oauth"} else None,
        needs_reauth=needs_reauth,
        call_price=call_price,
        tool_prices=tool_prices,
        mcp_status=mcp_status,
        room_bindings=room_bindings or [],
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


class MarketMcpCatalogItemOut(BaseModel):
    id: UUID
    slug: str
    name: str
    description: str | None = None
    category: str | None = None
    is_paid: bool = False
    price_tier: str | None = None
    version: str = "latest"
    installed: bool
    call_price: float = 0
    tool_prices: dict[str, float] | None = None


class MarketCategoryOut(BaseModel):
    slug: str
    emoji: str
    label_ru: str
    label_en: str
    count: int = 0
    empty: bool = False


class MarketMcpCatalogOut(BaseModel):
    items: list[MarketMcpCatalogItemOut]
    total: int
    page: int
    page_size: int


class MarketMcpInstallIn(BaseModel):
    slug: str = Field(min_length=1, max_length=100)


def _validate_custom_mcp_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise HTTPException(status_code=422, detail="custom_config is required for a custom MCP")
    if "headers" in config or "env" in config:
        raise HTTPException(
            status_code=422,
            detail="custom MCP headers/env must use a SecretStore binding",
        )
    try:
        assert_no_secret_fields(config, path="custom_config")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Hosted personal MCPs are launched by mcp_host and receive their URL at
    # start time.  They deliberately do not accept an arbitrary URL here.
    if config.get("mode") == "hosted":
        image = config.get("image")
        if not isinstance(image, str) or not image.strip() or len(image) > 512:
            raise HTTPException(status_code=422, detail="hosted MCP requires custom_config.image")
        port = config.get("port", 8000)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise HTTPException(status_code=422, detail="custom_config.port must be a TCP port")
        return dict(config)
    if config.get("mode") == "managed_stdio":
        raise HTTPException(
            status_code=422,
            detail="managed_stdio MCPs must be created through /api/v1/personal-mcps",
        )
    url = config.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="custom_config.url must be an absolute HTTP(S) URL")
    auth = config.get("auth")
    if auth is not None:
        if not isinstance(auth, dict):
            raise HTTPException(status_code=422, detail="custom_config.auth is invalid")
        mode = auth.get("mode")
        if mode == "bearer":
            if set(auth) - {"mode", "secret_binding_id"}:
                raise HTTPException(status_code=422, detail="custom_config.auth is invalid")
            if auth.get("secret_binding_id") is not None:
                try:
                    UUID(str(auth["secret_binding_id"]))
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail="MCP auth binding is invalid") from exc
        elif mode == "oauth":
            # internal design notes Stage 5 — this shape is written by
            # app.core.mcp_oauth.complete_login, never typed by a user; this
            # branch only lets it round-trip through a later PATCH that
            # resubmits the item's full custom_config unchanged.
            if set(auth) - {
                "mode", "issuer", "resource", "client_ref", "scope", "expires_at", "needs_reauth",
            }:
                raise HTTPException(status_code=422, detail="custom_config.auth is invalid")
        else:
            raise HTTPException(status_code=422, detail="Only bearer or oauth MCP auth is supported")
    return dict(config)


async def _check_link_mcp_limit(user_id: UUID, db: AsyncSession) -> None:
    """Refuse one more MCP-by-link past the tariff's ``link_mcp_max``.

    internal design notes §3c (TR9, T7): 10 on Hobby, 40 on Pro. Every link
    the user has installed counts, enabled or not — the cap guards the Hub
    as a proxy to arbitrary addresses, not the context of one room. The
    user's quota row is locked first so two concurrent adds cannot both
    take the last slot (Postgres; a no-op elsewhere).
    """
    from app.core.workspace_quota import EXPIRED_PLAN_TIER, get_effective_limits
    from app.models.user_quota import UserQuota

    limits = await get_effective_limits(user_id, db)
    cap = limits.get("link_mcp_max")
    if isinstance(cap, bool) or not isinstance(cap, int):
        return
    await db.execute(
        select(UserQuota.user_id).where(UserQuota.user_id == user_id).with_for_update()
    )
    items = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "mcp",
                UserInstalledItem.is_custom.is_(True),
            )
        )
    ).scalars().all()
    installed = sum(1 for item in items if is_http_link(item))
    if installed < cap:
        return
    tier = limits.get("plan_tier")
    raise HTTPException(
        status_code=402,
        detail=(
            "The paid plan has ended: link-based MCPs cannot be connected. "
            "Renew the plan to continue."
            if tier == EXPIRED_PLAN_TIER else
            f"The link-based MCP limit for the '{tier}' plan has been reached: {installed}/{cap}. "
            "Delete an unused server or switch to a higher plan."
        ),
    )


class PersonalStatusOut(BaseModel):
    installed_item_id: UUID
    status: str
    endpoint: str | None = None
    container_id: str | None = None
    port: int | None = None
    uptime_sec: float | None = None
    quota: dict[str, Any] | None = None


async def _owned_hosted_item(item_id: UUID, user_id: UUID, db: AsyncSession) -> UserInstalledItem:
    item = await db.get(UserInstalledItem, item_id)
    if item is None or item.user_id != user_id:
        raise HTTPException(status_code=404, detail="Installed item not found")
    if item.type != "mcp" or not item.is_custom or (item.custom_config or {}).get("mode") != "hosted":
        raise HTTPException(status_code=422, detail="Item is not a hosted personal MCP")
    return item


async def _host_request(path: str, user_id: UUID, item_id: UUID, *, method: str = "get", payload: dict | None = None) -> dict:
    base = os.getenv("MCP_HOST_URL", "http://mcp-host:8320").rstrip("/")
    token = configured_secret("MCP_HOST_TOKEN") or ""
    async with httpx.AsyncClient(timeout=float(os.getenv("MCP_HOST_TIMEOUT_SEC", "30"))) as client:
        response = await getattr(client, method)(f"{base}{path.format(user_id=user_id, item_id=item_id)}",
                                                  headers={"Authorization": f"Bearer {token}"}, json=payload)
    if response.status_code >= 400:
        try: detail = response.json().get("detail", response.text)
        except Exception: detail = response.text
        raise HTTPException(status_code=409, detail=detail)
    return response.json()


@router.post("/{user_installed_item_id}/personal/start", response_model=PersonalStatusOut)
async def start_personal_mcp(user_installed_item_id: UUID, db: AsyncSession = Depends(get_session), current_user: User = Depends(deps.get_current_user)):
    item = await _owned_hosted_item(user_installed_item_id, current_user.id, db)
    from app.core.mcp_personal import check_personal_start_allowed, update_runtime, utcnow_iso
    existing_runtime = (item.custom_config or {}).get("runtime") or {}
    if existing_runtime.get("status") == "running":
        return PersonalStatusOut(installed_item_id=item.id, status="running", endpoint=existing_runtime.get("endpoint"),
                                 container_id=existing_runtime.get("container_id"), port=existing_runtime.get("port"),
                                 uptime_sec=None)
    quota = await check_personal_start_allowed(current_user.id, db)
    cfg = item.custom_config or {}
    mem = min(int(cfg.get("mem_limit_mb", quota.get("mem_limit_mb") or 1024)), int(quota.get("mem_limit_mb") or 8192))
    # User secrets are never injected into a personal workspace container.
    payload = {"image": cfg["image"], "port": int(cfg.get("port", 8000)), "mem_limit_mb": mem,
               "cpu_limit_milli": int(cfg.get("cpu_limit_milli", 500)), "args": cfg.get("args", []), "env": {}}
    # internal design notes stage 12.1: the machine's memory, all users together.
    from app.core import host_capacity
    async with host_capacity.host_lock():
        try:
            await host_capacity.check_personal_mcp_start(db, item.id, mem)
        except host_capacity.HostCapacityExceeded:
            raise HTTPException(status_code=503, detail={"error_code": host_capacity.SERVER_BUSY,
                                                         "message": "The server is at capacity right now: the personal MCP cannot start. Try again in a few minutes."})
        # mem_limit_mb: what host_capacity counts while this server runs.
        update_runtime(item, status="starting", started_at=utcnow_iso(), mem_limit_mb=mem)
        await db.commit()
    try:
        result = await _host_request("/v1/personal/{user_id}/{item_id}/start", current_user.id, item.id, method="post", payload=payload)
    except Exception:
        update_runtime(item, status="error")
        await db.commit()
        raise
    update_runtime(item, status=result.get("status", "running"), endpoint=result.get("endpoint"), container_id=result.get("container_id"), port=result.get("port", payload["port"]))
    await db.commit()
    return PersonalStatusOut(installed_item_id=item.id, quota=quota, **{k: result.get(k) for k in ("status", "endpoint", "container_id", "port", "uptime_sec")})


@router.post("/{user_installed_item_id}/personal/stop", response_model=PersonalStatusOut)
async def stop_personal_mcp(user_installed_item_id: UUID, db: AsyncSession = Depends(get_session), current_user: User = Depends(deps.get_current_user)):
    item = await _owned_hosted_item(user_installed_item_id, current_user.id, db)
    from app.core.mcp_personal import update_runtime, utcnow_iso
    result = await _host_request("/v1/personal/{user_id}/{item_id}/stop", current_user.id, item.id, method="post")
    update_runtime(item, status="stopped", stopped_at=utcnow_iso())
    await db.commit()
    return PersonalStatusOut(installed_item_id=item.id, **{k: result.get(k) for k in ("status", "endpoint", "container_id", "port", "uptime_sec")})


@router.get("/{user_installed_item_id}/personal/status", response_model=PersonalStatusOut)
async def personal_mcp_status(user_installed_item_id: UUID, db: AsyncSession = Depends(get_session), current_user: User = Depends(deps.get_current_user)):
    item = await _owned_hosted_item(user_installed_item_id, current_user.id, db)
    cfg = item.custom_config or {}
    result = await _host_request("/v1/personal/{user_id}/{item_id}/status", current_user.id, item.id, method="get")
    return PersonalStatusOut(installed_item_id=item.id, **{k: result.get(k) for k in ("status", "endpoint", "container_id", "port", "uptime_sec")})


async def _require_owned_skill(skill_id: UUID, user_id: UUID, db: AsyncSession) -> Skill:
    skill = (
        await db.execute(select(Skill).where(Skill.id == skill_id, Skill.owner_id == user_id))
    ).scalar_one_or_none()
    if skill is None:
        raise HTTPException(status_code=404, detail="Uploaded skill not found")
    return skill


@router.get("/", response_model=list[InstalledItemOut])
async def list_user_installed_items(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[InstalledItemOut]:
    result = await db.execute(
        select(UserInstalledItem)
        .where(UserInstalledItem.user_id == current_user.id)
        .order_by(UserInstalledItem.created_at.desc())
    )
    items = result.scalars().all()
    market_item_ids = [item.market_item_id for item in items if item.market_item_id is not None]
    snapshots: dict[UUID, Any] = {}
    if market_item_ids:
        from app.models.mcp_credits import McpPriceSnapshot

        snap_result = await db.execute(
            select(McpPriceSnapshot).where(McpPriceSnapshot.market_item_id.in_(market_item_ids))
        )
        snapshots = {snap.market_item_id: snap for snap in snap_result.scalars().all()}
    link_items = [item for item in items if is_http_link(item)]
    connection_snapshots: dict[UUID, McpConnectionSnapshot] = {}
    if link_items:
        connection_result = await db.execute(
            select(McpConnectionSnapshot).where(
                McpConnectionSnapshot.installed_item_id.in_([item.id for item in link_items])
            )
        )
        connection_snapshots = {
            snapshot.installed_item_id: snapshot
            for snapshot in connection_result.scalars().all()
        }
    result: list[InstalledItemOut] = []
    for item in items:
        status_out = await mcp_connection_status(
            db,
            item,
            snapshot=connection_snapshots.get(item.id),
        )
        result.append(_as_out(item, snapshots.get(item.market_item_id), mcp_status=status_out))
    return result


@router.get("/mcp/market/categories", response_model=list[MarketCategoryOut])
async def list_market_mcp_categories(
    q: str | None = Query(default=None, max_length=100),
    publisher: Literal["legion", "vendor", "community"] | None = None,
    price: Literal["free", "per_call", "install"] | None = None,
    access: Literal["ready", "needs_account"] | None = None,
    current_user: User = Depends(deps.get_current_user),
) -> list[MarketCategoryOut]:
    """Market taxonomy with MCP counts under the other active filters."""
    from app.core import market_client

    try:
        rows = await market_client.list_categories(type="mcp", q=q, publisher=publisher, price=price, access=access)
        return [MarketCategoryOut(**row) for row in rows]
    except market_client.MarketClientError as exc:
        raise HTTPException(status_code=502, detail="Marketplace is unavailable") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Marketplace returned an invalid category list") from exc


@router.get("/mcp/market/catalog", response_model=MarketMcpCatalogOut)
async def browse_market_mcps(
    q: str | None = Query(default=None, max_length=100),
    category: list[str] = Query(default_factory=list),
    publisher: Literal["legion", "vendor", "community"] | None = None,
    price: Literal["free", "per_call", "install"] | None = None,
    access: Literal["ready", "needs_account"] | None = None,
    sort: Literal["relevance", "name", "new"] | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=12, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> MarketMcpCatalogOut:
    """Expose the MCP part of Legion Market in the same installed-first shape
    used by the Skills UI.  Market remains the source of MCP transport data;
    Hub stores only the user's installation snapshot.
    """
    from app.core import market_client, mcp_credits

    try:
        catalog = await market_client.list_catalog(
            type="mcp", q=q, category=category, publisher=publisher, price=price,
            access=access, sort=sort, page=page, page_size=page_size,
        )
    except market_client.MarketClientError as exc:
        raise HTTPException(status_code=502, detail="Marketplace is unavailable") from exc

    raw_items = catalog.get("items", [])
    try:
        item_ids = [UUID(str(item["id"])) for item in raw_items]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Marketplace returned an invalid MCP catalog") from exc
    installed_ids: set[UUID] = set()
    if item_ids:
        installed = await db.execute(
            select(UserInstalledItem.market_item_id).where(
                UserInstalledItem.user_id == current_user.id,
                UserInstalledItem.market_item_id.in_(item_ids),
            )
        )
        installed_ids = {row[0] for row in installed.all()}

    try:
        items = [
            MarketMcpCatalogItemOut(
                id=item["id"],
                slug=item["slug"],
                name=item["name"],
                description=item.get("description"),
                category=item.get("category"),
                is_paid=bool(item.get("is_paid", False)),
                price_tier=item.get("price_tier"),
                version=str(item.get("version") or "latest"),
                installed=UUID(str(item["id"])) in installed_ids,
                call_price=mcp_credits.mcr_to_credits(int(item.get("call_price_mcr") or 0)),
                tool_prices={
                    name: mcp_credits.mcr_to_credits(int(price))
                    for name, price in (item.get("tool_price_overrides") or {}).items()
                } or None,
            )
            for item in raw_items
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Marketplace returned an invalid MCP catalog") from exc
    return MarketMcpCatalogOut(
        items=items,
        total=int(catalog.get("total", len(items))),
        page=int(catalog.get("page", page)),
        page_size=int(catalog.get("page_size", page_size)),
    )


@router.post("/mcp/market/install", response_model=InstalledItemOut, status_code=status.HTTP_201_CREATED)
async def install_market_mcp(
    body: MarketMcpInstallIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> InstalledItemOut:
    """Install a Legion Market MCP into the caller's global registry.

    The gateway resolves the immutable Market item id server-side.  Keeping
    its transport configuration out of Hub prevents a browser from changing a
    Market MCP into an arbitrary endpoint.
    """
    from app.core import market_client
    from app.core.workspace_quota import get_effective_limits

    try:
        market_item = await market_client.get_catalog_item(body.slug)
    except market_client.MarketNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"MCP '{body.slug}' not found in Marketplace") from exc
    except market_client.MarketClientError as exc:
        raise HTTPException(status_code=502, detail="Marketplace is unavailable") from exc
    if market_item.get("type") != "mcp" or not market_item.get("is_active", True):
        raise HTTPException(status_code=404, detail=f"MCP '{body.slug}' not found in Marketplace")
    try:
        market_item_id = UUID(str(market_item["id"]))
        name = str(market_item["name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Marketplace returned an invalid MCP item") from exc

    existing = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == current_user.id,
                UserInstalledItem.market_item_id == market_item_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        from app.core import mcp_credits

        snapshot = await mcp_credits.get_price_snapshot(db, market_item_id)
        return _as_out(existing, snapshot)

    price_tier = market_item.get("price_tier")
    if price_tier:
        limits = await get_effective_limits(current_user.id, db)
        if price_tier not in (limits.get("market_paid_tiers") or []):
            raise HTTPException(status_code=402, detail="The current plan does not include this Market item")

    isolation = market_item.get("isolation")
    item = UserInstalledItem(
        user_id=current_user.id,
        market_item_id=market_item_id,
        type="mcp",
        name=name,
        market_slug=body.slug,
        isolation=isolation if isolation in {"stateless", "tenant_aware"} else None,
    )
    db.add(item)
    # Cache the call price now, from the catalog item already in hand — no
    # extra Market round-trip. execute_mcp_tool refuses to charge for a call
    # it can't price, so a missing snapshot would silently strand this item.
    from app.core import mcp_credits

    snapshot = await mcp_credits.store_price_snapshot(
        db,
        market_item_id=market_item_id,
        market_slug=body.slug,
        call_price_mcr=int(market_item.get("call_price_mcr") or 0),
        tool_price_overrides=market_item.get("tool_price_overrides") or {},
    )
    await db.commit()
    await db.refresh(item)
    return _as_out(item, snapshot)


@router.post("/", response_model=InstalledItemOut, status_code=status.HTTP_201_CREATED)
async def create_user_installed_item(
    body: CustomInstalledItemIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> InstalledItemOut:
    """Add a custom MCP, or register one of the caller's uploaded skills."""
    custom_config: dict[str, Any] | None = None
    isolation: str | None = None
    skill_id: UUID | None = None

    if body.type == "mcp":
        if body.skill_id is not None:
            raise HTTPException(status_code=422, detail="skill_id is only valid for skill items")
        custom_config = _validate_custom_mcp_config(body.custom_config)
        isolation = body.isolation or "stateless"
        if is_http_link(SimpleNamespace(type="mcp", is_custom=True, custom_config=custom_config)):
            await _check_link_mcp_limit(current_user.id, db)
    else:
        if body.custom_config is not None or body.isolation is not None:
            raise HTTPException(status_code=422, detail="Skill items do not accept MCP configuration")
        if body.skill_id is None:
            raise HTTPException(status_code=422, detail="skill_id is required for a custom skill")
        await _require_owned_skill(body.skill_id, current_user.id, db)
        skill_id = body.skill_id

    item = UserInstalledItem(
        user_id=current_user.id,
        type=body.type,
        is_custom=True,
        name=body.name,
        isolation=isolation,
        custom_config=custom_config,
        skill_id=skill_id,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return _as_out(item, mcp_status=await mcp_connection_status(db, item))


@router.post("/{user_installed_item_id}/mcp-discovery", response_model=McpConnectionStatusOut)
async def discover_user_installed_mcp(
    user_installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> McpConnectionStatusOut:
    """Manually verify one owned HTTP MCP link through the Gateway."""

    item = await db.get(UserInstalledItem, user_installed_item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Installed item not found")
    if not is_http_link(item):
        raise HTTPException(status_code=422, detail="Only HTTP link MCPs support this discovery endpoint")
    return await verify_mcp_connection(db, item, force=True)


class OAuthStartOut(BaseModel):
    authorization_url: str
    state: str


class OAuthStatusOut(BaseModel):
    status: Literal["not_started", "pending", "success", "error"]
    error_code: str | None = None


def _hub_origin_allowlist(request: Request) -> set[str]:
    """The origin(s) a redirect_uri may legitimately be minted for.

    internal design notes §2/§9: built from the Hub's own origin (never trusted
    from the request's ``Origin`` header as such — that would let a
    cross-origin page CSRF a login into registering a redirect_uri on its own
    domain) plus an explicit ``HUB_PUBLIC_URL`` for deployments behind a
    proxy that changes the origin the browser sees.
    """

    allowed = {f"{request.url.scheme}://{request.url.netloc}"}
    public_url = os.getenv("HUB_PUBLIC_URL")
    if public_url:
        parts = urlsplit(public_url)
        if parts.scheme and parts.netloc:
            allowed.add(urlunsplit((parts.scheme, parts.netloc, "", "", "")))
    return allowed


@router.post("/{user_installed_item_id}/oauth/start", response_model=OAuthStartOut)
async def start_mcp_oauth_login(
    user_installed_item_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> OAuthStartOut:
    """internal design notes Stage 3 — begin a provider login for one owned item."""

    item, redirect_uri = await _owned_oauth_item_and_redirect_uri(
        user_installed_item_id, request, db, current_user
    )
    try:
        result = await mcp_oauth.start_login(
            db, item=item, user_id=current_user.id, redirect_uri=redirect_uri
        )
    except mcp_oauth.NoDynamicRegistration as exc:
        raise HTTPException(
            status_code=422, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except mcp_oauth.McpOAuthError as exc:
        raise HTTPException(
            status_code=502, detail={"code": exc.code, "message": "OAuth provider is unavailable"}
        ) from exc
    return OAuthStartOut(authorization_url=result.authorization_url, state=result.flow.state)


async def _owned_oauth_item_and_redirect_uri(
    user_installed_item_id: UUID,
    request: Request,
    db: AsyncSession,
    current_user: User,
) -> tuple[UserInstalledItem, str]:
    """Shared ownership/origin checks for both the DCR and Stage 8 manual
    registration entry points — same 404/400 shape either way."""

    item = await db.get(UserInstalledItem, user_installed_item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Installed item not found")
    if not is_http_link(item):
        raise HTTPException(status_code=422, detail="Only HTTP link MCPs support an OAuth login")

    origin = request.headers.get("origin")
    if not origin or origin not in _hub_origin_allowlist(request):
        raise HTTPException(status_code=400, detail="Origin is not allowed for an OAuth login")

    redirect_uri = f"{origin}/api/v1/user-installed-items/mcp/oauth/callback"
    return item, redirect_uri


class ManualOAuthClientIn(BaseModel):
    """internal design notes Stage 8 — a client_id/client_secret a person typed in
    after creating an app in a provider's console that offers no DCR."""

    client_id: str = Field(min_length=1, max_length=512)
    client_secret: str | None = Field(default=None, min_length=1, max_length=4096)


class ManualOAuthClientOut(BaseModel):
    issuer: str
    redirect_uri: str


@router.post(
    "/{user_installed_item_id}/oauth/manual-client", response_model=ManualOAuthClientOut
)
async def register_mcp_oauth_manual_client(
    user_installed_item_id: UUID,
    body: ManualOAuthClientIn,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ManualOAuthClientOut:
    """internal design notes Stage 8 — register a manual client for a provider that
    has no ``registration_endpoint``. A subsequent ``oauth/start`` call finds
    this row and never attempts DCR against it."""

    item, redirect_uri = await _owned_oauth_item_and_redirect_uri(
        user_installed_item_id, request, db, current_user
    )
    keyring = getattr(request.app.state, "secret_keyring", None)
    if keyring is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "mcp_oauth_storage_unavailable", "message": "Secret storage is unavailable"},
        )
    try:
        result = await mcp_oauth.register_manual_client(
            db,
            item=item,
            user_id=current_user.id,
            redirect_uri=redirect_uri,
            client_id=body.client_id,
            client_secret=body.client_secret,
            keyring=keyring,
        )
    except mcp_oauth.ManualClientInvalid as exc:
        raise HTTPException(
            status_code=422, detail={"code": exc.code, "message": str(exc)}
        ) from exc
    except mcp_oauth.McpOAuthError as exc:
        raise HTTPException(
            status_code=502, detail={"code": exc.code, "message": "OAuth provider is unavailable"}
        ) from exc
    return ManualOAuthClientOut(issuer=result.client.issuer, redirect_uri=redirect_uri)


_OAUTH_CALLBACK_OK_HTML = (
    "<!doctype html><html><body>"
    "<script>window.close();</script>"
    "Signed in. You can close this window."
    "</body></html>"
)
_OAUTH_CALLBACK_ERROR_HTML = (
    "<!doctype html><html><body>"
    "Could not sign in at the provider ({code}). You can close this window."
    "</body></html>"
)


async def _fail_open_flow(db: AsyncSession, *, state: str, error_code: str) -> None:
    """Mark a not-yet-exchanged flow as failed so the status poll can see why,
    for the cases ``complete_login`` never gets to run (no ``code``, or the
    provider itself reported ``error=access_denied``)."""

    flow = await db.scalar(
        select(McpOAuthFlow).where(McpOAuthFlow.state == state, McpOAuthFlow.consumed_at.is_(None))
    )
    if flow is None:
        return
    flow.consumed_at = datetime.now(timezone.utc)
    flow.outcome = "error"
    flow.error_code = error_code
    flow.code_verifier = ""
    db.add(flow)
    await db.commit()


@router.get("/mcp/oauth/callback", response_class=HTMLResponse, include_in_schema=False)
async def mcp_oauth_callback(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """internal design notes Stage 3 — the provider redirects the browser here.

    Not behind ``get_current_user``: a full-page redirect from the provider
    carries no Authorization header. Ownership was already established when
    ``oauth/start`` created the flow row; ``state`` alone re-identifies it.
    """

    keyring = getattr(request.app.state, "secret_keyring", None)
    if keyring is None:
        return HTMLResponse(
            _OAUTH_CALLBACK_ERROR_HTML.format(code="mcp_oauth_storage_unavailable"),
            status_code=503,
        )
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    provider_error = request.query_params.get("error")
    if not state:
        return HTMLResponse(
            _OAUTH_CALLBACK_ERROR_HTML.format(code="mcp_oauth_flow_invalid"), status_code=400
        )
    if provider_error or not code:
        await _fail_open_flow(
            db, state=state, error_code="mcp_oauth_denied" if provider_error else "mcp_oauth_flow_invalid"
        )
        return HTMLResponse(
            _OAUTH_CALLBACK_ERROR_HTML.format(code="mcp_oauth_denied"), status_code=400
        )
    try:
        outcome = await mcp_oauth.complete_login(db, state=state, code=code, keyring=keyring)
    except mcp_oauth.InvalidFlow:
        return HTMLResponse(
            _OAUTH_CALLBACK_ERROR_HTML.format(code="mcp_oauth_flow_invalid"), status_code=400
        )
    if outcome.ok and outcome.installed_item_id is not None:
        # A stale "requires OAuth login" connection snapshot would otherwise
        # outlive a login that just succeeded — nothing else re-probes the
        # endpoint on its own, and the popup's own opener tab may not be
        # around any more (reloaded, or the login ran past its polling
        # window) to trigger the refresh itself. Best-effort: a failed probe
        # here must not turn a successful login into an error page.
        try:
            item = await db.get(UserInstalledItem, outcome.installed_item_id)
            if item is not None:
                await verify_mcp_connection(db, item, force=True)
        except Exception:
            logger.exception(
                "mcp_oauth_post_login_verify_failed item=%s", outcome.installed_item_id
            )
    if not outcome.ok:
        return HTMLResponse(
            _OAUTH_CALLBACK_ERROR_HTML.format(code=outcome.error_code or "mcp_oauth_failed"),
            status_code=502,
        )
    return HTMLResponse(_OAUTH_CALLBACK_OK_HTML)


@router.get("/{user_installed_item_id}/oauth/status", response_model=OAuthStatusOut)
async def mcp_oauth_status(
    user_installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> OAuthStatusOut:
    """internal design notes Stage 3 — what the frontend polls while the popup is open."""

    item = await db.get(UserInstalledItem, user_installed_item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Installed item not found")

    flow = (
        await db.execute(
            select(McpOAuthFlow)
            .where(
                McpOAuthFlow.installed_item_id == item.id,
                McpOAuthFlow.user_id == current_user.id,
            )
            .order_by(McpOAuthFlow.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if flow is None:
        return OAuthStatusOut(status="not_started")
    if flow.consumed_at is None:
        now = datetime.now(timezone.utc)
        expires_at = (
            flow.expires_at
            if flow.expires_at.tzinfo is not None
            else flow.expires_at.replace(tzinfo=timezone.utc)
        )
        if expires_at <= now:
            return OAuthStatusOut(status="error", error_code="mcp_oauth_flow_expired")
        return OAuthStatusOut(status="pending")
    if flow.outcome == "success":
        return OAuthStatusOut(status="success")
    return OAuthStatusOut(status="error", error_code=flow.error_code)


@router.patch("/{user_installed_item_id}", response_model=InstalledItemOut)
async def update_user_installed_item(
    user_installed_item_id: UUID,
    body: InstalledItemUpdateIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> InstalledItemOut:
    """Update safe item metadata and invalidate link verification on changes."""

    item = await db.get(UserInstalledItem, user_installed_item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Installed item not found")
    if body.custom_config is not None:
        if item.type != "mcp":
            raise HTTPException(status_code=422, detail="Only MCP items accept custom_config")
        new_config = _validate_custom_mcp_config(body.custom_config)
        if new_config != (item.custom_config or {}):
            # A hosted item rewritten into a URL becomes one more link (T7).
            if not is_http_link(item) and is_http_link(
                SimpleNamespace(type=item.type, is_custom=item.is_custom, custom_config=new_config)
            ):
                await _check_link_mcp_limit(current_user.id, db)
            item.custom_config = new_config
            if is_http_link(item):
                from app.core.mcp_discovery import invalidate_mcp_snapshot

                await invalidate_mcp_snapshot(db, item.id, owner_id=item.user_id)
    if body.name is not None:
        item.name = body.name
    if body.status is not None:
        item.status = body.status
        if body.status != "active" and is_http_link(item):
            from app.core.mcp_discovery import invalidate_mcp_snapshot

            await invalidate_mcp_snapshot(db, item.id, owner_id=item.user_id)
    item.updated_at = datetime.now(item.updated_at.tzinfo) if item.updated_at else datetime.now()
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return _as_out(item, mcp_status=await mcp_connection_status(db, item))


@router.delete("/{user_installed_item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_installed_item(
    user_installed_item_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    item = await db.get(UserInstalledItem, user_installed_item_id)
    if item is None or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Installed item not found")
    if (item.custom_config or {}).get("mode") == "managed_stdio":
        raise HTTPException(
            status_code=409,
            detail="Use /api/v1/personal-mcps/{id}?confirm_data_delete=true to delete managed MCP data",
        )
    await db.execute(
        delete(SecretBinding).where(
            SecretBinding.owner_id == current_user.id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == item.id,
            SecretBinding.purpose.in_({SecretPurpose.MCP_AUTH_HEADER, SecretPurpose.MCP_ENV}),
        )
    )
    await db.delete(item)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
