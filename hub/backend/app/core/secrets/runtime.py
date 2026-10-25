"""Short-lived secret resolution at outbound-call boundaries."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.secret_config import allow_legacy_secret_read
from app.core.secrets.crypto import SecretValue
from app.core.secrets.service import SecretService
from app.core.secrets.store import (
    BindingTarget,
    PostgresEncryptedSecretStore,
    SecretAccessDenied,
)
from app.models.market import UserInstalledItem
from app.models.secret import SecretBinding, SecretPurpose, SecretResourceType, SecretState, UserSecret

logger = logging.getLogger(__name__)

_RUNTIME_KEYRING = None

MCP_UPSTREAM_AUTH_HEADER = "X-Legion-MCP-Upstream-Authorization"

# internal design notes Stage 4 — refresh this far ahead of an OAuth access
# token's expiry, so a tool call almost never races a still-in-flight
# refresh against an upstream 401.
_OAUTH_REFRESH_SKEW_SEC = 120

# One lock per item: ten concurrent tool calls on the same expired token must
# produce exactly one refresh exchange (Stage 4 acceptance), never a burned
# one-shot refresh token racing itself.
_OAUTH_REFRESH_LOCKS: dict[UUID, asyncio.Lock] = {}


def _oauth_refresh_lock(installed_item_id: UUID) -> asyncio.Lock:
    return _OAUTH_REFRESH_LOCKS.setdefault(installed_item_id, asyncio.Lock())


class McpSecretUnavailable(RuntimeError):
    """A safe, stable failure for a missing or unusable MCP bearer binding."""

    code = "mcp_secret_unavailable"

    def __init__(self) -> None:
        super().__init__("MCP authorization is not configured")


def configure_runtime_keyring(keyring) -> None:
    """Install the startup-loaded keyring for non-HTTP outbound call-sites."""

    global _RUNTIME_KEYRING
    _RUNTIME_KEYRING = keyring


def binding_id_from_config(config: dict[str, Any], *keys: str) -> UUID | None:
    """Read a safe binding UUID from a resource config."""

    for key in keys:
        raw = config.get(key)
        if raw in (None, ""):
            continue
        try:
            return UUID(str(raw))
        except (TypeError, ValueError) as exc:
            raise SecretAccessDenied("secret binding is invalid") from exc
    return None


def legacy_secret_value(config: dict[str, Any], field: str) -> SecretValue | None:
    """Controlled dual-read fallback for records not migrated yet."""

    if not allow_legacy_secret_read():
        return None
    raw = config.get(field)
    if raw in (None, ""):
        return None
    try:
        from app.core.metrics import record_legacy_secret_read

        record_legacy_secret_read()
    except Exception:
        pass
    return SecretValue(str(raw))


def service_for(db: AsyncSession) -> SecretService:
    """Build a :class:`SecretService` using the process keyring set at startup.

    Avoids ``PostgresEncryptedSecretStore``'s default fallback to
    ``load_keyring()`` (blocking file I/O) on every call — the same reason
    ``resolve_for_use`` below builds its store this way.
    """

    store = (
        PostgresEncryptedSecretStore(db, keyring=_RUNTIME_KEYRING)
        if _RUNTIME_KEYRING is not None
        else None
    )
    return SecretService(db, store=store) if store is not None else SecretService(db)


async def resolve_for_use(
    *,
    db: AsyncSession,
    owner_id: UUID,
    resource_type: SecretResourceType | str,
    resource_id: UUID,
    purpose: SecretPurpose | str,
    binding_id: UUID | None,
    request_id: UUID,
    source_ip: str | None = None,
    injection_config: dict[str, Any] | None = None,
) -> SecretValue:
    """Resolve exactly one authorized binding for one outbound operation."""

    store = (
        PostgresEncryptedSecretStore(db, keyring=_RUNTIME_KEYRING)
        if _RUNTIME_KEYRING is not None
        else None
    )
    service = SecretService(db, store=store) if store is not None else SecretService(db)
    return await service.resolve_for_use(
        owner_id=owner_id,
        target=BindingTarget(
            resource_type=resource_type,
            resource_id=resource_id,
            purpose=purpose,
            binding_id=binding_id,
            injection_config=injection_config,
        ),
        request_id=request_id,
        source_ip=source_ip,
    )


def _active_secret(secret: UserSecret | None) -> bool:
    if secret is None or SecretState(secret.state) != SecretState.ACTIVE:
        return False
    if secret.expires_at is None:
        return True
    expires_at = secret.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at > datetime.now(timezone.utc)


async def _active_mcp_binding_id(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
) -> UUID | None:
    """Find the sole active item-scoped MCP header binding."""

    result = await db.execute(
        select(SecretBinding, UserSecret)
        .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
        .where(
            SecretBinding.owner_id == owner_id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == installed_item_id,
            SecretBinding.purpose == SecretPurpose.MCP_AUTH_HEADER,
            UserSecret.owner_id == owner_id,
        )
    )
    active_ids = [binding.id for binding, secret in result.all() if _active_secret(secret)]
    return active_ids[0] if len(active_ids) == 1 else None


async def _active_mcp_binding(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
    purpose: SecretPurpose,
    allow_expired: bool = False,
) -> tuple[UUID, UUID] | None:
    """Like ``_active_mcp_binding_id``, but for any item-scoped MCP purpose
    and returning ``(binding_id, secret_id)`` — Stage 4's refresh needs the
    underlying secret row, not just the binding used to authorize resolves.

    ``allow_expired`` keeps an ACTIVE-but-expired secret in the running: the
    refresh is exactly what overwrites an expired access token, so skipping
    it would make an OAuth item left idle past its token TTL unrefreshable."""

    result = await db.execute(
        select(SecretBinding, UserSecret)
        .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
        .where(
            SecretBinding.owner_id == owner_id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == installed_item_id,
            SecretBinding.purpose == purpose,
            UserSecret.owner_id == owner_id,
        )
    )
    active = [
        (binding, secret)
        for binding, secret in result.all()
        if _active_secret(secret)
        or (allow_expired and secret is not None and SecretState(secret.state) == SecretState.ACTIVE)
    ]
    if len(active) != 1:
        return None
    binding, secret = active[0]
    return binding.id, secret.id


def _oauth_needs_refresh(auth: dict[str, Any]) -> bool:
    """Whether ``custom_config.auth`` (mode ``"oauth"``) is close enough to
    its stored ``expires_at`` to refresh proactively. No expiry on record
    (some providers never send ``expires_in``) means there is nothing to
    schedule — Stage 5's 401-triggered retry is the fallback for those."""

    raw = auth.get("expires_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        expires_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - datetime.now(timezone.utc) <= timedelta(seconds=_OAUTH_REFRESH_SKEW_SEC)


async def _refresh_oauth_tokens(
    *,
    db: AsyncSession,
    owner_id: UUID,
    item: UserInstalledItem,
    auth: dict[str, Any],
    request_id: UUID,
    source_ip: str | None,
) -> bool:
    """Exchange the item's stored refresh token for a new access token.

    Never raises: a failure here (provider down, refresh token revoked,
    malformed state) must leave the still-active access token exactly as it
    was, per Stage 4 acceptance — the caller keeps using it and, if it truly
    no longer works, Stage 5's 401 path is what surfaces ``needs_reauth``.
    """

    from app.core import mcp_oauth  # local: mcp_oauth -> mcp_gateway -> this module
    from app.models.mcp_oauth import McpOAuthClient

    issuer = auth.get("issuer")
    client_ref = auth.get("client_ref")
    resource = auth.get("resource")
    if not isinstance(issuer, str) or not issuer or not isinstance(client_ref, str) or not client_ref:
        return False
    try:
        client_uuid = UUID(client_ref)
    except ValueError:
        return False

    client_row = await db.get(McpOAuthClient, client_uuid)
    if client_row is None:
        return False

    refresh_binding = await _active_mcp_binding(
        db=db,
        owner_id=owner_id,
        installed_item_id=item.id,
        purpose=SecretPurpose.MCP_OAUTH_REFRESH,
    )
    if refresh_binding is None:
        return False
    refresh_binding_id, refresh_secret_id = refresh_binding

    try:
        refresh_value = await resolve_for_use(
            db=db,
            owner_id=owner_id,
            resource_type=SecretResourceType.INSTALLED_MCP,
            resource_id=item.id,
            purpose=SecretPurpose.MCP_OAUTH_REFRESH,
            binding_id=refresh_binding_id,
            request_id=request_id,
            source_ip=source_ip,
        )
        refresh_token = refresh_value.reveal()
    except Exception:
        logger.info("mcp_oauth_refresh outcome=failed reason=refresh_secret item=%s", item.id)
        return False

    client_secret = await mcp_oauth.resolve_manual_client_secret(
        db,
        client=client_row,
        owner_id=owner_id,
        installed_item_id=item.id,
        keyring=_RUNTIME_KEYRING,
    )
    try:
        token_response = await mcp_oauth.refresh_access_token(
            issuer=issuer,
            client_id=client_row.client_id,
            refresh_token=refresh_token,
            resource=resource if isinstance(resource, str) else None,
            client_secret=client_secret,
        )
    except Exception:
        logger.info("mcp_oauth_refresh outcome=failed reason=token_exchange item=%s", item.id)
        return False
    finally:
        refresh_token = None

    access_token = token_response.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        logger.info("mcp_oauth_refresh outcome=failed reason=no_access_token item=%s", item.id)
        return False

    access_binding = await _active_mcp_binding(
        db=db,
        owner_id=owner_id,
        installed_item_id=item.id,
        purpose=SecretPurpose.MCP_AUTH_HEADER,
        allow_expired=True,
    )
    if access_binding is None:
        return False
    _, access_secret_id = access_binding

    now = datetime.now(timezone.utc)
    expires_in = token_response.get("expires_in")
    token_expires_at = (
        now + timedelta(seconds=int(expires_in))
        if isinstance(expires_in, (int, float)) and expires_in > 0
        else None
    )

    store = (
        PostgresEncryptedSecretStore(db, keyring=_RUNTIME_KEYRING)
        if _RUNTIME_KEYRING is not None
        else PostgresEncryptedSecretStore(db)
    )
    await store.replace_value(
        owner_id=owner_id,
        secret_id=access_secret_id,
        value=SecretValue(access_token),
        request_id=request_id,
        source_ip=source_ip,
    )
    await store.update_metadata(
        owner_id=owner_id,
        secret_id=access_secret_id,
        expires_at=token_expires_at,
        request_id=request_id,
        source_ip=source_ip,
    )

    new_refresh_token = token_response.get("refresh_token")
    if isinstance(new_refresh_token, str) and new_refresh_token:
        await store.replace_value(
            owner_id=owner_id,
            secret_id=refresh_secret_id,
            value=SecretValue(new_refresh_token),
            request_id=request_id,
            source_ip=source_ip,
        )

    config = dict(item.custom_config or {})
    new_auth = dict(auth)
    new_auth["expires_at"] = token_expires_at.isoformat() if token_expires_at else None
    new_auth["needs_reauth"] = False
    config["auth"] = new_auth
    item.custom_config = config
    db.add(item)
    await db.commit()
    logger.info("mcp_oauth_refresh outcome=success item=%s", item.id)
    return True


async def _maybe_refresh_oauth_token(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
    request_id: UUID,
    source_ip: str | None,
) -> None:
    """Proactively refresh an OAuth-authorized item's access token when it is
    close to expiring, before a bearer header is ever issued for it.

    A no-op for bearer-mode items (no ``auth.mode == "oauth"``) and for
    static-forever OAuth tokens (no ``expires_at`` recorded). Swallows every
    failure by design — see ``_refresh_oauth_tokens``.
    """

    try:
        item = await db.get(UserInstalledItem, installed_item_id)
        if item is None or item.user_id != owner_id:
            return
        config = item.custom_config if isinstance(item.custom_config, dict) else {}
        auth = config.get("auth")
        if not isinstance(auth, dict) or auth.get("mode") != "oauth":
            return
        if not _oauth_needs_refresh(auth):
            return

        async with _oauth_refresh_lock(installed_item_id):
            # Re-read after the lock: a concurrent call may have already
            # refreshed and committed while this one waited its turn.
            await db.refresh(item)
            config = item.custom_config if isinstance(item.custom_config, dict) else {}
            auth = config.get("auth")
            if not isinstance(auth, dict) or auth.get("mode") != "oauth":
                return
            if not _oauth_needs_refresh(auth):
                return
            await _refresh_oauth_tokens(
                db=db,
                owner_id=owner_id,
                item=item,
                auth=auth,
                request_id=request_id,
                source_ip=source_ip,
            )
    except Exception:
        logger.info("mcp_oauth_refresh outcome=failed reason=unexpected item=%s", installed_item_id)


async def force_refresh_mcp_oauth_token(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
    request_id: UUID,
    source_ip: str | None = None,
) -> bool:
    """Refresh an OAuth item's access token regardless of its recorded expiry.

    Stage 5's 401 path: the upstream just rejected the current access token,
    so the recorded ``expires_at`` (if any) is already stale — waiting for
    ``_oauth_needs_refresh`` to agree would retry with the same dead token.
    Returns whether a new access token is now in place; a caller that gets
    ``False`` back should surface ``needs_reauth`` instead of retrying.
    """

    try:
        item = await db.get(UserInstalledItem, installed_item_id)
        if item is None or item.user_id != owner_id:
            return False
        async with _oauth_refresh_lock(installed_item_id):
            await db.refresh(item)
            config = item.custom_config if isinstance(item.custom_config, dict) else {}
            auth = config.get("auth")
            if not isinstance(auth, dict) or auth.get("mode") != "oauth":
                return False
            return await _refresh_oauth_tokens(
                db=db,
                owner_id=owner_id,
                item=item,
                auth=auth,
                request_id=request_id,
                source_ip=source_ip,
            )
    except Exception:
        logger.info("mcp_oauth_refresh outcome=failed reason=unexpected item=%s", installed_item_id)
        return False


async def mark_mcp_oauth_needs_reauth(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
) -> None:
    """Record that this OAuth item's access token no longer works.

    Best-effort and silent: Stage 6's pause card reads ``needs_reauth`` off
    ``custom_config.auth``, but a failure to persist it here must not turn
    an upstream 401 into a Hub-side error of its own.
    """

    try:
        item = await db.get(UserInstalledItem, installed_item_id)
        if item is None or item.user_id != owner_id:
            return
        config = item.custom_config if isinstance(item.custom_config, dict) else {}
        auth = config.get("auth")
        if not isinstance(auth, dict) or auth.get("mode") != "oauth" or auth.get("needs_reauth") is True:
            return
        new_config = dict(config)
        new_config["auth"] = {**auth, "needs_reauth": True}
        item.custom_config = new_config
        db.add(item)
        await db.commit()
    except Exception:
        logger.info("mcp_oauth_needs_reauth outcome=failed item=%s", installed_item_id)


async def resolve_mcp_upstream_auth(
    *,
    db: AsyncSession,
    owner_id: UUID,
    installed_item_id: UUID,
    binding_id: UUID | None = None,
    request_id: UUID | None = None,
    source_ip: str | None = None,
) -> dict[str, str]:
    """Resolve one item-scoped bearer for one internal Gateway request.

    ``resolve_for_use`` remains the authorization and audit boundary.  The
    returned mapping is the only credential-bearing object crossing Hub's
    Gateway boundary; labels, binding references, and legacy config are never
    used as credentials.
    """

    value: SecretValue | None = None
    resolved_request_id = request_id or uuid4()
    try:
        await _maybe_refresh_oauth_token(
            db=db,
            owner_id=owner_id,
            installed_item_id=installed_item_id,
            request_id=resolved_request_id,
            source_ip=source_ip,
        )
        selected_binding_id = binding_id
        if selected_binding_id is None:
            selected_binding_id = await _active_mcp_binding_id(
                db=db,
                owner_id=owner_id,
                installed_item_id=installed_item_id,
            )
        if selected_binding_id is None:
            # Preserve the SecretStore's value-free denied audit for a
            # missing/disabled binding, while ignoring any test/double that
            # might incorrectly return a value for ``binding_id=None``.
            try:
                await resolve_for_use(
                    db=db,
                    owner_id=owner_id,
                    resource_type=SecretResourceType.INSTALLED_MCP,
                    resource_id=installed_item_id,
                    purpose=SecretPurpose.MCP_AUTH_HEADER,
                    binding_id=None,
                    request_id=request_id or uuid4(),
                    source_ip=source_ip,
                )
            except Exception:
                pass
            raise McpSecretUnavailable
        value = await resolve_for_use(
            db=db,
            owner_id=owner_id,
            resource_type=SecretResourceType.INSTALLED_MCP,
            resource_id=installed_item_id,
            purpose=SecretPurpose.MCP_AUTH_HEADER,
            binding_id=selected_binding_id,
            request_id=request_id or uuid4(),
            source_ip=source_ip,
        )
        token = value.reveal()
        if (
            not token
            or len(token.encode("utf-8")) > 4096
            or any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in token)
        ):
            raise McpSecretUnavailable
        return {MCP_UPSTREAM_AUTH_HEADER: f"Bearer {token}"}
    except McpSecretUnavailable:
        raise
    except Exception:
        # Storage, ownership, and decryption details are intentionally mapped
        # to one value-free code and always fail closed before network I/O.
        raise McpSecretUnavailable from None
    finally:
        value = None


async def resolve_owned_llm_for_validation(
    *,
    db: AsyncSession,
    owner_id: UUID,
    secret_id: UUID,
    request_id: UUID,
    source_ip: str | None = None,
) -> SecretValue:
    """Resolve one owner-selected LLM secret for a provider health check."""

    store = (
        PostgresEncryptedSecretStore(db, keyring=_RUNTIME_KEYRING)
        if _RUNTIME_KEYRING is not None
        else PostgresEncryptedSecretStore(db)
    )
    return await store.resolve_owned_llm_for_validation(
        owner_id=owner_id,
        secret_id=secret_id,
        request_id=request_id,
        source_ip=source_ip,
    )


async def resolve_agent_secret(
    *,
    db: AsyncSession,
    owner_id: UUID,
    agent_id: UUID,
    config: dict[str, Any],
    request_id: UUID,
    source_ip: str | None = None,
    purpose: SecretPurpose = SecretPurpose.LLM_PRIMARY,
    binding_keys: tuple[str, ...] = ("primary_secret_binding_id", "secret_binding_id"),
    legacy_field: str = "api_key",
) -> SecretValue | None:
    """Resolve one agent LLM binding, with a controlled migration fallback."""

    binding_id = binding_id_from_config(config, *binding_keys)
    if binding_id is None:
        return legacy_secret_value(config, legacy_field)
    return await resolve_for_use(
        db=db,
        owner_id=owner_id,
        resource_type=SecretResourceType.AGENT,
        resource_id=agent_id,
        purpose=purpose,
        binding_id=binding_id,
        request_id=request_id,
        source_ip=source_ip,
    )


__all__ = [
    "MCP_UPSTREAM_AUTH_HEADER",
    "McpSecretUnavailable",
    "binding_id_from_config",
    "configure_runtime_keyring",
    "force_refresh_mcp_oauth_token",
    "legacy_secret_value",
    "mark_mcp_oauth_needs_reauth",
    "resolve_agent_secret",
    "resolve_for_use",
    "resolve_mcp_upstream_auth",
    "resolve_owned_llm_for_validation",
    "service_for",
]
