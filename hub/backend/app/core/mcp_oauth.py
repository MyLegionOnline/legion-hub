"""internal design notes Stage 1 — provider metadata cache and dynamic client registration.

This module never talks to a provider directly. Per the plan's "who calls
authorize/token/register" decision, every outbound hop to a provider's OAuth
endpoints crosses Gateway's SSRF perimeter (Stage 2's ``/v1/oauth/metadata``
and ``/v1/oauth/register``) instead of a second, weaker one grown here. Until
those routes exist this degrades to ``mcp_unreachable`` like any other
Gateway outage — the same shape a caller already handles.

Registration is idempotent on (issuer, redirect_uri): the same Hub origin
logging into the same provider a second time reuses the row this module
already wrote, never re-registers.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets as _secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.mcp_gateway import MCP_GATEWAY_TOKEN, MCP_GATEWAY_URL
from app.core.secrets.crypto import SecretValue
from app.core.secrets.store import BindingTarget, PostgresEncryptedSecretStore, SecretAccessDenied
from app.models.market import UserInstalledItem
from app.models.mcp_oauth import McpOAuthClient, McpOAuthFlow
from app.models.secret import (
    SecretActorType,
    SecretBinding,
    SecretKind,
    SecretPurpose,
    SecretResourceType,
    SecretState,
    UserSecret,
)

logger = logging.getLogger(__name__)

_GATEWAY_TIMEOUT_SEC = 10.0

#: Metadata documents are provider-controlled but not secret; a short TTL
#: bounds staleness without a network round trip on every login attempt.
_METADATA_CACHE_TTL_SEC = 600.0
_metadata_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


class McpOAuthError(Exception):
    """Base for a recoverable OAuth-setup failure — never a bare 500 upstream."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class NoDynamicRegistration(McpOAuthError):
    """The provider's authorization server has no ``registration_endpoint``.

    An expected outcome, not an exception the caller should treat as a bug —
    Stage 8's manual client_id/client_secret path is the answer for it.
    """

    def __init__(self, issuer: str):
        super().__init__(
            "mcp_oauth_no_dynamic_registration",
            f"{issuer} does not support dynamic client registration",
        )
        self.issuer = issuer


class ProviderUnresolved(McpOAuthError):
    """The MCP endpoint's protected-resource/authorization-server metadata
    never yielded a usable issuer or the two endpoints a login needs."""

    def __init__(self, resource: str):
        super().__init__("mcp_oauth_provider_unresolved", f"could not resolve an OAuth issuer for {resource}")
        self.resource = resource


class ManualClientInvalid(McpOAuthError):
    """The manual client_id/client_secret Stage 8 form submitted is unusable."""

    def __init__(self, message: str) -> None:
        super().__init__("mcp_oauth_manual_client_invalid", message)


class InvalidFlow(McpOAuthError):
    """``state`` does not name a still-usable login (Stage 3 acceptance:
    a one-shot ``state``, a repeat callback is rejected, not silently retried)."""

    def __init__(self) -> None:
        super().__init__("mcp_oauth_flow_invalid", "OAuth login is unknown, expired, or already used")


@dataclass(frozen=True)
class ClientRegistration:
    client: McpOAuthClient
    created: bool


@dataclass(frozen=True)
class ProviderMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    scope: str | None


@dataclass(frozen=True)
class LoginStart:
    flow: McpOAuthFlow
    authorization_url: str


@dataclass(frozen=True)
class LoginOutcome:
    ok: bool
    error_code: str | None = None
    # Set on success so the callback route can refresh the item's MCP
    # connection snapshot immediately — without this, a stale
    # "mcp_oauth_required" status can outlive a login that already succeeded,
    # since nothing else re-probes the endpoint on its own.
    installed_item_id: UUID | None = None


async def _gateway_oauth_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to one of Gateway's OAuth service routes, under service auth only.

    Unlike ``mcp_gateway._gateway_post`` this carries no room/user headers:
    fetching provider metadata and registering a client are Hub-wide
    operations, not actions taken on behalf of one room.
    """

    async with httpx.AsyncClient(timeout=_GATEWAY_TIMEOUT_SEC) as client:
        response = await client.post(
            f"{MCP_GATEWAY_URL.rstrip('/')}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {MCP_GATEWAY_TOKEN}"},
        )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise httpx.HTTPError("Gateway returned a non-object OAuth response")
    return body


async def _gateway_oauth_token_post(payload: dict[str, Any]) -> dict[str, Any]:
    """Like ``_gateway_oauth_post`` for ``/v1/oauth/token``, but surfaces the
    Gateway's own error code (``mcp_oauth_token_rejected`` /
    ``_unreachable``) instead of collapsing every non-2xx into one generic
    exception — the caller (``complete_login``) needs to tell "the provider
    said no" apart from "the provider could not be reached" for the status
    poll.
    """

    async with httpx.AsyncClient(timeout=_GATEWAY_TIMEOUT_SEC) as client:
        try:
            response = await client.post(
                f"{MCP_GATEWAY_URL.rstrip('/')}/v1/oauth/token",
                json=payload,
                headers={"Authorization": f"Bearer {MCP_GATEWAY_TOKEN}"},
            )
        except httpx.HTTPError:
            raise McpOAuthError("mcp_oauth_token_unreachable", "OAuth Gateway is unavailable") from None
    if response.status_code >= 400:
        code = "mcp_oauth_token_rejected"
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, dict) and isinstance(detail.get("code"), str):
            code = detail["code"]
        raise McpOAuthError(code, "Provider rejected the token request")
    try:
        body = response.json()
    except ValueError:
        raise McpOAuthError("mcp_oauth_token_rejected", "OAuth Gateway returned an invalid response") from None
    if not isinstance(body, dict):
        raise McpOAuthError("mcp_oauth_token_rejected", "OAuth Gateway returned an invalid response")
    return body


def _cache_get(kind: str, key: str) -> dict[str, Any] | None:
    cached = _metadata_cache.get((kind, key))
    if cached is None:
        return None
    fetched_at, document = cached
    if (time.monotonic() - fetched_at) >= _METADATA_CACHE_TTL_SEC:
        return None
    return document


def _cache_put(kind: str, key: str, document: dict[str, Any]) -> None:
    _metadata_cache[(kind, key)] = (time.monotonic(), document)


async def fetch_authorization_server_metadata(issuer: str) -> dict[str, Any] | None:
    """Cached ``/.well-known/oauth-authorization-server`` for ``issuer``."""

    cached = _cache_get("asm", issuer)
    if cached is not None:
        return cached
    try:
        document = await _gateway_oauth_post(
            "/v1/oauth/metadata", {"kind": "authorization_server", "issuer": issuer}
        )
    except httpx.HTTPError:
        return None
    _cache_put("asm", issuer, document)
    return document


async def fetch_protected_resource_metadata(resource: str) -> dict[str, Any] | None:
    """Cached ``/.well-known/oauth-protected-resource`` for ``resource``."""

    cached = _cache_get("prm", resource)
    if cached is not None:
        return cached
    try:
        document = await _gateway_oauth_post(
            "/v1/oauth/metadata", {"kind": "protected_resource", "resource": resource}
        )
    except httpx.HTTPError:
        return None
    _cache_put("prm", resource, document)
    return document


def _safe_registration_snapshot(asm: dict[str, Any]) -> dict[str, Any]:
    """Trim the authorization-server document down to what a re-login needs.

    Never stores the provider's raw client-registration response: that is
    where a confidential client's ``client_secret`` would appear, and Stage 8
    is explicit that a secret belongs only in Vault.
    """

    return {
        key: asm[key]
        for key in ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint")
        if isinstance(asm.get(key), str)
    }


async def register_client(
    db: AsyncSession,
    *,
    issuer: str,
    redirect_uri: str,
) -> ClientRegistration:
    """Reuse the Hub's client registration at ``issuer``, or create one via DCR.

    Raises :class:`NoDynamicRegistration` when the provider's authorization
    server metadata has no ``registration_endpoint`` — a distinct, expected
    outcome the caller routes to Stage 8's manual path, not a 500.
    """

    existing = await db.scalar(
        select(McpOAuthClient).where(
            McpOAuthClient.issuer == issuer,
            McpOAuthClient.redirect_uri == redirect_uri,
        )
    )
    if existing is not None:
        return ClientRegistration(client=existing, created=False)

    asm = await fetch_authorization_server_metadata(issuer)
    registration_endpoint = asm.get("registration_endpoint") if isinstance(asm, dict) else None
    if not isinstance(registration_endpoint, str) or not registration_endpoint:
        raise NoDynamicRegistration(issuer)

    response = await _gateway_oauth_post(
        "/v1/oauth/register",
        {
            "issuer": issuer,
            "registration_endpoint": registration_endpoint,
            "redirect_uris": [redirect_uri],
            "client_name": "Legion Hub",
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
    )
    client_id = response.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise McpOAuthError("mcp_oauth_registration_failed", f"{issuer} returned no client_id")

    client = McpOAuthClient(
        issuer=issuer,
        redirect_uri=redirect_uri,
        client_id=client_id,
        registration_source="dcr",
        metadata_json=_safe_registration_snapshot(asm),
    )
    db.add(client)
    try:
        await db.commit()
    except IntegrityError:
        # Lost a race against a concurrent login from the same origin — the
        # winner's row is what both callers should use, not a second client.
        await db.rollback()
        existing = await db.scalar(
            select(McpOAuthClient).where(
                McpOAuthClient.issuer == issuer,
                McpOAuthClient.redirect_uri == redirect_uri,
            )
        )
        if existing is None:
            raise
        return ClientRegistration(client=existing, created=False)
    await db.refresh(client)
    logger.info("mcp_oauth_register outcome=created issuer=%s", issuer)
    return ClientRegistration(client=client, created=True)


async def register_manual_client(
    db: AsyncSession,
    *,
    item: UserInstalledItem,
    user_id: UUID,
    redirect_uri: str,
    client_id: str,
    client_secret: str | None,
    keyring: Any,
) -> ClientRegistration:
    """Stage 8 — register the Hub's client at a provider that offers no
    ``registration_endpoint``, from a client_id/client_secret a person typed
    in after creating an app in the provider's own console.

    Resolves the same issuer ``register_client`` would (the provider's own
    protected-resource/authorization-server metadata), so ``start_login``'s
    subsequent lookup for (issuer, redirect_uri) finds this row and never
    attempts DCR against it. A resubmission for the same (issuer,
    redirect_uri) rotates the existing row's credentials rather than
    creating a duplicate client Legion would then have to choose between.
    """

    config = item.custom_config if isinstance(item.custom_config, dict) else {}
    resource = config.get("url")
    if not isinstance(resource, str) or not resource:
        raise McpOAuthError("mcp_oauth_endpoint_missing", "item has no MCP endpoint configured")
    if not client_id.strip():
        raise ManualClientInvalid("client_id is required")

    provider = await resolve_provider(resource)

    secret_binding_id: UUID | None = None
    if client_secret:
        store = PostgresEncryptedSecretStore(db, keyring=keyring)
        provider_label = (_origin(provider.issuer) or provider.issuer)[:64]
        meta = await store.create(
            owner_id=user_id,
            kind=SecretKind.CUSTOM,
            label="OAuth client secret",
            provider=provider_label,
            value=SecretValue(client_secret),
            actor_type=SecretActorType.USER,
        )
        binding = await store.bind(
            owner_id=user_id,
            secret_id=meta.id,
            target=BindingTarget(
                SecretResourceType.INSTALLED_MCP,
                item.id,
                SecretPurpose.MCP_OAUTH_CLIENT_SECRET,
                None,
            ),
            actor_type=SecretActorType.USER,
        )
        secret_binding_id = binding.id

    existing = await db.scalar(
        select(McpOAuthClient).where(
            McpOAuthClient.issuer == provider.issuer,
            McpOAuthClient.redirect_uri == redirect_uri,
        )
    )
    if existing is not None:
        existing.client_id = client_id
        existing.registration_source = "manual"
        if secret_binding_id is not None:
            existing.client_secret_binding_id = secret_binding_id
        db.add(existing)
        await db.commit()
        await db.refresh(existing)
        logger.info("mcp_oauth_register outcome=manual_updated issuer=%s", provider.issuer)
        return ClientRegistration(client=existing, created=False)

    client = McpOAuthClient(
        issuer=provider.issuer,
        redirect_uri=redirect_uri,
        client_id=client_id,
        client_secret_binding_id=secret_binding_id,
        registration_source="manual",
        metadata_json={
            "issuer": provider.issuer,
            "authorization_endpoint": provider.authorization_endpoint,
            "token_endpoint": provider.token_endpoint,
        },
    )
    db.add(client)
    await db.commit()
    await db.refresh(client)
    logger.info("mcp_oauth_register outcome=manual_created issuer=%s", provider.issuer)
    return ClientRegistration(client=client, created=True)


async def resolve_manual_client_secret(
    db: AsyncSession,
    *,
    client: McpOAuthClient,
    owner_id: UUID,
    installed_item_id: UUID,
    keyring: Any,
) -> str | None:
    """The confidential secret of a manually registered client, or ``None``
    for a DCR (public) client or one whose secret was never stored."""

    if client.registration_source != "manual" or client.client_secret_binding_id is None:
        return None
    store = PostgresEncryptedSecretStore(db, keyring=keyring)
    try:
        value = await store.resolve_for_use(
            owner_id=owner_id,
            target=BindingTarget(
                SecretResourceType.INSTALLED_MCP,
                installed_item_id,
                SecretPurpose.MCP_OAUTH_CLIENT_SECRET,
                expected_kind=SecretKind.CUSTOM,
                binding_id=client.client_secret_binding_id,
            ),
            request_id=uuid4(),
        )
    except SecretAccessDenied:
        return None
    return value.reveal()


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


#: ``McpOAuthFlow.scope`` column width.
_MAX_SCOPE_CHARS = 512


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s for s in value if isinstance(s, str) and s]


def _login_scope(prm: Any, asm: Any) -> str | None:
    """The scope to request at login, per MCP authorization spec: the
    protected resource's own ``scopes_supported`` — never the authorization
    server's, which lists every scope the whole provider knows (Cloudflare's
    runs to ~7.5k chars and blew up the flow insert). ``offline_access`` is
    added when the AS supports it, so the login yields a refresh token.
    Omitted (server default) when the PRM names none or the result is too
    long to store."""

    scopes = _string_list(prm.get("scopes_supported")) if isinstance(prm, dict) else []
    if not scopes:
        return None
    asm_scopes = _string_list(asm.get("scopes_supported")) if isinstance(asm, dict) else []
    if "offline_access" in asm_scopes and "offline_access" not in scopes:
        scopes.append("offline_access")
    scope = " ".join(scopes)
    return scope if len(scope) <= _MAX_SCOPE_CHARS else None


async def resolve_provider(resource: str) -> ProviderMetadata:
    """Discover ``resource``'s issuer and its authorize/token endpoints.

    Mirrors ``mcp_gateway.oauth_probe.detect_provider``'s own walk (protected
    -resource metadata's ``authorization_servers[0]``, falling back to the
    endpoint's own origin) — the probe only ever needed a display label,
    Stage 3 needs the full authorization-server document to build a login.
    """

    prm = await fetch_protected_resource_metadata(resource)
    issuer: str | None = None
    if isinstance(prm, dict):
        servers = prm.get("authorization_servers")
        if isinstance(servers, list) and servers and isinstance(servers[0], str):
            issuer = servers[0]
    if issuer is None:
        issuer = _origin(resource)
    if issuer is None:
        raise ProviderUnresolved(resource)

    asm = await fetch_authorization_server_metadata(issuer)
    authorization_endpoint = asm.get("authorization_endpoint") if isinstance(asm, dict) else None
    token_endpoint = asm.get("token_endpoint") if isinstance(asm, dict) else None
    if not isinstance(authorization_endpoint, str) or not isinstance(token_endpoint, str):
        raise ProviderUnresolved(resource)
    scope = _login_scope(prm, asm)
    return ProviderMetadata(
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        scope=scope,
    )


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(_secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


async def start_login(
    db: AsyncSession,
    *,
    item: UserInstalledItem,
    user_id: UUID,
    redirect_uri: str,
) -> LoginStart:
    """Begin one login for ``item``: resolve its provider, register the Hub
    as a DCR client if needed, and mint a one-shot PKCE/state pair.

    Raises :class:`NoDynamicRegistration` when the provider has no
    ``registration_endpoint`` (Stage 8's manual path is the answer, not this
    function) and :class:`ProviderUnresolved` when the endpoint's own
    metadata never named a usable issuer.
    """

    config = item.custom_config if isinstance(item.custom_config, dict) else {}
    resource = config.get("url")
    if not isinstance(resource, str) or not resource:
        raise McpOAuthError("mcp_oauth_endpoint_missing", "item has no MCP endpoint configured")

    provider = await resolve_provider(resource)
    registration = await register_client(db, issuer=provider.issuer, redirect_uri=redirect_uri)
    code_verifier, code_challenge = _pkce_pair()
    state = _secrets.token_urlsafe(32)

    flow = McpOAuthFlow(
        state=state,
        code_verifier=code_verifier,
        installed_item_id=item.id,
        user_id=user_id,
        redirect_uri=redirect_uri,
        issuer=provider.issuer,
        client_id=registration.client.client_id,
        resource=resource,
        scope=provider.scope,
    )
    db.add(flow)
    await db.commit()
    await db.refresh(flow)

    query = {
        "response_type": "code",
        "client_id": registration.client.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if provider.scope:
        query["scope"] = provider.scope
    authorization_url = f"{provider.authorization_endpoint}?{urlencode(query)}"
    return LoginStart(flow=flow, authorization_url=authorization_url)


async def _exchange_authorization_code(
    flow: McpOAuthFlow, *, code: str, client_secret: str | None = None
) -> dict[str, Any]:
    payload = {
        "token_endpoint": (await _asm_token_endpoint(flow.issuer)),
        "grant_type": "authorization_code",
        "client_id": flow.client_id,
        "code": code,
        "redirect_uri": flow.redirect_uri,
        "code_verifier": flow.code_verifier,
        "resource": flow.resource,
    }
    if client_secret:
        payload["client_secret"] = client_secret
    return await _gateway_oauth_token_post(payload)


async def _asm_token_endpoint(issuer: str) -> str:
    asm = await fetch_authorization_server_metadata(issuer)
    token_endpoint = asm.get("token_endpoint") if isinstance(asm, dict) else None
    if not isinstance(token_endpoint, str) or not token_endpoint:
        raise ProviderUnresolved(issuer)
    return token_endpoint


async def refresh_access_token(
    *,
    issuer: str,
    client_id: str,
    refresh_token: str,
    resource: str | None,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Exchange a stored refresh token for a new access token (Stage 4).

    Raises :class:`McpOAuthError` on any rejection or transport failure —
    ``secrets.runtime``'s caller treats that as "refresh did not happen" and
    falls back to the still-active access token rather than erasing it.
    ``client_secret`` is only ever set for a manually registered
    (Stage 8) confidential client — a DCR public client never has one.
    """

    payload: dict[str, Any] = {
        "token_endpoint": await _asm_token_endpoint(issuer),
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if resource:
        payload["resource"] = resource
    if client_secret:
        payload["client_secret"] = client_secret
    return await _gateway_oauth_token_post(payload)


async def _deactivate_stale_oauth_secrets(
    db: AsyncSession,
    store: PostgresEncryptedSecretStore,
    *,
    owner_id: UUID,
    installed_item_id: UUID,
) -> None:
    """Disable any still-active MCP_AUTH_HEADER/MCP_OAUTH_REFRESH secret
    already bound to this item, before a fresh login binds new ones.

    Both purposes are deliberately outside ``_SINGLE_SECRET_PURPOSES``
    (``secrets/store.py``) because a plain bearer MCP can legitimately draw
    several secrets at once — but a repeat OAuth login is a replace, not an
    addition, and leaving the old one active breaks
    ``resolve_mcp_upstream_auth``'s exactly-one-active-binding lookup.
    """

    for purpose in (SecretPurpose.MCP_AUTH_HEADER, SecretPurpose.MCP_OAUTH_REFRESH):
        result = await db.execute(
            select(SecretBinding, UserSecret)
            .join(UserSecret, UserSecret.id == SecretBinding.secret_id)
            .where(
                SecretBinding.owner_id == owner_id,
                SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
                SecretBinding.resource_id == installed_item_id,
                SecretBinding.purpose == purpose,
                UserSecret.owner_id == owner_id,
                UserSecret.state == SecretState.ACTIVE,
            )
        )
        for _binding, secret in result.all():
            await store.update_metadata(owner_id=owner_id, secret_id=secret.id, enabled=False)


async def complete_login(
    db: AsyncSession,
    *,
    state: str,
    code: str,
    keyring: Any,
) -> LoginOutcome:
    """Consume one ``state``, exchange ``code`` for tokens, and persist them.

    One-shot by construction: the row is looked up with ``consumed_at IS
    NULL`` and marked consumed in the same transaction that commits the
    result, so a replayed callback (the same code from a doubled browser
    request, or an attacker replaying an observed redirect) finds nothing to
    consume a second time — Stage 3 acceptance: "a repeated callback is rejected".
    """

    flow = await db.scalar(
        select(McpOAuthFlow).where(McpOAuthFlow.state == state, McpOAuthFlow.consumed_at.is_(None))
    )
    if flow is None:
        raise InvalidFlow()
    now = datetime.now(timezone.utc)
    expires_at = flow.expires_at if flow.expires_at.tzinfo is not None else flow.expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        flow.consumed_at = now
        flow.outcome = "error"
        flow.error_code = "mcp_oauth_flow_expired"
        flow.code_verifier = ""
        db.add(flow)
        await db.commit()
        raise InvalidFlow()

    item = await db.get(UserInstalledItem, flow.installed_item_id)
    if item is None or item.user_id != flow.user_id:
        flow.consumed_at = now
        flow.outcome = "error"
        flow.error_code = "mcp_oauth_item_missing"
        flow.code_verifier = ""
        db.add(flow)
        await db.commit()
        return LoginOutcome(ok=False, error_code="mcp_oauth_item_missing")

    client_row = await db.scalar(
        select(McpOAuthClient).where(
            McpOAuthClient.issuer == flow.issuer,
            McpOAuthClient.redirect_uri == flow.redirect_uri,
        )
    )
    client_secret = (
        await resolve_manual_client_secret(
            db,
            client=client_row,
            owner_id=flow.user_id,
            installed_item_id=item.id,
            keyring=keyring,
        )
        if client_row is not None
        else None
    )

    try:
        token_response = await _exchange_authorization_code(flow, code=code, client_secret=client_secret)
    except McpOAuthError as exc:
        flow.consumed_at = now
        flow.outcome = "error"
        flow.error_code = exc.code
        flow.code_verifier = ""
        db.add(flow)
        await db.commit()
        return LoginOutcome(ok=False, error_code=exc.code)
    finally:
        client_secret = None

    access_token = token_response.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        flow.consumed_at = now
        flow.outcome = "error"
        flow.error_code = "mcp_oauth_token_rejected"
        flow.code_verifier = ""
        db.add(flow)
        await db.commit()
        return LoginOutcome(ok=False, error_code="mcp_oauth_token_rejected")

    refresh_token = token_response.get("refresh_token")
    expires_in = token_response.get("expires_in")
    token_expires_at = (
        now + timedelta(seconds=int(expires_in))
        if isinstance(expires_in, (int, float)) and expires_in > 0
        else None
    )

    store = PostgresEncryptedSecretStore(db, keyring=keyring)
    # A repeat login (a second "Sign in", or Stage 6's "Re-authorize") must
    # replace the previous access/refresh secret, not accumulate a second
    # active one next to it: resolve_mcp_upstream_auth's
    # _active_mcp_binding_id demands exactly one active binding for the
    # purpose and silently gives up (McpSecretUnavailable) the moment there
    # are two, permanently breaking a server that was already logged into
    # successfully once.
    await _deactivate_stale_oauth_secrets(db, store, owner_id=flow.user_id, installed_item_id=item.id)
    provider_label = (_origin(flow.issuer) or flow.issuer)[:64]
    access_meta = await store.create(
        owner_id=flow.user_id,
        kind=SecretKind.MCP_HEADER,
        label="OAuth access token",
        provider=provider_label,
        value=SecretValue(access_token),
        expires_at=token_expires_at,
        actor_type=SecretActorType.SYSTEM,
    )
    await store.bind(
        owner_id=flow.user_id,
        secret_id=access_meta.id,
        target=BindingTarget(
            SecretResourceType.INSTALLED_MCP,
            item.id,
            SecretPurpose.MCP_AUTH_HEADER,
            {"mode": "bearer"},
        ),
        actor_type=SecretActorType.SYSTEM,
    )
    if isinstance(refresh_token, str) and refresh_token:
        refresh_meta = await store.create(
            owner_id=flow.user_id,
            kind=SecretKind.MCP_HEADER,
            label="OAuth refresh token",
            provider=provider_label,
            value=SecretValue(refresh_token),
            actor_type=SecretActorType.SYSTEM,
        )
        await store.bind(
            owner_id=flow.user_id,
            secret_id=refresh_meta.id,
            target=BindingTarget(
                SecretResourceType.INSTALLED_MCP,
                item.id,
                SecretPurpose.MCP_OAUTH_REFRESH,
                None,
            ),
            actor_type=SecretActorType.SYSTEM,
        )

    config = dict(item.custom_config or {})
    config["auth"] = {
        "mode": "oauth",
        "issuer": flow.issuer,
        "resource": flow.resource,
        "client_ref": str(client_row.id) if client_row is not None else None,
        "scope": flow.scope,
        "expires_at": token_expires_at.isoformat() if token_expires_at else None,
        "needs_reauth": False,
    }
    item.custom_config = config
    db.add(item)

    flow.consumed_at = now
    flow.outcome = "success"
    flow.error_code = None
    flow.code_verifier = ""
    db.add(flow)
    await db.commit()
    logger.info("mcp_oauth_login outcome=success issuer=%s", flow.issuer)
    return LoginOutcome(ok=True, installed_item_id=item.id)


__all__ = [
    "ClientRegistration",
    "InvalidFlow",
    "LoginOutcome",
    "LoginStart",
    "ManualClientInvalid",
    "McpOAuthError",
    "NoDynamicRegistration",
    "ProviderMetadata",
    "ProviderUnresolved",
    "complete_login",
    "fetch_authorization_server_metadata",
    "fetch_protected_resource_metadata",
    "refresh_access_token",
    "register_client",
    "register_manual_client",
    "resolve_manual_client_secret",
    "resolve_provider",
    "start_login",
]
