"""Room-scoped VPS server inventory and connection resolver.

``internal design notes`` §3.4 / §3.3, Stage 3.  A "server" is not a new table —
it is a naming convention over the existing Room variables/secrets:

    VPS_<NAME>_HOST            variable  — IP or DNS name
    VPS_<NAME>_PORT            variable  — SSH port, default 22
    VPS_<NAME>_USER            variable  — everyday login (post-prepare: legion)
    VPS_<NAME>_FINGERPRINT     variable  — pinned ``SHA256:...`` host key
    VPS_<NAME>_PREPARED_AT     variable  — ISO timestamp, set by Stage 8
    VPS_<NAME>_DEPLOY_CREDENTIAL  variable  — ssh | https, set by Stage 13
    VPS_<NAME>_DEPLOY_KEY_ID      variable  — the forge's id for the deploy key
    VPS_<NAME>_DEPLOY_REMOTE      variable  — clone URL the server must use
    VPS_<NAME>_DEPLOY_TOKEN    secret, purpose=git_deploy, kind=room_env
    VPS_<NAME>_KEY             secret, purpose=vps_access, kind=ssh_private_key
    VPS_<NAME>_PASSWORD        secret, purpose=vps_access, kind=room_env
    VPS_<NAME>_ROOT_KEY        secret, purpose=vps_access, kind=ssh_private_key
    VPS_<NAME>_ROOT_PASSWORD   secret, purpose=vps_access, kind=room_env

``<NAME>`` is what an agent calls the server by (``server="prod"``).  The
root credentials are write-only from here: :func:`resolve_connection` never
reads a ``*_ROOT_*`` name — only the human-run bootstrap (Stage 8) does.

This module is the *only* place ``purpose=vps_access`` values are resolved.
They never flow through ``resolve_room_secrets_mixed`` (bash's ``secrets:``
list) or the ``.env`` sync — see ``room_secrets.vps_access_bindings``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote, unquote
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import room_config
from app.core.secret_config import configured_secret
from app.core.secrets import room_secrets
from app.core.secrets.room_secrets import EnvironmentLocked, RoomSecretNotAvailable
from app.core.secrets.service import SecretService
from app.models.room_variable import RoomVariable
from app.models.secret import SecretKind, SecretPurpose

SERVER_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")

_VAR_SUFFIXES = (
    "HOST", "PORT", "USER", "FINGERPRINT", "PREPARED_AT",
    # internal design notes Stage 13 — the repository credential this server holds.
    # All three are metadata, not secrets: which kind of credential was
    # issued, the forge's id for the public key (so "Revoke" deletes *that*
    # key rather than guessing by title), and the clone URL the server is
    # supposed to use, which for SSH is a per-Room alias, not the forge's own
    # host name.
    "DEPLOY_CREDENTIAL", "DEPLOY_KEY_ID", "DEPLOY_REMOTE",
)
_ROOT_SECRET_SUFFIXES = ("ROOT_KEY", "ROOT_PASSWORD")

#: Stage 13's HTTPS fallback credential (``purpose=git_deploy``). Named here
#: rather than in ``vps_deploy_key.py`` so :func:`remove_server` can clean it
#: up without importing that module.
DEPLOY_TOKEN_SUFFIX = "DEPLOY_TOKEN"

DEFAULT_PORT = 22
DEFAULT_USER = "root"
DEFAULT_IDLE_TTL_SEC = 60
DEFAULT_CONNECT_TIMEOUT_SEC = 15
DEFAULT_EGRESS_TICKET_TTL_SEC = 30

#: Operator secret for the egress-ticket HMAC (§3.13). The broker that
#: verifies these tickets is Stage 4; this module is the one place both the
#: credential resolve and the network permission are decided together, so the
#: ticket is minted here even before a broker exists to consume it.
#: No built-in default: without it no ticket is minted (``EgressNotConfigured``).
_EGRESS_TICKET_SECRET = configured_secret("VPS_EGRESS_TICKET_SECRET")


class VpsInventoryError(RuntimeError):
    code = "vps_inventory_error"
    http_status = 400


class ServerInvalid(VpsInventoryError):
    code = "server_invalid"
    http_status = 422


class ServerExists(VpsInventoryError):
    code = "server_exists"
    http_status = 409


class ServerNotFound(VpsInventoryError):
    """Unknown server name, or a known name with no ``HOST`` configured yet."""

    code = "server_not_found"
    http_status = 404


class ServerCredentialsMissing(VpsInventoryError):
    """The server has a host but neither an everyday key nor a password."""

    code = "server_credentials_missing"
    http_status = 422


class RootCredentialsMissing(VpsInventoryError):
    """No ``*_ROOT_KEY``/``*_ROOT_PASSWORD`` for a server being prepared.

    Raised only by :func:`resolve_root_connection` — the everyday resolver
    never looks at those names at all (§3.4).
    """

    code = "root_credentials_missing"
    http_status = 422


class EgressNotConfigured(VpsInventoryError):
    """``VPS_EGRESS_TICKET_SECRET`` is not set, so no ticket can be signed.

    A ticket signed with an empty or well-known key would let anyone with
    network access to ``legion-egress`` reach any host.
    """

    code = "egress_not_configured"
    http_status = 503


class EgressTicketInvalid(VpsInventoryError):
    """A ticket string that does not parse or verify against our own secret.

    Only used by :func:`verify_egress_ticket`, kept here for parity tests —
    the actual enforcement of this format lives in ``legion-egress``
    (Stage 4), a separate Node process that re-implements the same HMAC
    check against the same ``VPS_EGRESS_TICKET_SECRET``.
    """

    code = "egress_ticket_invalid"
    http_status = 400


# Re-exported so callers (``mcp_gateway.py``) only need one import for the
# whole resolve path.
__all__ = [
    "SERVER_NAME_RE",
    "VpsInventoryError",
    "ServerInvalid",
    "ServerExists",
    "ServerNotFound",
    "ServerCredentialsMissing",
    "EgressNotConfigured",
    "EgressTicketInvalid",
    "EnvironmentLocked",
    "VpsConnection",
    "EgressTicketClaims",
    "validate_server_name",
    "list_servers",
    "server_coordinates",
    "add_server",
    "remove_server",
    "forget_root_credentials",
    "set_server_variable",
    "clear_server_variable",
    "set_fingerprint",
    "DEPLOY_TOKEN_SUFFIX",
    "resolve_connection",
    "resolve_root_connection",
    "build_connection",
    "RootCredentialsMissing",
    "issue_egress_ticket",
    "verify_egress_ticket",
]


def validate_server_name(raw: str) -> str:
    name = str(raw or "").strip().upper()
    if not SERVER_NAME_RE.match(name):
        raise ServerInvalid(
            "server name must match ^[A-Z][A-Z0-9_]{0,31}$ (letters/digits/underscore, "
            f"starting with a letter; got {raw!r})"
        )
    return name


def _var_name(name: str, suffix: str) -> str:
    return f"VPS_{name}_{suffix}"


def _pick(rows: list[RoomVariable], environment: Optional[str]) -> Optional[RoomVariable]:
    """Exact-environment match wins; else the shared (``NULL``) one."""
    if environment is not None:
        for r in rows:
            if r.environment == environment:
                return r
    for r in rows:
        if r.environment is None:
            return r
    return None


async def _all_vps_variables(db: AsyncSession, room_id: UUID) -> dict[str, list[RoomVariable]]:
    """``{VPS_<NAME>_<SUFFIX>: [rows across environments]}`` for every server var."""
    rows = await room_config.list_variables(db, room_id, environment=room_config._UNSET)
    grouped: dict[str, list[RoomVariable]] = {}
    for row in rows:
        if not row.name.startswith("VPS_"):
            continue
        grouped.setdefault(row.name, []).append(row)
    return grouped


def _server_names_from_vars(grouped: dict[str, list[RoomVariable]]) -> set[str]:
    names: set[str] = set()
    for var_name in grouped:
        for suffix in _VAR_SUFFIXES:
            marker = f"_{suffix}"
            if var_name.endswith(marker) and var_name.startswith("VPS_"):
                candidate = var_name[len("VPS_"):-len(marker)]
                if candidate and SERVER_NAME_RE.match(candidate):
                    names.add(candidate)
                break
    return names


async def list_servers(
    db: AsyncSession, room_id: UUID, *, environment: Optional[str] = None
) -> list[dict[str, Any]]:
    """Metadata only — host/port/user/fingerprint/prepared_at + credential
    presence flags. Never a secret value.
    """
    grouped = await _all_vps_variables(db, room_id)
    names = sorted(_server_names_from_vars(grouped))
    bindings = await room_secrets.vps_access_bindings(db, room_id)

    def has_secret(name: str, suffix: str) -> bool:
        env_name = _var_name(name, suffix)
        for b in bindings:
            cfg = b.injection_config or {}
            if cfg.get("env_name") == env_name:
                return True
        return False

    out: list[dict[str, Any]] = []
    for name in names:
        host_row = _pick(grouped.get(_var_name(name, "HOST"), []), environment)
        port_row = _pick(grouped.get(_var_name(name, "PORT"), []), environment)
        user_row = _pick(grouped.get(_var_name(name, "USER"), []), environment)
        fp_row = _pick(grouped.get(_var_name(name, "FINGERPRINT"), []), environment)
        prepared_row = _pick(grouped.get(_var_name(name, "PREPARED_AT"), []), environment)
        deploy_row = _pick(grouped.get(_var_name(name, "DEPLOY_CREDENTIAL"), []), environment)
        deploy_key_row = _pick(grouped.get(_var_name(name, "DEPLOY_KEY_ID"), []), environment)
        deploy_remote_row = _pick(grouped.get(_var_name(name, "DEPLOY_REMOTE"), []), environment)
        out.append(
            {
                "name": name,
                "host": host_row.value if host_row else None,
                "port": int(port_row.value) if port_row and port_row.value.isdigit() else DEFAULT_PORT,
                "user": user_row.value if user_row else DEFAULT_USER,
                "fingerprint": fp_row.value if fp_row else None,
                "prepared_at": prepared_row.value if prepared_row else None,
                "has_key": has_secret(name, "KEY"),
                "has_password": has_secret(name, "PASSWORD"),
                "has_root_key": has_secret(name, "ROOT_KEY"),
                "has_root_password": has_secret(name, "ROOT_PASSWORD"),
                # Stage 13: which repository credential this server holds, if
                # any. `none` is a real answer an agent and the UI both act
                # on ("ask a person to issue one"), so it is never omitted.
                "deploy_credential": deploy_row.value if deploy_row else "none",
                "deploy_key_id": deploy_key_row.value if deploy_key_row else None,
                "deploy_remote": deploy_remote_row.value if deploy_remote_row else None,
            }
        )
    return out


async def _server_exists(db: AsyncSession, room_id: UUID, name: str, environment: Optional[str]) -> bool:
    """Whether ``VPS_<name>_HOST`` already has a row in this exact environment scope.

    Mirrors ``room_config._assert_unique``'s (room, name, environment) triple —
    the same server name may legitimately exist in the shared scope *and* in
    a named environment as separate rows.
    """
    norm_env = room_config.normalize_environment(environment)
    grouped = await _all_vps_variables(db, room_id)
    for row in grouped.get(_var_name(name, "HOST"), []):
        if row.environment == norm_env:
            return True
    return False


async def add_server(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    host: str,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    root_password: Optional[str] = None,
    root_key: Optional[str] = None,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> dict[str, Any]:
    """Add a server: variables for the coordinates, a secret for root access.

    Exactly one of ``root_password`` / ``root_key`` is required — this is the
    one-time credential Stage 8's bootstrap consumes; §3.4/§3.6 forbid the
    resolver (this module's own :func:`resolve_connection`) from ever reading
    it back.

    Every step after the existence check is best-effort-rolled-back on
    failure: ``create_variable``/``bind_room_secret`` each commit on their
    own (no shared transaction to wrap), so a failure partway through — most
    likely the credential bind — explicitly deletes the HOST/PORT/USER rows
    already committed rather than leaving a credential-less, undeletable
    server behind.
    """
    name = validate_server_name(name)
    host = str(host or "").strip()
    if not host:
        raise ServerInvalid("host is required")
    if bool(root_password) == bool(root_key):
        raise ServerInvalid("exactly one of root_password or root_key is required")
    port = int(port or DEFAULT_PORT)
    if not (1 <= port <= 65535):
        raise ServerInvalid("port must be between 1 and 65535")
    user = str(user or DEFAULT_USER).strip() or DEFAULT_USER

    if await _server_exists(db, room_id, name, environment):
        raise ServerExists(name)

    created_rows: list[RoomVariable] = []
    try:
        created_rows.append(await room_config.create_variable(
            db, room_id, name=_var_name(name, "HOST"), value=host,
            environment=environment, materialize_to_env=False, created_by=actor_user_id,
        ))
        created_rows.append(await room_config.create_variable(
            db, room_id, name=_var_name(name, "PORT"), value=str(port),
            environment=environment, materialize_to_env=False, created_by=actor_user_id,
        ))
        created_rows.append(await room_config.create_variable(
            db, room_id, name=_var_name(name, "USER"), value=user,
            environment=environment, materialize_to_env=False, created_by=actor_user_id,
        ))

        if root_key is not None:
            await room_secrets.bind_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(name, "ROOT_KEY"), value=root_key,
                environment=environment, purpose=SecretPurpose.VPS_ACCESS,
                kind_override=SecretKind.SSH_PRIVATE_KEY,
                request_id=request_id, source_ip=source_ip,
            )
        else:
            await room_secrets.bind_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(name, "ROOT_PASSWORD"), value=root_password,
                environment=environment, purpose=SecretPurpose.VPS_ACCESS,
                kind_override=SecretKind.ROOM_ENV,
                request_id=request_id, source_ip=source_ip,
            )
    except Exception:
        for row in created_rows:
            try:
                await room_config.delete_variable(db, row)
            except Exception:
                pass
        raise

    rows = await list_servers(db, room_id, environment=environment)
    return next(r for r in rows if r["name"] == name)


async def remove_server(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> None:
    """Remove a server from the inventory entirely: coordinates and every credential.

    Unlike :func:`forget_root_credentials` (root creds only), this also drops
    the everyday HOST/PORT/USER/FINGERPRINT/PREPARED_AT variables and the
    everyday KEY/PASSWORD secret, so the server disappears from
    :func:`list_servers` for this environment scope. Best-effort per row: a
    half-prepared server missing some of them is not an error.
    """
    name = validate_server_name(name)
    norm_env = room_config.normalize_environment(environment)
    grouped = await _all_vps_variables(db, room_id)
    for suffix in _VAR_SUFFIXES:
        for row in grouped.get(_var_name(name, suffix), []):
            if row.environment == norm_env:
                await room_config.delete_variable(db, row)
    for suffix in ("KEY", "PASSWORD", *_ROOT_SECRET_SUFFIXES):
        try:
            await room_secrets.unbind_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(name, suffix), environment=environment,
                purpose=SecretPurpose.VPS_ACCESS,
                request_id=request_id, source_ip=source_ip,
            )
        except RoomSecretNotAvailable:
            pass
    # Stage 13: the HTTPS deploy token is a `git_deploy` binding, so the
    # vps_access loop above does not see it. Left behind it would be a
    # credential for a server the Room no longer knows about — and a name
    # collision the next time the same server name is added.
    try:
        await room_secrets.unbind_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=_var_name(name, DEPLOY_TOKEN_SUFFIX), environment=environment,
            purpose=SecretPurpose.GIT_DEPLOY,
            request_id=request_id, source_ip=source_ip,
        )
    except RoomSecretNotAvailable:
        pass


async def forget_root_credentials(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> None:
    """Delete ``ROOT_KEY``/``ROOT_PASSWORD`` for one server (plan §3.6 step 10).

    Best-effort per suffix: whichever of the two exists is unbound, a missing
    one is not an error.
    """
    name = validate_server_name(name)
    for suffix in _ROOT_SECRET_SUFFIXES:
        try:
            await room_secrets.unbind_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(name, suffix), environment=environment,
                purpose=SecretPurpose.VPS_ACCESS,
                request_id=request_id, source_ip=source_ip,
            )
        except RoomSecretNotAvailable:
            pass


async def set_server_variable(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    suffix: str,
    value: str,
    environment: Optional[str] = None,
) -> None:
    """Upsert one ``VPS_<name>_<SUFFIX>`` variable in one environment scope.

    Only the suffixes this module recognises (:data:`_VAR_SUFFIXES`) may be
    written: a typo would otherwise create a variable that looks like part of
    a server but is invisible to :func:`list_servers` and
    :func:`resolve_connection`.
    """
    name = validate_server_name(name)
    if suffix not in _VAR_SUFFIXES:
        raise ServerInvalid(f"unknown server variable suffix {suffix!r}")
    value = str(value or "").strip()
    if not value:
        raise ServerInvalid(f"{suffix} value is required")
    norm_env = room_config.normalize_environment(environment)
    var_name = _var_name(name, suffix)
    grouped = await _all_vps_variables(db, room_id)
    existing = next(
        (row for row in grouped.get(var_name, []) if row.environment == norm_env), None
    )
    if existing is not None:
        await room_config.update_variable(db, existing, fields={"value": value})
    else:
        await room_config.create_variable(
            db, room_id, name=var_name, value=value,
            environment=environment, materialize_to_env=False, created_by=actor_user_id,
        )


async def clear_server_variable(
    db: AsyncSession,
    *,
    room_id: UUID,
    name: str,
    suffix: str,
    environment: Optional[str] = None,
) -> None:
    """Delete one ``VPS_<name>_<SUFFIX>`` variable if it exists (Stage 13 revoke).

    The mirror image of :func:`set_server_variable`: revoking a deploy
    credential has to leave the inventory saying "none", not "ssh, and the
    key id is one the forge no longer has".
    """
    name = validate_server_name(name)
    if suffix not in _VAR_SUFFIXES:
        raise ServerInvalid(f"unknown server variable suffix {suffix!r}")
    norm_env = room_config.normalize_environment(environment)
    grouped = await _all_vps_variables(db, room_id)
    for row in grouped.get(_var_name(name, suffix), []):
        if row.environment == norm_env:
            await room_config.delete_variable(db, row)


async def set_fingerprint(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    fingerprint: str,
    environment: Optional[str] = None,
) -> None:
    """Pin (or re-pin) ``VPS_<name>_FINGERPRINT`` — the §3.5 TOFU accept.

    Called only after an operator explicitly approves a ``host_key_unknown``
    or ``host_key_mismatch`` ask-user card (``mcp_gateway._confirm_vps_host_trust``)
    or the same decision inside the preparation wizard (``vps_bootstrap.py``,
    Stage 8); never invoked on the strength of the tool call alone. A
    fingerprint is a Room variable, not a secret — it identifies the server,
    it does not grant access to it.
    """
    await set_server_variable(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name,
        suffix="FINGERPRINT", value=fingerprint, environment=environment,
    )


@dataclass(frozen=True)
class VpsConnection:
    """Resolved ``_legion_conn`` payload plus what to mask in the result."""

    legion_conn: dict[str, Any]
    #: ``{name: value}`` for ``room_masking.track_room_secrets`` — never the
    #: host/user (not secret), only whatever credential was used.
    mask_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ServerCoordinates:
    """The non-secret half of a server: everything but the credential."""

    name: str
    host: str
    port: int
    user: str
    fingerprint: Optional[str]
    prepared_at: Optional[str]


async def server_coordinates(
    db: AsyncSession, room_id: UUID, server: str, *, environment: Optional[str] = None
) -> ServerCoordinates:
    """Host/port/user/fingerprint for one server, or :class:`ServerNotFound`."""
    name = validate_server_name(server)
    grouped = await _all_vps_variables(db, room_id)
    host_row = _pick(grouped.get(_var_name(name, "HOST"), []), environment)
    if host_row is None:
        raise ServerNotFound(name)
    port_row = _pick(grouped.get(_var_name(name, "PORT"), []), environment)
    user_row = _pick(grouped.get(_var_name(name, "USER"), []), environment)
    fp_row = _pick(grouped.get(_var_name(name, "FINGERPRINT"), []), environment)
    prepared_row = _pick(grouped.get(_var_name(name, "PREPARED_AT"), []), environment)
    return ServerCoordinates(
        name=name,
        host=host_row.value,
        port=int(port_row.value) if port_row and port_row.value.isdigit() else DEFAULT_PORT,
        user=user_row.value if user_row else DEFAULT_USER,
        fingerprint=fp_row.value if fp_row else None,
        prepared_at=prepared_row.value if prepared_row else None,
    )


def build_connection(
    *,
    room_id: UUID,
    server: str,
    host: str,
    port: int,
    user: str,
    auth: dict[str, Any],
    fingerprint: Optional[str],
    call_id: UUID,
    idle_ttl_sec: int = DEFAULT_IDLE_TTL_SEC,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Assemble one ``_legion_conn`` block, egress ticket included.

    Shared by the everyday resolver below and by ``vps_bootstrap.py``, which
    builds connections the inventory does not hold yet — the freshly
    generated ``legion`` key in step 8, before it has been written anywhere.
    Minting the ticket here keeps §3.13's property intact whichever caller it
    is: nothing reaches the network without also having passed a credential
    decision.
    """
    return {
        # `parseLegionConn` (vps_mcp/src/connection.ts) rejects a block without
        # these two outright: `(room_id, server, credential)` is what keys the
        # service's connection pool (§3.2 guarantee 4), so they are part of the
        # contract, not bookkeeping the Hub could keep to itself.
        "room_id": str(room_id),
        "server": server,
        "host": host,
        "port": port,
        "user": user,
        "auth": auth,
        "host_fingerprint": fingerprint,
        "idle_ttl_sec": idle_ttl_sec,
        "connect_timeout_sec": connect_timeout_sec,
        "call_id": str(call_id),
        "egress_ticket": issue_egress_ticket(
            room_id=room_id, server=server, host=host, port=port,
        ),
    }


async def resolve_root_connection(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    user: Optional[str] = None,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> VpsConnection:
    """The one read of ``VPS_<NAME>_ROOT_*`` in the codebase (§3.4, §3.6).

    Called only from ``core/vps_bootstrap.py``, which is reachable only from
    the human-run preparation endpoint — never from an agent's tool call, and
    never from :func:`resolve_connection`. ``actor_user_id`` is the person who
    pressed the button, so the ``SecretAudit(resolve)`` row names them and not
    a system actor.
    """
    coords = await server_coordinates(db, room_id, server, environment=environment)
    call_id = request_id or uuid4()

    auth: Optional[dict[str, Any]] = None
    mask_values: dict[str, str] = {}
    try:
        key_value = await room_secrets.resolve_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=_var_name(coords.name, "ROOT_KEY"), purpose=SecretPurpose.VPS_ACCESS,
            environment=environment, scope=None, request_id=call_id, source_ip=source_ip,
            expected_kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
        auth = {"kind": "key", "private_key": key_value, "passphrase": None}
        mask_values[_var_name(coords.name, "ROOT_KEY")] = key_value
    except RoomSecretNotAvailable:
        pass

    if auth is None:
        try:
            password_value = await room_secrets.resolve_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(coords.name, "ROOT_PASSWORD"), purpose=SecretPurpose.VPS_ACCESS,
                environment=environment, scope=None, request_id=call_id, source_ip=source_ip,
                expected_kind_override=SecretKind.ROOM_ENV,
            )
            auth = {"kind": "password", "password": password_value}
            mask_values[_var_name(coords.name, "ROOT_PASSWORD")] = password_value
        except RoomSecretNotAvailable:
            pass

    if auth is None:
        raise RootCredentialsMissing(coords.name)

    legion_conn = build_connection(
        room_id=room_id, server=coords.name, host=coords.host, port=coords.port,
        user=user or coords.user, auth=auth, fingerprint=coords.fingerprint,
        call_id=call_id, connect_timeout_sec=connect_timeout_sec,
    )
    return VpsConnection(legion_conn=legion_conn, mask_values=mask_values)


async def resolve_connection(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
    idle_ttl_sec: int = DEFAULT_IDLE_TTL_SEC,
    connect_timeout_sec: int = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> VpsConnection:
    """Build the ``_legion_conn`` block ``mcp_gateway.execute_mcp_tool`` injects.

    Never touches ``*_ROOT_*`` names. Raises :class:`ServerNotFound` for an
    unknown/host-less server, :class:`ServerCredentialsMissing` when the host
    is configured but neither an everyday key nor password is, and lets
    ``room_secrets.EnvironmentLocked`` propagate unchanged so the caller can
    raise the same ask-user card the rest of the MCP path uses.
    """
    coords = await server_coordinates(db, room_id, server, environment=environment)
    name = coords.name
    call_id = request_id or uuid4()

    auth: Optional[dict[str, Any]] = None
    mask_values: dict[str, str] = {}
    try:
        key_value = await room_secrets.resolve_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=_var_name(name, "KEY"), purpose=SecretPurpose.VPS_ACCESS,
            environment=environment, scope=None, request_id=call_id, source_ip=source_ip,
            expected_kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
        auth = {"kind": "key", "private_key": key_value, "passphrase": None}
        mask_values[_var_name(name, "KEY")] = key_value
    except RoomSecretNotAvailable:
        pass

    if auth is None:
        try:
            password_value = await room_secrets.resolve_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=_var_name(name, "PASSWORD"), purpose=SecretPurpose.VPS_ACCESS,
                environment=environment, scope=None, request_id=call_id, source_ip=source_ip,
                expected_kind_override=SecretKind.ROOM_ENV,
            )
            auth = {"kind": "password", "password": password_value}
            mask_values[_var_name(name, "PASSWORD")] = password_value
        except RoomSecretNotAvailable:
            pass

    if auth is None:
        raise ServerCredentialsMissing(name)

    legion_conn = build_connection(
        room_id=room_id, server=name, host=coords.host, port=coords.port,
        user=coords.user, auth=auth, fingerprint=coords.fingerprint,
        call_id=call_id, idle_ttl_sec=idle_ttl_sec, connect_timeout_sec=connect_timeout_sec,
    )
    return VpsConnection(legion_conn=legion_conn, mask_values=mask_values)


@dataclass(frozen=True)
class EgressTicketClaims:
    room_id: str
    server: str
    host: str
    port: int
    exp: int
    jti: str


_TICKET_FIELD_COUNT = 6  # room_id, server, host, port, exp, jti


def _ticket_payload(fields: list[str]) -> str:
    # Every field is percent-encoded before joining on "|" so a free-text
    # `host` (the VPS's IP/DNS name — no format restriction, see add_server)
    # can never be mistaken for the field separator: decoding always yields
    # exactly _TICKET_FIELD_COUNT + 1 (signature) parts. `legion-egress`
    # (Stage 4, a separate Node process) decodes with the mirror-image
    # `decodeURIComponent`, which happily reads anything `quote()` produces.
    return "|".join(quote(field, safe="") for field in fields)


def issue_egress_ticket(
    *, room_id: UUID, server: str, host: str, port: int, ttl_sec: int = DEFAULT_EGRESS_TICKET_TTL_SEC
) -> str:
    """HMAC ticket for ``legion-egress`` (§3.13, verified by Stage 4's broker).

    Minted by the same call that resolves credentials — access and network
    permission are one decision, not two, so a call that never reaches this
    point (denied, locked, no credentials) never gets a ticket either.
    """
    if not _EGRESS_TICKET_SECRET:
        raise EgressNotConfigured("VPS_EGRESS_TICKET_SECRET is not set; VPS connections are off")
    exp = int(time.time()) + max(1, int(ttl_sec))
    jti = uuid4().hex
    payload = _ticket_payload([str(room_id), server, host, str(port), str(exp), jti])
    secret = _EGRESS_TICKET_SECRET.encode("utf-8")
    signature = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(f"{payload}|{signature}".encode("utf-8")).decode("ascii")
    return token.rstrip("=")


def verify_egress_ticket(token: str, *, secret: Optional[str] = None) -> EgressTicketClaims:
    """Parity check for the format ``legion-egress`` enforces (Stage 4).

    Not on the hot path of any request — the broker is the actual enforcement
    point, in a separate process, reachable only from ``vps-mcp``'s isolated
    network. This exists so a change to the wire format is caught by a
    same-process test instead of only surfacing as a live broker rejection.
    """
    raw_secret = secret if secret is not None else (_EGRESS_TICKET_SECRET or "")
    padded = token + "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - any decode failure is just an invalid ticket
        raise EgressTicketInvalid("malformed ticket") from exc

    parts = decoded.split("|")
    if len(parts) != _TICKET_FIELD_COUNT + 1:
        raise EgressTicketInvalid("malformed ticket")

    *field_parts, signature = parts
    payload = "|".join(field_parts)
    expected_signature = hmac.new(raw_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise EgressTicketInvalid("ticket signature invalid")

    try:
        room_id_raw, server_raw, host_raw, port_raw, exp_raw, jti_raw = (unquote(part) for part in field_parts)
        claims = EgressTicketClaims(
            room_id=room_id_raw, server=server_raw, host=host_raw,
            port=int(port_raw), exp=int(exp_raw), jti=jti_raw,
        )
    except (ValueError, TypeError) as exc:
        raise EgressTicketInvalid("malformed ticket fields") from exc

    if not claims.room_id or not claims.host or not claims.jti:
        raise EgressTicketInvalid("malformed ticket fields")
    if claims.exp <= int(time.time()):
        raise EgressTicketInvalid("ticket expired")
    return claims
