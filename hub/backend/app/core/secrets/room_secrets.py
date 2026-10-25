"""Room-scoped secrets — bind, list, resolve, unbind (Stage 2).

``internal design notes`` §2.3 / §3 / Stage 2.

A Room secret is an ordinary :class:`UserSecret` (``kind=room_env``) plus a
:class:`SecretBinding` (``resource_type=room``, ``purpose=room_env``) whose
``injection_config`` carries the ``env_name``, the allowed ``scope`` list and
the ``environment``.  No second encrypted store: rotation, audit and redaction
already work for ``user_secret``.

Key rule (§2.3):

  Any Room member may *bind* a secret to the Room and any member may *use* it,
  but the value is always resolved under the **binding owner's** user id
  (``SecretBinding.owner_id``), not the caller.  Membership grants use — not
  read, and not unbind of someone else's secret.

Values never leave this module as plaintext except through
:func:`resolve_room_secret` / :func:`resolve_room_env`, which are called at an
outbound-execution boundary and hand the value straight to the exec env.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.secrets import room_environment_gate
from app.core.secrets.crypto import SecretValue
from app.core.secrets.service import SecretResourceAccessDenied, SecretService
from app.core.secrets.store import (
    ROOM_SECRET_SCOPES,
    BindingTarget,
    SecretAccessDenied,
    SecretNotFound,
    SecretStoreError,
    SecretUnavailable,
)
from app.models.room import Room, RoomUserLink
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretKind,
    SecretPurpose,
    SecretState,
    UserSecret,
)

_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ENVIRONMENT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# Short values would be pure noise for the output masker (Stage 3) and are more
# likely to collide with ordinary text, so they are rejected on the way in
# (plan §4.3).
MIN_SECRET_VALUE_LEN = 8
MAX_SECRET_VALUE_LEN = 8 * 1024
MAX_ROOM_SECRETS_PER_ENVIRONMENT = 100

# The purposes that show up in the Room "Secrets & variables" tab.
ROOM_SECRET_PURPOSES = (
    SecretPurpose.ROOM_ENV,
    SecretPurpose.DEPLOY_SSH_KEY,
    SecretPurpose.ROOM_MCP_ENV,
    SecretPurpose.ROOM_MCP_AUTH_HEADER,
)

# Purposes that name one specific MCP item via injection_config.mcp_item_id
# (plan §2.2/§3.4, Stage 7) instead of being usable by name from any exec.
MCP_SCOPED_PURPOSES = (SecretPurpose.ROOM_MCP_ENV, SecretPurpose.ROOM_MCP_AUTH_HEADER)

_UNSET = object()


class RoomSecretError(RuntimeError):
    """Base for value-free Room-secret failures mapped at the API boundary."""

    code = "room_secret_error"
    http_status = 400


class RoomSecretAccessDenied(RoomSecretError):
    code = "room_secret_access_denied"
    http_status = 403


class RoomSecretInvalid(RoomSecretError):
    code = "room_secret_invalid"
    http_status = 422


class RoomSecretExists(RoomSecretError):
    code = "room_secret_exists"
    http_status = 409


class RoomSecretNotAvailable(RoomSecretError):
    """Missing, disabled, expired, or out-of-scope — deliberately one code.

    The caller must not be able to tell "no such secret" from "not allowed
    here" (plan §3.1).
    """

    code = "secret_not_available"
    http_status = 404


class EnvironmentLocked(RoomSecretError):
    """The named environment is ``protected`` and has no active unlock window.

    Unlike :class:`RoomSecretNotAvailable`, this is deliberately distinct and
    named — the caller (a human clicking Sync, or the bash tool deciding
    whether to raise an ask-user confirmation) needs to tell "locked" apart
    from "no such secret" (plan §2.4 Stage 12).
    """

    code = "environment_locked"
    http_status = 423


# ── helpers ─────────────────────────────────────────────────────────────────


def _normalize_environment(environment: Optional[str]) -> Optional[str]:
    if environment is None:
        return None
    environment = environment.strip()
    if not environment:
        return None
    if not _ENVIRONMENT_RE.match(environment):
        raise RoomSecretInvalid(
            f"environment must match ^[a-z][a-z0-9_-]{{0,31}}$ (got {environment!r})"
        )
    return environment


def _validate_env_name(env_name: str) -> str:
    env_name = (env_name or "").strip()
    if (
        not env_name
        or not env_name.isascii()
        or env_name.upper() != env_name
        or len(env_name) > 128
        or not _ENV_NAME_RE.match(env_name)
    ):
        raise RoomSecretInvalid("env_name must match ^[A-Z_][A-Z0-9_]*$ and be ≤128 chars")
    return env_name


def _validate_scope(scope: Optional[Iterable[str]]) -> Optional[list[str]]:
    if scope is None:
        return None
    scope = list(scope)
    bad = [s for s in scope if s not in ROOM_SECRET_SCOPES]
    if bad:
        raise RoomSecretInvalid(
            f"scope entries must be one of {sorted(ROOM_SECRET_SCOPES)} (got {bad})"
        )
    # de-dupe, keep order
    seen: set[str] = set()
    return [s for s in scope if not (s in seen or seen.add(s))]


def _validate_value(value: str) -> SecretValue:
    if not isinstance(value, str):
        raise RoomSecretInvalid("value must be text")
    if len(value.encode("utf-8")) < MIN_SECRET_VALUE_LEN:
        raise RoomSecretInvalid(
            f"secret value must be at least {MIN_SECRET_VALUE_LEN} bytes"
        )
    if len(value.encode("utf-8")) > MAX_SECRET_VALUE_LEN:
        raise RoomSecretInvalid(
            f"secret value must be at most {MAX_SECRET_VALUE_LEN} bytes"
        )
    try:
        return SecretValue(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise RoomSecretInvalid("secret value is invalid") from exc


def _parse_ssh_private_key(value: str) -> tuple[str, str]:
    """Derive the OpenSSH public-key line + ``SHA256:...`` fingerprint (Stage 5).

    Only these two — never the private key itself — are safe to keep in
    ``injection_config`` and show in the UI (plan §3.3).
    """

    try:
        private_key = serialization.load_ssh_private_key(
            value.encode("utf-8"), password=None
        )
    except Exception as exc:
        raise RoomSecretInvalid(
            "value is not a valid, unencrypted OpenSSH private key"
        ) from exc

    pub_line = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    blob = base64.b64decode(pub_line.split()[1])
    digest = hashlib.sha256(blob).digest()
    fingerprint = "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")
    return pub_line, fingerprint


async def _assert_member(db: AsyncSession, room_id: UUID, user_id: UUID) -> None:
    row = (
        await db.execute(
            select(Room.id)
            .outerjoin(RoomUserLink, RoomUserLink.room_id == Room.id)
            .where(
                Room.id == room_id,
                or_(Room.creator_id == user_id, RoomUserLink.user_id == user_id),
            )
        )
    ).first()
    if row is None:
        raise RoomSecretAccessDenied("not a member of this room")


def _binding_env_name(binding: SecretBinding) -> Optional[str]:
    cfg = binding.injection_config or {}
    return cfg.get("env_name")


def _binding_environment(binding: SecretBinding) -> Optional[str]:
    cfg = binding.injection_config or {}
    return cfg.get("environment")


def _binding_scope(binding: SecretBinding) -> list[str]:
    cfg = binding.injection_config or {}
    return list(cfg.get("scope") or [])


def _binding_mcp_item_id(binding: SecretBinding) -> Optional[str]:
    cfg = binding.injection_config or {}
    value = cfg.get("mcp_item_id")
    return str(value) if value is not None else None


async def _room_bindings(
    db: AsyncSession,
    room_id: UUID,
    *,
    purposes: Iterable[SecretPurpose] = ROOM_SECRET_PURPOSES,
) -> list[SecretBinding]:
    purpose_values = [p.value if hasattr(p, "value") else p for p in purposes]
    rows = (
        await db.execute(
            select(SecretBinding).where(
                SecretBinding.resource_type == "room",
                SecretBinding.resource_id == room_id,
                SecretBinding.purpose.in_(purpose_values),
            )
        )
    ).scalars().all()
    return list(rows)


def _pick_by_environment(
    candidates: list[SecretBinding], environment: Optional[str]
) -> Optional[SecretBinding]:
    """Exact-environment match wins; else the shared (``NULL``) one (plan §2.4)."""

    if environment is not None:
        for b in candidates:
            if _binding_environment(b) == environment:
                return b
    for b in candidates:
        if _binding_environment(b) is None:
            return b
    return None


async def _audit_actor_resolve(
    db: AsyncSession,
    *,
    actor_user_id: UUID,
    secret_id: UUID,
    room_id: UUID,
    purpose: SecretPurpose,
    env_name: str,
    request_id: Optional[UUID],
    outcome: SecretAuditOutcome,
    error_code: Optional[str] = None,
) -> None:
    """Record *who* used a Room secret.

    The SecretStore already writes a value-free RESOLVE row under the binding
    owner; this second row keeps ``actor_id`` = the session user, which is what
    the "last used" column in the UI reads (plan §2.3 / §4.4).
    """

    db.add(
        SecretAudit(
            actor_type=SecretActorType.USER,
            actor_id=actor_user_id,
            action=SecretAuditAction.RESOLVE,
            outcome=outcome,
            secret_id=secret_id,
            resource_type="room",
            resource_id=room_id,
            purpose=purpose,
            request_id=request_id,
            error_code=error_code,
            env_name=env_name,
        )
    )
    await db.flush()


# ── bind / update / unbind ──────────────────────────────────────────────────


async def bind_room_secret(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    env_name: str,
    value: str,
    label: Optional[str] = None,
    scope: Optional[Iterable[str]] = None,
    environment: Optional[str] = None,
    materialize_to_env: bool = False,
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    mcp_item_id: Optional[UUID] = None,
    provider: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
    kind_override: Optional[SecretKind] = None,
) -> SecretBinding:
    """Create the encrypted secret and its Room binding in one call.

    ``actor_user_id`` becomes the binding owner: only they can later replace
    the value or unbind it, though any member can use it.

    ``mcp_item_id`` is required for ``room_mcp_env`` / ``room_mcp_auth_header``
    (plan §3.4, Stage 7) and forbidden otherwise; ``SecretService.bind`` →
    ``authorize_target`` additionally checks it names an MCP item
    ``actor_user_id`` owns, not merely any item.

    ``kind_override`` lets a caller (``vps_inventory.py``, Stage 3) pick the
    actual secret kind per call when one ``purpose`` covers more than one
    shape of value — e.g. ``vps_access`` is a password for one env_name and
    an SSH private key for another.
    """

    await _assert_member(db, room_id, actor_user_id)
    env_name = _validate_env_name(env_name)
    environment = _normalize_environment(environment)
    scope_list = _validate_scope(scope)
    secret_value = _validate_value(value)
    expected_kind = kind_override or _expected_kind(purpose)
    if purpose in MCP_SCOPED_PURPOSES:
        if mcp_item_id is None:
            raise RoomSecretInvalid(f"mcp_item_id is required for purpose {purpose.value!r}")
    elif mcp_item_id is not None:
        raise RoomSecretInvalid(f"mcp_item_id is not valid for purpose {purpose.value!r}")

    existing = await _room_bindings(db, room_id, purposes=[purpose])
    for b in existing:
        if _binding_env_name(b) == env_name and _binding_environment(b) == environment:
            where = f"environment {environment!r}" if environment else "the shared scope"
            raise RoomSecretExists(
                f"a {env_name!r} secret already exists in {where}"
            )
    same_env_count = sum(
        1 for b in existing if _binding_environment(b) == environment
    )
    if same_env_count >= MAX_ROOM_SECRETS_PER_ENVIRONMENT:
        raise RoomSecretInvalid(
            f"too many secrets for this environment "
            f"(limit {MAX_ROOM_SECRETS_PER_ENVIRONMENT})"
        )

    public_key: Optional[str] = None
    fingerprint: Optional[str] = None
    if purpose == SecretPurpose.DEPLOY_SSH_KEY or expected_kind == SecretKind.SSH_PRIVATE_KEY:
        public_key, fingerprint = _parse_ssh_private_key(value)
        if purpose == SecretPurpose.DEPLOY_SSH_KEY and scope_list is None:
            # Deploy keys are only ever pulled in by name for a deploy step
            # (plan §3.3) — default the scope instead of making every caller
            # remember to set it.
            scope_list = ["deploy"]

    injection_config: dict[str, Any] = {"env_name": env_name, "environment": environment}
    if scope_list is not None:
        injection_config["scope"] = scope_list
    if materialize_to_env:
        injection_config["materialize_to_env"] = True
    if public_key is not None:
        injection_config["public_key"] = public_key
        injection_config["fingerprint"] = fingerprint
    if mcp_item_id is not None:
        injection_config["mcp_item_id"] = str(mcp_item_id)

    try:
        meta = await service.create(
            owner_id=actor_user_id,
            kind=expected_kind,
            label=(label or env_name)[:120],
            provider=provider,
            value=secret_value,
            request_id=request_id,
            source_ip=source_ip,
        )
        binding = await service.bind(
            owner_id=actor_user_id,
            secret_id=meta.id,
            target=BindingTarget(
                resource_type="room",
                resource_id=room_id,
                purpose=purpose,
                injection_config=injection_config,
                expected_kind=expected_kind,
            ),
            request_id=request_id,
            source_ip=source_ip,
        )
    except (SecretAccessDenied, SecretResourceAccessDenied) as exc:
        raise RoomSecretAccessDenied(str(exc)) from None
    except SecretStoreError as exc:
        raise RoomSecretInvalid(str(exc)) from None

    row = (
        await db.execute(
            select(SecretBinding).where(SecretBinding.id == binding.id)
        )
    ).scalar_one()
    return row


def _expected_kind(purpose: SecretPurpose) -> SecretKind:
    return {
        SecretPurpose.ROOM_ENV: SecretKind.ROOM_ENV,
        SecretPurpose.ROOM_MCP_ENV: SecretKind.ROOM_ENV,
        SecretPurpose.DEPLOY_SSH_KEY: SecretKind.SSH_PRIVATE_KEY,
        SecretPurpose.ROOM_MCP_AUTH_HEADER: SecretKind.MCP_HEADER,
        # internal design notes Stage 3 — a vps_access binding is either a password
        # (room_env) or a private key (ssh_private_key); callers that know
        # which one pass ``kind_override`` instead of relying on this default.
        SecretPurpose.VPS_ACCESS: SecretKind.ROOM_ENV,
        # internal design notes Stage 13 — the HTTPS deploy token is an opaque
        # string, same shape as any other room_env value.
        SecretPurpose.GIT_DEPLOY: SecretKind.ROOM_ENV,
    }[purpose]


async def _owned_binding_for_write(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    binding_id: Optional[UUID] = None,
    env_name: Optional[str] = None,
    environment: Optional[str] = _UNSET,  # type: ignore[assignment]
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
) -> SecretBinding:
    """Locate a Room binding for a mutating call and enforce §2.3 ownership."""

    await _assert_member(db, room_id, actor_user_id)
    if binding_id is not None:
        all_bindings = await _room_bindings(db, room_id)
        match = next((b for b in all_bindings if b.id == binding_id), None)
        if match is None:
            raise RoomSecretNotAvailable(str(binding_id))
        if match.owner_id != actor_user_id:
            raise RoomSecretAccessDenied(
                "only the member who added this secret can change or remove it"
            )
        return match
    candidates = await _room_bindings(db, room_id, purposes=[purpose])
    if env_name is None:
        raise RoomSecretInvalid("env_name or binding_id is required")
    env_name = _validate_env_name(env_name)
    norm_env = (
        _normalize_environment(environment) if environment is not _UNSET else _UNSET
    )
    named = [b for b in candidates if _binding_env_name(b) == env_name]
    if norm_env is _UNSET:
        match = named[0] if len(named) == 1 else None
    else:
        match = next(
            (b for b in named if _binding_environment(b) == norm_env), None
        )
    if match is None:
        raise RoomSecretNotAvailable(env_name or str(binding_id))
    if match.owner_id != actor_user_id:
        # A member can use another member's secret but not change it.
        raise RoomSecretAccessDenied(
            "only the member who added this secret can change or remove it"
        )
    return match


async def replace_room_secret_value(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    value: str,
    binding_id: Optional[UUID] = None,
    env_name: Optional[str] = None,
    environment: Optional[str] = _UNSET,  # type: ignore[assignment]
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> SecretBinding:
    binding = await _owned_binding_for_write(
        db,
        room_id=room_id,
        actor_user_id=actor_user_id,
        binding_id=binding_id,
        env_name=env_name,
        environment=environment,
        purpose=purpose,
    )
    secret_value = _validate_value(value)
    try:
        await service.store.replace_value(
            owner_id=actor_user_id,
            secret_id=binding.secret_id,
            value=secret_value,
            request_id=request_id,
            source_ip=source_ip,
        )
    except (SecretNotFound, SecretAccessDenied) as exc:
        raise RoomSecretAccessDenied(str(exc)) from None
    except SecretStoreError as exc:
        raise RoomSecretInvalid(str(exc)) from None
    await db.refresh(binding)
    return binding


async def update_room_secret_config(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    binding_id: Optional[UUID] = None,
    env_name: Optional[str] = None,
    environment: Optional[str] = _UNSET,  # type: ignore[assignment]
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    scope: Optional[Iterable[str]] = _UNSET,  # type: ignore[assignment]
    materialize_to_env: Optional[bool] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> SecretBinding:
    """Update ``scope`` / ``materialize_to_env`` on a Room secret binding.

    ``env_name`` and ``environment`` are the binding's identity and are not
    editable here — unbind and bind again to move a secret.
    """

    binding = await _owned_binding_for_write(
        db,
        room_id=room_id,
        actor_user_id=actor_user_id,
        binding_id=binding_id,
        env_name=env_name,
        environment=environment,
        purpose=purpose,
    )
    config = dict(binding.injection_config or {})
    if scope is not _UNSET:
        scope_list = _validate_scope(scope)
        if scope_list is None:
            config.pop("scope", None)
        else:
            config["scope"] = scope_list
    if materialize_to_env is not None:
        if materialize_to_env:
            config["materialize_to_env"] = True
        else:
            config.pop("materialize_to_env", None)

    try:
        await service.bind(
            owner_id=actor_user_id,
            secret_id=binding.secret_id,
            target=BindingTarget(
                resource_type="room",
                resource_id=room_id,
                purpose=purpose,
                injection_config=config,
                expected_kind=_expected_kind(purpose),
            ),
            request_id=request_id,
            source_ip=source_ip,
        )
    except (SecretAccessDenied, SecretResourceAccessDenied) as exc:
        raise RoomSecretAccessDenied(str(exc)) from None
    except SecretStoreError as exc:
        raise RoomSecretInvalid(str(exc)) from None
    await db.refresh(binding)
    return binding


async def unbind_room_secret(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    binding_id: Optional[UUID] = None,
    env_name: Optional[str] = None,
    environment: Optional[str] = _UNSET,  # type: ignore[assignment]
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> None:
    """Detach a Room secret.  The underlying ``user_secret`` stays with its owner."""

    binding = await _owned_binding_for_write(
        db,
        room_id=room_id,
        actor_user_id=actor_user_id,
        binding_id=binding_id,
        env_name=env_name,
        environment=environment,
        purpose=purpose,
    )
    try:
        await service.store.unbind(
            owner_id=actor_user_id,
            binding_id=binding.id,
            request_id=request_id,
            source_ip=source_ip,
        )
    except SecretNotFound as exc:
        raise RoomSecretNotAvailable(str(exc)) from None


async def unbind_all_for_room(db: AsyncSession, room_id: UUID) -> int:
    """Detach every Room secret binding — called from the room-delete path.

    System path: no actor and no per-owner authorization.  The encrypted
    ``user_secret`` rows are left untouched; they belong to individual users.
    """

    bindings = await _room_bindings(db, room_id)
    for binding in bindings:
        secret_id = binding.secret_id
        await db.delete(binding)
        db.add(
            SecretAudit(
                actor_type=SecretActorType.SYSTEM,
                actor_id=None,
                action=SecretAuditAction.UNBIND,
                outcome=SecretAuditOutcome.SUCCESS,
                secret_id=secret_id,
                resource_type="room",
                resource_id=room_id,
                purpose=binding.purpose,
                error_code="room_deleted",
            )
        )
    if bindings:
        await db.flush()
    return len(bindings)


# ── list ────────────────────────────────────────────────────────────────────


async def list_room_secrets(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    environment: Any = _UNSET,
) -> list[dict[str, Any]]:
    """Metadata only — never a value.  One row per Room secret binding."""

    await _assert_member(db, room_id, actor_user_id)
    bindings = await _room_bindings(db, room_id)
    if environment is not _UNSET:
        want = _normalize_environment(environment)
        bindings = [b for b in bindings if _binding_environment(b) == want]
    if not bindings:
        return []

    secret_ids = [b.secret_id for b in bindings]
    secrets = {
        s.id: s
        for s in (
            await db.execute(
                select(UserSecret).where(UserSecret.id.in_(secret_ids))
            )
        ).scalars().all()
    }
    last_used_rows = (
        await db.execute(
            select(
                SecretAudit.secret_id,
                func.max(SecretAudit.occurred_at),
            )
            .where(
                SecretAudit.secret_id.in_(secret_ids),
                SecretAudit.action == SecretAuditAction.RESOLVE,
                SecretAudit.outcome == SecretAuditOutcome.SUCCESS,
            )
            .group_by(SecretAudit.secret_id)
        )
    ).all()
    last_used = {row[0]: row[1] for row in last_used_rows}

    out: list[dict[str, Any]] = []
    for b in sorted(
        bindings,
        key=lambda x: (_binding_environment(x) or "", _binding_env_name(x) or ""),
    ):
        secret = secrets.get(b.secret_id)
        out.append(
            {
                "binding_id": b.id,
                "env_name": _binding_env_name(b),
                "purpose": SecretPurpose(b.purpose).value,
                "scope": _binding_scope(b),
                "environment": _binding_environment(b),
                "materialize_to_env": bool(
                    (b.injection_config or {}).get("materialize_to_env", False)
                ),
                "public_key": (b.injection_config or {}).get("public_key"),
                "fingerprint": (b.injection_config or {}).get("fingerprint"),
                "mcp_item_id": _binding_mcp_item_id(b),
                "label": secret.label if secret else None,
                "secret_kind": SecretKind(secret.kind).value if secret else None,
                "state": SecretState(secret.state).value if secret else "deleted",
                "is_own": b.owner_id == actor_user_id,
                "created_at": b.created_at,
                "updated_at": b.updated_at,
                "last_used_at": last_used.get(b.secret_id),
            }
        )
    return out


async def list_used_environments(db: AsyncSession, room_id: UUID) -> list[str]:
    """Distinct non-shared environments referenced by this Room's secret bindings.

    Powers the Stage 12 environment switcher, which must offer an environment
    as soon as anything (a variable or a secret) uses it — not only ones that
    already have a ``RoomEnvironmentConfig`` row.
    """
    bindings = await _room_bindings(db, room_id)
    envs = {_binding_environment(b) for b in bindings}
    envs.discard(None)
    return sorted(envs)


async def list_room_secret_names(
    db: AsyncSession, room_id: UUID, *, environment: Optional[str] = None
) -> list[dict[str, Any]]:
    """``env_name`` + ``scope`` + ``environment`` for the memory-layer block.

    No membership check — the caller is the Hub itself building an agent
    prompt.  Still no values.
    """

    bindings = await _room_bindings(db, room_id, purposes=[SecretPurpose.ROOM_ENV,
                                                          SecretPurpose.DEPLOY_SSH_KEY])
    want = _normalize_environment(environment)
    seen: set[str] = set()
    names: list[dict[str, Any]] = []
    for b in sorted(bindings, key=lambda x: _binding_env_name(x) or ""):
        name = _binding_env_name(b)
        if not name:
            continue
        env = _binding_environment(b)
        if env is not None and env != want:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(
            {
                "env_name": name,
                "scope": _binding_scope(b),
                "purpose": SecretPurpose(b.purpose).value,
            }
        )
    return names


# ── resolve (outbound-execution boundary) ───────────────────────────────────


async def _assert_environment_unlocked(
    db: AsyncSession, room_id: UUID, environment: Optional[str]
) -> None:
    """Shared choke point for the protected-environment gate (plan §4, Stage 0).

    Every outbound-execution path — ``bash``, workspace services, CI, and
    both MCP paths below — must call this before touching a keyed value for
    a non-``None`` environment. Kept in one place so the MCP paths cannot
    drift from ``resolve_room_secret``'s behaviour by copy-paste.
    """
    environment = _normalize_environment(environment)
    if environment is not None and await room_environment_gate.is_protected(
        db, room_id, environment
    ):
        if not await room_environment_gate.has_active_unlock(db, room_id, environment):
            raise EnvironmentLocked(environment)


async def resolve_room_secret(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    env_name: str,
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    environment: Optional[str] = None,
    scope: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
    expected_kind_override: Optional[SecretKind] = None,
) -> str:
    """Resolve one Room secret for one outbound operation.

    ``scope`` is the call-site (``"deploy"`` / ``"services"`` / …).  When set,
    a binding whose ``scope`` list does not include it is treated as *not
    available* — as is a binding with an empty ``scope`` list, which is
    "explicit name request only" (plan §2.2).  ``scope=None`` is an explicit
    request and bypasses that filter.

    Returns plaintext.  The caller must place it straight into an exec env and
    forget it — never log it, never return it to a model.
    """

    await _assert_member(db, room_id, actor_user_id)
    environment = _normalize_environment(environment)
    await _assert_environment_unlocked(db, room_id, environment)
    try:
        env_name = _validate_env_name(env_name)
    except RoomSecretInvalid:
        raise RoomSecretNotAvailable(str(env_name)) from None

    candidates = [
        b
        for b in await _room_bindings(db, room_id, purposes=[purpose])
        if _binding_env_name(b) == env_name
    ]
    binding = _pick_by_environment(candidates, environment)
    if binding is None:
        raise RoomSecretNotAvailable(env_name)

    if scope is not None:
        allowed = _binding_scope(binding)
        if not allowed or scope not in allowed:
            raise RoomSecretNotAvailable(env_name)

    request_id = request_id or uuid4()
    try:
        value = await service.store.resolve_for_use(
            owner_id=binding.owner_id,
            target=BindingTarget(
                resource_type="room",
                resource_id=room_id,
                purpose=purpose,
                injection_config=dict(binding.injection_config or {}),
                expected_kind=expected_kind_override or _expected_kind(purpose),
                binding_id=binding.id,
            ),
            request_id=request_id,
            source_ip=source_ip,
        )
    except (SecretAccessDenied, SecretNotFound, SecretUnavailable):
        await _audit_actor_resolve(
            db,
            actor_user_id=actor_user_id,
            secret_id=binding.secret_id,
            room_id=room_id,
            purpose=purpose,
            env_name=env_name,
            request_id=request_id,
            outcome=SecretAuditOutcome.DENIED,
            error_code="secret_not_available",
        )
        raise RoomSecretNotAvailable(env_name) from None

    await _audit_actor_resolve(
        db,
        actor_user_id=actor_user_id,
        secret_id=binding.secret_id,
        room_id=room_id,
        purpose=purpose,
        env_name=env_name,
        request_id=request_id,
        outcome=SecretAuditOutcome.SUCCESS,
    )
    return value.reveal()


async def resolve_room_env(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    names: Iterable[str],
    scope: Optional[str] = None,
    environment: Optional[str] = None,
    purpose: SecretPurpose = SecretPurpose.ROOM_ENV,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> dict[str, str]:
    """Batch resolve for exactly one environment and one exec.

    All-or-nothing: an unknown / out-of-scope name fails the whole batch with
    ``secret_not_available`` naming that one name (plan §3.1).
    """

    names = list(dict.fromkeys(names))
    request_id = request_id or uuid4()
    resolved: dict[str, str] = {}
    for name in names:
        resolved[name] = await resolve_room_secret(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=actor_user_id,
            env_name=name,
            purpose=purpose,
            environment=environment,
            scope=scope,
            request_id=request_id,
            source_ip=source_ip,
        )
    return resolved


async def _purpose_for_name(
    db: AsyncSession, room_id: UUID, env_name: str, environment: Optional[str]
) -> Optional[SecretPurpose]:
    """Which Room-secret purpose (if any) owns this ``env_name`` right now.

    Deploy keys win over a same-named plain secret — there should never be
    both, but if there somehow were, a key is the more dangerous thing to
    silently drop into an env string.
    """

    for purpose in (SecretPurpose.DEPLOY_SSH_KEY, SecretPurpose.ROOM_ENV):
        candidates = [
            b
            for b in await _room_bindings(db, room_id, purposes=[purpose])
            if _binding_env_name(b) == env_name
        ]
        if _pick_by_environment(candidates, environment) is not None:
            return purpose
    return None


async def resolve_room_secrets_mixed(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    actor_user_id: UUID,
    names: Iterable[str],
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve a bash-style ``secrets: [names]`` list across purposes.

    An agent names a secret without saying what kind it is; this looks each
    name up to see whether it is a plain ``room_env`` value (goes straight
    into ``env``) or a ``deploy_ssh_key`` (must become a tmpfs file instead —
    see ``_bash.py``'s ``BashTool._build_exec_env``, plan §3.1/§3.3). All the
    usual rules still apply: deploy keys are scope-checked against
    ``"deploy"``, plain secrets are an explicit-name request (``scope=None``).

    Returns ``(env_values, ssh_key_values)``, both plaintext, both meant to
    be forgotten by the caller as soon as they cross the exec boundary.
    """

    names = list(dict.fromkeys(names))
    request_id = request_id or uuid4()
    norm_environment = _normalize_environment(environment)
    env_values: dict[str, str] = {}
    ssh_values: dict[str, str] = {}
    for raw_name in names:
        name = _validate_env_name(raw_name)
        purpose = await _purpose_for_name(db, room_id, name, norm_environment)
        if purpose is None:
            raise RoomSecretNotAvailable(name)
        scope = "deploy" if purpose == SecretPurpose.DEPLOY_SSH_KEY else None
        value = await resolve_room_secret(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=actor_user_id,
            env_name=name,
            purpose=purpose,
            environment=environment,
            scope=scope,
            request_id=request_id,
            source_ip=source_ip,
        )
        if purpose == SecretPurpose.DEPLOY_SSH_KEY:
            ssh_values[name] = value
        else:
            env_values[name] = value
    return env_values, ssh_values


# ── MCP (outbound-execution boundary, Stage 7) ──────────────────────────────


async def _room_mcp_bindings(
    db: AsyncSession,
    room_id: UUID,
    *,
    mcp_item_id: UUID,
    purpose: SecretPurpose,
) -> list[SecretBinding]:
    want = str(mcp_item_id)
    bindings = await _room_bindings(db, room_id, purposes=[purpose])
    return [b for b in bindings if _binding_mcp_item_id(b) == want]


async def room_mcp_env_bindings_for_lease(
    db: AsyncSession, *, room_id: UUID, mcp_item_id: UUID, environment: Optional[str] = None
) -> list[tuple[SecretBinding, Optional[UserSecret]]]:
    """Active ``(room, room_mcp_env)`` bindings pointed at one MCP item.

    Read by ``lease.py``'s ``ManagedMcpSecretLeaseService.issue`` to fold
    Room-supplied env into a managed-stdio MCP's one-time capability
    alongside the item owner's own ``mcp_env`` bindings (plan §3.4). No
    membership check here — the caller has already confirmed the runtime
    owner is a Room member and the item is attached to that Room.

    internal design notes Stage 0: gated on the same protected-environment window
    as ``resolve_room_secret`` — a locked ``environment`` raises
    ``EnvironmentLocked`` before the binding rows are even queried, so a
    prod credential never reaches ``store.resolve_for_use`` for this path.
    """

    await _assert_environment_unlocked(db, room_id, environment)
    want = str(mcp_item_id)
    rows = (
        await db.execute(
            select(SecretBinding, UserSecret)
            .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
            .where(
                SecretBinding.resource_type == "room",
                SecretBinding.resource_id == room_id,
                SecretBinding.purpose == SecretPurpose.ROOM_MCP_ENV,
            )
        )
    ).all()
    return [(binding, secret) for binding, secret in rows if _binding_mcp_item_id(binding) == want]


async def vps_access_bindings(db: AsyncSession, room_id: UUID) -> list[SecretBinding]:
    """Room bindings with ``purpose=vps_access`` (internal design notes Stage 3).

    Deliberately not folded into :data:`ROOM_SECRET_PURPOSES` — these never
    show up in the generic "Secrets" list or ``.env`` sync; only
    ``core/vps_inventory.py`` reads them.
    """
    return await _room_bindings(db, room_id, purposes=[SecretPurpose.VPS_ACCESS])


async def resolve_room_mcp_auth_header(
    db: AsyncSession,
    *,
    service: SecretService,
    room_id: UUID,
    mcp_item_id: UUID,
    actor_user_id: UUID,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Resolve the Room-supplied bearer for one HTTP-link MCP item (§3.4).

    Returns ``(env_name, token)`` when a ``(room, room_mcp_auth_header)``
    binding is configured for ``mcp_item_id``, or ``None`` when none is — a
    normal, silent no-op for ``execute_mcp_tool``, exactly like an item with
    no personal bearer configured. A binding that *is* present but no longer
    resolvable (disabled, deleted, decrypt failure) still fails closed to
    ``None`` rather than raising, since this header is additive to the
    item's own auth, not the only path to it.
    """

    try:
        await _assert_member(db, room_id, actor_user_id)
    except RoomSecretAccessDenied:
        return None
    await _assert_environment_unlocked(db, room_id, environment)

    candidates = await _room_mcp_bindings(
        db, room_id, mcp_item_id=mcp_item_id, purpose=SecretPurpose.ROOM_MCP_AUTH_HEADER
    )
    if not candidates:
        return None
    binding = _pick_by_environment(candidates, _normalize_environment(environment))
    if binding is None:
        return None
    name = _binding_env_name(binding) or "ROOM_MCP_AUTH_HEADER"
    request_id = request_id or uuid4()
    try:
        value = await service.store.resolve_for_use(
            owner_id=binding.owner_id,
            target=BindingTarget(
                resource_type="room",
                resource_id=room_id,
                purpose=SecretPurpose.ROOM_MCP_AUTH_HEADER,
                injection_config=dict(binding.injection_config or {}),
                expected_kind=_expected_kind(SecretPurpose.ROOM_MCP_AUTH_HEADER),
                binding_id=binding.id,
            ),
            request_id=request_id,
            source_ip=source_ip,
        )
    except (SecretAccessDenied, SecretNotFound, SecretUnavailable):
        await _audit_actor_resolve(
            db,
            actor_user_id=actor_user_id,
            secret_id=binding.secret_id,
            room_id=room_id,
            purpose=SecretPurpose.ROOM_MCP_AUTH_HEADER,
            env_name=name,
            request_id=request_id,
            outcome=SecretAuditOutcome.DENIED,
            error_code="secret_not_available",
        )
        return None
    await _audit_actor_resolve(
        db,
        actor_user_id=actor_user_id,
        secret_id=binding.secret_id,
        room_id=room_id,
        purpose=SecretPurpose.ROOM_MCP_AUTH_HEADER,
        env_name=name,
        request_id=request_id,
        outcome=SecretAuditOutcome.SUCCESS,
    )
    return name, value.reveal()


__all__ = [
    "RoomSecretError",
    "RoomSecretAccessDenied",
    "RoomSecretInvalid",
    "RoomSecretExists",
    "RoomSecretNotAvailable",
    "EnvironmentLocked",
    "ROOM_SECRET_SCOPES",
    "ROOM_SECRET_PURPOSES",
    "MCP_SCOPED_PURPOSES",
    "MIN_SECRET_VALUE_LEN",
    "bind_room_secret",
    "replace_room_secret_value",
    "update_room_secret_config",
    "unbind_room_secret",
    "unbind_all_for_room",
    "list_room_secrets",
    "list_room_secret_names",
    "list_used_environments",
    "resolve_room_secret",
    "resolve_room_env",
    "resolve_room_secrets_mixed",
    "room_mcp_env_bindings_for_lease",
    "resolve_room_mcp_auth_header",
    "vps_access_bindings",
]
