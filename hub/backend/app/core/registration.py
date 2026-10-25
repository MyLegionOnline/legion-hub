"""Who may create an account on a self-hosted Hub (internal design notes stage 4.3).

A user of the Hub gets ``bash`` in containers on the owner's machine, so
registration is closed unless the owner opens it:

* the first person to register becomes the owner of the Hub;
* after that ``closed`` (the default) refuses everyone, ``invite`` takes only
  people who hold an invitation the owner made, ``open`` takes anyone.

``mode`` comes from ``legion.toml`` ``[registration]`` and ``LEGION_REGISTRATION``
overrides it. A bad value stops the startup. Under ``AUTH_MODE=legion`` accounts
are made in Legion and none of this applies.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import secrets
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional
from uuid import UUID

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.limits_policy import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH
from app.models.registration_invite import RegistrationInvite
from app.models.user import User

logger = logging.getLogger(__name__)

MODES = ("closed", "invite", "open")
DEFAULT_MODE = "closed"
FIRST_USER_MODE = "first_user"  # not configurable: an empty Hub lets its owner in
MODE_ENV = "LEGION_REGISTRATION"
OWNER_EMAIL_ENV = "LEGION_OWNER_EMAIL"
_SECTION_KEYS = ("mode",)

INVITE_DEFAULT_TTL_DAYS = 7
INVITE_MAX_TTL_DAYS = 30
_INVITE_PREFIX = "lgi_"
# Serialises "is this the first user" across workers (transaction-scoped).
_REGISTRATION_LOCK_ID = 7_240_117


class RegistrationConfigError(RuntimeError):
    """legion.toml or LEGION_REGISTRATION is wrong; startup must stop."""


class RegistrationRefused(Exception):
    """A registration that must be turned down; ``status`` and ``code`` go to the client."""

    def __init__(self, message: str, *, status: int = 403, code: str = "registration_closed") -> None:
        super().__init__(message)
        self.status = status
        self.code = code

    def detail(self) -> dict:
        return {"code": self.code, "message": str(self)}


def _check_mode(value: object, source: str) -> str:
    mode = str(value).strip().lower()
    if mode in MODES:
        return mode
    hint = difflib.get_close_matches(mode, MODES, n=1)
    suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
    raise RegistrationConfigError(f"{source}: registration mode {value!r} is not one of {', '.join(MODES)}.{suggestion}")


def load_mode(path: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """The configured mode: the built-in default, then ``[registration]``, then the env."""
    env = os.environ if environ is None else environ
    mode = DEFAULT_MODE

    explicit = path or env.get(CONFIG_PATH_ENV)
    config_file = Path(explicit or DEFAULT_CONFIG_PATH)
    if config_file.is_file():
        try:
            with config_file.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise RegistrationConfigError(f"{config_file}: invalid TOML: {exc}") from exc
        table = data.get("registration", {})
        if not isinstance(table, dict):
            raise RegistrationConfigError(f"{config_file}: [registration] must be a table")
        for key in table:
            if key not in _SECTION_KEYS:
                hint = difflib.get_close_matches(key, _SECTION_KEYS, n=1)
                suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
                raise RegistrationConfigError(f"{config_file} [registration]: unknown key {key!r}.{suggestion}")
        if "mode" in table:
            mode = _check_mode(table["mode"], f"{config_file} [registration]")
    elif explicit:
        raise RegistrationConfigError(f"{CONFIG_PATH_ENV} points to {explicit!r}, which is not a file")

    if env.get(MODE_ENV, "").strip():
        mode = _check_mode(env[MODE_ENV], MODE_ENV)
    return mode


_mode: Optional[str] = None


def get_mode() -> str:
    """The configured mode, read once; call it at startup to fail early."""
    global _mode
    if _mode is None:
        _mode = load_mode()
    return _mode


def reset_mode() -> None:
    global _mode
    _mode = None


async def user_count(db: AsyncSession) -> int:
    return int((await db.execute(select(func.count()).select_from(User))).scalar() or 0)


async def effective_mode(db: AsyncSession) -> str:
    """What the login page should offer right now."""
    if await user_count(db) == 0:
        return FIRST_USER_MODE
    return get_mode()


# --- Invitations --------------------------------------------------------------

def hash_token(token: str) -> str:
    return hashlib.sha256(token.strip().encode()).hexdigest()


def new_token() -> str:
    return _INVITE_PREFIX + secrets.token_urlsafe(24)


def invite_status(invite: RegistrationInvite, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    if invite.revoked_at is not None:
        return "revoked"
    if invite.used_at is not None:
        return "used"
    if invite.expires_at <= now:
        return "expired"
    return "pending"


async def create_invite(
    db: AsyncSession, *, owner_id: UUID, email: Optional[str] = None, ttl_days: int = INVITE_DEFAULT_TTL_DAYS,
) -> tuple[RegistrationInvite, str]:
    """(row, plain token). The token is shown once; only its hash is kept."""
    ttl = max(1, min(int(ttl_days), INVITE_MAX_TTL_DAYS))
    token = new_token()
    invite = RegistrationInvite(
        token_hash=hash_token(token),
        email=email.strip().lower() if email else None,
        created_by=owner_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=ttl),
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)
    return invite, token


async def _consume_invite(db: AsyncSession, token: str, email: str) -> UUID:
    """Mark the invitation used, atomically: two registrations on one token cannot both win."""
    now = datetime.now(timezone.utc)
    stmt = (
        update(RegistrationInvite)
        .where(
            RegistrationInvite.token_hash == hash_token(token),
            RegistrationInvite.used_at.is_(None),
            RegistrationInvite.revoked_at.is_(None),
            RegistrationInvite.expires_at > now,
            (RegistrationInvite.email.is_(None)) | (RegistrationInvite.email == email.strip().lower()),
        )
        .values(used_at=now)
        .returning(RegistrationInvite.id)
    )
    invite_id = (await db.execute(stmt)).scalar()
    if invite_id is None:
        raise RegistrationRefused(
            "This invitation is not valid for this email address, or it has expired or been used.",
            code="invite_invalid",
        )
    return invite_id


# --- Registration -------------------------------------------------------------

async def admit(db: AsyncSession, *, email: str, invite_token: Optional[str]) -> tuple[bool, Optional[UUID]]:
    """Decide whether this registration may proceed.

    Returns ``(becomes_owner, invite_id)``. Raises ``RegistrationRefused``.
    The decision and the user insert share one transaction; the advisory lock
    keeps two simultaneous first registrations from both becoming the owner.
    """
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        await db.execute(text("SELECT pg_advisory_xact_lock(:id)").bindparams(id=_REGISTRATION_LOCK_ID))
    if await user_count(db) == 0:
        return True, None
    mode = get_mode()
    if mode == "open":
        return False, None
    if mode == "invite":
        if not invite_token or not invite_token.strip():
            raise RegistrationRefused("Registration on this Hub needs an invitation from its owner.", code="invite_required")
        return False, await _consume_invite(db, invite_token, email)
    raise RegistrationRefused("Registration is closed on this Hub. Its owner can open it in legion.toml.", code="registration_closed")


async def mark_invite_used_by(db: AsyncSession, invite_id: UUID, user_id: UUID) -> None:
    await db.execute(update(RegistrationInvite).where(RegistrationInvite.id == invite_id).values(used_by=user_id))


async def claim_owner_from_env(db: AsyncSession, environ: Optional[Mapping[str, str]] = None) -> Optional[User]:
    """A Hub that predates owners: ``LEGION_OWNER_EMAIL`` names the person who runs it.

    Does nothing once an owner exists, so the variable can stay in the file.
    """
    env = os.environ if environ is None else environ
    email = env.get(OWNER_EMAIL_ENV, "").strip()
    if not email:
        return None
    if (await db.execute(select(User.id).where(User.is_owner.is_(True)))).first() is not None:
        return None
    user = (await db.execute(select(User).where(func.lower(User.email) == email.lower()))).scalars().first()
    if user is None:
        logger.warning("%s=%s matches no user; nobody is the owner yet", OWNER_EMAIL_ENV, email)
        return None
    user.is_owner = True
    await db.commit()
    logger.info("%s is now the owner of this Hub", user.email)
    return user
