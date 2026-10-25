"""Sign in to the Hub with a Legion account (internal design notes §2.2, stage 2).

Market is the OpenID Connect issuer. The Hub is a confidential client:

1. ``GET /api/v1/auth/legion/start`` puts state, nonce and the PKCE verifier
   into a short signed cookie and sends the browser to Market's
   ``/oauth/authorize``.
2. Market brings the browser back to the SPA route ``/auth/legion/callback``;
   the SPA posts ``code`` and ``state`` to ``/api/v1/auth/legion/callback``.
3. The Hub swaps the code for tokens (server to server, with the client
   secret), checks the ``id_token`` against Market's JWKS, finds or creates
   the user by ``legion_account_id`` and returns its own JWT, as the password
   login always did. Nothing else in the Hub changes: it keeps working with
   ``user.id``.

A sign-in never creates a user (decision A10, stage 8): an account without a
Legion Cloud plan is sent to Market's ``/pricing?from=hub``. The user is made
by the first paid ``plan_period`` (``legion_cloud.cloud_users.provision_cloud_user``).
The cloud sign-in gate checks that the plan entitlement was delivered; it
also decides who may sign in at all.

``AUTH_MODE``: ``local`` (default; a self-hosted Hub, local users only),
``legion`` (Legion Cloud) or ``both`` (the transition).

Fallback login: Market pushes each account's bcrypt hash, token_version and
status to ``legion_credential``. While Market is unreachable (checked every
LEGION_HEALTH_INTERVAL_SEC) the Hub's own password form checks the copy.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import UUID, uuid4

import httpx
from jose import JWTError, jwt
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security, sign_in_gates
from app.core.market_switch import DEFAULT_MARKET_URL, market_enabled
from app.core.secret_config import configured_secret
from app.models.legion_credential import LegionCredential
from app.models.user import User

logger = logging.getLogger(__name__)


def _optional_secret(name: str, default: Optional[str]) -> Optional[str]:
    try:
        return configured_secret(name, default)
    except RuntimeError:  # strict config without the value: the feature is off
        return None


AUTH_MODES = ("local", "legion", "both")
AUTH_MODE = os.getenv("AUTH_MODE", "local").strip().lower()
if AUTH_MODE not in AUTH_MODES:
    logger.error("AUTH_MODE=%r is not one of %s; using local", AUTH_MODE, AUTH_MODES)
    AUTH_MODE = "local"
if AUTH_MODE != "local" and not market_enabled():
    # Signing in through Legion is a Market call; with the switch off it could
    # only fail at the first login. Stop at startup instead.
    raise RuntimeError(f"AUTH_MODE={AUTH_MODE} needs MARKET_ENABLED=true")

# The browser-facing Market address: the issuer in id_tokens and where
# /oauth/authorize, the profile and Market itself are opened.
LEGION_ISSUER = (os.getenv("LEGION_ISSUER", "").strip() or DEFAULT_MARKET_URL).rstrip("/")
# Server-to-server address of the same Market (token, JWKS, health).
LEGION_INTERNAL_URL = (
    os.getenv("LEGION_INTERNAL_URL", "").strip() or os.getenv("MARKET_API_URL", "").strip() or DEFAULT_MARKET_URL
).rstrip("/")
LEGION_CLIENT_ID = os.getenv("LEGION_CLIENT_ID", "legion-hub")
LEGION_CLIENT_SECRET = _optional_secret("LEGION_CLIENT_SECRET", None)
LEGION_REDIRECT_URI = os.getenv("LEGION_REDIRECT_URI", "http://localhost:3000/auth/legion/callback")
# Market -> Hub credential push (X-Legion-Accounts-Token). Unset = refused.
LEGION_ACCOUNTS_TOKEN = _optional_secret("LEGION_ACCOUNTS_TOKEN", None)
LEGION_HEALTH_INTERVAL_SEC = float(os.getenv("LEGION_HEALTH_INTERVAL_SEC", "30"))
LEGION_HTTP_TIMEOUT_SEC = float(os.getenv("LEGION_HTTP_TIMEOUT_SEC", "10"))

FLOW_COOKIE = "legion_oidc"
FLOW_COOKIE_PATH = "/api/v1/auth/legion"
FLOW_TTL_SEC = 600
SCOPE = "openid email profile"
# hashed_password of a user without a password copy yet; bcrypt never matches it.
NO_PASSWORD = "!legion"


def legion_enabled() -> bool:
    return AUTH_MODE in ("legion", "both")


def legion_url(path: str = "") -> str:
    return f"{LEGION_ISSUER}{path}"


def legal_urls() -> dict[str, str]:
    """Where Market publishes the documents of its services (internal design notes §2.1).

    The Hub only links to them: they govern the account and the Market tools,
    not a self-hosted Hub, whose operator answers to its own users.
    """
    return {
        "terms": legion_url("/terms"),
        "privacy": legion_url("/privacy"),
        "acceptable_use": legion_url("/acceptable-use"),
    }


class LegionAuthError(Exception):
    """A sign-in that must be refused; ``status`` and ``code`` go to the client."""

    def __init__(self, message: str, *, status: int = 400, code: str = "legion_error",
                 extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.extra = extra or {}

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.extra}


def no_cloud_plan() -> LegionAuthError:
    """An account with no Legion Cloud user: it has never paid for a plan here."""
    return LegionAuthError(
        "This Legion account has no Legion Cloud plan yet.", status=403, code="no_cloud_plan",
        extra={"pricing_url": legion_url("/pricing?from=hub")},
    )


# --- Market health (the switch for the fallback form) ------------------------

@dataclass
class _Health:
    available: Optional[bool] = None  # None: not checked yet
    checked_at: float = 0.0


_health = _Health()


def market_available() -> bool:
    """Optimistic until the first check: a fresh Hub assumes Market is up."""
    return _health.available is not False


def set_market_available(value: Optional[bool]) -> None:
    _health.available = value
    _health.checked_at = time.monotonic()


async def check_market_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=min(LEGION_HTTP_TIMEOUT_SEC, 5.0)) as client:
            response = await client.get(f"{LEGION_INTERNAL_URL}/health")
        ok = response.status_code == 200
    except httpx.HTTPError:
        ok = False
    if ok != _health.available:
        logger.log(
            logging.INFO if ok else logging.WARNING,
            "Legion (Market) is %s; the Hub's fallback password form is %s",
            "reachable" if ok else "unreachable", "off" if ok else "on",
        )
    set_market_available(ok)
    return ok


async def run_health_loop() -> None:
    while True:
        try:
            await check_market_health()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop must outlive one bad round
            logger.exception("Legion health check failed")
        await asyncio.sleep(LEGION_HEALTH_INTERVAL_SEC)


def auth_config() -> dict[str, Any]:
    """What the login page needs to know (GET /api/v1/auth/config)."""
    if not legion_enabled():
        return {"mode": AUTH_MODE, "legion": None, "password_login": True, "registration": "local"}
    available = market_available()
    return {
        "mode": AUTH_MODE,
        "legion": {
            "available": available,
            "market_url": legion_url(),
            "account_url": legion_url("/account"),
            # Payments live in Market (internal design notes §3.8).
            "billing_url": legion_url("/account/billing"),
            "pricing_url": legion_url("/pricing"),
            "legal_urls": legal_urls(),
        },
        "password_login": AUTH_MODE == "both" or not available,
        "registration": "legion" if available else "unavailable",
    }


# --- The flow cookie and PKCE -------------------------------------------------

def _flow_key() -> bytes:
    # Derived, so a flow cookie can never pass for a Hub access token.
    return hmac.new(security.SECRET_KEY.encode(), b"legion-oidc-flow", hashlib.sha256).digest()


def pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


def start_flow(intent: str) -> tuple[str, str]:
    """(authorize URL, flow cookie value) for intent ``login`` or ``register``."""
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    now = int(time.time())
    cookie = jwt.encode(
        {"state": state, "nonce": nonce, "verifier": verifier, "iat": now, "exp": now + FLOW_TTL_SEC},
        _flow_key(), algorithm="HS256",
    )
    params = {
        "response_type": "code",
        "client_id": LEGION_CLIENT_ID,
        "redirect_uri": LEGION_REDIRECT_URI,
        "scope": SCOPE,
        "state": state,
        "nonce": nonce,
        "code_challenge": pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    if intent == "register":
        params["prompt"] = "create"
    return f"{legion_url('/oauth/authorize')}?{urlencode(params)}", cookie


def read_flow(cookie: Optional[str], state: str) -> dict[str, Any]:
    if not cookie:
        raise LegionAuthError("The sign-in has expired. Start again.", code="flow_missing")
    try:
        flow = jwt.decode(cookie, _flow_key(), algorithms=["HS256"])
    except JWTError:
        raise LegionAuthError("The sign-in has expired. Start again.", code="flow_expired") from None
    if not state or not hmac.compare_digest(str(flow.get("state", "")), state):
        raise LegionAuthError("The sign-in does not match this browser. Start again.", code="state_mismatch")
    return flow


# --- Token exchange and id_token verification ---------------------------------

_jwks_cache: dict[str, Any] = {"keys": {}, "fetched_at": 0.0}
_JWKS_MIN_REFRESH_SEC = 30.0


async def _fetch_jwks() -> None:
    async with httpx.AsyncClient(timeout=LEGION_HTTP_TIMEOUT_SEC) as client:
        response = await client.get(f"{LEGION_INTERNAL_URL}/.well-known/jwks.json")
    response.raise_for_status()
    _jwks_cache["keys"] = {key["kid"]: key for key in response.json().get("keys", []) if key.get("kid")}
    _jwks_cache["fetched_at"] = time.monotonic()


async def _signing_key(kid: Optional[str]) -> dict[str, Any]:
    """The JWKS key for ``kid``; an unknown kid triggers one refetch (rotation)."""
    if kid not in _jwks_cache["keys"] and time.monotonic() - _jwks_cache["fetched_at"] >= _JWKS_MIN_REFRESH_SEC:
        await _fetch_jwks()
    key = _jwks_cache["keys"].get(kid)
    if key is None:
        raise LegionAuthError("Legion signed the sign-in with an unknown key.", status=502, code="unknown_key")
    return key


def reset_jwks_cache() -> None:
    _jwks_cache["keys"] = {}
    _jwks_cache["fetched_at"] = 0.0


async def exchange_code(code: str, verifier: str) -> dict[str, Any]:
    if not LEGION_CLIENT_SECRET:
        raise LegionAuthError("Sign-in with Legion is not configured.", status=503, code="not_configured")
    try:
        async with httpx.AsyncClient(timeout=LEGION_HTTP_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{LEGION_INTERNAL_URL}/oauth/token",
                data={
                    "grant_type": "authorization_code", "code": code,
                    "redirect_uri": LEGION_REDIRECT_URI, "code_verifier": verifier,
                },
                auth=(LEGION_CLIENT_ID, LEGION_CLIENT_SECRET),
            )
    except httpx.HTTPError as exc:
        raise LegionAuthError("Legion is unreachable. Try again shortly.", status=503, code="unreachable") from exc
    if response.status_code != 200:
        error = (response.json() if response.headers.get("content-type", "").startswith("application/json")
                 else {}).get("error", "")
        logger.warning("Legion token exchange refused: %s %s", response.status_code, error)
        if error == "invalid_client":
            raise LegionAuthError("The Hub is not registered with Legion.", status=502, code="invalid_client")
        raise LegionAuthError("The sign-in has expired. Start again.", code="invalid_grant")
    return response.json()


async def verify_id_token(id_token: str, nonce: str) -> dict[str, Any]:
    try:
        header = jwt.get_unverified_header(id_token)
    except JWTError:
        raise LegionAuthError("Legion returned a malformed sign-in.", status=502, code="bad_token") from None
    if header.get("alg") != "RS256":
        raise LegionAuthError("Legion returned a malformed sign-in.", status=502, code="bad_token")
    try:
        key = await _signing_key(header.get("kid"))
    except httpx.HTTPError as exc:
        raise LegionAuthError("Legion is unreachable. Try again shortly.", status=503, code="unreachable") from exc
    try:
        claims = jwt.decode(
            id_token, key, algorithms=["RS256"], audience=LEGION_CLIENT_ID, issuer=LEGION_ISSUER,
            options={"verify_at_hash": False},
        )
    except JWTError as exc:
        logger.warning("Legion id_token rejected: %s", exc)
        raise LegionAuthError("Legion returned an invalid sign-in.", status=502, code="bad_token") from None
    if not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
        raise LegionAuthError("The sign-in does not match this browser. Start again.", code="nonce_mismatch")
    try:
        UUID(str(claims.get("sub")))
    except ValueError:
        raise LegionAuthError("Legion returned an invalid sign-in.", status=502, code="bad_token") from None
    if not claims.get("email") or not isinstance(claims.get("tv"), int):
        raise LegionAuthError("Legion returned an invalid sign-in.", status=502, code="bad_token")
    return claims


# --- The credentials copy -----------------------------------------------------

async def store_credentials(db: AsyncSession, account_id: UUID, payload: dict[str, Any]) -> bool:
    """Upsert the copy of an account's credentials; False if ``payload`` is
    older than what is stored (a delayed push after a newer one). Also
    refreshes the linked user's ``hashed_password``, so the plain password
    form keeps matching Legion in ``both`` mode. Commit is the caller's.

    Status ``erased`` (the account was deleted in Legion) is final: the copy
    becomes a tombstone, the linked user is closed and an erasure of their data
    is requested (``app.core.user_erasure``); no later push revives it."""
    tv = int(payload.get("token_version") or 0)
    status = payload.get("status") or "active"
    row = await db.get(LegionCredential, account_id)
    if row is not None and tv < row.token_version:
        return False
    # An erased account stays erased: nothing a delayed push carries brings it back.
    if row is not None and row.status == "erased" and status != "erased":
        return False
    if row is None:
        row = LegionCredential(account_id=account_id)
        db.add(row)
    row.email = str(payload["email"]).strip().lower()
    row.password_hash = None if status == "erased" else payload.get("password_hash")
    row.token_version = tv
    row.status = status
    row.updated_at = datetime.now(timezone.utc)
    user = await linked_user(db, account_id)
    if status == "erased":
        # The account is gone in Legion (internal design notes stage 13): the user is
        # closed now and everything they left here is erased by the reconciler.
        if user is not None:
            from app.core import user_erasure

            await user_erasure.request_erasure(db, user, source=user_erasure.SOURCE_MARKET, account_id=account_id)
        return True
    if user is not None:
        # No password in Legion (Google/GitHub only, or one Legion dropped,
        # stage 7) means none here either, also after a rollback to `local`.
        user.hashed_password = row.password_hash or NO_PASSWORD
        # A block in Legion also ends the Hub sessions already issued
        # (deps.get_current_user refuses an inactive user); unblock lifts it.
        # A sign-in gate that keeps the account out (a taken-back invitation)
        # is not undone by the push either.
        user.is_active = row.status == "active" and not await sign_in_gates.refused(db, account_id)
    return True


async def linked_user(db: AsyncSession, account_id: UUID) -> Optional[User]:
    return (await db.execute(select(User).where(User.legion_account_id == account_id))).scalars().first()


async def credential_by_email(db: AsyncSession, email: str) -> Optional[LegionCredential]:
    return (
        await db.execute(select(LegionCredential).where(LegionCredential.email == email.strip().lower()))
    ).scalars().first()


# --- Account -> Hub user -------------------------------------------------------

async def user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    return (await db.execute(select(User).where(func.lower(User.email) == email))).scalars().first()


async def resolve_user(
    db: AsyncSession,
    account_id: UUID,
    email: str,
    *,
    email_verified: bool,
    display_name: Optional[str] = None,
    create: bool = False,
    paid_provision: bool = False,
) -> tuple[User, bool]:
    """The Hub user of a Legion account, linked now or (``create``) made now.
    Returns (user, created).

    Order: the linked user; a user moved before Legion (same id, same email,
    not linked yet); a user with the same email, linked only when the account
    has proved the mailbox (otherwise anyone could register someone else's
    email in Legion and walk into their Hub user); else a new user, with the
    account's id when it is free. Only a paid plan creates one (A10): without
    ``create`` the last step raises ``no_cloud_plan`` — unless a sign-in gate
    admitted the account with the right to a user. A gate that refuses the account stops this
    function before any user is linked or made.
    """
    email = email.strip().lower()
    # Only the authenticated internal billing route sets paid_provision after
    # validating a plan_period. Sign-in always passes the cloud admission gate.
    if not paid_provision and await sign_in_gates.admit(
        db, account_id=account_id, email=email, email_verified=email_verified
    ):
        create = True
    user = await linked_user(db, account_id)
    if user is not None:
        return user, False

    same_id = await db.get(User, account_id)
    if same_id is not None and same_id.legion_account_id is None and same_id.email.lower() == email:
        same_id.legion_account_id = account_id
        await db.commit()
        logger.info("legion: linked moved user %s to its account", same_id.id)
        return same_id, False

    by_email = await user_by_email(db, email)
    if by_email is not None:
        if by_email.legion_account_id is not None:
            raise LegionAuthError(
                "This email belongs to a Hub user linked to another Legion account.",
                status=409, code="email_linked_elsewhere",
            )
        if not email_verified:
            raise LegionAuthError(
                "The Hub already has a user with this email. Confirm the email in your Legion "
                "account to connect them.", status=409, code="email_unverified",
            )
        by_email.legion_account_id = account_id
        await db.commit()
        logger.info("legion: linked existing user %s to account %s by verified email", by_email.id, account_id)
        return by_email, False

    if not create:
        raise no_cloud_plan()
    credential = await db.get(LegionCredential, account_id)

    user = User(
        id=account_id if same_id is None else uuid4(),
        email=email,
        full_name=display_name,
        hashed_password=(credential.password_hash if credential and credential.password_hash else NO_PASSWORD),
        legion_account_id=account_id,
        is_active=credential is None or credential.status == "active",
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # A parallel sign-in of the same account created it first.
        await db.rollback()
        user = await linked_user(db, account_id)
        if user is None:
            raise
        return user, False
    await db.refresh(user)
    logger.info("legion: created user %s for account %s", user.id, account_id)
    await sign_in_gates.user_created(db, user)
    await db.refresh(user)
    return user, True


async def check_not_revoked(db: AsyncSession, account_id: UUID, token_version: int) -> None:
    """Refuse a blocked account or an id_token older than the last sign-out everywhere."""
    credential = await db.get(LegionCredential, account_id)
    if credential is None:
        return
    if credential.status != "active":
        raise LegionAuthError("This Legion account is blocked.", status=403, code="blocked")
    if token_version < credential.token_version:
        raise LegionAuthError("The sign-in has expired. Start again.", code="stale_token")


async def current_token_version(db: AsyncSession, legion_account_id: Optional[UUID]) -> Optional[int]:
    """The ``tv`` to put in a Hub session token: None for a local user."""
    if legion_account_id is None:
        return None
    credential = await db.get(LegionCredential, legion_account_id)
    return credential.token_version if credential is not None else 0


async def session_revoked(db: AsyncSession, legion_account_id: Optional[UUID], claims: dict[str, Any]) -> bool:
    """A Hub session token issued before the account's last "sign out
    everywhere" (or password change, block) in Legion. A token without ``tv``
    counts as 0, so sessions from before this check end on the next sign-out.
    """
    if legion_account_id is None:
        return False
    credential = await db.get(LegionCredential, legion_account_id)
    if credential is None:
        return False
    try:
        tv = int(claims.get("tv") or 0)
    except (TypeError, ValueError):
        return True
    return tv < credential.token_version


# --- Fallback password form: a small per-process limit ------------------------

_FALLBACK_LIMIT = int(os.getenv("LEGION_FALLBACK_LOGIN_LIMIT", "5"))
_FALLBACK_WINDOW_SEC = float(os.getenv("LEGION_FALLBACK_LOGIN_WINDOW_SEC", "900"))
_attempts: dict[str, list[float]] = {}


def login_attempt_allowed(key: str) -> bool:
    now = time.monotonic()
    window = [t for t in _attempts.get(key, []) if t > now - _FALLBACK_WINDOW_SEC]
    allowed = len(window) < _FALLBACK_LIMIT
    if allowed:
        window.append(now)
    _attempts[key] = window
    return allowed


def clear_login_attempts(key: str) -> None:
    _attempts.pop(key, None)
