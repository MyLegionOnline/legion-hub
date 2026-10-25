"""Link a self-hosted Hub user to a Legion account (internal design notes §2.3, stage 6).

A self-hosted Hub works with local users (``AUTH_MODE=local``). A user who
wants Legion credits and paid Market items issues an API key in their Legion
profile and pastes it into Settings → Legion account. The Hub asks Market
whose key it is (``GET /api/account/me`` with ``Authorization: Bearer``),
keeps the key encrypted (the secret store's keyring and AEAD) and shows the
account's email. Opening the settings asks Market again, so a revoked key
shows up as rejected and Market records the key's last use.

The key never leaves the Hub except to Market; the API returns its prefix only.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import legion_auth
from app.core.market_switch import market_enabled
from app.core.secret_keyring import SecretKeyringError, load_keyring
from app.core.secrets.crypto import SecretCryptoError, SecretValue, decrypt_secret, encrypt_secret
from app.models.legion_link import LegionLink

logger = logging.getLogger(__name__)

# The AAD "kind" of the ciphertext: a key copied into another row (another
# user, another kind of secret) does not decrypt there.
AAD_KIND = "legion_api_key"
# Market's keys: "lgn_" + lower-case base32; the first 12 characters are the
# prefix Market shows in the profile.
_KEY_RE = re.compile(r"^lgn_[a-z2-7]{16,100}$")
_PREFIX_LEN = 12


class LegionLinkError(Exception):
    """A refusal for the client: ``status`` and ``code`` go into the response."""

    def __init__(self, message: str, *, status: int = 400, code: str = "legion_link_error") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def available() -> bool:
    """Only a Hub with local users links accounts; a Legion-signed-in user already is one.
    Nothing is linked while the Market integration is off (MARKET_ENABLED)."""
    return legion_auth.AUTH_MODE == "local" and market_enabled()


def profile_url() -> str:
    return legion_auth.legion_url("/account")


def normalize_key(value: str) -> str:
    key = (value or "").strip()
    if not _KEY_RE.match(key):
        raise LegionLinkError(
            "This is not a Legion API key. Copy the whole key (it starts with lgn_) from your Legion profile.",
            status=422, code="bad_key_format",
        )
    return key


async def fetch_account(key: str) -> dict[str, Any]:
    """The account behind ``key``, as Market's ``/api/account/me`` reports it."""
    try:
        async with httpx.AsyncClient(timeout=legion_auth.LEGION_HTTP_TIMEOUT_SEC) as client:
            response = await client.get(
                f"{legion_auth.LEGION_INTERNAL_URL}/api/account/me",
                headers={"Authorization": f"Bearer {key}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("legion link: Market unreachable (%s)", type(exc).__name__)
        raise LegionLinkError("Legion is unreachable. Try again shortly.", status=503, code="unreachable") from exc
    if response.status_code in (401, 403):
        raise LegionLinkError(
            "Legion did not accept this key: it was revoked, mistyped, or the account is blocked.",
            code="key_rejected",
        )
    try:
        body = response.json() if response.status_code == 200 else None
        account = body["account"] if body and body.get("auth") == "api_key" else None
        UUID(str(account["id"]))
        if not account.get("email"):
            raise ValueError("no email")
    except (ValueError, KeyError, TypeError):
        logger.warning("legion link: unexpected answer from Market: HTTP %s", response.status_code)
        raise LegionLinkError("Legion gave an unexpected answer. Try again shortly.", status=502,
                              code="bad_answer") from None
    return account


def _keyring():
    try:
        return load_keyring()
    except SecretKeyringError as exc:
        raise LegionLinkError("Secret storage is not configured on this Hub.", status=503,
                              code="storage_unavailable") from exc


def _encrypt(user_id: UUID, key: str) -> tuple[bytes, str, int]:
    try:
        return encrypt_secret(SecretValue(key), secret_id=user_id, owner_id=user_id, kind=AAD_KIND,
                              keyring=_keyring())
    except SecretCryptoError as exc:
        raise LegionLinkError("Could not store the key.", status=500, code="storage_failed") from exc


def stored_key(row: LegionLink) -> SecretValue:
    try:
        return decrypt_secret(row.ciphertext, secret_id=row.user_id, owner_id=row.user_id, kind=AAD_KIND,
                              key_id=row.key_id, aad_version=row.aad_version, keyring=_keyring())
    except SecretCryptoError as exc:
        raise LegionLinkError("The stored key cannot be read; link the account again.", status=500,
                              code="storage_failed") from exc


def _apply_account(row: LegionLink, account: dict[str, Any]) -> None:
    row.account_id = UUID(str(account["id"]))
    row.email = str(account["email"])
    row.display_name = account.get("display_name")
    row.status = "linked"
    row.checked_at = datetime.now(timezone.utc)


async def link(db: AsyncSession, user_id: UUID, raw_key: str) -> LegionLink:
    """Check the key with Market and store it; replaces an earlier link. Commits."""
    key = normalize_key(raw_key)
    account = await fetch_account(key)
    ciphertext, key_id, aad_version = _encrypt(user_id, key)
    row = await db.get(LegionLink, user_id)
    if row is None:
        row = LegionLink(user_id=user_id, ciphertext=ciphertext, key_id=key_id, aad_version=aad_version,
                         key_prefix=key[:_PREFIX_LEN], account_id=UUID(str(account["id"])), email="")
        db.add(row)
    else:
        row.ciphertext, row.key_id, row.aad_version = ciphertext, key_id, aad_version
        row.key_prefix = key[:_PREFIX_LEN]
        row.created_at = datetime.now(timezone.utc)
    _apply_account(row, account)
    await db.commit()
    logger.info("legion link: user %s linked to account %s with key %s", user_id, row.account_id, row.key_prefix)
    return row


async def refresh(db: AsyncSession, row: LegionLink) -> Optional[str]:
    """Ask Market about the stored key again. Returns None when Market answered
    (the row is updated and committed), else the error code (``unreachable``…);
    the row then keeps what the last answer said."""
    key = stored_key(row).reveal()
    try:
        account = await fetch_account(key)
    except LegionLinkError as exc:
        if exc.code != "key_rejected":
            return exc.code
        row.status = "rejected"
        row.checked_at = datetime.now(timezone.utc)
        await db.commit()
        logger.info("legion link: Market rejected key %s of user %s", row.key_prefix, row.user_id)
        return None
    _apply_account(row, account)
    await db.commit()
    return None


async def unlink(db: AsyncSession, user_id: UUID) -> bool:
    row = await db.get(LegionLink, user_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    logger.info("legion link: user %s unlinked key %s", user_id, row.key_prefix)
    return True


def link_out(row: Optional[LegionLink]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {
        "account_id": str(row.account_id),
        "email": row.email,
        "display_name": row.display_name,
        "key_prefix": row.key_prefix,
        "status": row.status,
        "checked_at": row.checked_at.isoformat() if row.checked_at else None,
        "linked_at": row.created_at.isoformat() if row.created_at else None,
    }
