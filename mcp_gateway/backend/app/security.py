from __future__ import annotations

import re
import secrets
from uuid import uuid4

from fastapi import Depends, Header, HTTPException

from app import account_keys, config
from app.account_keys import Caller
from app.config import GATEWAY_ADMIN_TOKEN, GATEWAY_TOKEN


UPSTREAM_AUTHORIZATION_HEADER = "X-Legion-MCP-Upstream-Authorization"
CORRELATION_ID_HEADER = "X-Legion-Correlation-ID"
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def require_gateway_token(
    authorization: str | None = Header(default=None),
    x_mcp_gateway_token: str | None = Header(default=None),
) -> None:
    """Authenticate Hub→Gateway service calls.

    Authorization Bearer is the documented contract. The explicit header is
    accepted as a compatibility aid for internal health probes, never a user
    credential and never logged.
    """
    candidate = x_mcp_gateway_token
    if not candidate and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = value
    if not candidate or not secrets.compare_digest(candidate, GATEWAY_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing Gateway service token")


async def gateway_caller(
    authorization: str | None = Header(default=None),
    x_mcp_gateway_token: str | None = Header(default=None),
) -> Caller:
    """Our Hub's service token, or — on the routes a self-hosted Hub uses —
    a Legion account key (payments plan §3.10, stage 3b).

    The service token is compared first and never sent anywhere; a value
    that looks like an account key goes to Market for checking. Anything
    else is the same 401 as before.
    """
    candidate = x_mcp_gateway_token
    if not candidate and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = value.strip()
    if candidate and secrets.compare_digest(candidate, GATEWAY_TOKEN):
        return account_keys.SERVICE
    if config.ACCOUNT_KEYS_ENABLED and not x_mcp_gateway_token and account_keys.looks_like_key(candidate):
        return await account_keys.caller_for_key(candidate)
    raise HTTPException(status_code=401, detail="Invalid or missing Gateway service token")


def caller_upstream_authorization(
    caller: Caller = Depends(gateway_caller),
    value: str | None = Header(default=None, alias=UPSTREAM_AUTHORIZATION_HEADER),
) -> str | None:
    """:func:`upstream_authorization` for routes open to account keys. An
    account caller reaches catalogue items only, whose credentials live in
    Gateway config — it has no upstream credential to send."""
    if caller.is_account:
        if value is not None:
            raise HTTPException(status_code=403, detail="Upstream credentials are not accepted with a Legion API key")
        return None
    return _parse_upstream_authorization(value)


def caller_correlation_id(
    _: Caller = Depends(gateway_caller),
    value: str | None = Header(default=None, alias=CORRELATION_ID_HEADER),
) -> str:
    return _parse_correlation_id(value)


def require_gateway_admin_token(
    authorization: str | None = Header(default=None),
    x_mcp_gateway_admin_token: str | None = Header(default=None),
) -> None:
    """Authenticate the narrow Admin Center → Gateway operator channel."""

    candidate = x_mcp_gateway_admin_token
    if not candidate and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = value
    if not candidate or not secrets.compare_digest(candidate, GATEWAY_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing Gateway admin token")


def required_identity(
    x_legion_user_id: str | None = Header(default=None),
    x_legion_room_id: str | None = Header(default=None),
) -> tuple[str, str | None]:
    if not x_legion_user_id or len(x_legion_user_id) > 128:
        raise HTTPException(status_code=400, detail="X-Legion-User-Id is required")
    if x_legion_room_id and len(x_legion_room_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid X-Legion-Room-Id")
    return x_legion_user_id, x_legion_room_id


def upstream_authorization(
    _: None = Depends(require_gateway_token),
    value: str | None = Header(default=None, alias=UPSTREAM_AUTHORIZATION_HEADER),
) -> str | None:
    """Read the resolved upstream bearer only after Hub service auth succeeds."""

    return _parse_upstream_authorization(value)


def _parse_upstream_authorization(value: str | None) -> str | None:
    if value is None:
        return None
    scheme, separator, credential = value.partition(" ")
    if (
        len(credential.encode("utf-8")) > 4096
        or scheme.lower() != "bearer"
        or separator != " "
        or not credential
        or any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in credential)
    ):
        raise HTTPException(status_code=400, detail="Invalid upstream authorization header")
    return credential


def request_correlation_id(
    _: None = Depends(require_gateway_token),
    value: str | None = Header(default=None, alias=CORRELATION_ID_HEADER),
) -> str:
    """Accept only opaque, value-free correlation metadata after auth."""

    return _parse_correlation_id(value)


def _parse_correlation_id(value: str | None) -> str:
    if value is None:
        return f"mcpd_gateway_{uuid4().hex}"
    if not _CORRELATION_ID.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid correlation ID")
    return value
