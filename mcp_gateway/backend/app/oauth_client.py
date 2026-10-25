"""internal design notes Stage 2 — the actual provider egress for OAuth setup.

``oauth_probe`` (Stage 0) is a read-only diagnostic on an already-failed MCP
handshake. This module is its counterpart with state at stake: it is what
Hub's ``app.core.mcp_oauth`` (Stage 1) actually calls to read provider
metadata, register the Hub as a DCR client, and exchange an authorization
code or refresh token. Per the plan's "who calls authorize/token/register"
decision, Hub never makes any of these three calls directly — only Gateway
does, so every one of them crosses the same SSRF perimeter (``urlguard``) as
the MCP endpoint itself: HTTPS-only in production, no loopback / private /
link-local / metadata destination either literally or after DNS resolution,
and ``follow_redirects=False``.

A code, a refresh token and an access token all pass through this module.
None of them is ever logged — only the outcome (rejected / unreachable) and
an HTTP status code are, exactly like ``mcp_client``'s own error path.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException

from app import urlguard

logger = logging.getLogger("mcp_gateway")

TIMEOUT_SEC = 10.0
#: Metadata, a DCR response and a token response are all small by spec —
#: anything past this is not one, and buffering it further would just be a
#: memory sink handed to us by an untrusted provider.
MAX_RESPONSE_BYTES = 64 * 1024

_WELL_KNOWN_ASM = "/.well-known/oauth-authorization-server"
_WELL_KNOWN_PRM = "/.well-known/oauth-protected-resource"


def _unavailable(code: str) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={"code": code, "message": "OAuth provider is unavailable"},
    )


def _rejected(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=502, detail={"code": code, "message": message})


def _guarded(url: str) -> str:
    endpoint = urlguard.validate_endpoint(url)
    urlguard.validate_public_endpoint(endpoint)
    return endpoint


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _well_known_candidates(base: str, well_known: str) -> list[str]:
    """Where an RFC 8414/9728 metadata document may live, best guess first."""

    origin = _origin(base)
    path = urlsplit(base).path.rstrip("/")
    candidates = [f"{origin}{well_known}{path}"] if path else []
    candidates.append(f"{origin}{well_known}")
    return candidates


async def fetch_metadata(
    *, kind: Literal["authorization_server", "protected_resource"], target: str
) -> dict[str, Any]:
    """Read the authorization-server or protected-resource metadata document."""

    well_known = _WELL_KNOWN_ASM if kind == "authorization_server" else _WELL_KNOWN_PRM
    guarded = _guarded(target)
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC, follow_redirects=False) as client:
        for candidate in _well_known_candidates(guarded, well_known):
            try:
                response = await client.get(candidate, headers={"accept": "application/json"})
            except httpx.HTTPError:
                continue
            if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                continue
            try:
                document = response.json()
            except ValueError:
                continue
            if isinstance(document, dict):
                return document
    raise _unavailable("mcp_oauth_metadata_unavailable")


async def register_client(*, registration_endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST a Dynamic Client Registration (RFC 7591) request."""

    guarded = _guarded(registration_endpoint)
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC, follow_redirects=False) as client:
        try:
            response = await client.post(
                guarded, json=payload, headers={"accept": "application/json"}
            )
        except httpx.HTTPError:
            raise _unavailable("mcp_oauth_registration_unreachable") from None
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise _unavailable("mcp_oauth_registration_unreachable")
    try:
        document = response.json()
    except ValueError:
        document = None
    if response.status_code >= 400 or not isinstance(document, dict):
        logger.info("mcp_oauth_register outcome=rejected status=%s", response.status_code)
        raise _rejected("mcp_oauth_registration_rejected", "Provider rejected client registration")
    return document


async def exchange_token(*, token_endpoint: str, payload: dict[str, str]) -> dict[str, Any]:
    """POST a token-endpoint request (authorization_code or refresh_token grant).

    ``payload`` carries the authorization code / refresh token / PKCE
    verifier as form fields — never logged, never echoed back in an
    exception, and forwarded as-is to exactly one guarded endpoint.
    """

    guarded = _guarded(token_endpoint)
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC, follow_redirects=False) as client:
        try:
            response = await client.post(
                guarded,
                data=payload,
                headers={
                    "accept": "application/json",
                    "content-type": "application/x-www-form-urlencoded",
                },
            )
        except httpx.HTTPError:
            raise _unavailable("mcp_oauth_token_unreachable") from None
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise _unavailable("mcp_oauth_token_unreachable")
    try:
        document = response.json()
    except ValueError:
        document = None
    if response.status_code >= 400 or not isinstance(document, dict):
        logger.info("mcp_oauth_token outcome=rejected status=%s", response.status_code)
        raise _rejected("mcp_oauth_token_rejected", "Provider rejected the token request")
    return document


__all__ = ["exchange_token", "fetch_metadata", "register_client"]
