"""Tell "this MCP server wants a login" apart from "your token is wrong".

internal design notes Stage 0.  An MCP server that requires OAuth answers the
handshake with ``401`` and a ``WWW-Authenticate: Bearer`` challenge (RFC 9728 /
the MCP Authorization spec).  Today that is indistinguishable from a bad bearer
token, so Hub shows a mute verification failure and the user has nothing to
press.  This module turns the challenge into one fact: *who* is asking to be
logged into.

Deliberately read-only and stateless: nothing is registered, nothing is cached,
nothing is persisted.  The probe follows the challenge to the protected-resource
metadata and, if needed, to the authorization-server metadata, and returns at
most a short provider label for the card.

Everything read here is upstream-controlled data.  It is never executed, never
forwarded as a URL to any other component, and only ever leaves as a label that
:func:`_safe_label` has reduced to printable characters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException

from app import urlguard

logger = logging.getLogger("mcp_gateway")

#: The whole probe is a diagnostic on an already-failed request, so it stays
#: far below the discovery budget Hub allows (10s for the entire operation).
PROBE_TIMEOUT_SEC = 3.0
#: Metadata documents are small by specification; anything larger is not one.
MAX_METADATA_BYTES = 64 * 1024
MAX_LABEL_CHARS = 64

_WELL_KNOWN_PRM = "/.well-known/oauth-protected-resource"
_WELL_KNOWN_ASM = "/.well-known/oauth-authorization-server"


@dataclass(frozen=True)
class Challenge:
    """One parsed ``WWW-Authenticate`` header."""

    scheme: str
    params: dict[str, str]

    @property
    def resource_metadata(self) -> str | None:
        return self.params.get("resource_metadata")


def parse_challenge(header: Any) -> Challenge | None:
    """Parse ``WWW-Authenticate`` far enough to recognise a Bearer challenge.

    Tolerant by design: a server that sends a bare ``Bearer`` with no
    parameters still gets recognised, because the well-known paths are
    derivable from the endpoint alone.
    """

    if not isinstance(header, str) or not header.strip():
        return None
    if len(header) > 4096:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in header):
        return None
    head, _, rest = header.strip().partition(" ")
    scheme = head.strip().casefold()
    if scheme != "bearer":
        return None
    params: dict[str, str] = {}
    for chunk in rest.split(","):
        key, sep, value = chunk.partition("=")
        if not sep:
            continue
        key = key.strip().casefold()
        value = value.strip().strip('"')
        if key and value:
            params[key] = value
    return Challenge(scheme=scheme, params=params)


def _safe_label(value: Any, *, fallback: str | None = None) -> str | None:
    """Reduce an upstream-controlled name to something safe to display."""

    if not isinstance(value, str):
        return fallback
    cleaned = "".join(
        char for char in value.strip() if ord(char) >= 0x20 and ord(char) != 0x7F
    ).strip()
    # A label is a provider name, not a sentence, a URL or a payload.
    if not cleaned or "://" in cleaned or "\n" in cleaned:
        return fallback
    if len(cleaned) > MAX_LABEL_CHARS:
        cleaned = cleaned[:MAX_LABEL_CHARS].rstrip() + "…"
    return cleaned


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _host_label(url: str | None) -> str | None:
    if not isinstance(url, str):
        return None
    host = urlsplit(url).hostname
    return _safe_label(host)


def _metadata_candidates(endpoint: str, challenge: Challenge) -> list[str]:
    """Where the protected-resource metadata may live, best guess first."""

    candidates: list[str] = []
    advertised = challenge.resource_metadata
    if isinstance(advertised, str) and advertised:
        candidates.append(advertised)
    origin = _origin(endpoint)
    if origin:
        path = urlsplit(endpoint).path.rstrip("/")
        # RFC 9728 places the document at the well-known path *prefixed* to the
        # resource path; servers mounted at the origin use the bare form.
        if path:
            candidates.append(f"{origin}{_WELL_KNOWN_PRM}{path}")
        candidates.append(f"{origin}{_WELL_KNOWN_PRM}")
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


async def _fetch_metadata(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    """GET one metadata document through the shared SSRF perimeter."""

    try:
        guarded = urlguard.validate_endpoint(url)
        urlguard.validate_public_endpoint(guarded)
    except HTTPException:
        # A metadata URL pointing at loopback/private space is exactly what the
        # perimeter exists for.  Not an error to report — just not followed.
        return None
    try:
        response = await client.get(guarded, headers={"accept": "application/json"})
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    if len(response.content) > MAX_METADATA_BYTES:
        return None
    try:
        document = response.json()
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


async def detect_provider(endpoint: str | None, www_authenticate: Any) -> str | None:
    """Return a provider label when this endpoint wants an OAuth login.

    ``None`` means "not an OAuth challenge" — the caller keeps its existing
    ``mcp_auth_failed`` classification, so a genuinely wrong bearer token still
    reads as a wrong bearer token.
    """

    challenge = parse_challenge(www_authenticate)
    if challenge is None:
        return None
    if not isinstance(endpoint, str) or not endpoint:
        return None

    # From here on the answer is already "OAuth login required" — the metadata
    # only improves the *name* we show, so every failure below degrades to the
    # endpoint's own hostname rather than losing the finding.
    fallback = _host_label(endpoint) or "the external provider"
    try:
        async with httpx.AsyncClient(
            timeout=PROBE_TIMEOUT_SEC,
            follow_redirects=False,
        ) as client:
            resource_document: dict[str, Any] | None = None
            for candidate in _metadata_candidates(endpoint, challenge):
                resource_document = await _fetch_metadata(client, candidate)
                if resource_document is not None:
                    break

            issuer: str | None = None
            if resource_document is not None:
                label = _safe_label(resource_document.get("resource_name"))
                if label:
                    return label
                servers = resource_document.get("authorization_servers")
                if isinstance(servers, list) and servers and isinstance(servers[0], str):
                    issuer = servers[0]

            if issuer is None:
                issuer = _origin(endpoint)
            if issuer is None:
                return fallback

            server_document = await _fetch_metadata(
                client, f"{issuer.rstrip('/')}{_WELL_KNOWN_ASM}"
            )
            if server_document is not None:
                label = _host_label(str(server_document.get("issuer") or issuer))
                if label:
                    return label
            return _host_label(issuer) or fallback
    except Exception:
        # Never let a diagnostic probe change the outcome of discovery.  No
        # exception text is logged: it can carry an upstream URL or body.
        logger.info("mcp_oauth_probe outcome=degraded")
        return fallback


__all__ = ["Challenge", "detect_provider", "parse_challenge"]
