"""The one SSRF perimeter Gateway applies to user-supplied URLs.

Extracted from ``main.py`` unchanged (internal design notes Stage 0) so that the
OAuth metadata probe crosses the *same* barrier as the MCP endpoint itself
rather than growing a second, weaker copy of it: HTTP(S) only, no credentials
in the URL, HTTPS required in production, and no loopback / private /
link-local / metadata destination either literally or after DNS resolution.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException

from app import config

#: Hostnames that are not IP literals but still resolve inward by convention.
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "metadata.google.internal"}


def _unreachable() -> HTTPException:
    """One value-free failure: never tell a caller *which* host was blocked."""

    return HTTPException(
        status_code=502,
        detail={
            "code": "mcp_unreachable",
            "message": "MCP server is unavailable",
            "reached_upstream": False,
        },
    )


def validate_endpoint(endpoint: Any, *, reject_query: bool = False) -> str:
    if not isinstance(endpoint, str) or len(endpoint.encode("utf-8")) > 2048:
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in endpoint):
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    if reject_query and parts.query:
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    return endpoint


def is_blocked_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def validate_public_endpoint(endpoint: str) -> None:
    """Enforce the custom-link SSRF perimeter without exposing host details."""

    parts = urlsplit(endpoint)
    if config.MCP_GATEWAY_REQUIRE_HTTPS and parts.scheme != "https":
        raise _unreachable()
    host = parts.hostname
    if not host:
        raise HTTPException(status_code=422, detail="MCP endpoint is invalid")
    if host.casefold() in BLOCKED_HOSTNAMES:
        raise _unreachable()
    if is_blocked_address(host):
        raise _unreachable()
    if not config.MCP_GATEWAY_DNS_REBIND_PROTECTION:
        return
    try:
        addresses = socket.getaddrinfo(
            host,
            parts.port or (443 if parts.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        # Let the bounded HTTP request classify an unresolved public hostname
        # as mcp_unreachable.  A failed lookup does not grant private access.
        return
    except OSError:
        raise _unreachable() from None
    if any(is_blocked_address(str(info[4][0])) for info in addresses):
        raise _unreachable()


__all__ = [
    "BLOCKED_HOSTNAMES",
    "is_blocked_address",
    "validate_endpoint",
    "validate_public_endpoint",
]
