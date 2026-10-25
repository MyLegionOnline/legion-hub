"""One switch for every call the Hub makes to Legion Market (internal design notes stage 4.1).

``MARKET_ENABLED`` is off unless the operator turns it on: a self-hosted Hub
does not contact Legion services by default (docs/SELF_HOSTING_PRIVACY.md,
section 1). Off means the catalogue is empty, the background catalogue sync does
not run, nothing can be linked to a Legion account, and the Hub never opens a
connection to Market or to Legion's MCP gateway on its own. Legion Cloud and any
Hub that signs in with Legion accounts set it on.

Defaults for the addresses below are the public services; they matter only
once the switch is on.
"""

from __future__ import annotations

import os

DEFAULT_MARKET_URL = "https://market.mylegion.online"
DEFAULT_MCP_GATEWAY_URL = "https://gateway.mylegion.online"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


class MarketSwitchError(RuntimeError):
    """MARKET_ENABLED holds something that is neither true nor false."""


def market_enabled(environ=None) -> bool:
    raw = (os.environ if environ is None else environ).get("MARKET_ENABLED", "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise MarketSwitchError(f"MARKET_ENABLED={raw!r} must be true or false")


def market_url() -> str:
    """Server-to-server address of Market."""
    return (os.getenv("MARKET_API_URL", "").strip() or DEFAULT_MARKET_URL).rstrip("/")
