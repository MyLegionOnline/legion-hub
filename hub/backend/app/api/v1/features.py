"""GET /api/v1/features: what this Hub offers, so the UI can hide what is absent.

Public on purpose (the login page asks before there is a token) and carries
only on/off facts, no per-user data. The core answers for itself; a plugin adds
or overrides flags through ``PluginHost.add_features`` (internal design notes §3).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter

from app.core.market_switch import MarketSwitchError, market_enabled
from app.plugins import get_registry

router = APIRouter()

# What a Hub without plugins offers: nothing is sold there, so the UI hides
# the hour packs, the pricing links and the plan banners.
CORE_FEATURES: dict[str, Any] = {"billing": False}

# AGPL section 13: whoever runs a changed Hub must offer its users the source.
# The UI links here; a fork points LEGION_SOURCE_URL at its own repository.
DEFAULT_SOURCE_URL = "https://github.com/MyLegionOnline/legion-hub"


def source_url() -> str:
    """LEGION_SOURCE_URL when it is an http(s) address, else the public repository."""
    raw = os.getenv("LEGION_SOURCE_URL", "").strip()
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return raw
    return DEFAULT_SOURCE_URL


def market_flag() -> bool:
    """Whether the Legion Market integration is on (MARKET_ENABLED).

    Off by default: the UI then has no Marketplace entry, because following it
    would send the user to a Market this Hub is not meant to talk to.
    """
    try:
        return market_enabled()
    except MarketSwitchError:
        return False


def features_payload() -> dict[str, Any]:
    return {**CORE_FEATURES, "market": market_flag(), "source_url": source_url(), **get_registry().features}


@router.get("/features")
async def features() -> dict[str, Any]:
    return features_payload()
