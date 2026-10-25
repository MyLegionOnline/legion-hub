from __future__ import annotations

import os


def _env_or_file(name: str, default: str = "") -> str:
    """``NAME_FILE`` (a Docker secret path) wins over ``NAME``, as in Hub and Market."""
    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        with open(file_path, encoding="utf-8") as fh:
            return fh.read().strip()
    return os.getenv(name, default)


# No built-in defaults: an empty token matches nothing (see app/security.py).
GATEWAY_TOKEN = _env_or_file("MCP_GATEWAY_TOKEN")
GATEWAY_ADMIN_TOKEN = _env_or_file("MCP_GATEWAY_ADMIN_TOKEN")
MARKET_API_URL = os.getenv("MARKET_API_URL", "http://legion-market:8300")
MARKET_API_TOKEN = _env_or_file("MARKET_API_TOKEN", os.getenv("HUB_MARKET_TOKEN", ""))
ADMIN_API_URL = os.getenv("ADMIN_API_URL", "")
ADMIN_API_TOKEN = _env_or_file("ADMIN_API_TOKEN", "")
# Explicit local-dev escape hatch (internal design notes Stage 7): set to
# "0" to skip the Hub credit-reservation verification call even when
# ADMIN_API_URL is configured. Verification is fail-closed whenever it does
# run, so this is the only way to run Gateway against a Hub that isn't up.
MCP_CREDITS_VERIFY = os.getenv("MCP_CREDITS_VERIFY", "1").strip().lower() not in {"0", "false", "no"}
# Self-hosted Hubs (internal design notes §3.10, stage 3b): a Legion
# account's API key (`Authorization: Bearer lgn_…`) is the second way in, paid
# from the account's central wallet in Market. "0" turns it off.
ACCOUNT_KEYS_ENABLED = os.getenv("MCP_GATEWAY_ACCOUNT_KEYS", "1").strip().lower() not in {"0", "false", "no", "off"}
# Calls per minute one key may make, with bursts of this many (per process).
KEY_RATE_PER_MIN = float(os.getenv("MCP_GATEWAY_KEY_RATE_PER_MIN", "120"))
KEY_RATE_BURST = int(os.getenv("MCP_GATEWAY_KEY_RATE_BURST", "30"))
# How long an accepted key is believed without asking Market again. Paid calls
# always ask (the reservation); this only spares the tool listing.
KEY_CACHE_SEC = float(os.getenv("MCP_GATEWAY_KEY_CACHE_SEC", "30"))
# How long an account's acceptance of a server's Server Terms is believed
# (app.server_terms). An acceptance is never withdrawn, so all this delays is
# a new version of the terms reaching a run of calls already under way.
SERVER_TERMS_CACHE_SEC = float(os.getenv("MCP_GATEWAY_SERVER_TERMS_CACHE_SEC", "60"))
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_SEC = float(os.getenv("MCP_GATEWAY_CACHE_TTL_SEC", "60"))
REQUEST_TIMEOUT_SEC = float(os.getenv("MCP_GATEWAY_REQUEST_TIMEOUT_SEC", "30"))
# Vision's provider deadline is 45 seconds.  Keep this separate from the
# ordinary MCP request limit so a slow image model cannot change search/toolbox
# behavior. extract_design_spec/critique_design (Design Vision Toolkit
# Stage 3) run a bigger prompt and a bigger structured output than
# describe_image, so this carries more margin than the old 60s; Hub's own
# MCP_VISION_GATEWAY_TIMEOUT_SEC must stay strictly greater than this value.
VISION_REQUEST_TIMEOUT_SEC = float(os.getenv("MCP_VISION_REQUEST_TIMEOUT_SEC", "90"))
# Design Vision Toolkit plan R10 (Stage 3): raised globally from 1 MiB to
# 8 MiB for every MCP server, not just mcp-vision — image/UI-analysis
# artifacts need the headroom and a per-slug ceiling would just be a second
# branch to maintain. Hub's structural safety net (execute_mcp_tool /
# MCP_MAX_RESULT_CHARS) is what keeps a large response out of agent context;
# this limit only bounds what Gateway is willing to buffer in memory.
MAX_RESPONSE_BYTES = int(os.getenv("MCP_GATEWAY_MAX_RESPONSE_BYTES", "8388608"))
MCP_SEARXNG_AUTH_TOKEN = os.getenv("MCP_SEARXNG_AUTH_TOKEN", os.getenv("MCP_HTTP_AUTH_TOKEN", ""))
MCP_VISION_AUTH_TOKEN = os.getenv("MCP_VISION_AUTH_TOKEN", "")
# legion-vps-mcp (internal design notes Stage 9): the first-party VPS transport is a
# shared internal service exactly like mcp-vision — its bearer token lives in
# Gateway configuration, never in the Market catalog record.
MCP_VPS_AUTH_TOKEN = os.getenv("MCP_VPS_AUTH_TOKEN", "")
MCP_VISION_ADMIN_TOKEN = os.getenv("MCP_VISION_ADMIN_TOKEN", "")
MCP_VISION_ADMIN_URL = os.getenv("MCP_VISION_ADMIN_URL", "http://mcp-vision:3000/admin/config")
# internal design notes §3.13/Stage 9 — the operator's *declared* egress posture for
# this deployment: "broker" (legion-egress enforces it outside the MCP
# process), "host_firewall" (the operator enforces it in the host's kernel;
# Legion cannot verify that) or "off" (no enforcement). Declared here rather
# than in the Hub because Gateway is the component that actually sits next to
# vps-mcp and legion-egress — one source of truth, so the two cannot drift.
# What is *observed* comes from the service's own /health; the posture
# endpoint reports both and says plainly when they disagree.
MCP_EGRESS_MODE = (os.getenv("MCP_EGRESS_MODE", "broker") or "").strip().lower() or "broker"
# Fallback when the Market catalog has no legion-vps-mcp row (or is down):
# posture must still be answerable, because "I cannot tell you" is itself an
# honest answer an operator needs.
MCP_VPS_INTERNAL_ENDPOINT = os.getenv("MCP_VPS_INTERNAL_ENDPOINT", "http://vps-mcp:3000/mcp")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


# Custom HTTP links are an SSRF boundary.  Production requires HTTPS and
# rejects private/link-local/metadata destinations after DNS resolution.
# Local development can explicitly opt into HTTP, but not via request data.
MCP_GATEWAY_REQUIRE_HTTPS = _env_bool("MCP_GATEWAY_REQUIRE_HTTPS", True)
MCP_GATEWAY_DNS_REBIND_PROTECTION = _env_bool("MCP_GATEWAY_DNS_REBIND_PROTECTION", True)
