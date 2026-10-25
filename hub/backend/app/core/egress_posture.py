"""Honest egress posture of this Hub (internal design notes §3.13, Stage 9).

The posture is *not* stored anywhere and is deliberately not a Hub setting.
Two facts make it up, and both live outside the Hub:

1. what the operator **declared** — ``MCP_EGRESS_MODE`` in the Gateway's
   environment, because Gateway is the component that actually stands next to
   ``vps-mcp`` and ``legion-egress``; a second copy in the Hub's own env would
   be a second truth, free to drift; and
2. what the service **does** — reported by ``legion-vps-mcp``'s own
   ``/health``, which is the only party that knows whether it has a broker.

This module fetches both through the Gateway and turns them into a verdict
plus one sentence a human can read. The rule the plan sets (§3.13) is "not 'we
are secure', but 'here is what you have enabled'": when the two disagree, or when
the service cannot be reached, that is what gets displayed — never an assumed
posture, and never silence.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.mcp_gateway import MCP_GATEWAY_TOKEN, MCP_GATEWAY_URL

logger = logging.getLogger(__name__)

#: A posture read is a health probe of an internal service, not a hot path;
#: this keeps a Room panel refresh (and an agent-visible line) from turning
#: into a probe per render, while staying short enough that an operator who
#: just changed the deployment sees it within a minute.
_CACHE_TTL_SEC = 30.0
_PROBE_TIMEOUT_SEC = 10.0

#: Verdicts, worst-to-best in terms of what a human must do about them.
#: ``mismatch``/``misconfigured`` mean the deployment is not what it claims —
#: they are the whole reason this exists.
VERDICT_SUMMARY = {
    "enforced": "Outgoing connections are restricted by the broker: a server can only be opened at an address from the Room's inventory.",
    "claimed_by_operator": "The broker is off — egress restriction is claimed at the host firewall level. Legion does not verify this.",
    "not_enforced": "The service's outgoing traffic is not restricted: it can connect to any address.",
    "mismatch": "The declared mode does not match the service's actual behavior — the posture is unreliable, check the deployment configuration.",
    "misconfigured": "MCP_EGRESS_MODE is set to an unknown value — the posture is not configured.",
    "unverified": "The posture could not be verified: the deployment service is unavailable or does not report its mode.",
}

_ENFORCED_VERDICTS = {"enforced"}


@dataclass(frozen=True)
class EgressPosture:
    declared_mode: str
    declared_mode_valid: bool
    verdict: str
    service: dict[str, Any]

    @property
    def enforced_by_legion(self) -> bool:
        """True only when Legion itself enforces egress and the service confirms it."""

        return self.verdict in _ENFORCED_VERDICTS

    @property
    def summary_text(self) -> str:
        return VERDICT_SUMMARY.get(self.verdict, VERDICT_SUMMARY["unverified"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "declared_mode": self.declared_mode,
            "declared_mode_valid": self.declared_mode_valid,
            "verdict": self.verdict,
            "enforced": self.enforced_by_legion,
            "summary": self.summary_text,
            "service": self.service,
        }


#: Returned when the Gateway itself cannot be asked. Still a posture answer —
#: "we don't know" — never a fallback to "broker".
_UNKNOWN = EgressPosture(
    declared_mode="unknown",
    declared_mode_valid=False,
    verdict="unverified",
    service={"reachable": False, "reason": "gateway_unreachable"},
)

_cache: tuple[float, EgressPosture] | None = None


def _parse(payload: Any) -> EgressPosture:
    if not isinstance(payload, dict):
        return _UNKNOWN
    service = payload.get("service")
    verdict = payload.get("verdict")
    if verdict not in VERDICT_SUMMARY:
        verdict = "unverified"
    return EgressPosture(
        declared_mode=str(payload.get("declared_mode") or "unknown"),
        declared_mode_valid=bool(payload.get("declared_mode_valid")),
        verdict=verdict,
        service=service if isinstance(service, dict) else {"reachable": False, "reason": "bad_response"},
    )


async def get_posture(*, refresh: bool = False) -> EgressPosture:
    """Read this deployment's egress posture through the Gateway."""

    global _cache
    now = time.monotonic()
    if not refresh and _cache is not None and (now - _cache[0]) < _CACHE_TTL_SEC:
        return _cache[1]
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC) as client:
            response = await client.get(
                f"{MCP_GATEWAY_URL.rstrip('/')}/v1/egress-posture",
                headers={"Authorization": f"Bearer {MCP_GATEWAY_TOKEN}"},
            )
        response.raise_for_status()
        posture = _parse(response.json())
    except (httpx.HTTPError, ValueError):
        logger.warning("egress posture probe failed", exc_info=True)
        posture = _UNKNOWN
    _cache = (now, posture)
    return posture


def reset_cache() -> None:
    """Drop the memoized posture (tests, and after a deployment change)."""

    global _cache
    _cache = None
