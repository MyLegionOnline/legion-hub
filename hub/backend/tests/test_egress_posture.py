"""internal design notes Stage 9 — the Hub never invents a posture.

`core/egress_posture` is a thin, memoized read of the Gateway's answer, whose
only real job is to refuse to be reassuring: an unreachable Gateway, a
Gateway that answers nonsense, or a deployment whose declared mode disagrees
with what the service does must all come out as "not enforced by us", with a
sentence a human can act on.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core import egress_posture


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if self._payload is _BAD_JSON:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]


_BAD_JSON = object()


def _patch_gateway(monkeypatch, payload: Any, *, status_code: int = 200, boom: bool = False) -> list[dict]:
    calls: list[dict] = []

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str, headers: dict | None = None) -> Any:
            calls.append({"url": url, "headers": headers or {}})
            if boom:
                raise httpx.ConnectError("no route to gateway")
            return _Response(payload, status_code)

    monkeypatch.setattr(egress_posture.httpx, "AsyncClient", _Client)
    egress_posture.reset_cache()
    return calls


@pytest.fixture(autouse=True)
def _clean_cache():
    egress_posture.reset_cache()
    yield
    egress_posture.reset_cache()


@pytest.mark.asyncio
async def test_broker_confirmed_by_the_service_is_the_only_enforced_verdict(monkeypatch):
    calls = _patch_gateway(monkeypatch, {
        "declared_mode": "broker",
        "declared_mode_valid": True,
        "verdict": "enforced",
        "service": {"reachable": True, "version": "0.1.0", "egress_mode": "broker"},
    })
    posture = await egress_posture.get_posture()
    assert posture.enforced_by_legion is True
    assert posture.declared_mode == "broker"
    assert "broker" in posture.summary_text.lower()
    assert calls[0]["url"].endswith("/v1/egress-posture")
    assert calls[0]["headers"]["Authorization"].startswith("Bearer ")


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["not_enforced", "claimed_by_operator", "mismatch", "misconfigured", "unverified"])
async def test_nothing_but_enforced_claims_enforcement(monkeypatch, verdict):
    _patch_gateway(monkeypatch, {
        "declared_mode": "off",
        "declared_mode_valid": True,
        "verdict": verdict,
        "service": {"reachable": True, "version": "0.1.0", "egress_mode": "direct"},
    })
    posture = await egress_posture.get_posture()
    assert posture.verdict == verdict
    assert posture.enforced_by_legion is False
    assert posture.summary_text == egress_posture.VERDICT_SUMMARY[verdict]


@pytest.mark.asyncio
async def test_unreachable_gateway_is_unverified_not_broker(monkeypatch):
    _patch_gateway(monkeypatch, None, boom=True)
    posture = await egress_posture.get_posture()
    assert posture.verdict == "unverified"
    assert posture.enforced_by_legion is False
    assert posture.service == {"reachable": False, "reason": "gateway_unreachable"}


@pytest.mark.asyncio
async def test_garbage_from_the_gateway_is_unverified(monkeypatch):
    _patch_gateway(monkeypatch, _BAD_JSON)
    assert (await egress_posture.get_posture()).verdict == "unverified"

    _patch_gateway(monkeypatch, {"verdict": "totally-fine-trust-me"})
    posture = await egress_posture.get_posture()
    assert posture.verdict == "unverified"
    assert posture.enforced_by_legion is False


@pytest.mark.asyncio
async def test_result_is_memoized_and_refresh_forces_a_new_probe(monkeypatch):
    payload = {
        "declared_mode": "broker",
        "declared_mode_valid": True,
        "verdict": "enforced",
        "service": {"reachable": True, "version": "0.1.0", "egress_mode": "broker"},
    }
    calls = _patch_gateway(monkeypatch, payload)
    await egress_posture.get_posture()
    await egress_posture.get_posture()
    assert len(calls) == 1
    await egress_posture.get_posture(refresh=True)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_room_facing_payload_omits_deployment_internals(monkeypatch):
    _patch_gateway(monkeypatch, {
        "declared_mode": "host_firewall",
        "declared_mode_valid": True,
        "verdict": "claimed_by_operator",
        "service": {"reachable": True, "version": "0.1.0", "egress_mode": "direct"},
    })
    body = (await egress_posture.get_posture()).as_dict()
    # The operator view keeps the label and the raw service report; the Room
    # endpoint (api/v1/room_config.py) forwards only these three keys.
    assert {"verdict", "enforced", "summary"} <= set(body)
    assert body["enforced"] is False
    assert "firewall" in body["summary"].lower()


@pytest.mark.asyncio
async def test_room_endpoint_forwards_exactly_three_keys(monkeypatch):
    """The Room line is for members, not operators: it carries the verdict and
    a sentence, never the declared label or the names of internal services."""
    from unittest.mock import AsyncMock

    from app.api.v1 import room_config

    _patch_gateway(monkeypatch, {
        "declared_mode": "broker",
        "declared_mode_valid": True,
        "verdict": "enforced",
        "service": {"reachable": True, "version": "0.1.0", "egress_mode": "broker"},
    })
    monkeypatch.setattr(room_config, "_verify_membership", AsyncMock(return_value=None))

    body = await room_config.room_vps_egress_posture(
        room_id=None, db=None, current_user=None,  # type: ignore[arg-type]
    )
    assert set(body) == {"verdict", "enforced", "summary"}
    assert body["enforced"] is True
