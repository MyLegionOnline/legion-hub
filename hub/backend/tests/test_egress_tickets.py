"""internal design notes §3.13 / Stage 4 — egress ticket format.

No DB required: `issue_egress_ticket`/`verify_egress_ticket` are pure
functions over `core.secret_config.configured_secret`. The actual
enforcement point is `legion-egress` (a separate Node process, tested in
`legion-egress/__tests__/`) — this file exists so a change to the wire
format both processes agree on is caught here first.
"""
from __future__ import annotations

import time
from uuid import uuid4

import pytest

from app.core.vps_inventory import (
    EgressTicketInvalid,
    issue_egress_ticket,
    verify_egress_ticket,
)

SECRET = "unit-test-secret"


def _issue(**overrides):
    fields = dict(room_id=uuid4(), server="PROD", host="203.0.113.10", port=22, ttl_sec=30)
    fields.update(overrides)
    return issue_egress_ticket(**fields)


def test_round_trip_claims_match_input():
    room_id = uuid4()
    token = issue_egress_ticket(room_id=room_id, server="PROD", host="203.0.113.10", port=22)
    claims = verify_egress_ticket(token, secret=None)
    assert claims.room_id == str(room_id)
    assert claims.server == "PROD"
    assert claims.host == "203.0.113.10"
    assert claims.port == 22
    assert claims.jti


def test_two_tickets_for_the_same_call_get_different_jti():
    room_id = uuid4()
    token_a = issue_egress_ticket(room_id=room_id, server="PROD", host="203.0.113.10", port=22)
    token_b = issue_egress_ticket(room_id=room_id, server="PROD", host="203.0.113.10", port=22)
    assert verify_egress_ticket(token_a).jti != verify_egress_ticket(token_b).jti
    assert token_a != token_b


def test_tampered_signature_is_rejected():
    token = _issue()
    tampered = token[:-4] + ("a" if token[-4] != "a" else "b") + token[-3:]
    with pytest.raises(EgressTicketInvalid):
        verify_egress_ticket(tampered)


def test_wrong_secret_is_rejected():
    token = issue_egress_ticket(room_id=uuid4(), server="PROD", host="203.0.113.10", port=22)
    with pytest.raises(EgressTicketInvalid):
        verify_egress_ticket(token, secret="not-the-configured-secret")


def test_expired_ticket_is_rejected():
    token = _issue(ttl_sec=1)
    time.sleep(1.2)
    with pytest.raises(EgressTicketInvalid):
        verify_egress_ticket(token)


def test_garbage_input_is_rejected():
    with pytest.raises(EgressTicketInvalid):
        verify_egress_ticket("not-a-valid-ticket")
    with pytest.raises(EgressTicketInvalid):
        verify_egress_ticket("")


def test_host_containing_the_field_delimiter_round_trips_safely():
    # Nothing in add_server() restricts the HOST variable's charset — a host
    # value that happens to contain "|" must not desync field parsing.
    token = issue_egress_ticket(room_id=uuid4(), server="PROD", host="evil|injected|host", port=22)
    claims = verify_egress_ticket(token)
    assert claims.host == "evil|injected|host"
