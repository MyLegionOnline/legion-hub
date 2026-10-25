"""The room zip download link (internal design notes §5.5): a one-time,
room-bound ticket that is never a sign-in — and sign-in refuses every
purpose token signed with the same key."""

from __future__ import annotations

import time
from uuid import uuid4

import pytest
from fastapi import HTTPException
from jose import JWTError, jwt

from app.api.v1 import workspace_files as wf
from app.core import security


@pytest.fixture(autouse=True)
def _fresh_tickets(monkeypatch):
    monkeypatch.setattr(wf, "_used_export_tickets", {})


def test_ticket_opens_once_for_its_room():
    user_id, room_id = uuid4(), uuid4()
    ticket, exp = wf._issue_export_ticket(user_id, room_id)
    assert exp - time.time() <= wf.EXPORT_TICKET_TTL_SEC
    assert wf._redeem_export_ticket(ticket, room_id) == user_id
    with pytest.raises(HTTPException) as again:
        wf._redeem_export_ticket(ticket, room_id)
    assert again.value.status_code == 403


def test_ticket_of_another_room_or_expired_is_refused():
    user_id, room_id = uuid4(), uuid4()
    ticket, _ = wf._issue_export_ticket(user_id, room_id)
    with pytest.raises(HTTPException):
        wf._redeem_export_ticket(ticket, uuid4())
    stale = jwt.encode(
        {"typ": wf.EXPORT_TICKET_TYPE, "uid": str(user_id), "room_id": str(room_id),
         "exp": int(time.time()) - 1, "jti": "x"},
        security.SECRET_KEY, algorithm=security.ALGORITHM,
    )
    with pytest.raises(HTTPException):
        wf._redeem_export_ticket(stale, room_id)


def test_a_sign_in_token_is_not_a_ticket():
    room_id = uuid4()
    with pytest.raises(HTTPException):
        wf._redeem_export_ticket(security.create_access_token(str(uuid4())), room_id)


def test_purpose_tokens_never_sign_in():
    user_id = uuid4()
    ticket, _ = wf._issue_export_ticket(user_id, uuid4())
    preview = jwt.encode(
        {"typ": "workspace_preview_ticket", "sub": str(user_id), "exp": int(time.time()) + 60},
        security.SECRET_KEY, algorithm=security.ALGORITHM,
    )
    for token in (ticket, preview):
        with pytest.raises(JWTError):
            security.decode_access_token(token)
    assert security.decode_access_token(security.create_access_token(str(user_id)))["sub"] == str(user_id)
