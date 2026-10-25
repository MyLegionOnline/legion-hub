"""Without a configured token workspaced refuses every call (no built-in default)."""

import pytest
from fastapi import HTTPException

from app import auth


@pytest.mark.asyncio
async def test_empty_token_matches_nothing(monkeypatch):
    monkeypatch.setattr(auth, "WORKSPACED_TOKEN", "")
    # "Bearer " carries an empty token, which must not equal an unset one.
    for header in ("Bearer ", "Bearer anything", "Basic abc"):
        with pytest.raises(HTTPException) as exc:
            await auth.verify_token(header)
        assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_configured_token_passes(monkeypatch):
    monkeypatch.setattr(auth, "WORKSPACED_TOKEN", "secret")
    await auth.verify_token("Bearer secret")
    with pytest.raises(HTTPException):
        await auth.verify_token("Bearer other")
