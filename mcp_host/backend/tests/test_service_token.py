"""Without a configured token mcp_host refuses every call (no built-in default)."""

import pytest
from fastapi import HTTPException

from app import auth


@pytest.mark.asyncio
async def test_empty_token_matches_nothing(monkeypatch):
    monkeypatch.setattr(auth, "MCP_HOST_TOKEN", "")
    # "Bearer " is exactly what an unset token would expect.
    for header in ("Bearer ", "Bearer anything", None):
        with pytest.raises(HTTPException) as exc:
            await auth.verify_token(header)
        assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_configured_token_passes(monkeypatch):
    monkeypatch.setattr(auth, "MCP_HOST_TOKEN", "secret")
    await auth.verify_token("Bearer secret")
    with pytest.raises(HTTPException):
        await auth.verify_token("Bearer other")
