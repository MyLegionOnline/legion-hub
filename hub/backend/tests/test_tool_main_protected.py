"""A workspaced 409 on a file write (HEAD is ``main``) reaches the agent as a
tool result, not as a session-ending "model provider error"."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from app.core import workspaced_client as wsd
from app.core.proxy_session import RunStopped, _execute_registry_tool
from app.core.tools import REGISTRY, ToolContext
from app.core.tools._session_state import clear_session, mark_read


def _ctx(session_id=None) -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()
    return ToolContext(room_id=uuid4(), session_id=session_id or uuid4(), task_id=uuid4(), db=db)


def _protected_error() -> wsd.WorkspaceMainProtectedError:
    request = httpx.Request("PUT", "http://workspaced/workspaces/r/file")
    response = httpx.Response(409, json={"detail": "main is protected"}, request=request)
    return wsd.WorkspaceMainProtectedError("main is protected", request=request, response=response)


def test_client_maps_409_to_main_protected():
    request = httpx.Request("PUT", "http://workspaced/workspaces/r/file")
    response = httpx.Response(409, json={"detail": "main is protected"}, request=request)
    with pytest.raises(wsd.WorkspaceMainProtectedError) as info:
        wsd._raise_if_main_protected(response)
    # Still an HTTPStatusError, so status-mapping callers keep working.
    assert isinstance(info.value, httpx.HTTPStatusError)
    assert info.value.response.status_code == 409
    wsd._raise_if_main_protected(httpx.Response(200, json={}, request=request))


@pytest.mark.asyncio
async def test_write_file_explains_protected_main():
    ctx = _ctx()
    with (
        patch("app.core.room_storage.room_storage_refusal", new=AsyncMock(return_value=None)),
        patch("app.core.workspaced_client.write_workspace_file", new=AsyncMock(side_effect=_protected_error())),
    ):
        result = await REGISTRY.get("write_file").execute(ctx, {"path": "report.md", "content": "x"})
    assert result.startswith("Error: report.md was written to disk but not committed")
    assert f"git checkout task/{ctx.task_id}" in result


@pytest.mark.asyncio
async def test_edit_file_explains_protected_main():
    sid = uuid4()
    ctx = _ctx(sid)
    mark_read(str(sid), "a.txt")
    with (
        patch("app.core.workspaced_client.read_workspace_file", new=AsyncMock(return_value="old\n")),
        patch("app.core.workspaced_client.write_workspace_file", new=AsyncMock(side_effect=_protected_error())),
    ):
        result = await REGISTRY.get("edit_file").execute(
            ctx, {"path": "a.txt", "old_string": "old", "new_string": "new"}
        )
    assert "not committed" in result and f"task/{ctx.task_id}" in result
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_proxy_tool_exception_becomes_error_result():
    tool = MagicMock()
    tool.execute = AsyncMock(side_effect=httpx.HTTPStatusError(
        "Client error '409 Conflict'",
        request=httpx.Request("PUT", "http://workspaced/x"),
        response=httpx.Response(409),
    ))
    with patch.object(REGISTRY, "get", return_value=tool):
        result = await _execute_registry_tool("write_file", _ctx(), {})
    assert result.startswith("Error: Client error '409 Conflict'")


@pytest.mark.asyncio
async def test_proxy_tool_control_flow_still_propagates():
    tool = MagicMock()
    tool.execute = AsyncMock(side_effect=RunStopped(uuid4(), uuid4()))
    with patch.object(REGISTRY, "get", return_value=tool):
        with pytest.raises(RunStopped):
            await _execute_registry_tool("write_file", _ctx(), {})
