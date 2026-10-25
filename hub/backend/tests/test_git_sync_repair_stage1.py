"""
Stage 1 of internal design notes — correct semantics for
"Sync room files" (``sync_files_to_workspace``).

Stage 6 rewrote this endpoint: the workspace volume is the single source of
truth, so "sync" now refreshes the snapshot from the volume (a heartbeat
ping) and returns ``{"ok": True, "mode": "workspace", "synced": 0}``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.v1 import workspace as workspace_api
from app.models.workspace import Workspace


@pytest.mark.asyncio
async def test_sync_files_refreshes_snapshot_and_returns_ok():
    room_id = uuid4()
    ws = Workspace(room_id=room_id, status="running")
    db = AsyncMock()

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "_get_or_create_workspace", new_callable=AsyncMock, return_value=ws
         ), \
         patch.object(workspace_api.wsd, "get_workspace_snapshot", new_callable=AsyncMock) as msnap:
        msnap.return_value = {"files": [], "generation": {}}
        result = await workspace_api.sync_files_to_workspace(
            room_id=room_id,
            db=db,
            current_user=MagicMock(),
        )

    assert result == {"ok": True, "mode": "workspace", "synced": 0}
    msnap.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_files_surfaces_workspaced_error():
    room_id = uuid4()
    ws = Workspace(room_id=room_id, status="running")
    db = AsyncMock()

    request = httpx.Request("GET", "http://workspaced/tree")
    response = httpx.Response(500, request=request, text="boom")

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "_get_or_create_workspace", new_callable=AsyncMock, return_value=ws
         ), \
         patch.object(workspace_api.wsd, "get_workspace_snapshot", new_callable=AsyncMock) as msnap, \
         patch.object(workspace_api, "broadcast_workspace_event", new_callable=AsyncMock):
        msnap.side_effect = httpx.HTTPStatusError("boom", request=request, response=response)
        with pytest.raises(HTTPException) as exc_info:
            await workspace_api.sync_files_to_workspace(
                room_id=room_id,
                db=db,
                current_user=MagicMock(),
            )

    assert exc_info.value.status_code == 502
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_workspace_reports_fs_backend():
    room_id = uuid4()
    ws = Workspace(room_id=room_id, status="stopped")
    db = AsyncMock()
    res = MagicMock()
    res.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=res)

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "_get_or_create_workspace", new_callable=AsyncMock, return_value=ws
         ):
        result = await workspace_api.get_workspace(
            room_id=room_id, db=db, current_user=MagicMock()
        )

    assert result.fs_backend == "workspace"
