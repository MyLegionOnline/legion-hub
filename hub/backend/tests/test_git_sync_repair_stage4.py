"""
Stage 4 of internal design notes — recovery tool.

Incident recap: "Sync room files" zeroed out live files in the workspace
volume in workspace-mode. Stages 0-3 stop it from happening again; Stage 4
adds a way to repair a room that already got hit (or any future volume
corruption) by restoring from the room's own git history.

Covers:
  4.1/4.2 `workspace_recovery.compute_restore_preview` diffs the current
      volume tree against a candidate ref's tree (via the new
      `git_ls_tree_workspace` primitive) purely by size, without touching
      anything, and classifies every file as restore / dangerous / change /
      unchanged.
  4.2 `workspace_recovery.apply_restore` performs the actual restore
      (`git_revert_workspace` — checkout ref over the whole tree, auto-commit)
      and then re-materializes RepoFile metadata via `fs_sync_workspace_tree`
      so it doesn't go stale relative to the volume.
  4.3 the `/workspace/restore/apply` endpoint refuses to proceed past a
      dangerous overwrite (a file that would go from non-empty to empty)
      unless the caller explicitly passes confirm_dangerous=True.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import workspace as workspace_api
from app.core import workspace_recovery


# ---------------------------------------------------------------------------
# 4.1/4.2 — compute_restore_preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_restore_preview_classifies_every_file():
    room_id = uuid4()
    current_tree = [
        {"path": "/zeroed-now.js", "size": 0},          # corrupted today, good at ref -> restore
        {"path": "/keep.md", "size": 10},                # unchanged
        {"path": "/changed.py", "size": 20},              # both non-empty, differ -> change
        {"path": "/live-but-empty-at-ref.js", "size": 12},  # non-empty now, EMPTY at ref -> dangerous
    ]
    target_entries = [
        {"path": "zeroed-now.js", "size": 15},           # good historical content
        {"path": "keep.md", "size": 10},                 # same size
        {"path": "changed.py", "size": 8},                # differs from current
        {"path": "live-but-empty-at-ref.js", "size": 0},  # restoring this would zero it out
    ]

    with patch.object(
        workspace_recovery.wsd, "get_workspace_tree", new_callable=AsyncMock
    ) as mt, patch.object(
        workspace_recovery.wsd, "git_ls_tree_workspace", new_callable=AsyncMock
    ) as ml:
        mt.return_value = current_tree
        ml.return_value = target_entries
        preview = await workspace_recovery.compute_restore_preview(room_id, "goodsha", AsyncMock())

    by_path = {f["path"]: f for f in preview["files"]}
    assert by_path["/zeroed-now.js"]["status"] == "restore"
    assert by_path["/keep.md"]["status"] == "unchanged"
    assert by_path["/changed.py"]["status"] == "change"
    assert by_path["/live-but-empty-at-ref.js"]["status"] == "dangerous"
    assert preview["dangerous"] == 1
    assert preview["unchanged"] == 1
    assert preview["changed"] == 1
    assert preview["restored"] == 1
    ml.assert_awaited_once_with(str(room_id), "goodsha")


@pytest.mark.asyncio
async def test_compute_restore_preview_ignores_files_absent_from_target():
    """A file that exists now but not at the target ref is untouched by
    `git checkout <ref> -- .` (it never deletes), so it must not appear in
    the preview at all — showing it as "dangerous" would be a false alarm."""
    room_id = uuid4()
    with patch.object(
        workspace_recovery.wsd, "get_workspace_tree", new_callable=AsyncMock
    ) as mt, patch.object(
        workspace_recovery.wsd, "git_ls_tree_workspace", new_callable=AsyncMock
    ) as ml:
        mt.return_value = [{"path": "/new-file-not-in-history.txt", "size": 42}]
        ml.return_value = []
        preview = await workspace_recovery.compute_restore_preview(room_id, "goodsha", AsyncMock())

    assert preview["files"] == []
    assert preview["dangerous"] == 0


# ---------------------------------------------------------------------------
# 4.2 — apply_restore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_restore_reverts_tree():
    room_id = uuid4()
    db = AsyncMock()

    with patch.object(
        workspace_recovery.wsd, "git_revert_workspace", new_callable=AsyncMock
    ) as mr:
        mr.return_value = {"ok": True}
        result = await workspace_recovery.apply_restore(room_id, "goodsha", db)

    mr.assert_awaited_once_with(str(room_id), "goodsha", path=None)
    assert result == {"revert": {"ok": True}}


# ---------------------------------------------------------------------------
# 4.3 — /workspace/restore/apply endpoint gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_apply_endpoint_blocks_dangerous_without_confirmation():
    room_id = uuid4()
    dangerous_preview = {
        "ref": "goodsha",
        "files": [{"path": "/app.js", "current_size": 0, "target_size": 15, "status": "dangerous"}],
        "dangerous": 1,
        "restored": 0,
        "changed": 0,
        "unchanged": 0,
    }

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "compute_restore_preview", new_callable=AsyncMock,
             return_value=dangerous_preview,
         ), \
         patch.object(workspace_api, "apply_restore", new_callable=AsyncMock) as ma, \
         patch.object(workspace_api, "emit_event", new_callable=AsyncMock) as me:
        with pytest.raises(HTTPException) as exc_info:
            await workspace_api.workspace_restore_apply(
                room_id=room_id,
                body=workspace_api.RestoreApplyIn(ref="goodsha", confirm_dangerous=False),
                db=AsyncMock(),
                current_user=MagicMock(),
            )

    assert exc_info.value.status_code == 409
    ma.assert_not_awaited()
    me.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_apply_endpoint_proceeds_when_confirmed():
    room_id = uuid4()
    dangerous_preview = {
        "ref": "goodsha",
        "files": [{"path": "/app.js", "current_size": 0, "target_size": 15, "status": "dangerous"}],
        "dangerous": 1,
        "restored": 0,
        "changed": 0,
        "unchanged": 0,
    }

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "compute_restore_preview", new_callable=AsyncMock,
             return_value=dangerous_preview,
         ), \
          patch.object(
              workspace_api, "apply_restore", new_callable=AsyncMock,
              return_value={"revert": {"ok": True}},
          ) as ma, \
         patch.object(workspace_api, "emit_event", new_callable=AsyncMock) as me:
        result = await workspace_api.workspace_restore_apply(
            room_id=room_id,
            body=workspace_api.RestoreApplyIn(ref="goodsha", confirm_dangerous=True),
            db=AsyncMock(),
            current_user=MagicMock(),
        )

    ma.assert_awaited_once()
    me.assert_awaited_once()
    assert result.ok is True
    assert result.preview.dangerous == 1


@pytest.mark.asyncio
async def test_restore_apply_endpoint_does_not_require_confirmation_when_safe():
    room_id = uuid4()
    safe_preview = {
        "ref": "goodsha",
        "files": [{"path": "/missing.txt", "current_size": 0, "target_size": 5, "status": "restore"}],
        "dangerous": 0,
        "restored": 1,
        "changed": 0,
        "unchanged": 0,
    }

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(
             workspace_api, "compute_restore_preview", new_callable=AsyncMock,
             return_value=safe_preview,
         ), \
         patch.object(
             workspace_api, "apply_restore", new_callable=AsyncMock,
             return_value={"revert": {"ok": True}},
         ) as ma, \
         patch.object(workspace_api, "emit_event", new_callable=AsyncMock):
        result = await workspace_api.workspace_restore_apply(
            room_id=room_id,
            body=workspace_api.RestoreApplyIn(ref="goodsha", confirm_dangerous=False),
            db=AsyncMock(),
            current_user=MagicMock(),
        )

    ma.assert_awaited_once()
    assert result.ok is True
