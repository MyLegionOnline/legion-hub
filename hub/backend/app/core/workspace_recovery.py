"""Room-files recovery: dry-run compare + apply restore from workspace git history.

See internal design notes Stage 4. The volume is the source of truth, and
every write through fs_write_file/bash auto-commits to the room's own git
repo — so a corrupted volume can be repaired by checking out an earlier
commit's tree. compute_restore_preview() compares the current volume against
a candidate ref *without changing anything*, flagging any file that would go
from non-empty to empty (N -> 0): the exact failure mode of the incident
this guards against. apply_restore() should only be called after that
preview has been shown to (and confirmed by) the caller.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import workspaced_client as wsd
from app.core.workspace_fs import normalize_ws_path


async def compute_restore_preview(room_id: UUID, ref: str, db: AsyncSession) -> dict[str, Any]:
    """Compare the current volume tree against `ref` without touching it.

    Returns {"ref", "files", "dangerous", "restored", "changed", "unchanged"}.
    Each entry in "files" is {"path", "current_size", "target_size", "status"}
    where status is one of:
      - "restore":   empty/missing now, non-empty at ref (fixes a prior loss)
      - "dangerous": non-empty now, empty at ref (restoring would zero it out)
      - "change":    both non-empty but sizes differ
      - "unchanged": same size (a cheap check, not a byte-for-byte guarantee,
        but enough to keep the list focused on what actually matters)

    Files that exist now but not at `ref` are omitted: `git checkout <ref>
    -- .` never deletes them, so a restore leaves them untouched.
    """
    current_tree = await wsd.get_workspace_tree(str(room_id))
    target_entries = await wsd.git_ls_tree_workspace(str(room_id), ref)

    current_sizes = {
        normalize_ws_path(item.get("path", "")): int(item.get("size", 0))
        for item in current_tree
    }

    files: list[dict[str, Any]] = []
    dangerous = restored = changed = unchanged = 0

    for entry in target_entries:
        path = normalize_ws_path(entry["path"])
        target_size = int(entry["size"])
        current_size = current_sizes.get(path, 0)

        if current_size == target_size:
            status = "unchanged"
            unchanged += 1
        elif current_size > 0 and target_size == 0:
            status = "dangerous"
            dangerous += 1
        elif current_size == 0 and target_size > 0:
            status = "restore"
            restored += 1
        else:
            status = "change"
            changed += 1

        files.append({
            "path": path,
            "current_size": current_size,
            "target_size": target_size,
            "status": status,
        })

    return {
        "ref": ref,
        "files": files,
        "dangerous": dangerous,
        "restored": restored,
        "changed": changed,
        "unchanged": unchanged,
    }


async def apply_restore(
    room_id: UUID,
    ref: str,
    db: AsyncSession,
    agent_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """Check out ``ref`` over the whole tree.

    The caller is expected to have already shown compute_restore_preview() to
    the user and gotten confirmation for any "dangerous" entries — this
    function performs the restore unconditionally.

    Stage 6: the ``repofile`` metadata re-materialization that used to run
    here is gone (the table no longer exists). The volume is the single
    source of truth; ``git_revert_workspace`` checks out the ref directly
    on the volume and the snapshot reflects it on the next read.
    """
    revert_result = await wsd.git_revert_workspace(str(room_id), ref, path=None)
    return {"revert": revert_result}
