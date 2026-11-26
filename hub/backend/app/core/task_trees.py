"""Per-task git worktrees: lifecycle on the backend side.

internal design notes, stage E1. With ``ROOM_TASK_WORKTREES``
on, a task works in its own ``git worktree`` (``/workspace/.legion/wt/<task8>``)
instead of switching HEAD of the room's shared tree. This module decides *when*
a tree exists:

* :func:`ensure_task_tree` — created (or reused: Resume / Retry / rework) when
  the task branch is ensured; recorded in ``task.worktree``;
* :func:`remove_task_tree` — dropped once the task's branch is merged;
* :func:`sweep_task_trees` — the watchdog's safety net: trees of terminal tasks.

A task with ``task.worktree`` empty is in the old mode (shared tree, HEAD pins)
and stays there for its whole life. Every failure here is soft: a tree that
cannot be made leaves the task on the old path, a tree that cannot be removed is
retried on the next pass.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset({"completed", "cancelled", "continued", "rejected"})


def task_worktrees_enabled() -> bool:
    """``ROOM_TASK_WORKTREES`` — new tasks get their own worktree (default off)."""
    return os.getenv("ROOM_TASK_WORKTREES", "").strip().lower() in {"1", "true", "yes", "on"}


def task8_of(task_id: "UUID | str") -> str:
    """The short tree name of a task: the first 8 characters of its id."""
    return str(task_id).replace("-", "")[:8]


def tree_container_path(tree: str) -> str:
    """Where ``<task8>`` lives inside the room container."""
    name = str(tree).strip().removeprefix("wt/")
    return f"/workspace/.legion/wt/{name}"


# Room-local dirs that live only in the shared /workspace, never in a task tree
# (bash saves large outputs to /workspace/.legion/logs and points read_file there).
_SHARED_ONLY_PREFIXES = (".legion/logs/",)


def tree_for_path(tree: str | None, path: str) -> str | None:
    """The tree a file tool should read ``path`` from: ``None`` for shared-only dirs."""
    if tree and path.lstrip("/").removeprefix("workspace/").startswith(_SHARED_ONLY_PREFIXES):
        return None
    return tree


_WS_REF = re.compile(r"(?<![\w./~-])/workspace(?![\w-]|/\.legion(?:/|(?![\w-])))")


def tree_cwd(cwd: str, tree: str) -> str:
    """Map an absolute ``/workspace[/…]`` cwd into the task tree (``.legion/…`` stays)."""
    root = tree_container_path(tree)
    c = (cwd or "/workspace").rstrip("/") or "/workspace"
    if c == root or c.startswith(root + "/"):
        return c
    if c == "/workspace":
        return root
    if c.startswith("/workspace/.legion"):
        return c
    if c.startswith("/workspace/"):
        return root + c[len("/workspace"):]
    return c


def rewrite_cmd_for_tree(cmd: str, tree: str) -> tuple[str, bool]:
    """Point ``/workspace`` in a task's command at its tree (``/workspace/.legion/…`` kept)."""
    root = tree_container_path(tree)
    new, n = _WS_REF.subn(root, cmd)
    return new, n > 0


# Dependencies shared with the room's main tree (plan E4): a task tree's
# ``node_modules`` / ``.venv`` may be a link to the shared directory, and an
# install written through it would change every task's dependencies at once.
_CMD_START = r"(?:^|[;&|(]\s*|\bsudo\s+)"
_NODE_INSTALL_RE = re.compile(
    _CMD_START
    + r"(?:(?:npm|pnpm|bun)\s+(?:install|i|ci|add|remove|rm|uninstall|update|up|upgrade|dedupe|prune|link)"
    r"|yarn\s+(?:install|add|remove|upgrade|up)|yarn(?=\s*(?:$|[;&|)])))\b"
)
_PY_INSTALL_RE = re.compile(
    _CMD_START
    + r"(?:(?:python3?\s+-m\s+)?pip3?\s+(?:install|uninstall)|uv\s+(?:pip\s+(?:install|uninstall)|sync|add|remove|venv)"
    r"|poetry\s+(?:install|add|remove|update)|python3?\s+-m\s+venv|virtualenv)\b"
)
_CD_RE = re.compile(r"\bcd\s+(?:--\s+)?(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|()]+))")
DEP_UNLINK_MARK = "__LEGION_DEP_UNLINKED__"
_DEP_UNLINK_LINE = re.compile(rf"^{DEP_UNLINK_MARK} (\S+)\n?", re.M)


def _dep_dirs(cmd: str, cwd: str, tree: str) -> list[str]:
    """Directories inside the tree where ``cmd`` may install: its cwd, the root, ``cd`` targets."""
    import posixpath

    root = tree_container_path(tree)
    found = [root, cwd]
    here = cwd
    for m in _CD_RE.finditer(cmd):
        target = next(g for g in m.groups() if g)
        if target.startswith(("~", "$", "-")) or "`" in target:
            continue
        here = posixpath.normpath(target if target.startswith("/") else posixpath.join(here, target))
        found.append(here)
    out: list[str] = []
    for d in found:
        if (d == root or d.startswith(root + "/")) and d not in out and "'" not in d:
            out.append(d)
    return out


def dep_unlink_prelude(cmd: str, cwd: str, tree: str) -> str:
    """Shell prefix that detaches a task tree from the shared dependencies before an install.

    Empty unless ``cmd`` installs packages. A ``node_modules`` / ``.venv`` that
    is still a symbolic link is removed, so the install builds the task's own
    copy; each removal is reported through ``DEP_UNLINK_MARK`` on stdout (see
    :func:`take_dep_unlink_note`).
    """
    kinds = []
    if _NODE_INSTALL_RE.search(cmd):
        kinds.append("node_modules")
    if _PY_INSTALL_RE.search(cmd):
        kinds.append(".venv")
    if not kinds:
        return ""
    dirs = " ".join(f"'{d}'" for d in _dep_dirs(cmd, cwd, tree))
    names = " ".join(kinds)
    return (
        f"for _d in {dirs}; do for _k in {names}; do "
        f'if [ -L "$_d/$_k" ]; then rm -f -- "$_d/$_k" && echo "{DEP_UNLINK_MARK} $_d/$_k"; fi; '
        "done; done; unset _d _k; "
    )


def take_dep_unlink_note(out: str) -> tuple[str, str]:
    """Strip the unlink markers from ``out``; return ``(output, note)`` (note empty if none)."""
    paths = _DEP_UNLINK_LINE.findall(out)
    if not paths:
        return out, ""
    cleaned = _DEP_UNLINK_LINE.sub("", out)
    venv = any(p.endswith("/.venv") for p in paths)
    note = (
        "\n[note: " + ", ".join(paths) + " was a link to the room's shared dependencies; "
        "it is replaced by your own install here, so the shared copy stays untouched"
        + ("; recreate the venv with `python -m venv .venv` if you need it" if venv else "")
        + "]"
    )
    return cleaned, note


def wants_task_tree(task) -> bool:
    """True when ``task`` should work (or already works) in its own tree.

    A task that already has a branch but no tree began before the flag: it
    finishes in the shared tree, so HEAD never changes under it mid-task.
    """
    if getattr(task, "worktree", None):
        return True
    if not task_worktrees_enabled():
        return False
    return not getattr(task, "branch_name", None)


def tree_of_task(task) -> str | None:
    """``<task8>`` of ``task``'s own worktree, or None (old mode)."""
    return task8_of(task.id) if getattr(task, "worktree", None) else None


async def tree_of_task_id(task_id: "UUID | str", db) -> str | None:
    """``<task8>`` of the task's own worktree, or None (old mode / unknown task)."""
    from app.models.task import Task

    try:
        tid = task_id if isinstance(task_id, UUID) else UUID(str(task_id))
    except ValueError:
        return None
    path = (
        await db.execute(select(Task.worktree).where(Task.id == tid))
    ).scalar_one_or_none()
    return task8_of(tid) if path else None


async def ensure_task_tree(room_id: "UUID | str", task, branch: str, base: str | None, db) -> bool:
    """Create or reuse the task's worktree on ``branch``; record it on the task.

    Returns True when the task now works in its tree. False — and nothing
    changed — when workspaced refused (the caller falls back to a checkout).
    Flush only; the caller commits.
    """
    from app.core import workspaced_client as wsd

    try:
        res = await wsd.add_task_worktree(str(room_id), task8_of(task.id), branch, base=base)
    except Exception as exc:  # noqa: BLE001 — soft: the old path still works
        logger.warning(
            "ensure_task_tree: room %s task %s: no worktree on %s (%s); using the shared tree",
            room_id, task.id, branch, exc,
        )
        return False
    path = res.get("path")
    if not path:
        return False
    if getattr(task, "worktree", None) != path:
        task.worktree = path
        db.add(task)
        await db.flush()
    logger.info(
        "ensure_task_tree: room %s task %s → %s (%s)",
        room_id, task.id, path, "created" if res.get("created") else "reused",
    )
    return True


async def remove_task_tree(room_id: "UUID | str", task, db, *, force: bool = False) -> bool:
    """Remove the task's worktree and clear ``task.worktree``; True when it is gone.

    Pending files are parked in a commit by workspaced first; a tree that cannot
    be parked stays (``task.worktree`` kept) for the next pass. Flush only.
    """
    from app.core import workspaced_client as wsd

    if not getattr(task, "worktree", None):
        return True
    try:
        res = await wsd.remove_task_worktree(str(room_id), task8_of(task.id), force=force)
    except Exception as exc:  # noqa: BLE001
        logger.warning("remove_task_tree: room %s task %s: %s", room_id, task.id, exc)
        return False
    if not res.get("ok"):
        logger.warning(
            "remove_task_tree: room %s task %s kept its tree: %s",
            room_id, task.id, res.get("error") or res,
        )
        return False
    task.worktree = None
    db.add(task)
    await db.flush()
    logger.info("remove_task_tree: room %s task %s — tree removed", room_id, task.id)
    return True


def _state(task: Any) -> str:
    return str(getattr(task.state, "value", task.state) or "").lower()


async def sweep_task_trees(db: AsyncSession) -> int:
    """Remove the trees of tasks that are over; return how many were removed.

    The watchdog's safety net for every path that ends a task without going
    through a merge (Cancel, Reject, a continuation taking over, a restart in
    between). Paused / reviewing / checkpoint tasks keep their tree.
    """
    from app.models.task import Task

    rows = (
        await db.execute(select(Task).where(Task.worktree.is_not(None)))
    ).scalars().all()
    removed = 0
    for task in rows:
        if _state(task) not in _TERMINAL_STATES:
            continue
        try:
            if await remove_task_tree(task.room_id, task, db):
                removed += 1
            await db.commit()
        except Exception:  # noqa: BLE001
            await db.rollback()
            logger.exception("task_trees: sweeping task %s failed", task.id)
    return removed
