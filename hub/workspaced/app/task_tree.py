"""Per-task git worktrees (internal design notes, E0).

A task of a room works in its own ``git worktree`` at
``<room volume>/.legion/wt/<task8>`` — inside the container
``/workspace/.legion/wt/<task8>`` — so the room's shared tree (``/workspace``)
keeps one HEAD that nobody moves under a live agent.

This module only names and validates those trees; the git work lives in
``git_ops`` and the file work in ``file_ops``. Callers pass the tree as an
optional ``tree`` argument — the task id as ``<task8>`` or ``wt/<task8>`` — and
get the room root back when it is ``None``.

The agent owns everything under the volume, ``.legion/wt`` included, so every
path built here is checked link by link: no component may be a symbolic link,
and a tree must be a real directory.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Optional

# Where the trees live, relative to the room volume.
TREES_REL = (".legion", "wt")
TREES_DIR = "/".join(TREES_REL)
_PREFIX = "wt/"
# ``<task8>``: the first characters of a task id (hex/uuid), kept conservative
# so it is also a safe directory name and a safe git worktree name.
_TASK8 = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")


class TaskTreeError(ValueError):
    """A task tree that is malformed, missing, or unsafe to touch.

    ``status`` is the HTTP code the service answers with: 400 for a bad name
    or a link where a directory must be, 404 for a tree that does not exist.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def normalize_tree(tree: Optional[str]) -> Optional[str]:
    """``<task8>`` for ``tree`` (``<task8>`` or ``wt/<task8>``); ``None`` for none."""
    if tree is None:
        return None
    value = str(tree).strip()
    if not value:
        return None
    if value.startswith(_PREFIX):
        value = value[len(_PREFIX):]
    if not _TASK8.fullmatch(value):
        raise TaskTreeError(f"invalid task tree name: {tree!r}")
    return value


def require_task8(task8: str) -> str:
    value = normalize_tree(task8)
    if value is None:
        raise TaskTreeError("task tree name is required")
    return value


def tree_relative(task8: str) -> str:
    """``.legion/wt/<task8>`` — the tree's path inside the room volume."""
    return f"{TREES_DIR}/{require_task8(task8)}"


def container_path(task8: str) -> str:
    """The tree's path as the room container sees it."""
    return f"/workspace/{tree_relative(task8)}"


def _no_links(root: Path, parts: tuple[str, ...]) -> Path:
    """``root/parts…`` when no component is a link; raises ``TaskTreeError``."""
    path = root
    for part in parts:
        path = path / part
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise TaskTreeError(f"task tree does not exist: {path.name}", 404) from None
        except OSError as exc:
            raise TaskTreeError(f"task tree is unreadable: {exc}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise TaskTreeError(f"{path.name} is a symbolic link; refusing to use it")
        if not stat.S_ISDIR(st.st_mode):
            raise TaskTreeError(f"{path.name} is not a directory")
    return path


def trees_dir(room_root: Path) -> Path:
    """The room's ``.legion/wt`` directory; must exist and not be a link."""
    return _no_links(room_root, TREES_REL)


def tree_path(room_root: Path, tree: Optional[str]) -> Path:
    """The directory of ``tree`` in the room volume ``room_root``.

    ``tree`` ``None`` gives ``room_root`` itself. Otherwise the tree must exist
    as a real directory under ``.legion/wt`` with no link on the way.
    """
    task8 = normalize_tree(tree)
    if task8 is None:
        return room_root
    return _no_links(room_root, (*TREES_REL, task8))


def planned_tree_path(room_root: Path, task8: str) -> Path:
    """Where ``task8``'s tree would live — it need not exist yet."""
    return room_root.joinpath(*TREES_REL, require_task8(task8))


def room_of_tree(tree_root: Path) -> Optional[Path]:
    """The room volume a task tree directory belongs to, else ``None``."""
    parts = tree_root.parts
    if len(parts) >= 4 and parts[-3:-1] == TREES_REL and _TASK8.fullmatch(parts[-1]):
        return tree_root.parents[2]
    return None
