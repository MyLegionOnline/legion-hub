"""Room disk accounting, level 1 (internal design notes §5.1–5.2, T3).

* :func:`measure_room_disk` — the bytes a room's directory takes on the
  volume, everything included (``.git``, dependencies, uploads): disk is
  disk, and exclusions are dodged by renaming a directory.
* :func:`clean_dependencies` — the "Clean dependencies" button: removes
  the reproducible directories the B1 backup skips (``npm install`` brings
  them back). A directory holding files tracked by git is the project's own
  content, not a dependency, and is left alone.
* :func:`data_volume_usage` — size and free space of the filesystem that
  holds the rooms, for the backend's disk watchdog (T0.16, T10).

The backend stores the result on ``workspace.disk_bytes`` and sums it per
owner; the hard limit is the kernel's (XFS project quota, T9).
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from app.generation import workspace_root
from app.room_id import require_safe_room_id

logger = logging.getLogger("workspaced.disk_usage")

#: The B1 backup exclusions (internal design notes B1, tariffs plan §5.1).
DEPENDENCY_DIRS = frozenset({
    "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", "target", ".cache",
})
#: A measurement past this budget returns what it counted, marked partial.
MEASURE_BUDGET_SEC = float(os.getenv("DISK_MEASURE_BUDGET_SEC", "30"))


def _entry_bytes(st: os.stat_result) -> int:
    """Allocated bytes where the platform reports blocks (what a quota
    counts), the apparent size elsewhere."""
    blocks = getattr(st, "st_blocks", None)
    return blocks * 512 if blocks is not None else st.st_size


def _tree_bytes(root: Path, deadline: float) -> tuple[int, bool]:
    """Bytes under ``root`` (symlinks not followed, hard links once)."""
    total = 0
    seen: set[tuple[int, int]] = set()
    stack = [str(root)]
    while stack:
        if time.monotonic() > deadline:
            return total, True
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if st.st_nlink > 1 and not stat.S_ISDIR(st.st_mode):
                        key = (st.st_dev, st.st_ino)
                        if key in seen:
                            continue
                        seen.add(key)
                    total += _entry_bytes(st)
                    if stat.S_ISDIR(st.st_mode):
                        stack.append(entry.path)
        except OSError:
            continue
    return total, False


def measure_room_disk(room_id: str) -> dict[str, Any]:
    room_id = require_safe_room_id(room_id)
    root = workspace_root(room_id)
    if not root.is_dir():
        return {"room_id": room_id, "bytes": 0, "partial": False, "exists": False}
    total, partial = _tree_bytes(root, time.monotonic() + MEASURE_BUDGET_SEC)
    if partial:
        logger.warning("disk measure of room %s hit the %.0fs budget", room_id, MEASURE_BUDGET_SEC)
    return {"room_id": room_id, "bytes": total, "partial": partial, "exists": True}


def data_volume_usage(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Size, used and free bytes of the filesystem holding ``path``.

    ``free_bytes`` is what an unprivileged writer can still take
    (``f_bavail``): room containers do not write as root, and the root
    reserve is not theirs.
    """
    st = os.statvfs(path)
    size = st.f_blocks * st.f_frsize
    return {
        "path": str(path),
        "size_bytes": size,
        "used_bytes": size - st.f_bfree * st.f_frsize,
        "free_bytes": st.f_bavail * st.f_frsize,
    }


def _tracked_dirs(root: Path) -> set[str] | None:
    """Every directory (relative, posix) that holds a git-tracked file;
    None when the room is not a git repository."""
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "ls-files", "-z"],
            capture_output=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # Unknown tracking: refuse to guess, keep everything.
        logger.warning("clean deps: git ls-files failed in %s: %s", root, exc)
        raise RuntimeError("cannot read the tracked files of the room") from exc
    dirs: set[str] = set()
    for raw in out.split(b"\0"):
        if not raw:
            continue
        parts = raw.decode("utf-8", "surrogateescape").split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    return dirs


def _rmtree(path: Path) -> None:
    def _retry_writable(func, target, _exc):
        try:
            os.chmod(target, stat.S_IRWXU)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_retry_writable)


def find_dependency_dirs(root: Path, tracked: set[str] | None) -> list[Path]:
    """Dependency directories under ``root``, outermost only, ``.git`` skipped."""
    found: list[Path] = []
    for current, dirnames, _ in os.walk(root):
        keep = []
        for name in dirnames:
            full = Path(current) / name
            if name == ".git" or full.is_symlink():
                continue
            if name in DEPENDENCY_DIRS:
                rel = full.relative_to(root).as_posix()
                if tracked is None or rel not in tracked:
                    found.append(full)
                    continue
            keep.append(name)
        dirnames[:] = keep
    return found


def clean_dependencies(room_id: str) -> dict[str, Any]:
    room_id = require_safe_room_id(room_id)
    root = workspace_root(room_id)
    if not root.is_dir():
        return {"room_id": room_id, "removed": [], "freed_bytes": 0, "bytes": 0}
    targets = find_dependency_dirs(root, _tracked_dirs(root))
    removed: list[str] = []
    freed = 0
    for target in targets:
        size, _ = _tree_bytes(target, time.monotonic() + MEASURE_BUDGET_SEC)
        _rmtree(target)
        if not target.exists():
            removed.append(target.relative_to(root).as_posix())
            freed += size
    after = measure_room_disk(room_id)
    logger.info("clean deps: room %s removed %d dir(s), freed %d bytes", room_id, len(removed), freed)
    return {"room_id": room_id, "removed": removed, "freed_bytes": freed, "bytes": after["bytes"]}
