"""Serialise HEAD-moving git operations on a room's single working tree.

internal design notes §5 (Stage 3). The room volume is the one working
directory and many code paths run ``git checkout`` on it — task dispatch, task
merge, promotion, ``git_fetch``, revert, the direct-session branch pin. Nothing
serialises them, so two operations in the same room can switch each other's
branch mid-run and leave a half-merged tree behind.

The lock lives here, in ``workspaced``, because ``workspaced`` is the sole owner
of the volume: an ``asyncio.Lock`` in the backend would not hold across multiple
backend workers, and neither side can see the other's in-process locks.

Implementation — an advisory lock *file* at ``<workspace>/.legion/git.lock``:

* created with ``O_CREAT | O_EXCL`` so exactly one caller (across processes and
  threads) wins the race;
* carries the holder pid and an ISO timestamp, so a crashed holder is
  detectable — a lock file older than ``STALE_TTL`` is broken and re-taken;
* acquisition waits up to ``ACQUIRE_TIMEOUT`` seconds, then raises
  ``GitWorkspaceBusyError`` — callers turn that into a ``busy`` response rather
  than a silent skip;
* re-entrant within a single thread: ``git_revert`` takes the lock and then
  calls ``git_commit``, which takes it again — the nested acquire only bumps a
  depth counter.

The lock file is added to ``.git/info/exclude`` on first acquire so ``git add
-A`` never stages it and ``git status`` never reports the tree as dirty because
of it.
"""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("workspaced.git.lock")

ACQUIRE_TIMEOUT = 60.0   # seconds a caller will wait for the lock
STALE_TTL = 300.0        # a lock file older than this is presumed abandoned
_POLL_INTERVAL = 0.1

_LOCK_REL_PATH = ".legion/git.lock"

# Per-thread re-entrancy: room_id -> held depth for this thread.
_local = threading.local()


class GitWorkspaceBusyError(RuntimeError):
    """Raised when the room git lock could not be acquired within the timeout."""

    def __init__(self, room_id: str):
        super().__init__(
            f"workspace git tree for room {room_id!r} is busy — another git "
            f"operation is already in progress"
        )
        self.room_id = room_id


def _depths() -> dict[str, int]:
    d = getattr(_local, "depths", None)
    if d is None:
        d = {}
        _local.depths = d
    return d


def _workspace_root(room_id: str) -> Path:
    # Lazy import: git_ops imports this module for the wrappers, and reading the
    # root through git_ops keeps a monkeypatched WORKSPACE_DATA_DIR (tests) in
    # effect for the lock file too.
    from app import git_ops

    return git_ops._workspace_root(room_id)


def _lock_path(room_id: str) -> Path:
    return _workspace_root(room_id) / _LOCK_REL_PATH


def _ensure_lock_ignored(room_id: str) -> None:
    """Best-effort: keep the lock file out of ``git add -A`` / ``git status``.

    ``git_init`` already seeds ``.git/info/exclude`` with the full local-exclude
    set (internal design notes Stage 1), so this is just a thin safety net for
    a repo initialised outside ``git_init``. The write logic lives in
    ``git_ops._write_local_excludes`` — not duplicated here.
    """
    from app.git_ops import _write_local_excludes

    _write_local_excludes(_workspace_root(room_id))


def _lock_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _try_create(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except PermissionError:
        # Windows: opening a file that another thread is deleting (or holds
        # open) raises EACCES rather than EEXIST — treat it as "still locked".
        return False
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.EACCES):
            return False
        raise
    try:
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n".encode())
    finally:
        os.close(fd)
    return True


@contextlib.contextmanager
def room_git_lock(room_id: str, timeout: float | None = None):
    """Hold the room's git lock for the duration of the ``with`` block.

    Raises ``GitWorkspaceBusyError`` if the lock is not free within ``timeout``
    (``ACQUIRE_TIMEOUT`` when unset — read at call time so tests can shorten it).
    Re-entrant per thread.
    """
    if timeout is None:
        timeout = ACQUIRE_TIMEOUT
    depths = _depths()
    if depths.get(room_id, 0) > 0:
        depths[room_id] += 1
        try:
            yield
        finally:
            depths[room_id] -= 1
        return

    path = _lock_path(room_id)
    deadline = time.monotonic() + timeout
    while True:
        if _try_create(path):
            break
        age = _lock_age(path)
        if age is not None and age > STALE_TTL:
            logger.warning(
                "room_git_lock: breaking stale lock for room %s (age %.0fs)",
                room_id, age,
            )
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        if time.monotonic() >= deadline:
            raise GitWorkspaceBusyError(room_id)
        time.sleep(_POLL_INTERVAL)

    _ensure_lock_ignored(room_id)
    depths[room_id] = 1
    try:
        yield
    finally:
        depths[room_id] = 0
        with contextlib.suppress(OSError):
            path.unlink()
