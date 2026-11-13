"""Serialise HEAD-moving git operations on a room's single working tree.

internal design notes §5 (Stage 3). The room volume is the one working
directory and many code paths run ``git checkout`` on it — task dispatch, task
merge, promotion, ``git_fetch``, revert, the direct-session branch pin. Nothing
serialises them, so two operations in the same room can switch each other's
branch mid-run and leave a half-merged tree behind.

The lock lives here, in ``workspaced``, because ``workspaced`` is the sole owner
of the volume: an ``asyncio.Lock`` in the backend would not hold across multiple
backend workers, and neither side can see the other's in-process locks.

Implementation — an advisory lock *file* at
``<WORKSPACE_DATA_DIR>/.workspaced-locks/<room_id>.lock``, beside the room
directories rather than inside one (it used to be ``.legion/git.lock`` in the
room, where the room's agents could delete or plant it):

* created with ``O_CREAT | O_EXCL`` so exactly one caller (across processes and
  threads) wins the race;
* carries the holder's pid, process instance id and acquisition token. A lock
  is broken only when its holder is provably gone — a dead pid, an earlier
  incarnation of this process, or (unidentifiable file) older than
  ``STALE_TTL``. A live holder keeps it however long it runs, up to
  ``HOLDER_MAX_AGE``: the old age-only rule took the lock away from a slow
  push after five minutes and let a second operation run beside it. Every git
  call now has a timeout (``app/git_exec.py``), so a holder cannot hang;
* released only by the acquisition that took it (token check);
* acquisition waits up to ``ACQUIRE_TIMEOUT`` seconds, then raises
  ``GitWorkspaceBusyError`` — callers turn that into a ``busy`` response rather
  than a silent skip;
* re-entrant within a single thread: ``git_revert`` takes the lock and then
  calls ``git_commit``, which takes it again — the nested acquire only bumps a
  depth counter.

The first acquire also backfills ``.git/info/exclude`` and the guard hooks for
a repo initialised outside ``git_init`` (``.legion/git.lock`` stays excluded
for rooms that still carry an old lock file).
"""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("workspaced.git.lock")

ACQUIRE_TIMEOUT = 60.0   # seconds a caller will wait for the lock
# A lock whose holder cannot be identified (unparseable file) is presumed
# abandoned after this long. A holder that is identified is judged by whether
# it is still alive, never by age: a long push must not lose its lock.
STALE_TTL = 300.0
# Last resort for a holder in *another* live process (several workspaced
# workers): every git call has a timeout (app/git_exec.py), so no operation
# legitimately holds the lock this long.
HOLDER_MAX_AGE = 3 * 3600.0
_POLL_INTERVAL = 0.1

# The lock lives next to the room directories, not inside one: the room
# container mounts only its own subdirectory, so an agent can neither delete
# the lock mid-operation nor plant one to block the Git tab.
_LOCK_DIR_NAME = ".workspaced-locks"

# One id per workspaced process. A lock carrying our pid but another instance
# id was left by an earlier incarnation of this container (pid 1 again).
_INSTANCE = uuid.uuid4().hex

# Per-thread re-entrancy: room_id -> held depth for this thread.
_local = threading.local()
# Locks this process holds right now (any thread).
_held: set[str] = set()
_held_guard = threading.Lock()


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
    root = _workspace_root(room_id)
    return root.parent / _LOCK_DIR_NAME / f"{root.name}.lock"


def _ensure_lock_ignored(room_id: str) -> None:
    """Best-effort: keep the lock file out of ``git add -A`` / ``git status``.

    ``git_init`` already seeds ``.git/info/exclude`` with the full local-exclude
    set (internal design notes Stage 1), so this is just a thin safety net for
    a repo initialised outside ``git_init``. The write logic lives in
    ``git_ops._write_local_excludes`` — not duplicated here.
    """
    from app.git_ops import _write_local_excludes
    from app.release_guard import ensure_ref_guard

    _write_local_excludes(_workspace_root(room_id))
    # Same safety net for the guard hooks: a repo cloned or initialised
    # outside ``git_init`` gets them before its next git operation.
    ensure_ref_guard(_workspace_root(room_id))


def _lock_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    if os.name != "posix":
        return True  # os.kill(pid, 0) is not a probe on Windows
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _stale_reason(path: Path, room_id: str) -> str | None:
    """Why the existing lock may be broken, or ``None`` while its holder lives."""
    age = _lock_age(path)
    if age is None:
        return None  # vanished meanwhile: the next create attempt decides
    try:
        fields = path.read_text(encoding="utf-8").split()
        pid, instance = int(fields[0]), fields[1]
    except (OSError, ValueError, IndexError):
        return f"unidentified holder, age {age:.0f}s" if age > STALE_TTL else None
    if instance == _INSTANCE:
        with _held_guard:
            held = room_id in _held
        return None if held else "left by this process"
    if pid == os.getpid():
        return "left by an earlier run of workspaced"
    if not _pid_alive(pid):
        return f"holder pid {pid} is gone"
    if age > HOLDER_MAX_AGE:
        return f"holder pid {pid} has held it {age:.0f}s"
    return None


def _try_create(path: Path, token: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
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
        os.write(
            fd,
            f"{os.getpid()} {_INSTANCE} {token} {datetime.now(timezone.utc).isoformat()}\n".encode(),
        )
    finally:
        os.close(fd)
    return True


def _release(path: Path, token: str) -> None:
    """Remove the lock file only if it is still this acquisition's."""
    try:
        fields = path.read_text(encoding="utf-8").split()
    except OSError:
        return
    if len(fields) >= 3 and fields[2] == token:
        with contextlib.suppress(OSError):
            path.unlink()


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
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    while True:
        with _held_guard:
            # Threads of this process queue here instead of racing the file.
            mine = room_id in _held
            if not mine and _try_create(path, token):
                _held.add(room_id)
                break
        if not mine:
            reason = _stale_reason(path, room_id)
            if reason:
                logger.warning("room_git_lock: breaking lock for room %s (%s)", room_id, reason)
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
        if time.monotonic() >= deadline:
            raise GitWorkspaceBusyError(room_id)
        time.sleep(_POLL_INTERVAL)

    depths[room_id] = 1
    try:
        _ensure_lock_ignored(room_id)
        yield
    finally:
        depths[room_id] = 0
        _release(path, token)
        with _held_guard:
            _held.discard(room_id)
