"""Pool-level instruments for stage 1.2.0 of internal design notes (§8.2).

The 0.4 ladder showed the ceiling is the pool: every running proxy session
held one connection ``idle in transaction`` for the whole task.  Stage 1.2
releases the connection before every long ``await``; these instruments are
what proves it did, on the same meters before and after the fix.

Two things live here:

* **Pool wait** — ``hub_db_pool_wait_seconds``.  ``hub_db_wait_seconds`` times
  cursors, so a statement that first waited 10 s for a free connection showed
  up as ``other`` in the step profile.  The timing wraps ``Pool._do_get``, the
  one method that blocks on the queue (or opens an overflow connection).  It
  is a private SQLAlchemy API; ``tests/test_db_pool_instrumentation.py`` fails
  loudly if a version bump renames it.

* **Connection ownership by asyncio task.**  A checkout records the task that
  took the connection, a checkin forgets it.  ``current_task_holds_connection``
  then answers "is the coroutine that is about to wait on the model holding a
  pool connection?" for *every* session in the process — the proxy loop, user
  sessions, request handlers, background jobs — without threading the session
  through contextvars at each of the dozens of places sessions are opened.  A
  session holds a pooled connection exactly while its transaction is open, so
  this is the ``in_transaction()`` the plan asks about, measured where it
  costs a connection.

The pool events fire inside SQLAlchemy's greenlet, which runs on the event
loop thread within the awaiting task's step, so ``asyncio.current_task()`` is
the coroutine that issued the query.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref

from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

logger = logging.getLogger(__name__)

_OWNER_KEY = "legion_owner_task"

# Task -> number of pooled connections it currently holds.  Weak so a task
# that died holding a connection (the pool reclaims it on GC) cannot pin memory.
_held: "weakref.WeakKeyDictionary[asyncio.Task, int]" = weakref.WeakKeyDictionary()


def _current_task() -> asyncio.Task | None:
    try:
        return asyncio.current_task()
    except RuntimeError:  # no running loop (sync code, tests)
        return None


def _observe_pool_wait(seconds: float) -> None:
    try:
        from app.core.metrics import db_pool_wait_seconds

        db_pool_wait_seconds.observe(seconds)
    except Exception:  # pragma: no cover - instrumentation must never raise
        logger.debug("pool_instrumentation: pool wait not recorded", exc_info=True)


class _TimedGetMixin:
    """Times the blocking part of a checkout: queue wait or new connection."""

    def _do_get(self):  # noqa: D401 - SQLAlchemy private override
        started = time.perf_counter()
        try:
            return super()._do_get()  # type: ignore[misc]
        finally:
            _observe_pool_wait(time.perf_counter() - started)


class TimedQueuePool(_TimedGetMixin, QueuePool):
    """Sync twin, used by tests: same override, no event loop needed."""


class TimedAsyncAdaptedQueuePool(_TimedGetMixin, AsyncAdaptedQueuePool):
    """The pool ``create_async_engine`` uses by default, with the timing."""


# ---------------------------------------------------------------------------
# Connection ownership
# ---------------------------------------------------------------------------


def _on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
    try:
        task = _current_task()
        if task is None:
            return
        connection_record.info[_OWNER_KEY] = weakref.ref(task)
        _held[task] = _held.get(task, 0) + 1
    except Exception:  # pragma: no cover
        logger.debug("pool_instrumentation: checkout not tracked", exc_info=True)


def _on_checkin(dbapi_connection, connection_record) -> None:
    try:
        ref = connection_record.info.pop(_OWNER_KEY, None)
        task = ref() if ref is not None else None
        if task is None:
            return
        remaining = _held.get(task, 0) - 1
        if remaining > 0:
            _held[task] = remaining
        else:
            _held.pop(task, None)
    except Exception:  # pragma: no cover
        logger.debug("pool_instrumentation: checkin not tracked", exc_info=True)


def install_connection_tracking(target) -> None:
    """Attach ownership listeners to an Engine (sync) or a Pool."""
    from sqlalchemy import event

    event.listen(target, "checkout", _on_checkout)
    event.listen(target, "checkin", _on_checkin)


def current_task_holds_connection() -> bool:
    task = _current_task()
    if task is None:
        return False
    return _held.get(task, 0) > 0
