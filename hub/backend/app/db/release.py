"""Return the pooled connection before a long ``await`` (internal design notes §8.2).

The rule of stage 1.2: no open transaction survives an ``await`` into the
outside world — the model, ``workspaced``, MCP, a sleep.  The 0.4 ladder
measured what happens otherwise: every running task held one connection
``idle in transaction`` for its whole duration, so the room ceiling was exactly
``pool_size + max_overflow``.

:func:`release_db` commits.  Not ``rollback``: the step's own changes (tokens,
statuses, ``last_activity_at``) must survive, and ``emit_event`` already
commits on the same session, so an early commit inside a loop is normal here.
SQLAlchemy checks a connection out again lazily on the next query.
``expire_on_commit=False`` keeps loaded objects usable — which is also the
trap: after the wait, state that another actor may have changed (watchdog,
Cancel) must be re-read before it is written.  See ``_ensure_run_active`` and
``_pair_closed_externally`` in proxy_session.py.

Transaction-scoped locks are the exception: ``pg_advisory_xact_lock`` and
``SELECT … FOR UPDATE`` live exactly as long as the transaction, so committing
would silently drop them — and with them the guarantee they give (a second
click on a session button waits for the first to finish).  Code that takes such
a lock flags the session — automatically for any ``FOR UPDATE`` or
``pg_advisory_xact_lock`` statement run through the ORM session, or explicitly
with :func:`mark_xact_lock` — and :func:`release_db` then keeps the transaction
and logs, instead of breaking the lock.  The flag clears itself when the root
transaction ends.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

XACT_LOCK_INFO_KEY = "legion_xact_lock"
# Task ids whose row this transaction has already locked (``before_flush``).
_TASK_LOCKS_INFO_KEY = "legion_task_locks"

# Kill switch, read once at import.  ``off`` reproduces the pre-1.2 behaviour
# with the 1.2.0 instruments still in place — the A-local run of the §8.2
# protocol — and is the fast rollback if early commits misbehave in production.
_ENABLED = os.getenv("DB_RELEASE_BEFORE_AWAIT", "on").strip().lower() not in {"off", "0", "false", "no"}


def mark_xact_lock(db: Any) -> None:
    """Record that ``db``'s current transaction holds a transaction-scoped lock."""
    try:
        db.info[XACT_LOCK_INFO_KEY] = True
    except Exception:  # pragma: no cover - a mock without .info
        pass


@event.listens_for(Session, "do_orm_execute")
def _flag_xact_locks(orm_execute_state) -> None:
    """Flag the session when a statement takes a transaction-scoped lock."""
    try:
        stmt = orm_execute_state.statement
        locking = getattr(stmt, "_for_update_arg", None) is not None or (
            "pg_advisory_xact_lock" in (getattr(stmt, "text", None) or "")
        )
        if locking:
            orm_execute_state.session.info[XACT_LOCK_INFO_KEY] = True
    except Exception:  # pragma: no cover - never let the guard break a query
        logger.debug("release: xact-lock detection failed", exc_info=True)


@event.listens_for(Session, "before_commit")
def _flush_tasks_before_the_rest(session) -> None:
    """Lock order task → session for every commit in the process.

    The unit of work is free to write ``UPDATE session`` before ``UPDATE task``
    when both are dirty.  Code that takes the task row first (the proxy loop's
    ``_flush_task_first``) then deadlocks against any commit that closes
    sessions and touches their tasks in one go — the A3 ladder caught
    ``checkpoint_workspace`` doing exactly that when a workspace was paused.
    ``before_commit`` runs before the commit's own flush, so flushing the dirty
    tasks here puts their UPDATEs first without touching any call site.
    """
    try:
        from app.models.task import Task

        tasks = [
            obj for obj in session.dirty
            if isinstance(obj, Task) and session.is_modified(obj)
        ]
    except Exception:  # pragma: no cover - ordering is best effort
        logger.debug("release: task-first flush skipped", exc_info=True)
        return
    if tasks:
        session.flush(tasks)


@event.listens_for(Session, "before_flush")
def _lock_tasks_before_sessions(session, flush_context, instances) -> None:
    """Lock order task → session for every flush, autoflush included.

    ``before_commit`` only orders the commit's own flush.  A session row can
    reach the database earlier: after every model reply
    ``record_context_measurement`` writes the trace into
    ``session.context_snapshot``, and the next SELECT autoflushes it — the
    repeated A3 run caught the proxy loop holding ``UPDATE session`` from
    ``_pair_closed_externally``'s autoflush while ``_flush_task_first`` waited
    for the task row a workspace pause had already locked before closing that
    very session.  So a dirty session row with a task takes the task row lock
    first, unless this transaction already holds it.  ``FOR NO KEY UPDATE`` is
    the lock a plain ``UPDATE task`` takes; it goes through the connection, not
    the ORM, so ``_flag_xact_locks`` does not pin the transaction.
    """
    try:
        from app.models.session.base import Session as SessionRow

        locked = session.info.setdefault(_TASK_LOCKS_INFO_KEY, set())
        task_ids = {
            task_id
            for obj in session.dirty
            if isinstance(obj, SessionRow)
            and (task_id := getattr(obj, "task_id", None)) is not None
            and task_id not in locked
            and session.is_modified(obj)
        }
        if not task_ids:
            return
        conn = session.connection()
        if conn.dialect.name != "postgresql":
            return
        conn.execute(
            text("SELECT 1 FROM task WHERE id = ANY(:ids) ORDER BY id FOR NO KEY UPDATE"),
            {"ids": sorted(task_ids)},
        )
        locked.update(task_ids)
    except Exception:  # pragma: no cover - ordering is best effort
        logger.debug("release: task lock before session flush skipped", exc_info=True)


@event.listens_for(Session, "after_transaction_end")
def _clear_xact_lock_flag(session, transaction) -> None:
    if transaction.parent is None:
        session.info.pop(XACT_LOCK_INFO_KEY, None)
        session.info.pop(_TASK_LOCKS_INFO_KEY, None)


async def release_db(db: Any, *, reason: str) -> bool:
    """Commit ``db`` if it has an open transaction; return whether it did.

    ``reason`` names the wait for the log line on the locked path.
    Anything that is not a real open transaction (``None``, a test double)
    is left alone: the check is ``is True``, not truthiness.
    """
    if db is None or not _ENABLED:
        return False
    try:
        in_tx = db.in_transaction()
    except Exception:
        return False
    if inspect.iscoroutine(in_tx):  # an AsyncMock standing in for a session
        in_tx.close()
        return False
    if in_tx is not True:
        return False
    try:
        locked = db.info.get(XACT_LOCK_INFO_KEY) is True
    except Exception:
        locked = False
    if locked:
        logger.warning(
            "release_db(%s): transaction-scoped lock held — keeping the transaction "
            "(and its connection) across the wait",
            reason,
        )
        return False
    await db.commit()
    return True
