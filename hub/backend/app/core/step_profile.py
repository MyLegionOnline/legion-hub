"""Time profile of one agent-loop step (internal design notes §7, stage 0.3).

Stage 0.2 answered "how much does a room cost"; this stage answers "where does
a step's wall-clock actually go".  The split is the input to stage 9 of the
plan — the decision whether any part of the backend is worth optimizing at all:

    hub_other = step_wall - llm_wait - workspaced_wait - db_wait

``hub_other`` is the Hub's own processing: prompt assembly, tool dispatch,
serialization, event fan-out, scheduling.  Everything else is waiting on
somebody else and no backend language changes it.

Mechanics
---------
A *step* is one iteration of an agent tool loop.  Call :func:`step_tick` at the
top of the loop body: it closes the previous iteration (recording it) and opens
a new one.  There is no explicit end call, because these loops exit through a
dozen ``return``/``break`` paths and a missed ``finally`` would be a leak.
Instead the last, unfinished iteration of each series is simply never recorded
— one step lost per session, out of the hundreds the acceptance sample needs.

Accumulation rides on a :class:`contextvars.ContextVar`, so each asyncio task
(one per session) accumulates into its own step without any plumbing through
call signatures.  A step opened in a task that then dies is dropped when the
task is garbage collected: the open-step registry is keyed weakly by task.

Nothing here changes behaviour.  Every recorder swallows its own errors — a
metrics bug must never take down a session.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import logging
import os
import re
import time
import weakref
from contextlib import contextmanager
from typing import Iterator

from app.core.metrics import (
    db_wait_seconds,
    llm_wait_open_tx_total,
    llm_wait_seconds,
    process_asyncio_tasks,
    process_gc_objects,
    process_open_fds,
    step_component_seconds,
    step_overrun_total,
    step_wall_seconds,
    steps_in_flight,
    steps_total,
    workspaced_wait_seconds,
)

logger = logging.getLogger(__name__)

# Sides of the system that run a step loop.  Bounded on purpose: an unknown
# value is folded into "other" rather than becoming a new Prometheus series.
SIDE_AGENT = "agent"
SIDE_CHAT_MASTER = "chat_master"
SIDE_PLAN = "plan"
SIDE_ROUTER = "router"
_KNOWN_SIDES = frozenset({SIDE_AGENT, SIDE_CHAT_MASTER, SIDE_PLAN, SIDE_ROUTER, "other"})


def _side(value: str | None) -> str:
    name = (value or "other").strip().lower()
    return name if name in _KNOWN_SIDES else "other"


def _provider(value: str | None) -> str:
    name = (value or "unknown").strip().lower()
    # Provider ids come from a fixed vocabulary in llm_providers.py, but a
    # misconfigured agent could carry anything; keep the label short and safe.
    return re.sub(r"[^a-z0-9_.-]", "_", name)[:40] or "unknown"


class _Step:
    """Accumulators for one in-progress step."""

    __slots__ = ("side", "t0", "llm", "workspaced", "db", "closed")

    def __init__(self, side: str) -> None:
        self.side = side
        self.t0 = time.perf_counter()
        self.llm = 0.0
        self.workspaced = 0.0
        self.db = 0.0
        self.closed = False


_current: contextvars.ContextVar["_Step | None"] = contextvars.ContextVar(
    "legion_step_profile_current", default=None
)

# Weakly keyed by the owning asyncio task so an abandoned step cannot pin
# memory or inflate the in-flight gauge forever.
_open_steps: "weakref.WeakKeyDictionary[asyncio.Task, _Step]" = weakref.WeakKeyDictionary()


def _register(step: _Step) -> None:
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no running loop
        return
    if task is not None:
        _open_steps[task] = step


# ---------------------------------------------------------------------------
# Step boundaries
# ---------------------------------------------------------------------------


def step_tick(side: str) -> None:
    """Close the current step (if any) and open the next one.

    Called at the top of every agent tool-loop iteration.
    """
    try:
        _close_current()
        step = _Step(_side(side))
        _current.set(step)
        _register(step)
    except Exception:  # pragma: no cover - instrumentation must never raise
        logger.debug("step_profile: step_tick failed", exc_info=True)


def step_end() -> None:
    """Close the current step without opening another.

    Optional: the loops do not need it, but a caller that knows the series is
    over can record its last iteration instead of dropping it.
    """
    try:
        _close_current()
    except Exception:  # pragma: no cover
        logger.debug("step_profile: step_end failed", exc_info=True)


def _close_current() -> None:
    step = _current.get()
    _current.set(None)
    if step is None or step.closed:
        return
    step.closed = True

    wall = max(time.perf_counter() - step.t0, 0.0)
    side = step.side
    measured = step.llm + step.workspaced + step.db
    other = wall - measured

    step_wall_seconds.labels(side=side).observe(wall)
    db_wait_seconds.labels(side=side).observe(step.db)
    step_component_seconds.labels(side=side, component="llm").observe(step.llm)
    step_component_seconds.labels(side=side, component="workspaced").observe(step.workspaced)
    step_component_seconds.labels(side=side, component="db").observe(step.db)
    if other < 0:
        # Concurrent waits inside one step (gather of several tool calls) can
        # sum past the wall-clock.  Clamping to zero keeps the histogram honest
        # about "the Hub itself spent nothing here"; the counter says how often
        # the subtraction had to be clamped, so the report can state whether
        # the hub_other number is trustworthy.
        step_overrun_total.labels(side=side).inc()
        other = 0.0
    step_component_seconds.labels(side=side, component="other").observe(other)
    steps_total.labels(side=side).inc()


# ---------------------------------------------------------------------------
# Component recorders
# ---------------------------------------------------------------------------


def record_llm_wait(provider: str | None, side: str, seconds: float) -> None:
    try:
        llm_wait_seconds.labels(provider=_provider(provider), side=_side(side)).observe(seconds)
        step = _current.get()
        if step is not None and not step.closed:
            step.llm += seconds
    except Exception:  # pragma: no cover
        logger.debug("step_profile: record_llm_wait failed", exc_info=True)


def record_workspaced_wait(op: str, seconds: float) -> None:
    try:
        workspaced_wait_seconds.labels(op=op).observe(seconds)
        step = _current.get()
        if step is not None and not step.closed:
            step.workspaced += seconds
    except Exception:  # pragma: no cover
        logger.debug("step_profile: record_workspaced_wait failed", exc_info=True)


def record_db_wait(seconds: float) -> None:
    """Add one database statement's time to the open step.

    No standalone histogram: per-statement latency is not what stage 0.3 asks
    for, and a histogram observed on every query would be the most expensive
    metric in the process.
    """
    try:
        step = _current.get()
        if step is not None and not step.closed:
            step.db += seconds
    except Exception:  # pragma: no cover
        logger.debug("step_profile: record_db_wait failed", exc_info=True)


def note_llm_call(side: str) -> None:
    """Stage 1.2.0: count a model call that starts holding a DB connection.

    Called at the *start* of every LLM request (the timers below and the proxy
    retry helper).  The side comes from the open step when there is one: the
    proxy retry helper serves both the agent and the ChatMaster but only knows
    ``agent``, while the step opened by the loop knows which side it is.
    """
    try:
        from app.db.pool_instrumentation import current_task_holds_connection

        if not current_task_holds_connection():
            return
        step = _current.get()
        effective = step.side if step is not None and not step.closed else _side(side)
        llm_wait_open_tx_total.labels(side=effective).inc()
    except Exception:  # pragma: no cover
        logger.debug("step_profile: note_llm_call failed", exc_info=True)


@contextmanager
def llm_span(provider: str | None, side: str) -> Iterator[None]:
    """Time an LLM request written as a block of statements."""
    note_llm_call(side)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_llm_wait(provider, side, time.perf_counter() - t0)


async def timed_llm(awaitable, *, provider: str | None, side: str):
    """Time a single ``await`` on an LLM client call."""
    note_llm_call(side)
    t0 = time.perf_counter()
    try:
        return await awaitable
    finally:
        record_llm_wait(provider, side, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# workspaced route -> bounded `op` label
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_HEXISH_RE = re.compile(r"^[0-9a-f]{12,}$", re.I)


def workspaced_op(path: str) -> str:
    """Collapse identifier segments so the `op` label stays a fixed vocabulary.

    ``/workspaces/2f1c.../files/read`` becomes ``/workspaces/{id}/files/read``.
    """
    out: list[str] = []
    for segment in (path or "").split("/"):
        if not segment:
            continue
        if _UUID_RE.match(segment) or _HEXISH_RE.match(segment) or segment.isdigit():
            out.append("{id}")
        else:
            out.append(segment[:40])
    return "/" + "/".join(out) if out else "/"


# ---------------------------------------------------------------------------
# Gauges polled from the background loop in metrics.py
# ---------------------------------------------------------------------------


def refresh_in_flight_gauge() -> None:
    """Recompute `hub_steps_in_flight` from the live registry.

    Recomputed rather than incremented/decremented: a step whose task died
    mid-flight never closes, and a counter would drift upward forever.
    """
    try:
        counts = {side: 0 for side in _KNOWN_SIDES}
        for task, step in list(_open_steps.items()):
            if step.closed or task.done():
                continue
            counts[step.side] = counts.get(step.side, 0) + 1
        for side, value in counts.items():
            steps_in_flight.labels(side=side).set(value)
    except Exception:  # pragma: no cover
        logger.debug("step_profile: in-flight gauge refresh failed", exc_info=True)


_diag_ticks = 0
# Full GC object census only every Nth poll: gc.get_objects() materializes a
# list of every tracked object and blocks the event loop while it does.  At the
# 30 s gauge interval that is once every 5 minutes — plenty to see a trend that
# takes hours to develop.
_HEAVY_DIAG_EVERY = 10


def refresh_process_diagnostics() -> None:
    """Coarse leak signals for the stage 0.2 finding about backend RSS.

    Deliberately cheap: descriptor count, GC generation counts and live task
    count, plus an occasional full object census.  If RSS climbs while all of
    them stay flat, the growth is in native allocations rather than in retained
    Python objects — that is the single most useful fork in the diagnosis, and
    it costs one poll.
    """
    global _diag_ticks
    try:
        try:
            process_open_fds.set(len(os.listdir("/proc/self/fd")))
        except OSError:
            pass  # not Linux; the deployed backend is, the dev box may not be

        for generation, count in enumerate(gc.get_count()):
            process_gc_objects.labels(generation=str(generation)).set(count)

        _diag_ticks += 1
        if _diag_ticks % _HEAVY_DIAG_EVERY == 1:
            process_gc_objects.labels(generation="tracked").set(len(gc.get_objects()))

        try:
            process_asyncio_tasks.set(len(asyncio.all_tasks()))
        except RuntimeError:
            pass
    except Exception:  # pragma: no cover
        logger.debug("step_profile: process diagnostics refresh failed", exc_info=True)
