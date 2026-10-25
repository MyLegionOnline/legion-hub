"""Wall-clock credit for a proxy run that is only waiting on a background process.

``PROXY_MAX_DURATION_SECONDS`` bounds how long a worker may stay alive. Time the
agent spends blocked in ``bash_wait`` (a package install, a long build) is not
work the model is doing, so it is credited back: the effective deadline of the
task moves by the seconds spent waiting. The credit is capped so a hung process
cannot keep a worker alive forever.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from uuid import UUID

MAX_CREDIT_SECONDS = float(os.getenv("PROXY_MAX_WAIT_CREDIT_SECONDS", "10800"))
_MAX_TRACKED = 1000

_credit: dict[str, float] = {}


def _key(task_id: UUID | str | None) -> str | None:
    return str(task_id) if task_id is not None else None


def reset(task_id: UUID | str | None) -> None:
    key = _key(task_id)
    if key is not None:
        _credit.pop(key, None)


def add_credit(task_id: UUID | str | None, seconds: float) -> float:
    """Credit ``seconds`` to the task's deadline; returns the total credit."""
    key = _key(task_id)
    if key is None or seconds <= 0:
        return 0.0
    if key not in _credit and len(_credit) >= _MAX_TRACKED:
        _credit.pop(next(iter(_credit)))
    total = min(MAX_CREDIT_SECONDS, _credit.get(key, 0.0) + seconds)
    _credit[key] = total
    return total


@asynccontextmanager
async def accruing(task_id: UUID | str | None, interval: float = 5.0):
    """Credit wall-clock time to ``task_id`` *while* the block runs.

    Crediting once at the end is too late: the deadline is enforced by the LLM
    heartbeat while the tool is still blocked, so a wait that starts near the
    deadline would be cut off mid-way. Accruing every ``interval`` seconds keeps
    the effective deadline ahead of the wait.
    """
    if _key(task_id) is None:
        yield
        return
    stop = asyncio.Event()
    last = time.monotonic()

    async def _tick() -> None:
        nonlocal last
        while True:
            try:
                await asyncio.wait_for(stop.wait(), interval)
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            add_credit(task_id, now - last)
            last = now
            if stop.is_set():
                return

    ticker = asyncio.ensure_future(_tick())
    try:
        yield
    finally:
        stop.set()
        await asyncio.gather(ticker, return_exceptions=True)


def remaining_credit(task_id: UUID | str | None) -> float:
    key = _key(task_id)
    return max(0.0, MAX_CREDIT_SECONDS - _credit.get(key, 0.0)) if key else 0.0


def effective_deadline(deadline: float | None, task_id: UUID | str | None) -> float | None:
    """``deadline`` (a ``time.monotonic()`` value) shifted by the task's credit."""
    if deadline is None:
        return None
    key = _key(task_id)
    return deadline + _credit.get(key, 0.0) if key else deadline
