"""Memory gate for heavy bash commands.

internal design notes, stage E3. Several agents
share one workspace container; when its memory is nearly full, starting another
dev server, build or install only gets something OOM-killed. Before such a
command the bash tool asks the room's memory pressure (stage E0) and, while the
room is over the gate, waits — a few seconds at a time, up to a bound. The wait
is credited to the run's deadline (``run_clock.accruing``) and happens before
the command's own timeout starts, so it costs the agent neither. If memory is
still short at the end, the command is not started and the agent gets an error
naming the largest processes, so it can stop what it no longer needs.

Memory the agents cannot free is not waited for (E4 run 2): when the processes
the agents started (``agent_rss_mb``, tagged ``LEGION_PROC_OWNER``) hold less
than ``BASH_MEMORY_GATE_AGENT_SHARE`` of the room, the rest is the room's own
service (a ``next dev`` of 450-700 MB in a 1 GB room) or files in ``/tmp``.
Waiting cannot change that, and refusing made an agent loop on retries for
millions of tokens, so the command runs with a note. Either way the room gets
one ``ROOM_MEMORY_LOW`` notice (at most every ``ROOM_MEMORY_LOW_EVERY_HOURS``)
telling the user to raise the room's memory limit.

What is never gated: anything that is not a recognised heavy command (reads,
``ps``, ``free``, edits), and any command that kills processes — the agent must
always be able to free memory. An unknown reading never blocks: a failing
measurement must not stop work.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOM_MEMORY_LOW = "ROOM_MEMORY_LOW"

_POLL_SECONDS = 3.0
_TOUCH_EVERY_SECONDS = 30.0
_FENCE_EVERY_SECONDS = 10.0
_MIN_LISTED_MB = 10.0

# Commands that start something that may hold a lot of memory: dev servers,
# builds, bundlers, type checkers, test runners, compilers, containers. Package
# installs are matched by the bash tool's own install pattern, and every
# background launch is gated regardless of what it runs.
_HEAVY_RE = re.compile(
    r"(?:^|[;&|(`]\s*|\b(?:sudo|exec|time|nice|env)\s+|\$\(\s*)"
    r"(?:"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|build|start|test|serve|preview|watch|e2e|lint)\b|"
    r"(?:npm|pnpm|yarn|bun)\s+run\b|"
    r"npx\s+\S+|bunx\s+\S+|"
    r"next\s+(?:dev|build|start)\b|vite\b|webpack\b|rollup\b|esbuild\b|tsc\b|"
    r"jest\b|vitest\b|playwright\s+test\b|cypress\s+run\b|"
    r"node\s+\S+|deno\s+(?:run|task|test)\b|"
    r"cargo\s+(?:build|run|test|check|clippy)\b|rustc\b|"
    r"go\s+(?:build|run|test)\b|"
    r"(?:python3?\s+-m\s+)?pytest\b|python3?\s+\S+\.py\b|"
    r"mvn\b|gradle\b|\./gradlew\b|dotnet\s+(?:build|run|test)\b|"
    r"make\b|cmake\b|gcc\b|g\+\+\b|clang\b|"
    r"docker\b|remotion\b|ffmpeg\b|chromium\b|google-chrome\b"
    r")"
)

# A command that stops processes is how memory gets freed — never hold it back.
_KILL_RE = re.compile(r"(?:^|[;&|(]\s*|\bsudo\s+)(?:kill|pkill|killall)\b")


def gate_pct() -> float:
    try:
        return float(os.getenv("BASH_MEMORY_GATE_PCT", "80"))
    except ValueError:
        return 80.0


def gate_psi_full() -> float:
    try:
        return float(os.getenv("BASH_MEMORY_GATE_PSI_FULL", "10"))
    except ValueError:
        return 10.0


def gate_wait_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("BASH_MEMORY_GATE_WAIT_SECONDS", "60")))
    except ValueError:
        return 60.0


def gate_agent_share() -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv("BASH_MEMORY_GATE_AGENT_SHARE", "0.3"))))
    except ValueError:
        return 0.3


def notice_every_hours() -> float:
    try:
        return max(0.0, float(os.getenv("ROOM_MEMORY_LOW_EVERY_HOURS", "6")))
    except ValueError:
        return 6.0


def held_by_room(pressure: Optional[dict[str, Any]]) -> bool:
    """Most of the room's memory is not held by agent processes.

    Unknown agent figures (Docker stats fallback, old workspaced) say no: the
    gate then behaves as before.
    """
    if not pressure:
        return False
    agents = pressure.get("agent_rss_mb")
    limit = pressure.get("mem_max_mb")
    if agents is None or not limit:
        return False
    return float(agents) < gate_agent_share() * float(limit)


def needs_gate(cmd: str, *, background: bool, install: bool) -> bool:
    """Whether ``cmd`` should wait for memory before it starts."""
    if _KILL_RE.search(cmd):
        return False
    return background or install or bool(_HEAVY_RE.search(cmd))


def is_busy(pressure: Optional[dict[str, Any]]) -> bool:
    """The room is over the gate. An unknown reading is never busy."""
    if not pressure:
        return False
    pct = pressure.get("pct")
    psi_full = pressure.get("psi_full_avg10")
    if pct is not None and pct >= gate_pct():
        return True
    return psi_full is not None and psi_full >= gate_psi_full()


async def _pressure(room_id: str) -> Optional[dict[str, Any]]:
    from app.core.room_slots import room_pressure

    return await room_pressure(room_id)


async def _touch_session(ctx: Any) -> None:
    from sqlalchemy import update

    from app.models.session.base import Session

    await ctx.db.execute(
        update(Session)
        .where(Session.id == ctx.session_id)
        .values(last_activity_at=datetime.now(timezone.utc))
    )
    await ctx.db.commit()


class GateResult:
    """``ok`` — run the command; else ``stopped`` (run fenced) or still busy."""

    __slots__ = ("ok", "stopped", "pressure", "waited", "held_by_room")

    def __init__(self, ok: bool, *, stopped: bool = False,
                 pressure: Optional[dict[str, Any]] = None, waited: float = 0.0,
                 held_by_room: bool = False) -> None:
        self.ok = ok
        self.stopped = stopped
        self.pressure = pressure
        self.waited = waited
        # ok although memory is full: the agents' processes hold little of it.
        self.held_by_room = held_by_room


async def wait_for_memory(ctx: Any, room_id: str) -> GateResult:
    """Wait while the room's memory is over the gate, up to the configured bound."""
    from app.core import run_clock
    from app.core.tools.registry import release_tool_db

    pressure = await _pressure(room_id)
    if not is_busy(pressure):
        return GateResult(True, pressure=pressure)
    if held_by_room(pressure):
        logger.info(
            "memory_gate: room %s at %s%% but agents hold %s MB - not waiting",
            room_id, pressure.get("pct"), pressure.get("agent_rss_mb"),
        )
        return GateResult(True, pressure=pressure, held_by_room=True)

    max_wait = gate_wait_seconds()
    started = time.monotonic()
    last_touch = started
    last_fence = started
    logger.info(
        "memory_gate: room %s at %s%% (psi full %s) — holding a heavy command up to %.0fs",
        room_id, pressure.get("pct"), pressure.get("psi_full_avg10"), max_wait,
    )
    # Never hold a pool connection across the wait.
    await release_tool_db(ctx, reason="bash/memory_gate")
    async with run_clock.accruing(getattr(ctx, "task_id", None)):
        while True:
            now = time.monotonic()
            if now - started >= max_wait:
                return GateResult(False, pressure=pressure, waited=now - started)
            await asyncio.sleep(min(_POLL_SECONDS, max(0.1, max_wait - (now - started))))
            now = time.monotonic()
            run_id = getattr(ctx, "run_id", None)
            if run_id and now - last_fence >= _FENCE_EVERY_SECONDS:
                # Stop / Pause / Retry must not wait out the gate.
                last_fence = now
                from app.core.task_runtime import is_run_current

                try:
                    if not await is_run_current(ctx.task_id, run_id, ctx.db):
                        return GateResult(False, stopped=True, pressure=pressure,
                                          waited=now - started)
                except Exception:
                    logger.debug("memory_gate: run fence check failed", exc_info=True)
                finally:
                    await release_tool_db(ctx, reason="bash/memory_gate")
            if now - last_touch >= _TOUCH_EVERY_SECONDS:
                # The session is alive, just waiting: keep the stall watchdog quiet.
                last_touch = now
                try:
                    await _touch_session(ctx)
                except Exception:
                    logger.debug("memory_gate: activity touch failed", exc_info=True)
                finally:
                    await release_tool_db(ctx, reason="bash/memory_gate")
            fresh = await _pressure(room_id)
            if fresh is None or not is_busy(fresh):
                waited = time.monotonic() - started
                logger.info("memory_gate: room %s freed after %.0fs", room_id, waited)
                return GateResult(True, pressure=fresh, waited=waited)
            if held_by_room(fresh):
                # The agents freed their share; what is left they cannot free.
                return GateResult(True, pressure=fresh, waited=time.monotonic() - started,
                                  held_by_room=True)
            pressure = fresh


def busy_message(pressure: dict[str, Any], waited: float, own_bg: list[tuple[str, str]]) -> str:
    """The error an agent gets when memory did not free up in time."""
    used = pressure.get("mem_used_mb")
    limit = pressure.get("mem_max_mb")
    pct = pressure.get("pct")
    head = "Error: workspace memory is busy"
    if used is not None and limit:
        head += f": {used:.0f} of {limit} MB ({pct:.0f}%)"
    elif pct is not None:
        head += f" ({pct:.0f}%)"
    lines = [head + f"; waited {waited:.0f}s, the command was not started."]
    # Skip the measurement's own shell and small fry: only what is worth stopping.
    top = [
        p for p in (pressure.get("top") or [])
        if (p.get("rss_mb") or 0) >= _MIN_LISTED_MB
        and "/sys/fs/cgroup" not in (p.get("cmd") or "")
    ]
    if top:
        procs = ", ".join(
            f"{(p.get('cmd') or '?')[:60]} (pid {p.get('pid')}, {p.get('rss_mb', 0):.0f} MB)"
            for p in top
        )
        lines.append(f"Largest processes: {procs}.")
    if own_bg:
        lines.append(
            "Your background processes: "
            + ", ".join(f"{bg_id} ({cmd[:60]})" for bg_id, cmd in own_bg)
            + "."
        )
    lines.append(
        "Stop what you no longer need (bash_kill, or kill <pid> for your own processes), "
        "reuse the room's running service instead of starting another copy of the app, "
        "then retry. Do not copy the project to work around this."
    )
    return "\n".join(lines)


def _figures(pressure: dict[str, Any]) -> str:
    used = pressure.get("mem_used_mb")
    limit = pressure.get("mem_max_mb")
    pct = pressure.get("pct")
    if used is not None and limit:
        return f"{used:.0f} of {limit} MB ({pct:.0f}%)"
    return f"{pct:.0f}%" if pct is not None else "nearly full"


def held_by_room_note(pressure: dict[str, Any]) -> str:
    """Appended to a command that ran although the room's memory is full."""
    agents = pressure.get("agent_rss_mb")
    tmp = pressure.get("tmp_mb")
    parts = [f"[workspace memory is {_figures(pressure)}"]
    if agents is not None:
        parts.append(f"; agent processes hold {agents:.0f} MB, the rest is the room's "
                     "own service and other processes")
    if tmp:
        parts.append(f"; /tmp holds {tmp:.0f} MB of it - delete scratch files you no "
                     "longer need")
    parts.append(". The command ran anyway. If it gets killed for memory, keep heavy "
                 "steps one at a time; the user has been told the room may need more memory.]")
    return "".join(parts)


async def notify_room_memory_low(db: Any, room_id: Any, pressure: dict[str, Any]) -> bool:
    """One ROOM_MEMORY_LOW notice per room every ``notice_every_hours``.

    Best effort: a failure is logged and the command goes on. Commits ``db``.
    """
    from datetime import timedelta
    from uuid import UUID

    from sqlalchemy import select

    from app.core.events import emit_event
    from app.models.message import Message

    try:
        rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
        # message.timestamp is naive UTC (datetime.utcnow).
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=notice_every_hours())
        recent = (
            await db.execute(
                select(Message.id)
                .where(
                    Message.room_id == rid,
                    Message.event_type == ROOM_MEMORY_LOW,
                    Message.timestamp >= since,
                )
                .limit(1)
            )
        ).first()
        if recent is not None:
            return False
        agents = pressure.get("agent_rss_mb")
        held = ""
        if agents is not None and pressure.get("mem_used_mb") is not None:
            held = (f" Agents' processes hold {agents:.0f} MB; the rest is the room's "
                    "own service and files.")
        await emit_event(
            room_id=rid,
            event_type=ROOM_MEMORY_LOW,
            content=(
                f"The room's memory is nearly full: {_figures(pressure)}.{held} Agents' "
                "builds and tests may wait or fail. Raise the room's memory limit on the "
                "Workspace tab."
            ),
            db=db,
            metadata={
                "pct": pressure.get("pct"),
                "mem_used_mb": pressure.get("mem_used_mb"),
                "mem_max_mb": pressure.get("mem_max_mb"),
                "agent_rss_mb": agents,
                "tmp_mb": pressure.get("tmp_mb"),
            },
        )
        await db.commit()
        return True
    except Exception:
        logger.warning("memory_gate: room memory notice failed for %s", room_id, exc_info=True)
        try:
            await db.rollback()
        except Exception:
            pass
        return False
