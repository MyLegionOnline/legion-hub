"""Memory pressure inside one room's workspace container.

internal design notes, stage E0. The backend
asks this before admitting a third (or later) parallel task into a room and,
later, before a heavy command: how full the room's memory is right now, whether
the kernel is already stalling on it (PSI), how many out-of-memory kills the
cgroup has seen, and which processes hold the most.

The figures are read *inside* the container with one short exec: under cgroup
v2 with a private cgroup namespace (Docker's default) ``/sys/fs/cgroup`` there
is the room's own cgroup, so no host mount is needed. A host on cgroup v1 or
without PSI yields ``None`` for what it lacks; callers treat a missing figure
as "unknown", never as "full".
"""
from __future__ import annotations

from typing import Any, Optional

_MIB = 1024 * 1024
TOP_PROCESSES = 5

# One exec, one round trip. Each block is introduced by a ``#name`` marker so a
# missing file (cgroup v1, no PSI, no ps) leaves its block empty instead of
# shifting the others.
PRESSURE_SCRIPT = (
    "cd /sys/fs/cgroup 2>/dev/null; "
    "echo '#current'; cat memory.current 2>/dev/null; "
    "echo '#max'; cat memory.max 2>/dev/null; "
    "echo '#stat'; grep -E '^(anon|inactive_file|shmem) ' memory.stat 2>/dev/null; "
    "echo '#pressure'; cat memory.pressure 2>/dev/null; "
    "echo '#events'; cat memory.events 2>/dev/null; "
    # Memory of agent processes: everything a bash tool call started carries
    # LEGION_PROC_OWNER (E2), a room service or the user's terminal does not.
    "echo '#agents'; s=0; for d in /proc/[0-9]*; do "
    "if tr '\\000' '\\n' <\"$d/environ\" 2>/dev/null | grep -q '^LEGION_PROC_OWNER='; then "
    "r=$(awk '/^VmRSS:/{print $2}' \"$d/status\" 2>/dev/null); s=$((s+${r:-0})); fi; "
    "done; echo $s; "
    # The workspace image has no procps: read /proc directly. VmRSS is in KiB.
    "echo '#ps'; for d in /proc/[0-9]*; do "
    "r=$(awk '/^VmRSS:/{print $2}' \"$d/status\" 2>/dev/null); "
    "[ -n \"$r\" ] && echo \"${d#/proc/} $r $(tr '\\000' ' ' <\"$d/cmdline\" 2>/dev/null | cut -c1-160)\"; "
    f"done | sort -k2 -nr | head -n {TOP_PROCESSES}"
)


def _blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") and line[1:].isidentifier():
            current = line[1:]
            blocks[current] = []
        elif current is not None and line:
            blocks[current].append(line)
    return blocks


def _int(lines: list[str]) -> Optional[int]:
    if not lines:
        return None
    try:
        return int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None


def _kv(lines: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _psi(lines: list[str]) -> dict[str, Optional[float]]:
    """``some avg10=1.23 avg60=… total=…`` → ``{"some": 1.23, "full": …}``."""
    out: dict[str, Optional[float]] = {"some": None, "full": None}
    for line in lines:
        kind, _, rest = line.partition(" ")
        if kind not in out:
            continue
        for field in rest.split():
            name, _, value = field.partition("=")
            if name == "avg10":
                try:
                    out[kind] = float(value)
                except ValueError:
                    pass
    return out


def _top(lines: list[str]) -> list[dict[str, Any]]:
    top: list[dict[str, Any]] = []
    for line in lines[:TOP_PROCESSES]:
        parts = line.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        cmd = parts[2] if len(parts) > 2 else ""
        top.append({
            "pid": int(parts[0]),
            "rss_mb": round(int(parts[1]) / 1024, 1),  # ps reports KiB
            "cmd": cmd[:160],
        })
    return top


def pressure_from_stats(memory: dict[str, Any]) -> dict[str, Any]:
    """Pressure payload from a Docker stats sample when the exec could not run.

    Under load the exec runs inside the room's own CPU quota and can miss its
    timeout exactly when memory matters most (E4: a ``next build`` with eleven
    workers). Docker stats is read host-side, so the percentage survives; PSI,
    OOM kills and the process list are unknown. ``memory`` is
    :func:`app.docker_manager.memory_from_stats` output.
    """
    used = memory.get("memory_rss_mb")
    limit = memory.get("memory_limit_mb")
    pct = round(100.0 * used / limit, 1) if used is not None and limit else None
    return {
        "mem_used_mb": used,
        "mem_max_mb": limit,
        "pct": pct,
        "psi_some_avg10": None,
        "psi_full_avg10": None,
        "oom_kills": None,
        "agent_rss_mb": None,
        "tmp_mb": None,
        "top": [],
        "source": "docker_stats",
    }


def parse_pressure(text: str, *, docker_limit_mb: Optional[int] = None) -> dict[str, Any]:
    """Turn the output of :data:`PRESSURE_SCRIPT` into the pressure payload.

    ``used`` is what the room holds: ``memory.current`` minus the inactive page
    cache the kernel drops under pressure (the same figure ``docker stats`` and
    :func:`app.docker_manager.memory_from_stats` show). ``memory.max`` of
    ``max`` means no cgroup limit; ``docker_limit_mb`` then stands in.
    """
    b = _blocks(text)
    current = _int(b.get("current", []))
    max_raw = (b.get("max") or [""])[0]
    limit = int(max_raw) if max_raw.isdigit() else None
    stat = _kv(b.get("stat", []))
    used = None
    if current is not None:
        used = max(0, current - stat.get("inactive_file", 0))
    used_mb = round(used / _MIB, 1) if used is not None else None
    limit_mb = limit // _MIB if limit is not None else docker_limit_mb
    pct = None
    if used_mb is not None and limit_mb:
        pct = round(100.0 * used_mb / limit_mb, 1)
    psi = _psi(b.get("pressure", []))
    events = _kv(b.get("events", []))
    agents = _int(b.get("agents", []))
    return {
        "mem_used_mb": used_mb,
        "mem_max_mb": limit_mb,
        "pct": pct,
        "psi_some_avg10": psi["some"],
        "psi_full_avg10": psi["full"],
        "oom_kills": events.get("oom_kill") if "events" in b and b["events"] else None,
        # RSS of processes the agents started (KiB in the script); None when unknown.
        "agent_rss_mb": round(agents / 1024, 1) if agents is not None else None,
        # tmpfs (/tmp) pages are charged to the room as shmem.
        "tmp_mb": round(stat["shmem"] / _MIB, 1) if "shmem" in stat else None,
        "top": _top(b.get("ps", [])),
    }
