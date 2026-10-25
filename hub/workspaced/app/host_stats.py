"""Host memory and out-of-memory kills, for sizing ``[capacity]`` on the VPS.

The server-wide cap (backend ``core/host_capacity.py``) stays unset until a
measurement on the VPS: how much memory the host has, how much the running
rooms and personal MCPs reserve, and whether the kernel kills anything. This
module supplies the host side of that, read by the backend reconciler through
``GET /host/stats`` and published to Prometheus there:

* :func:`read_meminfo` — total and available memory of the host. A container
  sees the host's ``/proc/meminfo`` (the cgroup limit is elsewhere), so this
  is the machine, not workspaced's own allowance;
* :class:`OomWatch` — counts Docker ``oom`` events since workspaced started,
  by the kind of container: room workspaces (``ws-…``), personal MCP servers
  (``mcp-personal-…``) and the rest (the Hub's own services). Docker emits the
  event when the kernel kills a process of the container's cgroup, also when
  the container itself survives it, which an exited container's
  ``State.OOMKilled`` would miss.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiodocker
from aiodocker.events import DockerEvents

logger = logging.getLogger(__name__)

MEMINFO_PATH = "/proc/meminfo"
KINDS = ("workspace", "personal_mcp", "other")
RECONNECT_SEC = 5.0


def read_meminfo(path: str = MEMINFO_PATH) -> dict[str, int]:
    values: dict[str, int] = {}
    with open(path, encoding="ascii") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0]) * (1024 if parts[1:2] == ["kB"] else 1)
    return {"memory_total_bytes": values["MemTotal"],
            "memory_available_bytes": values.get("MemAvailable", values.get("MemFree", 0))}


def container_kind(name: str) -> str:
    name = name.lstrip("/")
    if name.startswith("ws-"):
        return "workspace"
    if name.startswith("mcp-personal-"):
        return "personal_mcp"
    return "other"


class OomWatch:
    def __init__(self) -> None:
        self.counts = {kind: 0 for kind in KINDS}
        self.last: Optional[dict] = None
        self.since = datetime.now(timezone.utc)
        self.connected = False

    def record(self, event: dict) -> None:
        name = ((event.get("Actor") or {}).get("Attributes") or {}).get("name", "")
        kind = container_kind(name)
        self.counts[kind] += 1
        self.last = {"kind": kind, "at": datetime.now(timezone.utc).isoformat()}
        # The name says which room or server; the room id is not personal data.
        logger.warning("out-of-memory kill in %s container %s", kind, name)

    def snapshot(self) -> dict:
        return {"oom_kills": dict(self.counts), "oom_last": self.last,
                "oom_watch_since": self.since.isoformat(), "oom_watch_connected": self.connected}

    async def run(self) -> None:
        """Follow the Docker event stream for good, reconnecting after a drop."""
        filters = json.dumps({"type": ["container"], "event": ["oom"]})
        while True:
            docker = aiodocker.Docker()
            events = DockerEvents(docker)
            try:
                subscriber = events.subscribe(filters=filters)
                self.connected = True
                while True:
                    event = await subscriber.get()
                    if event is None:
                        break  # the stream ended; reconnect
                    self.record(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Docker oom event stream failed; reconnecting", exc_info=True)
            finally:
                self.connected = False
                await events.stop()
                await docker.close()
            await asyncio.sleep(RECONNECT_SEC)


oom_watch = OomWatch()
