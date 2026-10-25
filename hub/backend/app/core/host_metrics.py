"""What the host holds and loses, for sizing ``[capacity]`` on the VPS.

``core/host_capacity.py`` can cap the workspaces and the memory all users run
together, but the caps stay unset until they are measured on the VPS. This
module gives that measurement, once per reconciler tick:

* the host's memory, total and available (workspaced ``/host/stats``);
* the memory limits of running room workspaces and personal MCP servers added
  up, the number :mod:`host_capacity` compares with ``memory_mb``;
* Docker out-of-memory kills by container kind. workspaced counts them since
  its start; this process turns the counts into a Prometheus counter, so a
  workspaced restart (counts back to zero) loses nothing already published.

The running workspaces are ``workspace_running_total`` (the reconciler's own
probe). The last snapshot is kept for the operator's dashboard (Admin Center
→ Dashboard). The Hub runs one backend process, so module state is enough.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_published: dict[str, Any] = {"since": None, "counts": {}}
_last: Optional[dict[str, Any]] = None


def _publish_oom(since: Optional[str], counts: dict[str, int]) -> None:
    from app.core.metrics import container_oom_kills_total

    fresh = since != _published["since"]
    for kind, count in counts.items():
        seen = 0 if fresh else _published["counts"].get(kind, 0)
        if count > seen:
            container_oom_kills_total.labels(kind=kind).inc(count - seen)
        else:
            container_oom_kills_total.labels(kind=kind)  # the series exists at 0
    _published["since"], _published["counts"] = since, dict(counts)


async def publish_host_metrics(db: AsyncSession) -> Optional[dict[str, Any]]:
    """One probe: host memory and OOM kills from workspaced, capacity from the DB."""
    global _last
    from app.core import host_capacity
    from app.core import metrics
    from app.core import workspaced_client as wsd
    from app.db.release import release_db

    capacity = await host_capacity.usage_summary(db)
    metrics.capacity_memory_allocated_mb.labels(kind="workspace").set(capacity["workspace_memory_mb"])
    metrics.capacity_memory_allocated_mb.labels(kind="personal_mcp").set(capacity["personal_mcp_memory_mb"])
    for key in host_capacity.KEYS:
        if capacity[key] is None:
            try:
                metrics.capacity_limit.remove(key)
            except KeyError:
                pass
        else:
            metrics.capacity_limit.labels(key=key).set(capacity[key])
    await release_db(db, reason="host_metrics/probe")
    try:
        host = await wsd.get_host_stats()
    except Exception as exc:
        # legion_host_stats_checked_timestamp_seconds goes stale and HostStatsStale fires.
        logger.warning("host_metrics: workspaced host probe failed: %s", exc)
        _last = {**capacity, "host": None}
        return _last
    metrics.host_memory_total_bytes.set(host["memory_total_bytes"])
    metrics.host_memory_available_bytes.set(host["memory_available_bytes"])
    metrics.host_stats_checked_timestamp_seconds.set(time.time())
    _publish_oom(host.get("oom_watch_since"), host.get("oom_kills") or {})
    _last = {**capacity, "host": host}
    return _last


def last_snapshot() -> Optional[dict[str, Any]]:
    return _last
