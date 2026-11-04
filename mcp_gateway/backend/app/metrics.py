"""Gateway metrics for Prometheus: ``GET /metrics``.

Counted in process: the Gateway runs as one uvicorn worker, and a restart
resetting the counters is what Prometheus ``increase()`` expects. The text
format is written by hand, as in Market's billing metrics; a few counters do
not need a client library. The edge nginx forwards only ``/v1/execute``,
``/v1/tools`` and ``/health``, so ``/metrics`` is reachable inside the
compose network only (Prometheus scrapes ``mcp-gateway:8310``).

- ``legion_gateway_requests_total{path,status}``: every request by route
  template (never the raw URL: query strings carry keys) and response status.
  Upstream MCP failures come out as 502/504 on ``/v1/execute``.
- ``legion_gateway_request_duration_seconds_sum``/``_count{path}``: time
  spent per route, for an average latency.
- ``legion_gateway_start_time_seconds``: when this process started.
"""
from __future__ import annotations

import time
from collections import defaultdict

START_TIME = time.time()

_requests: defaultdict[tuple[str, int], int] = defaultdict(int)
_duration_sum: defaultdict[str, float] = defaultdict(float)
_duration_count: defaultdict[str, int] = defaultdict(int)


def observe(path: str, status: int, seconds: float) -> None:
    _requests[(path, status)] += 1
    _duration_sum[path] += seconds
    _duration_count[path] += 1


def reset() -> None:
    """For tests."""
    _requests.clear()
    _duration_sum.clear()
    _duration_count.clear()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render() -> str:
    lines = [
        "# HELP legion_gateway_requests_total Gateway requests by route and response status.",
        "# TYPE legion_gateway_requests_total counter",
    ]
    for (path, status), count in sorted(_requests.items()):
        lines.append(f'legion_gateway_requests_total{{path="{_escape(path)}",status="{status}"}} {count}')
    lines += [
        "# HELP legion_gateway_request_duration_seconds Time spent per route.",
        "# TYPE legion_gateway_request_duration_seconds summary",
    ]
    for path in sorted(_duration_count):
        label = f'path="{_escape(path)}"'
        lines.append(f"legion_gateway_request_duration_seconds_sum{{{label}}} {_duration_sum[path]!r}")
        lines.append(f"legion_gateway_request_duration_seconds_count{{{label}}} {_duration_count[path]}")
    lines += [
        "# HELP legion_gateway_start_time_seconds Unix time this Gateway process started.",
        "# TYPE legion_gateway_start_time_seconds gauge",
        f"legion_gateway_start_time_seconds {START_TIME!r}",
    ]
    return "\n".join(lines) + "\n"
