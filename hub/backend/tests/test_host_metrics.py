"""Host memory and OOM kills to Prometheus."""
from __future__ import annotations

from collections import Counter

from app.core import host_metrics, metrics


class _Counter:
    def __init__(self) -> None:
        self.values: Counter = Counter()
        self._kind = None

    def labels(self, kind):
        self._kind = kind
        self.values[kind] += 0
        return self

    def inc(self, amount=1):
        self.values[self._kind] += amount


def test_oom_counts_become_a_counter_that_survives_a_workspaced_restart(monkeypatch):
    counter = _Counter()
    monkeypatch.setattr(metrics, "container_oom_kills_total", counter)
    monkeypatch.setattr(host_metrics, "_published", {"since": None, "counts": {}})
    host_metrics._publish_oom("t1", {"workspace": 2, "personal_mcp": 0, "other": 0})
    host_metrics._publish_oom("t1", {"workspace": 3, "personal_mcp": 0, "other": 0})
    assert counter.values == {"workspace": 3, "personal_mcp": 0, "other": 0}  # deltas, not totals twice
    # workspaced restarted: its counts start from zero again, nothing is lost or undone.
    host_metrics._publish_oom("t2", {"workspace": 1, "personal_mcp": 0, "other": 0})
    host_metrics._publish_oom("t2", {"workspace": 1, "personal_mcp": 0, "other": 0})
    assert counter.values["workspace"] == 4
