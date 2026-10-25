"""Host memory and OOM kills for sizing [capacity]."""
from __future__ import annotations

from app.host_stats import OomWatch, container_kind, read_meminfo


def test_meminfo_is_read_in_bytes(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:        8000000 kB\nMemFree:  100 kB\nMemAvailable:    6000000 kB\nHugePages_Total: 0\n")
    assert read_meminfo(str(path)) == {"memory_total_bytes": 8_192_000_000,
                                       "memory_available_bytes": 6_144_000_000}


def test_containers_are_told_apart_by_name():
    assert container_kind("/ws-0b1c") == "workspace"
    assert container_kind("mcp-personal-42") == "personal_mcp"
    assert container_kind("deployment-backend-1") == "other"


def test_oom_events_are_counted_by_kind():
    watch = OomWatch()
    watch.record({"Actor": {"Attributes": {"name": "ws-room"}}})
    watch.record({"Actor": {"Attributes": {"name": "ws-room"}}})
    watch.record({"Actor": {"Attributes": {"name": "mcp-personal-1"}}})
    watch.record({})
    snapshot = watch.snapshot()
    assert snapshot["oom_kills"] == {"workspace": 2, "personal_mcp": 1, "other": 1}
    assert snapshot["oom_last"]["kind"] == "other" and snapshot["oom_watch_connected"] is False
