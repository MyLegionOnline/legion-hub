"""Parallel-task slot rule (internal design notes, E1) — no DB."""

import pytest

from app.core.room_slots import Admission, decide, queue_reason_text


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("ROOM_EXTRA_SLOT_MEM_PCT", raising=False)
    monkeypatch.delenv("ROOM_EXTRA_SLOT_PSI_SOME", raising=False)
    monkeypatch.delenv("ROOM_EXTRA_SLOT_SETTLE_SEC", raising=False)


def test_base_slots_need_no_memory_reading():
    assert decide(0, 2, 4, None) == Admission(True, 0, 2, 4, "base")
    assert decide(1, 2, 4, None).ok


def test_extra_slot_only_with_spare_memory():
    assert decide(2, 2, 4, {"pct": 45.0, "psi_some_avg10": 0.0}).reason == "memory"
    busy = decide(2, 2, 4, {"pct": 60.0, "psi_some_avg10": 0.0})
    assert not busy.ok and busy.reason == "memory_busy" and busy.pct == 60.0


def test_memory_stall_blocks_an_extra_slot_even_below_the_threshold():
    a = decide(3, 2, 4, {"pct": 30.0, "psi_some_avg10": 25.0})
    assert not a.ok and a.reason == "memory_busy"


def test_unknown_memory_never_admits_an_extra_task():
    assert decide(2, 2, 4, None).reason == "memory_unknown"
    assert decide(2, 2, 4, {"pct": None}).reason == "memory_unknown"


def test_cap_holds_whatever_the_memory():
    a = decide(4, 2, 4, {"pct": 1.0, "psi_some_avg10": 0.0})
    assert not a.ok and a.reason == "cap"


def test_base_is_clamped_to_the_cap():
    assert decide(3, 9, 4, None).reason == "base"  # base 9 acts as 4
    assert decide(4, 9, 4, None).reason == "cap"
    assert decide(0, 0, 4, None).ok  # a base below 1 still lets one task run


def test_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("ROOM_EXTRA_SLOT_MEM_PCT", "80")
    assert decide(2, 2, 4, {"pct": 70.0}).ok


def test_queue_reason_texts_are_english_and_specific():
    assert "2 of 4" in queue_reason_text(Admission(False, 2, 2, 4, "cap"))
    assert "71% full" in queue_reason_text(Admission(False, 2, 2, 4, "memory_busy", 71.2))
    assert "cannot be checked" in queue_reason_text(Admission(False, 2, 2, 4, "memory_unknown"))


def test_extra_slot_waits_for_the_previous_start_to_settle():
    """E4: a cold room admitted four tasks at 30 % before any build had begun."""
    calm = {"pct": 30.0, "psi_some_avg10": 0.0}
    a = decide(2, 2, 4, calm, since_last_start=20.0)
    assert not a.ok and a.reason == "settling"
    assert decide(2, 2, 4, calm, since_last_start=200.0).ok
    assert decide(2, 2, 4, calm, since_last_start=None).ok
    # the base never waits
    assert decide(1, 2, 4, None, since_last_start=1.0).reason == "base"
    assert "moments ago" in queue_reason_text(Admission(False, 2, 2, 4, "settling"))
