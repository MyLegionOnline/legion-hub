"""Stage 0.3 of internal design notes — the agent-step time profile.

These tests pin the arithmetic the stage-0.3 report is read off: a step's
wall-clock split into llm / workspaced / db / other, the clamp when concurrent
waits overlap, and the bounded label vocabulary.  They touch neither the
database nor any provider — the instrumentation is pure accounting.

The metric objects are replaced with recording fakes rather than read out of
the Prometheus registry: conftest stubs prometheus_client with a MagicMock for
local runs, so the real histograms cannot be asserted on portably.
"""

import asyncio
import collections

import pytest

from app.core import step_profile as sp


class _FakeMetric:
    """Stand-in for a labelled Histogram/Counter/Gauge."""

    def __init__(self) -> None:
        self.observations: dict[tuple, list[float]] = collections.defaultdict(list)
        self.values: dict[tuple, float] = collections.defaultdict(float)

    def labels(self, **labels):
        key = tuple(sorted(labels.items()))
        metric = self

        class _Child:
            def observe(self, value):
                metric.observations[key].append(value)

            def inc(self, value=1):
                metric.values[key] += value

            def set(self, value):
                metric.values[key] = value

        return _Child()

    def sum(self, **labels) -> float:
        return sum(self.observations[tuple(sorted(labels.items()))])

    def count(self, **labels) -> int:
        return len(self.observations[tuple(sorted(labels.items()))])

    def value(self, **labels) -> float:
        return self.values[tuple(sorted(labels.items()))]


@pytest.fixture
def m(monkeypatch):
    """Swap every metric step_profile writes to for a recording fake."""
    fakes = {
        name: _FakeMetric()
        for name in (
            "llm_wait_seconds",
            "workspaced_wait_seconds",
            "db_wait_seconds",
            "step_wall_seconds",
            "step_component_seconds",
            "steps_total",
            "step_overrun_total",
            "steps_in_flight",
        )
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(sp, name, fake)
    sp._current.set(None)
    try:
        yield type("Metrics", (), fakes)
    finally:
        sp._current.set(None)


def test_step_splits_wall_clock_into_four_parts(m):
    side = sp.SIDE_AGENT

    async def run():
        sp.step_tick(side)
        sp.record_llm_wait("anthropic", side, 0.05)
        sp.record_workspaced_wait("/workspaces/{id}/files/read", 0.02)
        sp.record_db_wait(0.01)
        sp.record_db_wait(0.01)
        # The recorded waits are synthetic, so let real time pass beyond their
        # sum: otherwise `other` would be clamped and the identity below would
        # be testing the clamp instead of the split.
        await asyncio.sleep(0.12)
        sp.step_end()

    asyncio.run(run())

    assert m.step_component_seconds.sum(side=side, component="llm") == pytest.approx(0.05)
    assert m.step_component_seconds.sum(side=side, component="workspaced") == pytest.approx(0.02)
    assert m.step_component_seconds.sum(side=side, component="db") == pytest.approx(0.02)
    assert m.db_wait_seconds.sum(side=side) == pytest.approx(0.02)
    assert m.steps_total.value(side=side) == 1

    # other = wall - measured, and the four parts add back up to the wall-clock.
    wall = m.step_wall_seconds.sum(side=side)
    parts = sum(
        m.step_component_seconds.sum(side=side, component=c)
        for c in ("llm", "workspaced", "db", "other")
    )
    assert parts == pytest.approx(wall, abs=1e-6)


def test_tick_closes_the_previous_step_and_opens_the_next(m):
    side = sp.SIDE_CHAT_MASTER
    sp.step_tick(side)
    sp.record_llm_wait("openai", side, 0.01)
    sp.step_tick(side)  # closes the first, opens the second
    sp.record_llm_wait("openai", side, 0.02)

    # Two ticks, one close so far: the open step is recorded only when it ends.
    assert m.steps_total.value(side=side) == 1
    assert m.step_component_seconds.sum(side=side, component="llm") == pytest.approx(0.01)

    sp.step_end()
    assert m.steps_total.value(side=side) == 2
    assert m.step_component_seconds.sum(side=side, component="llm") == pytest.approx(0.03)


def test_closing_twice_does_not_record_twice(m):
    side = sp.SIDE_PLAN
    sp.step_tick(side)
    sp.step_end()
    sp.step_end()
    assert m.steps_total.value(side=side) == 1


def test_overlapping_waits_clamp_other_to_zero_and_are_counted(m):
    """Two tool calls awaited concurrently can sum past the step's wall-clock.

    The report must not read that as negative Hub time, and it must know how
    often the clamp fired before trusting the hub_other number.
    """
    side = sp.SIDE_AGENT
    sp.step_tick(side)
    sp.record_llm_wait("anthropic", side, 30.0)  # far beyond the real wall-clock
    sp.step_end()

    assert m.step_component_seconds.sum(side=side, component="other") == 0.0
    assert m.step_overrun_total.value(side=side) == 1


def test_wait_outside_a_step_is_observed_but_charged_to_nobody(m):
    """Router and background LLM calls happen outside any tool loop."""
    sp.record_llm_wait("openrouter", sp.SIDE_ROUTER, 0.5)
    sp.record_workspaced_wait("/workspaces/{id}/status", 0.1)
    sp.record_db_wait(0.3)

    assert m.llm_wait_seconds.sum(provider="openrouter", side=sp.SIDE_ROUTER) == pytest.approx(0.5)
    assert m.workspaced_wait_seconds.sum(op="/workspaces/{id}/status") == pytest.approx(0.1)
    assert m.steps_total.value(side=sp.SIDE_ROUTER) == 0


def test_unknown_side_and_provider_fold_into_bounded_labels(m):
    sp.step_tick("room-7f3a-user-42")  # would be unbounded cardinality
    sp.record_llm_wait("Weird Provider/v2 ✨", "room-7f3a-user-42", 0.01)
    sp.step_end()

    assert m.steps_total.value(side="other") == 1
    (key,) = list(m.llm_wait_seconds.observations)
    provider = dict(key)["provider"]
    assert provider == "weird_provider_v2__"
    assert dict(key)["side"] == "other"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/workspaces/2f1c8a0e-1111-4222-8333-444455556666/files/read",
         "/workspaces/{id}/files/read"),
        ("/workspaces", "/workspaces"),
        ("/workspaces/abc/services/9f2b1c4d5e6f/relay", "/workspaces/abc/services/{id}/relay"),
        ("/workspaces/12345/status", "/workspaces/{id}/status"),
        ("", "/"),
    ],
)
def test_workspaced_op_collapses_identifier_segments(path, expected):
    assert sp.workspaced_op(path) == expected


def test_wall_clock_covers_real_elapsed_time(m):
    side = sp.SIDE_AGENT

    async def run():
        sp.step_tick(side)
        await asyncio.sleep(0.05)
        sp.step_end()

    asyncio.run(run())
    assert m.step_wall_seconds.sum(side=side) >= 0.04


def test_two_sessions_accumulate_independently(m):
    """Each session runs in its own task; a step must not absorb another's wait."""

    async def session(provider: str, llm_seconds: float):
        sp.step_tick(sp.SIDE_AGENT)
        await asyncio.sleep(0)
        sp.record_llm_wait(provider, sp.SIDE_AGENT, llm_seconds)
        await asyncio.sleep(0)
        sp.step_end()

    async def run():
        await asyncio.gather(session("anthropic", 1.0), session("openai", 2.0))

    asyncio.run(run())

    assert m.steps_total.value(side=sp.SIDE_AGENT) == 2
    per_step = sorted(m.step_component_seconds.observations[
        tuple(sorted({"side": sp.SIDE_AGENT, "component": "llm"}.items()))
    ])
    assert per_step == pytest.approx([1.0, 2.0])


def test_in_flight_gauge_counts_running_steps_not_session_rows(m):
    """Stage 0.2 found sessions_open_by_kind unusable as a capacity signal."""

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def worker():
            sp.step_tick(sp.SIDE_AGENT)
            started.set()
            await release.wait()
            sp.step_end()

        task = asyncio.create_task(worker())
        await started.wait()
        sp.refresh_in_flight_gauge()
        during = m.steps_in_flight.value(side=sp.SIDE_AGENT)
        release.set()
        await task
        sp.refresh_in_flight_gauge()
        return during, m.steps_in_flight.value(side=sp.SIDE_AGENT)

    during, after = asyncio.run(run())
    assert during == 1
    assert after == 0


def test_abandoned_step_does_not_pin_the_gauge_up(m):
    """A task that dies mid-step must not inflate the gauge forever."""

    async def run():
        async def worker():
            sp.step_tick(sp.SIDE_AGENT)
            raise RuntimeError("session died mid-step")

        task = asyncio.create_task(worker())
        with pytest.raises(RuntimeError):
            await task
        sp.refresh_in_flight_gauge()
        return m.steps_in_flight.value(side=sp.SIDE_AGENT)

    assert asyncio.run(run()) == 0
