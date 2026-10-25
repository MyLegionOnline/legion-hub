from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.subagent_benchmark import (
    SessionTrace,
    ToolCallTrace,
    compare_benchmarks,
    summarize_benchmark,
)


def _session(
    session_id: str,
    *,
    started: datetime,
    seconds: int = 10,
    reason: str | None = "task_done",
    tokens_in: int = 100,
    tokens_out: int = 20,
) -> SessionTrace:
    return SessionTrace(
        session_id=session_id,
        started_at=started,
        closed_at=started + timedelta(seconds=seconds),
        close_reason=reason,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def test_summarize_counts_selected_tool_events_and_rollup_once():
    started = datetime(2026, 8, 9, tzinfo=timezone.utc)
    sessions = [
        _session("one", started=started, tokens_in=100, tokens_out=20),
        _session("two", started=started + timedelta(seconds=15), reason="tool_limit"),
    ]
    events = [
        ToolCallTrace("one", "read_file"),
        ToolCallTrace("one", "write_file"),
        ToolCallTrace("two", "edit_file"),
        ToolCallTrace("other", "write_file"),
    ]

    report = summarize_benchmark(sessions, events, label="baseline", task_id="task")

    assert report.tool_calls == 3
    assert report.productive_tool_calls == 2
    assert report.productive_ratio == 0.666667
    assert report.total_tokens == 240
    assert report.wall_clock_seconds == 25.0
    assert report.interruptions == 1
    assert report.close_reasons == {"task_done": 1, "tool_limit": 1}


def test_compare_requires_all_hard_metrics_and_is_conservative():
    baseline = summarize_benchmark(
        [_session("one", started=datetime(2026, 8, 9, tzinfo=timezone.utc))],
        [ToolCallTrace("one", "read_file"), ToolCallTrace("one", "write_file")],
        label="baseline",
    )
    candidate = summarize_benchmark(
        [_session("two", started=datetime(2026, 8, 9, tzinfo=timezone.utc), seconds=8, tokens_in=80)],
        [ToolCallTrace("two", "write_file")],
        label="candidate",
    )

    candidate_data = candidate.as_dict()
    candidate_data["run_subtask_calls"] = 1
    comparison = compare_benchmarks(baseline, candidate_data)

    assert comparison["decision"]["status"] == "enable_by_default"
    assert comparison["changes"]["tool_calls"]["delta"] == -1
    assert comparison["changes"]["total_tokens"]["delta"] == -20


def test_open_run_is_inconclusive_and_does_not_enable_by_default():
    started = datetime(2026, 8, 9, tzinfo=timezone.utc)
    baseline = summarize_benchmark(
        [_session("one", started=started)],
        [ToolCallTrace("one", "write_file")],
        label="baseline",
    )
    candidate = summarize_benchmark(
        [
            SessionTrace(
                session_id="two",
                started_at=started,
                closed_at=None,
                close_reason=None,
                tokens_in=1,
                tokens_out=1,
            )
        ],
        [ToolCallTrace("two", "write_file")],
        label="candidate",
    )

    assert compare_benchmarks(baseline, candidate)["decision"]["status"] == "inconclusive"


def test_fewer_calls_without_productive_work_stays_on_hold():
    started = datetime(2026, 8, 9, tzinfo=timezone.utc)
    baseline = summarize_benchmark(
        [_session("one", started=started)],
        [ToolCallTrace("one", "read_file"), ToolCallTrace("one", "write_file")],
        label="baseline",
    )
    candidate = summarize_benchmark(
        [_session("two", started=started, seconds=8, tokens_in=80)],
        [ToolCallTrace("two", "read_file")],
        label="candidate",
    )

    candidate_data = candidate.as_dict()
    candidate_data["run_subtask_calls"] = 1
    assert compare_benchmarks(baseline, candidate_data)["decision"]["status"] == "hold"
