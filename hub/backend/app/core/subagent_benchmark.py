"""Metrics and comparison helpers for the Stage-6 subagent benchmark.

The benchmark is deliberately based on persisted task events rather than
process-local counters.  This makes it possible to compare an interrupted
task, a resumed task, and a task using ``run_subtask`` with the same rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping


PRODUCTIVE_TOOLS = frozenset({"write_file", "edit_file", "git_apply"})
INTERRUPTION_REASONS = frozenset({"tool_limit", "orphaned_restart"})


@dataclass(frozen=True)
class SessionTrace:
    """Persisted metrics for one agent-side proxy session."""

    session_id: str
    started_at: datetime
    closed_at: datetime | None
    close_reason: str | None
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class ToolCallTrace:
    """One persisted ``AGENT_TOOL_CALL`` event."""

    session_id: str
    tool: str


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Aggregated metrics for one task execution."""

    label: str
    task_id: str | None
    session_ids: tuple[str, ...]
    session_count: int
    tool_calls: int
    productive_tool_calls: int
    productive_ratio: float
    tokens_in: int
    tokens_out: int
    total_tokens: int
    wall_clock_seconds: float | None
    interruptions: int
    close_reasons: dict[str, int]
    run_subtask_calls: int
    started_at: str | None
    closed_at: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def summarize_benchmark(
    sessions: Iterable[SessionTrace],
    tool_calls: Iterable[ToolCallTrace],
    *,
    label: str,
    task_id: str | None = None,
) -> BenchmarkMetrics:
    """Aggregate the persisted trace using the Stage-6 metric definitions.

    ``sessions`` must contain agent-side proxy sessions only.  Child
    ``run_subtask`` usage is already rolled into its parent session by the
    runtime, while the child tool events use the parent's session id as their
    durable correlation point.  Counting the persisted events and the parent
    token totals therefore avoids double-counting both kinds of usage.
    """

    session_rows = list(sessions)
    session_ids = {row.session_id for row in session_rows}
    event_rows = [row for row in tool_calls if row.session_id in session_ids]

    productive = sum(row.tool in PRODUCTIVE_TOOLS for row in event_rows)
    tool_count = len(event_rows)
    ratio = round(productive / tool_count, 6) if tool_count else 0.0

    started = min((row.started_at for row in session_rows), default=None)
    closed_values = [row.closed_at for row in session_rows if row.closed_at is not None]
    closed = max(closed_values, default=None)
    wall_clock = (closed - started).total_seconds() if started and closed else None

    reasons: dict[str, int] = {}
    for row in session_rows:
        reason = row.close_reason or "open"
        reasons[reason] = reasons.get(reason, 0) + 1

    return BenchmarkMetrics(
        label=label,
        task_id=task_id,
        session_ids=tuple(sorted(session_ids)),
        session_count=len(session_rows),
        tool_calls=tool_count,
        productive_tool_calls=productive,
        productive_ratio=ratio,
        tokens_in=sum(max(0, row.tokens_in) for row in session_rows),
        tokens_out=sum(max(0, row.tokens_out) for row in session_rows),
        total_tokens=sum(max(0, row.tokens_in) + max(0, row.tokens_out) for row in session_rows),
        wall_clock_seconds=round(wall_clock, 3) if wall_clock is not None else None,
        interruptions=sum(row.close_reason in INTERRUPTION_REASONS for row in session_rows),
        close_reasons=dict(sorted(reasons.items())),
        run_subtask_calls=sum(row.tool == "run_subtask" for row in event_rows),
        started_at=_iso(started),
        closed_at=_iso(closed),
    )


def _change(after: float | int | None, before: float | int | None) -> dict[str, Any]:
    """Return absolute and relative change, handling zero baselines."""

    if after is None or before is None:
        return {"before": before, "after": after, "delta": None, "relative": None}
    delta = after - before
    relative = None if before == 0 else round(delta / before, 6)
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "relative": relative,
    }


def compare_benchmarks(
    baseline: BenchmarkMetrics | Mapping[str, Any],
    candidate: BenchmarkMetrics | Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two runs and apply the conservative Stage-6 default gate.

    The feature is eligible for default enablement only when the candidate
    actually used ``run_subtask``, every hard efficiency metric is no worse,
    tool calls and tokens both decrease, and productive-call ratio does not
    decrease.  A missing wall-clock value, or a candidate that never used the
    mechanism under test, makes the result inconclusive instead of silently
    passing the gate.
    """

    before = baseline.as_dict() if isinstance(baseline, BenchmarkMetrics) else dict(baseline)
    after = candidate.as_dict() if isinstance(candidate, BenchmarkMetrics) else dict(candidate)

    metrics = {
        "tool_calls": _change(after.get("tool_calls"), before.get("tool_calls")),
        "total_tokens": _change(after.get("total_tokens"), before.get("total_tokens")),
        "wall_clock_seconds": _change(
            after.get("wall_clock_seconds"), before.get("wall_clock_seconds")
        ),
        "productive_ratio": _change(after.get("productive_ratio"), before.get("productive_ratio")),
        "interruptions": _change(after.get("interruptions"), before.get("interruptions")),
    }

    complete = all(
        before.get(name) is not None and after.get(name) is not None
        for name in ("tool_calls", "total_tokens", "wall_clock_seconds")
    )
    hard_metrics_non_worse = complete and all(
        metrics[name]["after"] <= metrics[name]["before"]
        for name in ("tool_calls", "total_tokens", "wall_clock_seconds", "interruptions")
    )
    productivity_non_worse = (
        after.get("productive_ratio") is not None
        and before.get("productive_ratio") is not None
        and after["productive_ratio"] >= before["productive_ratio"]
    )
    strictly_better = (
        complete
        and after.get("tool_calls", 0) < before.get("tool_calls", 0)
        and after.get("total_tokens", 0) < before.get("total_tokens", 0)
    )

    if not complete:
        status = "inconclusive"
        reason = "Both runs need completed wall-clock and usage metrics."
    elif int(after.get("run_subtask_calls", 0) or 0) <= 0:
        status = "inconclusive"
        reason = "The candidate did not use run_subtask; its improvement is not attributable to the feature."
    elif hard_metrics_non_worse and productivity_non_worse and strictly_better:
        status = "enable_by_default"
        reason = "Tool calls and tokens decreased without regression in any hard metric."
    else:
        status = "hold"
        reason = "The candidate did not pass the no-regression and strict-efficiency gate."

    return {
        "version": 1,
        "baseline": before,
        "candidate": after,
        "changes": metrics,
        "decision": {"status": status, "reason": reason},
    }
