"""Export and compare persisted Stage-6 subagent benchmark metrics.

Examples (inside the backend container):

    python -m scripts.subagent_benchmark \
        --task-id e3c36115-1862-41a9-8ec5-ab295a01c743 \
        --session-id 89c13b55-a616-430a-8719-019175e8c2d8 \
        --session-id 11f7cdf5-e01d-4f13-bbe6-b60afe1fbec7 \
        --session-id 9414e161-0833-464a-8a2d-375971e0ab42 \
        --session-id 2f9795e8-2090-42a3-be8b-d831f597be31 \
        --label baseline --output /app/reports/subagent-stage6-baseline.json

    python -m scripts.subagent_benchmark \
        --task-id <candidate-task-id> \
        --baseline-report /app/reports/subagent-stage6-baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.subagent_benchmark import (
    BenchmarkMetrics,
    SessionTrace,
    ToolCallTrace,
    compare_benchmarks,
    summarize_benchmark,
)
from app.db.session import engine


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


async def _load_metrics(
    db: AsyncSession,
    *,
    task_id: str,
    label: str,
    session_ids: list[str] | None,
) -> BenchmarkMetrics:
    session_result = await db.execute(
        text(
            """
            SELECT s.id::text AS session_id, s.started_at, s.closed_at,
                   s.close_reason, s.tokens_in, s.tokens_out
            FROM session s
            JOIN cm_agent_proxy_session p ON p.session_id = s.id
            WHERE p.task_id = :task_id AND p.side = 'agent'
            ORDER BY s.started_at
            """
        ),
        {"task_id": task_id},
    )
    rows = list(session_result.mappings())
    selected = {str(value) for value in session_ids} if session_ids else None
    if selected is not None:
        found = {str(row["session_id"]) for row in rows}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(
                f"session ids do not belong to task {task_id}: {', '.join(missing)}"
            )
        rows = [row for row in rows if str(row["session_id"]) in selected]

    sessions = [
        SessionTrace(
            session_id=str(row["session_id"]),
            started_at=_parse_datetime(row["started_at"]),
            closed_at=_parse_datetime(row["closed_at"]) if row["closed_at"] else None,
            close_reason=str(row["close_reason"]) if row["close_reason"] else None,
            tokens_in=int(row["tokens_in"] or 0),
            tokens_out=int(row["tokens_out"] or 0),
        )
        for row in rows
    ]
    if not sessions:
        raise ValueError(f"no agent proxy sessions found for task {task_id}")

    event_result = await db.execute(
        text(
            """
            SELECT event_metadata->>'session_id' AS session_id,
                   event_metadata->>'tool' AS tool
            FROM message
            WHERE task_id = :task_id AND event_type = 'AGENT_TOOL_CALL'
            """
        ),
        {"task_id": task_id},
    )
    events = [
        ToolCallTrace(session_id=str(row["session_id"]), tool=str(row["tool"]))
        for row in event_result.mappings()
        if row["session_id"] and row["tool"]
    ]
    return summarize_benchmark(sessions, events, label=label, task_id=task_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure and compare persisted agent subtask benchmark traces."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--session-id",
        action="append",
        dest="session_ids",
        help="Restrict the run to selected agent session ids; repeat as needed.",
    )
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        metrics = await _load_metrics(
            db,
            task_id=args.task_id,
            label=args.label,
            session_ids=args.session_ids,
        )

    if args.baseline_report:
        raw = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline = raw.get("metrics", raw.get("candidate", raw))
        return compare_benchmarks(baseline, metrics)
    return {"version": 1, "metrics": metrics.as_dict()}


def main() -> None:
    args = _parser().parse_args()
    report = asyncio.run(_run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"report: {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
