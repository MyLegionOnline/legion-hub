"""Offline Stage-0 context baseline and replay command.

Examples (inside the backend container):

    python -m scripts.context_replay export --output /tmp/context-corpus.jsonl
    python -m scripts.context_replay replay --input /tmp/context-corpus.jsonl
    python -m scripts.context_replay all --corpus /tmp/context-corpus.jsonl \
        --report /tmp/context-report.json

Export reads ``session.context_snapshot`` and writes an anonymized corpus.  It
never invokes an LLM and never writes message, file, command, pattern or tool
result text to disk.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.context_replay import build_anonymized_trajectory, replay_corpus

_TYPES = (
    "code_edit",
    "research_docs",
    "generated_content",
    "long_test_log",
    "resume",
)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


async def export_corpus(output: Path, *, per_type: int, scan_limit: int) -> list[dict[str, Any]]:
    from app.db.session import engine

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        rows = (
            await db.execute(
                text(
                    "SELECT s.id, s.context_snapshot, s.close_reason, s.kind, "
                    "       CAST(t.state AS TEXT) AS task_state, COALESCE(t.title, '') AS task_title "
                    "FROM session s "
                    "LEFT JOIN cm_agent_proxy_session p ON p.session_id = s.id "
                    "LEFT JOIN task t ON t.id = p.task_id "
                    "WHERE s.context_snapshot IS NOT NULL "
                    "  AND s.context_snapshot::jsonb <> '{}'::jsonb "
                    "ORDER BY s.started_at DESC "
                    "LIMIT :scan_limit"
                ),
                {"scan_limit": max(1, scan_limit)},
            )
        ).all()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory = build_anonymized_trajectory(
            session_id=row.id,
            snapshot=_json_object(row.context_snapshot),
            close_reason=row.close_reason,
            task_state=row.task_state,
            task_title=row.task_title or "",
        )
        if trajectory["requests"]:
            grouped[trajectory["trajectory_type"]].append(trajectory)

    selected: list[dict[str, Any]] = []
    for trajectory_type in _TYPES:
        selected.extend(grouped.get(trajectory_type, [])[: max(1, per_type)])
    for trajectory_type in sorted(set(grouped) - set(_TYPES)):
        selected.extend(grouped[trajectory_type][: max(1, per_type)])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trajectory in selected:
            handle.write(json.dumps(trajectory, ensure_ascii=False, separators=(",", ":")) + "\n")
    return selected


def load_corpus(path: Path) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            trajectories.append(value)
    return trajectories


def write_report(
    trajectories: list[dict[str, Any]],
    *,
    output: Path | None,
    keep_raw_results: int,
    receipt_tokens: int,
) -> dict[str, Any]:
    report = replay_corpus(
        trajectories,
        keep_raw_results=keep_raw_results,
        receipt_tokens=receipt_tokens,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"report: {output} ({len(trajectories)} trajectories)")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an anonymized context corpus and replay current vs proposed packing."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export anonymized trajectories from context_snapshot")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--per-type", type=int, default=20)
    export.add_argument("--scan-limit", type=int, default=1000)

    replay = sub.add_parser("replay", help="Compare current and proposed context without an LLM")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--keep-raw-results", type=int, default=3)
    replay.add_argument("--receipt-tokens", type=int, default=32)

    all_cmd = sub.add_parser("all", help="Export a corpus and immediately replay it")
    all_cmd.add_argument("--corpus", type=Path, required=True)
    all_cmd.add_argument("--report", type=Path, required=True)
    all_cmd.add_argument("--per-type", type=int, default=20)
    all_cmd.add_argument("--scan-limit", type=int, default=1000)
    all_cmd.add_argument("--keep-raw-results", type=int, default=3)
    all_cmd.add_argument("--receipt-tokens", type=int, default=32)
    return parser


async def _run(args: argparse.Namespace) -> None:
    if args.command == "export":
        trajectories = await export_corpus(
            args.output,
            per_type=args.per_type,
            scan_limit=args.scan_limit,
        )
        counts: dict[str, int] = defaultdict(int)
        for trajectory in trajectories:
            counts[trajectory["trajectory_type"]] += 1
        print(f"corpus: {args.output} ({len(trajectories)} trajectories; {dict(counts)})")
        return

    if args.command == "replay":
        write_report(
            load_corpus(args.input),
            output=args.output,
            keep_raw_results=args.keep_raw_results,
            receipt_tokens=args.receipt_tokens,
        )
        return

    trajectories = await export_corpus(
        args.corpus,
        per_type=args.per_type,
        scan_limit=args.scan_limit,
    )
    print(f"corpus: {args.corpus} ({len(trajectories)} trajectories)")
    write_report(
        trajectories,
        output=args.report,
        keep_raw_results=args.keep_raw_results,
        receipt_tokens=args.receipt_tokens,
    )


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
