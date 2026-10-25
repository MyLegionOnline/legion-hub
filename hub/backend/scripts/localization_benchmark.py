"""Run the token-free Stage-5 repository localization benchmark.

Example (inside the backend container):

    python -m scripts.localization_benchmark \
        --root /app/app --output /app/reports/localization-report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from app.core.localization_benchmark import LocalizationCase, benchmark_localization
from app.core.repository_index import extract_outline

_IGNORED = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "coverage",
    }
)
_SUFFIXES = frozenset(
    {
        ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
        ".ini", ".java", ".js", ".json", ".jsx", ".kt", ".md", ".php", ".py",
        ".pyi", ".rb", ".rs", ".rst", ".sh", ".sql", ".toml", ".ts", ".tsx",
        ".txt", ".xml", ".yaml", ".yml",
    }
)
_NAMES = frozenset({"dockerfile", "makefile", "readme", "license"})

_DEFAULT_CASES = (
    LocalizationCase(
        "context receipt result sha evicted tokens active tool result cap",
        ("core/context_assembler.py",),
    ),
    LocalizationCase(
        "artifact storage key stale pinned read ledger merge intervals observation payload",
        ("core/context_observations.py",),
    ),
    LocalizationCase(
        "offline anonymized trajectories compare current and proposed prompt tokens",
        ("core/context_replay.py",),
    ),
    LocalizationCase(
        "conservative utf8 token estimate with twenty percent safety margin",
        ("core/token_counter.py", "core/context_baseline.py"),
    ),
    LocalizationCase(
        "stable cache key from room role system version and tool schema version",
        ("core/provider_cache.py",),
    ),
    LocalizationCase(
        "compact ranked paths symbols imports and dependency edges",
        ("core/repository_index.py",),
    ),
)


def _files(root: Path, max_bytes: int) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED or part.startswith(".") for part in relative.parts):
            continue
        if path.stat().st_size > max_bytes:
            continue
        if path.suffix.casefold() not in _SUFFIXES and path.name.casefold() not in _NAMES:
            continue
        yield path


def _load_cases(path: Path | None) -> list[LocalizationCase]:
    if path is None:
        return list(_DEFAULT_CASES)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases file must contain a JSON array")
    return [
        LocalizationCase(
            query=str(item["query"]),
            target_paths=tuple(str(value) for value in item["target_paths"]),
        )
        for item in raw
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exact/symbol localization with lexical BM25 without an LLM."
    )
    parser.add_argument("--root", type=Path, default=Path("app"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=1_500_000)
    parser.add_argument("--required-recall-at-10", type=float, default=0.90)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    outlines = [
        extract_outline(
            "/" + path.relative_to(root).as_posix(),
            path.read_text(encoding="utf-8"),
        )
        for path in _files(root, max(1, args.max_file_bytes))
    ]
    report = benchmark_localization(
        outlines,
        _load_cases(args.cases),
        required_recall_at_10=args.required_recall_at_10,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(
            f"report: {args.output} "
            f"({report['corpus']['cases']} cases, {report['corpus']['indexed_files']} files)"
        )
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
