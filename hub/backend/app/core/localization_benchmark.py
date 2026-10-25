"""Offline localization benchmark for the conditional Stage-5 decisions.

The benchmark never calls an LLM.  It compares the Stage-3 exact/path/symbol
ranking with the Stage-5 lexical BM25 fallback on the same indexed files.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from app.core.repository_index import FileOutline, normalize_path, rank_outlines


@dataclass(frozen=True)
class LocalizationCase:
    query: str
    target_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("localization query cannot be empty")
        if not self.target_paths:
            raise ValueError("localization case needs at least one target path")


def _rank(
    outlines: list[FileOutline],
    case: LocalizationCase,
    *,
    use_bm25: bool,
) -> tuple[int | None, list[str]]:
    ranked = rank_outlines(outlines, query=case.query, use_bm25=use_bm25)
    targets = {normalize_path(path) for path in case.target_paths}
    paths = [item.path for item in ranked]
    first_rank = next(
        (index for index, path in enumerate(paths, start=1) if path in targets),
        None,
    )
    return first_rank, paths[:10]


def _summary(ranks: list[int | None]) -> dict[str, float | int]:
    total = len(ranks)
    reciprocal = [(1.0 / rank) if rank else 0.0 for rank in ranks]
    return {
        "cases": total,
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / total, 4) if total else 0.0,
        "recall_at_5": round(
            sum(rank is not None and rank <= 5 for rank in ranks) / total, 4
        ) if total else 0.0,
        "recall_at_10": round(
            sum(rank is not None and rank <= 10 for rank in ranks) / total, 4
        ) if total else 0.0,
        "mean_reciprocal_rank": round(mean(reciprocal), 4) if reciprocal else 0.0,
    }


def benchmark_localization(
    outlines: Iterable[FileOutline],
    cases: Iterable[LocalizationCase],
    *,
    required_recall_at_10: float = 0.90,
) -> dict[str, Any]:
    """Compare deterministic localization policies and record Stage-5 gates."""
    indexed = list(outlines)
    benchmark_cases = list(cases)
    baseline_ranks: list[int | None] = []
    bm25_ranks: list[int | None] = []
    rows: list[dict[str, Any]] = []

    for case in benchmark_cases:
        baseline_rank, baseline_top10 = _rank(indexed, case, use_bm25=False)
        bm25_rank, bm25_top10 = _rank(indexed, case, use_bm25=True)
        baseline_ranks.append(baseline_rank)
        bm25_ranks.append(bm25_rank)
        rows.append(
            {
                "query": case.query,
                "targets": [normalize_path(path) for path in case.target_paths],
                "baseline_rank": baseline_rank,
                "bm25_rank": bm25_rank,
                "baseline_top10": baseline_top10,
                "bm25_top10": bm25_top10,
            }
        )

    baseline = _summary(baseline_ranks)
    bm25 = _summary(bm25_ranks)
    recall_ok = float(bm25["recall_at_10"]) >= float(required_recall_at_10)
    improved = (
        float(bm25["recall_at_10"]) > float(baseline["recall_at_10"])
        or float(bm25["mean_reciprocal_rank"]) > float(baseline["mean_reciprocal_rank"])
    )
    return {
        "version": 1,
        "corpus": {
            "indexed_files": len(indexed),
            "cases": len(benchmark_cases),
        },
        "required_recall_at_10": required_recall_at_10,
        "baseline_exact_symbol": baseline,
        "bm25": bm25,
        "decisions": {
            "bm25": {
                "status": "enabled",
                "reason": (
                    "lexical ranking improves localization on the benchmark"
                    if improved
                    else "enabled as the bounded prose fallback; exact/symbol hits remain first"
                ),
            },
            "embeddings_hybrid": {
                "status": "not_required" if recall_ok else "benchmark_required",
                "reason": (
                    "BM25 meets the localization recall gate"
                    if recall_ok
                    else "BM25 misses the recall gate; evaluate embeddings on a larger labelled corpus"
                ),
            },
            "localization_agent": {
                "status": "not_required" if recall_ok else "ab_test_required",
                "reason": (
                    "deterministic repo_map meets the recall gate without provider tokens"
                    if recall_ok
                    else "deterministic localization misses the gate; an agent still requires a costed A/B test"
                ),
            },
        },
        "cases": rows,
    }
