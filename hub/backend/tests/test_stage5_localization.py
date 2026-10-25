from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.localization_benchmark import LocalizationCase, benchmark_localization
from app.core.repository_index import extract_outline, rank_outlines


def test_bm25_localizes_prose_missed_by_exact_symbol_baseline():
    distractors = [
        extract_outline(f"docs/{index:02d}.md", "generic installation notes")
        for index in range(12)
    ]
    target = extract_outline(
        "z/answer.md",
        "When a request is evicted, retain a valid provider placeholder for its tool result.",
    )
    case = LocalizationCase(
        "valid provider placeholder after tool result eviction",
        ("z/answer.md",),
    )

    report = benchmark_localization([*distractors, target], [case])

    assert report["baseline_exact_symbol"]["recall_at_10"] == 0.0
    assert report["bm25"]["recall_at_1"] == 1.0
    assert report["decisions"]["embeddings_hybrid"]["status"] == "not_required"
    assert report["decisions"]["localization_agent"]["status"] == "not_required"


def test_exact_path_and_symbol_hits_remain_ahead_of_bm25_content():
    exact = extract_outline("src/auth_service.py", "class AuthService:\n    pass\n")
    prose = extract_outline(
        "docs/security.md",
        "auth service " * 100,
    )

    ranked = rank_outlines([prose, exact], query="AuthService", use_bm25=True)

    assert ranked[0].path == "/src/auth_service.py"


def test_bm25_supports_non_ascii_queries():
    target = extract_outline(
        "docs/context.md",
        "Les anciens résultats d'outils sont remplacés par un reçu court.",
    )
    other = extract_outline("docs/other.md", "Configuration des conteneurs réseau.")

    ranked = rank_outlines([other, target], query="reçu court résultats")

    assert ranked[0].path == "/docs/context.md"


@pytest.mark.asyncio
async def test_repo_map_reuses_unchanged_index_and_invalidates_mutation():
    from app.core.tools._repo_map import (
        RepoMapTool,
        invalidate_repo_map_path,
    )
    from app.core.tools.registry import ToolContext

    scope = str(uuid.uuid4())
    tree = [
        {"path": "/src/a.py", "size": 20, "mtime": 1.0},
        {"path": "/docs/guide.md", "size": 30, "mtime": 1.0},
    ]
    contents = {
        "src/a.py": "class Service:\n    pass\n",
        "docs/guide.md": "provider placeholder eviction guide",
    }
    read = AsyncMock(side_effect=lambda _scope, path: contents[path])
    context = ToolContext(
        room_id=uuid.UUID(scope),
        session_id=uuid.uuid4(),
        task_id=None,
        db=AsyncMock(),
    )

    with (
        patch(
            "app.core.workspaced_client.get_workspace_tree",
            new=AsyncMock(return_value=tree),
        ),
        patch("app.core.workspaced_client.read_workspace_file", new=read),
        patch(
            "app.core.tools._repo_map.record_observation",
            new=AsyncMock(side_effect=RuntimeError("offline")),
        ),
    ):
        tool = RepoMapTool()
        await tool.execute(context, {"query": "placeholder", "max_tokens": 300})
        await tool.execute(context, {"query": "placeholder", "max_tokens": 300})
        assert read.await_count == 2

        invalidate_repo_map_path(scope, "docs/guide.md")
        await tool.execute(context, {"query": "placeholder", "max_tokens": 300})
        assert read.await_count == 3


def test_context_waterfall_projects_latest_trace_and_gateway_totals():
    from app.core.context_waterfall import context_waterfall

    result = context_waterfall(
        {
            "context_baseline": [
                {
                    "blocks": {
                        "system": {"estimated_tokens": 100},
                        "schemas": {"estimated_tokens": 50},
                        "conversation": {"estimated_tokens": 200},
                        "tool_results": {"estimated_tokens": 999},
                    }
                }
            ],
            "context_gateway": {
                "last_active_tool_tokens": 300,
                "tokens_evicted_total": 400,
                "provider_cache_read_tokens": 500,
            },
        }
    )

    assert result == {
        "pinned": 150,
        "conversation": 200,
        "active_tools": 300,
        "evicted": 400,
        "provider_cached": 500,
    }
