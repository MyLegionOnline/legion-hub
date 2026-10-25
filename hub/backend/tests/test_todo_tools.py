from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_observations import build_task_todo, format_todo_note
from app.core.tools import REGISTRY, ToolContext


def _ctx() -> ToolContext:
    return ToolContext(
        room_id=uuid4(),
        session_id=uuid4(),
        task_id=uuid4(),
        db=AsyncMock(),
    )


def _db_returning(rows: list[MagicMock]) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    db.execute = AsyncMock(return_value=result)
    return db


def test_todo_tools_are_agent_only_and_registered() -> None:
    assert REGISTRY.get("todo_read") is not None
    assert REGISTRY.get("todo_write") is not None
    assert REGISTRY.get("todo_read").roles == frozenset({"agent"})
    assert REGISTRY.get("todo_write").roles == frozenset({"agent"})


@pytest.mark.asyncio
async def test_todo_write_replaces_list_and_preserves_other_context_state() -> None:
    ctx = _ctx()
    items = [
        {"id": "inspect", "title": "Inspect files", "status": "done"},
        {"id": "edit", "title": "Apply changes", "status": "in_progress"},
    ]
    with (
        patch(
            "app.core.tools._todo_write.load_context_state",
            new_callable=AsyncMock,
            return_value={"read_ledger": {"a.py": {"ranges": [[1, 3]]}}},
        ) as load_state,
        patch(
            "app.core.tools._todo_write.save_context_state",
            new_callable=AsyncMock,
        ) as save_state,
    ):
        result = await REGISTRY.get("todo_write").execute(ctx, {"items": items})

    assert json.loads(result) == {"updated": True, "items": items}
    load_state.assert_awaited_once_with(ctx.db, ctx.session_id)
    save_state.assert_awaited_once_with(
        ctx.db,
        ctx.session_id,
        {"read_ledger": {"a.py": {"ranges": [[1, 3]]}}, "todo": items},
    )


@pytest.mark.asyncio
async def test_todo_read_returns_empty_list_when_missing() -> None:
    ctx = _ctx()
    with (
        patch(
            "app.core.tools._todo_read.load_context_state",
            new_callable=AsyncMock,
            return_value={"read_ledger": {}},
        ),
        patch(
            "app.core.tools._todo_read.build_task_todo",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await REGISTRY.get("todo_read").execute(ctx, {})

    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_todo_write_rejects_invalid_items_without_persisting() -> None:
    ctx = _ctx()
    with patch(
        "app.core.tools._todo_write.save_context_state",
        new_callable=AsyncMock,
    ) as save_state:
        result = await REGISTRY.get("todo_write").execute(
            ctx,
            {"items": [{"id": "x", "title": "X", "status": "blocked"}]},
        )

    assert result.startswith("Error:")
    save_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_task_todo_uses_latest_list_and_keeps_explicit_clear() -> None:
    older = MagicMock(context_state={"todo": [{"id": "old", "title": "Old", "status": "done"}]})
    latest = MagicMock(context_state={"todo": []})
    todo = await build_task_todo(_db_returning([older, latest]), task_id=uuid4())
    assert todo == []


def test_format_todo_note_includes_status_ids_and_titles() -> None:
    text = format_todo_note([
        {"id": "inspect", "title": "Inspect files", "status": "done"},
        {"id": "edit", "title": "Apply changes", "status": "in_progress"},
    ])
    assert "durable" in text
    assert "[done] Inspect files (id: inspect)" in text
    assert "[in_progress] Apply changes (id: edit)" in text
