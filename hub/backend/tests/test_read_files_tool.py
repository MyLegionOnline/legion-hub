"""Tests for the batch read_files tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools._session_state import clear_session, was_read


def _make_ctx(session_id=None) -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    execute_mock = AsyncMock()
    execute_mock.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    db.execute = execute_mock
    return ToolContext(room_id=uuid4(), session_id=session_id or uuid4(), task_id=uuid4(), db=db)


def test_read_files_is_registered():
    tool = REGISTRY.get("read_files")
    assert tool is not None
    assert tool.roles == frozenset({"agent", "cm"})
    assert tool.input_schema["properties"]["paths"]["maxItems"] == 8


@pytest.mark.asyncio
async def test_read_files_returns_a_section_per_path():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_files")

    async def fake_read(_room_id, path):
        return {
            "/workspace/a.py": "a1\na2\n",
            "/workspace/b.py": "b1\n",
        }.get(path)

    with patch("app.core.workspaced_client.read_workspace_file", new=AsyncMock(side_effect=fake_read)):
        result = await tool.execute(ctx, {"paths": ["a.py", "b.py"]})

    assert "path='a.py'" in result
    assert "1\ta1" in result and "2\ta2" in result
    assert "path='b.py'" in result
    assert "1\tb1" in result


@pytest.mark.asyncio
async def test_read_files_missing_file_does_not_fail_whole_batch():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_files")

    async def fake_read(_room_id, path):
        return "content\n" if path == "/workspace/exists.py" else None

    with patch("app.core.workspaced_client.read_workspace_file", new=AsyncMock(side_effect=fake_read)):
        result = await tool.execute(ctx, {"paths": ["exists.py", "missing.py"]})

    assert "1\tcontent" in result
    assert "File not found: 'missing.py'" in result


@pytest.mark.asyncio
async def test_read_files_marks_every_path_read_for_edit_file():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("read_files")
    clear_session(str(sid))

    async def fake_read(_room_id, path):
        return {"/workspace/a.py": "a\n", "/workspace/b.py": "b\n"}.get(path)

    with patch("app.core.workspaced_client.read_workspace_file", new=AsyncMock(side_effect=fake_read)):
        await tool.execute(ctx, {"paths": ["a.py", "b.py"]})

    assert was_read(str(sid), "a.py")
    assert was_read(str(sid), "b.py")
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_read_files_large_file_is_truncated_with_partial_tag():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_files")
    big = "".join(f"line {i}\n" for i in range(1, 400))

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = big
        result = await tool.execute(ctx, {"paths": ["big.py"]})

    assert "[partial]" in result
    assert "line 100" in result
    assert "line 101" not in result


@pytest.mark.asyncio
async def test_read_files_rejects_more_than_max_paths():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_files")
    result = await tool.execute(ctx, {"paths": [f"f{i}.py" for i in range(9)]})
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_read_files_rejects_empty_paths():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_files")
    result = await tool.execute(ctx, {"paths": []})
    assert result.startswith("Error:")
