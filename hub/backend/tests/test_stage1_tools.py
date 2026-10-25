"""
Tests for Stage 1: edit_file, grep (fallback), glob, improved read_file.

All tests use mocked workspaced_client so no real workspace is needed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools._session_state import clear_session, mark_read, was_read


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(session_id=None) -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    # Simulate no workspace row (triggers grep fallback)
    execute_mock = AsyncMock()
    execute_mock.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    db.execute = execute_mock
    return ToolContext(
        room_id=uuid4(),
        session_id=session_id or uuid4(),
        task_id=uuid4(),
        db=db,
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def test_session_state_mark_and_check():
    sid = str(uuid4())
    assert not was_read(sid, "src/foo.py")
    mark_read(sid, "src/foo.py")
    assert was_read(sid, "src/foo.py")
    clear_session(sid)
    assert not was_read(sid, "src/foo.py")


def test_session_state_normalizes_paths():
    sid = str(uuid4())
    mark_read(sid, "/workspace/src/foo.py")
    # Both forms should match
    assert was_read(sid, "src/foo.py")
    assert was_read(sid, "/workspace/src/foo.py")
    assert was_read(sid, "/src/foo.py")
    clear_session(sid)


def test_session_state_unknown_session():
    assert not was_read("nonexistent-session-id", "anything.py")


# ---------------------------------------------------------------------------
# read_file — line numbers, offset/limit, binary rejection, tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_line_numbers():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    content = "line one\nline two\nline three\n"
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = content
        result = await tool.execute(ctx, {"path": "foo.py"})
    assert "1\tline one" in result
    assert "2\tline two" in result
    assert "3\tline three" in result


@pytest.mark.asyncio
async def test_read_file_offset_and_limit():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    lines = [f"line {i}\n" for i in range(1, 11)]
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "".join(lines)
        result = await tool.execute(ctx, {"path": "big.py", "offset": 3, "limit": 4})
    assert "3\tline 3" in result
    assert "4\tline 4" in result
    assert "6\tline 6" in result
    assert "7\tline 7" not in result
    assert "[partial]" in result


@pytest.mark.asyncio
async def test_read_file_full_read_no_partial_tag():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "only line\n"
        result = await tool.execute(ctx, {"path": "x.py"})
    assert "[partial]" not in result


@pytest.mark.asyncio
async def test_read_file_binary_rejection():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "data:image/png;base64,iVBORw0KGgo="
        result = await tool.execute(ctx, {"path": "image.png"})
    assert "Error" in result
    assert "binary" in result.lower()


@pytest.mark.asyncio
async def test_read_file_tracks_path_in_session():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("read_file")
    clear_session(str(sid))
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "hello\n"
        await tool.execute(ctx, {"path": "tracked.py"})
    assert was_read(str(sid), "tracked.py")
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_read_file_binary_does_not_track():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("read_file")
    clear_session(str(sid))
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "data:application/octet-stream;base64,AAAA"
        await tool.execute(ctx, {"path": "binary.bin"})
    assert not was_read(str(sid), "binary.bin")
    clear_session(str(sid))


# ---------------------------------------------------------------------------
# edit_file — all error branches + happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_file_requires_prior_read():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    clear_session(str(sid))
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock):
        result = await tool.execute(ctx, {
            "path": "unread.py", "old_string": "x", "new_string": "y",
        })
    assert "Error" in result
    assert "read_file" in result
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_old_string_not_found():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "foo.py")
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = "def bar(): pass\n"
        result = await tool.execute(ctx, {
            "path": "foo.py", "old_string": "def foo", "new_string": "def baz",
        })
    assert "Error" in result
    assert "not found" in result.lower()
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_multiple_matches_without_replace_all():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "foo.py")
    content = "x = 1\nx = 2\n"
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = content
        result = await tool.execute(ctx, {
            "path": "foo.py", "old_string": "x", "new_string": "y",
        })
    assert "Error" in result
    assert "2" in result  # mentions count
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_replace_all():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "foo.py")
    content = "x = 1\nx = 2\n"
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = content
        with patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mw:
            result = await tool.execute(ctx, {
                "path": "foo.py", "old_string": "x", "new_string": "z", "replace_all": True,
            })
    assert "Edited" in result
    assert "2 replacements" in result
    written_content = mw.call_args[0][2]  # positional: room_id, path, content
    assert written_content == "z = 1\nz = 2\n"
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_happy_path():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "app.py")
    content = "def hello():\n    return 'world'\n"
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = content
        with patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mw:
            result = await tool.execute(ctx, {
                "path": "app.py",
                "old_string": "return 'world'",
                "new_string": "return 'legion'",
            })
    assert "Edited" in result
    assert "1 replacement" in result
    written = mw.call_args[0][2]
    assert "return 'legion'" in written
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_identical_strings():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "x.py")
    result = await tool.execute(ctx, {
        "path": "x.py", "old_string": "same", "new_string": "same",
    })
    assert "Error" in result
    assert "identical" in result.lower()
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_edit_file_file_not_found():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("edit_file")
    mark_read(str(sid), "ghost.py")
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = None
        result = await tool.execute(ctx, {
            "path": "ghost.py", "old_string": "x", "new_string": "y",
        })
    assert "not found" in result.lower()
    clear_session(str(sid))


# ---------------------------------------------------------------------------
# grep — fallback (no container)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_fallback_finds_match():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    fake_tree = [{"path": "/src/main.py", "size": 100, "mtime": 1.0}]
    fake_content = "def hello_world():\n    pass\n"

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.return_value = fake_content
            result = await tool.execute(ctx, {"pattern": "hello_world"})

    assert "src/main.py" in result
    assert "hello_world" in result


@pytest.mark.asyncio
async def test_grep_fallback_no_match():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    fake_tree = [{"path": "/src/main.py", "size": 50, "mtime": 1.0}]

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.return_value = "x = 1\n"
            result = await tool.execute(ctx, {"pattern": "missing_symbol"})

    assert "No matches" in result


@pytest.mark.asyncio
async def test_grep_fallback_files_mode():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    fake_tree = [
        {"path": "/a.py", "size": 50, "mtime": 1.0},
        {"path": "/b.py", "size": 50, "mtime": 1.0},
    ]

    async def _read(room_id, path):
        return "TARGET = 1\n" if "a.py" in path else "other = 2\n"

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.side_effect = _read
            result = await tool.execute(ctx, {"pattern": "TARGET", "output_mode": "files"})

    assert "a.py" in result
    assert "b.py" not in result


@pytest.mark.asyncio
async def test_grep_fallback_glob_filter():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    fake_tree = [
        {"path": "/src/app.py", "size": 50, "mtime": 1.0},
        {"path": "/src/app.ts", "size": 50, "mtime": 1.0},
    ]

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.return_value = "const x = useEffect;\n"
            result = await tool.execute(ctx, {"pattern": "useEffect", "glob": "**/*.ts"})

    assert "app.ts" in result
    assert "app.py" not in result


@pytest.mark.asyncio
async def test_grep_skips_binary_files():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    fake_tree = [{"path": "/image.png", "size": 50, "mtime": 1.0}]

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.return_value = "data:image/png;base64,iVBORw0KGgo="
            result = await tool.execute(ctx, {"pattern": "png"})

    assert "No matches" in result


@pytest.mark.asyncio
async def test_grep_skips_oversized_files():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    # File is 300 KB > 256 KB limit
    fake_tree = [{"path": "/big.py", "size": 300 * 1024, "mtime": 1.0}]

    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mr:
            mr.return_value = "def target(): pass\n"
            result = await tool.execute(ctx, {"pattern": "target"})

    # read_workspace_file should NOT have been called for oversized file
    mr.assert_not_awaited()
    assert "No matches" in result


@pytest.mark.asyncio
async def test_grep_invalid_regex():
    ctx = _make_ctx()
    tool = REGISTRY.get("grep")
    result = await tool.execute(ctx, {"pattern": "[invalid"})
    assert "Error" in result
    assert "regex" in result.lower()


# ---------------------------------------------------------------------------
# glob tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_returns_matching_files():
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    fake_tree = [
        {"path": "/src/main.py", "mtime": 1000.0, "size": 100},
        {"path": "/README.md", "mtime": 900.0, "size": 50},
        {"path": "/tests/test_foo.py", "mtime": 800.0, "size": 80},
    ]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        result = await tool.execute(ctx, {"pattern": "**/*.py"})

    assert "src/main.py" in result
    assert "tests/test_foo.py" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_glob_double_star_spans_zero_or_more_dirs():
    """Regression: PurePosixPath.match treated ``**`` as one segment, so
    ``bible/**/*.md`` missed ``bible/Kate_arc.md`` and ``**/*.md`` missed root files."""
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    fake_tree = [
        {"path": "/bible/Kate_arc.md", "mtime": 5.0, "size": 1},
        {"path": "/bible/arcs/deep/plan.md", "mtime": 4.0, "size": 1},
        {"path": "/episodes/s1/ep1.md", "mtime": 3.0, "size": 1},
        {"path": "/README.md", "mtime": 2.0, "size": 1},
        {"path": "/bible/notes.txt", "mtime": 1.0, "size": 1},
    ]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        bible = await tool.execute(ctx, {"pattern": "bible/**/*.md"})
        all_md = await tool.execute(ctx, {"pattern": "**/*.md"})
        everything = await tool.execute(ctx, {})

    assert bible.splitlines() == ["bible/Kate_arc.md", "bible/arcs/deep/plan.md"]
    assert set(all_md.splitlines()) == {
        "bible/Kate_arc.md", "bible/arcs/deep/plan.md", "episodes/s1/ep1.md", "README.md",
    }
    assert len(everything.splitlines()) == 5


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("bible/Kate_arc.md", "bible/**/*.md", True),
        ("bible/a/b/c.md", "bible/**/*.md", True),
        ("bible_old/a.md", "bible/**/*.md", False),
        ("README.md", "**/*.md", True),
        ("README.md", "**/*", True),
        ("src/a/b.py", "src/**", True),
        ("src/a.py", "src/*.py", True),
        ("src/a/b.py", "src/*.py", False),
        ("deep/x/config.json", "*.json", True),
        ("/src/main.py", "src/*.py", True),
        ("src/a.ts", "src/*.[jt]s", True),
        ("src/a.py", "src/*.[!p]y", False),
        ("docs/a.md", "docs/?.md", True),
    ],
)
def test_workspace_glob_match(path, pattern, expected):
    from app.core.path_contract import workspace_glob_match

    assert workspace_glob_match(path, pattern) is expected


def test_grep_include_from_glob():
    from app.core.tools._grep import _glob_to_include

    assert _glob_to_include("**/*.py") == "*.py"
    assert _glob_to_include("src/**/*.ts") == "*.ts"
    assert _glob_to_include("*.json") == "*.json"
    assert _glob_to_include("src/**") == "*"


@pytest.mark.asyncio
async def test_glob_sorted_by_mtime_newest_first():
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    fake_tree = [
        {"path": "/src/old.py", "mtime": 100.0, "size": 10},
        {"path": "/src/new.py", "mtime": 999.0, "size": 10},
        {"path": "/src/mid.py", "mtime": 500.0, "size": 10},
    ]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        result = await tool.execute(ctx, {"pattern": "**/*.py"})

    lines = [l for l in result.splitlines() if l.strip() and not l.startswith("[")]
    assert lines[0].endswith("new.py")
    assert lines[1].endswith("mid.py")
    assert lines[2].endswith("old.py")


@pytest.mark.asyncio
async def test_glob_no_match():
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    fake_tree = [{"path": "/README.md", "mtime": 100.0, "size": 10}]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        result = await tool.execute(ctx, {"pattern": "**/*.py"})

    assert "No files match" in result


@pytest.mark.asyncio
async def test_glob_truncation_at_100():
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    # 110 files
    fake_tree = [
        {"path": f"/src/file{i:03d}.py", "mtime": float(i), "size": 10}
        for i in range(110)
    ]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mt:
        mt.return_value = fake_tree
        result = await tool.execute(ctx, {"pattern": "**/*.py"})

    lines = [l for l in result.splitlines() if not l.startswith("[")]
    assert len(lines) == 100
    assert "10 more" in result or "more files" in result
