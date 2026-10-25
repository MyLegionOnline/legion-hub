"""CM review verification block: _build_review_verification.

The CM's system-prompt repo snapshot is taken at pair start and goes stale the
moment the agent writes files — reviews based on it produced false rejects.
The verification block must present fresh, system-trusted facts: persisted
file_writes with previews and a fresh-vs-start repo diff.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.proxy_session import (
    _REVIEW_CHANGED_FILES_MAX,
    _REVIEW_FILE_WRITES_MAX,
    _REVIEW_PREVIEW_CHARS,
    _REVIEW_TOTAL_MAX_CHARS,
    _build_review_verification,
    _is_review_path_ignored,
)
from app.core.session_close import FileWrite, SessionCloseActions

ROOM_ID = uuid.uuid4()


def _actions(file_writes=None) -> SessionCloseActions:
    return SessionCloseActions(summary="done", file_writes=file_writes or [])


@pytest.mark.asyncio
async def test_lists_persisted_file_writes_with_preview():
    actions = _actions([FileWrite(path="/project/index.html", content="<html>hello</html>")])
    with patch(
        "app.core.memory_layer._repo_snapshot",
        new=AsyncMock(return_value=[{"path": "/project/index.html", "version": 2}]),
    ):
        text = await _build_review_verification(actions, [], ROOM_ID, AsyncMock())
    assert "ALREADY saved 1 file(s)" in text
    assert "/project/index.html" in text
    assert "<html>hello</html>" in text


@pytest.mark.asyncio
async def test_long_content_is_truncated_in_preview():
    long_content = "x" * (_REVIEW_PREVIEW_CHARS + 500)
    actions = _actions([FileWrite(path="/project/big.js", content=long_content)])
    with patch("app.core.memory_layer._repo_snapshot", new=AsyncMock(return_value=[])):
        text = await _build_review_verification(actions, [], ROOM_ID, AsyncMock())
    assert "500 more chars" in text
    assert long_content not in text


@pytest.mark.asyncio
async def test_repo_diff_reports_created_and_bumped_versions():
    start = [{"path": "/project/README.md", "version": 1}]
    fresh = [
        {"path": "/project/README.md", "version": 2},
        {"path": "/project/index.html", "version": 1},
    ]
    with patch("app.core.memory_layer._repo_snapshot", new=AsyncMock(return_value=fresh)):
        text = await _build_review_verification(_actions(), start, ROOM_ID, AsyncMock())
    assert "`/project/README.md`: v1 → v2" in text
    assert "`/project/index.html`: created (v1)" in text
    assert "NO file_writes" in text


@pytest.mark.asyncio
async def test_snapshot_fallback_filters_and_caps_changed_files():
    fresh = [
        {"path": f"/workspace/node_modules/pkg{i}/index.js", "version": 1}
        for i in range(80)
    ] + [
        {"path": f"/workspace/src/file{i}.ts", "version": 1}
        for i in range(60)
    ]
    with patch("app.core.memory_layer._repo_snapshot", new=AsyncMock(return_value=fresh)):
        text = await _build_review_verification(_actions(), [], ROOM_ID, AsyncMock())

    assert "node_modules" not in text
    assert text.count(": created (v1)") == _REVIEW_CHANGED_FILES_MAX
    assert "… and 20 more files" in text
    assert "+80 generated/build files ignored" in text


@pytest.mark.asyncio
async def test_no_changes_and_snapshot_failure_are_graceful():
    start = [{"path": "/project/a.md", "version": 1}]
    with patch(
        "app.core.memory_layer._repo_snapshot",
        new=AsyncMock(return_value=[{"path": "/project/a.md", "version": 1}]),
    ):
        text = await _build_review_verification(_actions(), start, ROOM_ID, AsyncMock())
    assert "no repository changes" in text

    with patch(
        "app.core.memory_layer._repo_snapshot",
        new=AsyncMock(side_effect=RuntimeError("db down")),
    ):
        text = await _build_review_verification(_actions(), start, ROOM_ID, AsyncMock())
    assert "SYSTEM VERIFICATION" in text  # block still returned, no raise


# ---------------------------------------------------------------------------
# Task-scoped file changes (survive pair restarts after timeout/tool_limit)
# ---------------------------------------------------------------------------


from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app.core.proxy_session import _task_scoped_file_changes

TASK_ID = uuid.uuid4()


def _db_returning(*results):
    """AsyncMock db whose successive execute() calls yield *results*."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=list(results))
    return db


def _started(task_start):
    """db whose single execute() yields the first proxy pair's start."""
    started_result = MagicMock()
    started_result.scalar = MagicMock(return_value=task_start)
    return _db_returning(started_result)


def _snapshot(*files):
    """Patch the workspace volume snapshot with ``(path, datetime)`` entries."""
    return patch(
        "app.core.workspaced_client.get_workspace_snapshot",
        new=AsyncMock(return_value={
            "files": [{"path": path, "mtime": when.timestamp()} for path, when in files]
        }),
    )


@pytest.mark.asyncio
async def test_task_scoped_changes_use_first_pair_start():
    task_start = datetime(2026, 7, 20, 20, 29, tzinfo=timezone.utc)

    with _snapshot(
        ("apps/web/src/page.tsx", task_start + timedelta(minutes=11)),
        ("apps/web/src/layout.tsx", task_start + timedelta(minutes=30)),
        ("package.json", task_start - timedelta(days=2)),
    ):
        changes = await _task_scoped_file_changes(TASK_ID, ROOM_ID, _started(task_start))

    # Newest first; a file untouched since before the task is not a change.
    assert changes == [
        "- `/apps/web/src/layout.tsx`: changed",
        "- `/apps/web/src/page.tsx`: changed",
    ]


@pytest.mark.asyncio
async def test_task_scoped_changes_read_a_naive_start_as_utc():
    task_start = datetime(2026, 7, 20, 20, 29, tzinfo=timezone.utc)

    with _snapshot(("src/a.ts", task_start + timedelta(minutes=1))):
        changes = await _task_scoped_file_changes(
            TASK_ID, ROOM_ID, _started(task_start.replace(tzinfo=None))
        )

    assert changes == ["- `/src/a.ts`: changed"]


@pytest.mark.asyncio
async def test_task_scoped_changes_none_when_snapshot_unavailable():
    with patch(
        "app.core.workspaced_client.get_workspace_snapshot",
        new=AsyncMock(side_effect=RuntimeError("workspaced down")),
    ):
        changes = await _task_scoped_file_changes(
            TASK_ID, ROOM_ID, _started(datetime(2026, 7, 20, tzinfo=timezone.utc))
        )

    # None, not [] — the caller falls back to the pair-start comparison.
    assert changes is None


@pytest.mark.asyncio
async def test_task_scoped_changes_none_without_sessions():
    started_result = MagicMock()
    started_result.scalar = MagicMock(return_value=None)
    db = _db_returning(started_result)
    assert await _task_scoped_file_changes(TASK_ID, ROOM_ID, db) is None


@pytest.mark.asyncio
async def test_verification_prefers_task_scope_and_falls_back_on_error():
    # Task-scoped path succeeds → snapshot comparison is NOT used.
    with patch(
        "app.core.proxy_session._task_scoped_file_changes",
        new=AsyncMock(return_value=["- `/workspace/a.tsx`: created (v1)"]),
    ):
        text = await _build_review_verification(
            _actions(), [], ROOM_ID, AsyncMock(), task_id=TASK_ID
        )
    assert "changed since the task started" in text
    assert "/workspace/a.tsx" in text

    # Task-scoped path blows up → graceful fallback to snapshot comparison.
    with patch(
        "app.core.proxy_session._task_scoped_file_changes",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ), patch(
        "app.core.memory_layer._repo_snapshot",
        new=AsyncMock(return_value=[{"path": "/workspace/b.tsx", "version": 1}]),
    ):
        text = await _build_review_verification(
            _actions(), [], ROOM_ID, AsyncMock(), task_id=TASK_ID
        )
    assert "`/workspace/b.tsx`: created (v1)" in text


@pytest.mark.asyncio
async def test_verification_includes_fresh_git_stat_in_workspace_mode():
    stat = " workspace/apps/web/src/page.tsx | 120 ++++\n 15 files changed"
    with patch(
        "app.core.workspaced_client.git_diff_workspace",
        new=AsyncMock(return_value={"diff": stat}),
    ), patch(
        "app.core.proxy_session._task_scoped_file_changes",
        new=AsyncMock(return_value=[]),
    ):
        text = await _build_review_verification(
            _actions(), [], ROOM_ID, AsyncMock(), task_id=TASK_ID
        )
    assert "diff --stat" in text
    assert "workspace/apps/web/src/page.tsx" in text


# ---------------------------------------------------------------------------
# Stage-3 caps: file_writes, changed-files, ignore filter, hard total cap
# ---------------------------------------------------------------------------


# --- _is_review_path_ignored unit tests ---


@pytest.mark.parametrize("path", [
    "/workspace/node_modules/lodash/index.js",
    "/workspace/dist/bundle.js",
    "/workspace/build/output.css",
    "/workspace/.next/cache/something",
    "/workspace/__pycache__/mod.cpython-311.pyc",
    "/workspace/.git/COMMIT_EDITMSG",
    "/workspace/venv/lib/site-packages/pip/__init__.py",
    "/workspace/.venv/bin/python",
    "/workspace/target/debug/main",
    "/workspace/.idea/workspace.xml",
    "/workspace/.vscode/settings.json",
    "/workspace/app.log",
    "/workspace/temp.tmp",
    "/workspace/mod.pyc",
    "/workspace/.DS_Store",
    "/workspace/Thumbs.db",
    "/workspace/package-lock.json",
    "/workspace/yarn.lock",
])
def test_is_review_path_ignored_rejects_build_artifacts(path):
    assert _is_review_path_ignored(path)


@pytest.mark.parametrize("path", [
    "/workspace/src/main.ts",
    "/workspace/README.md",
    "/workspace/poetry.lock",
    "/workspace/package.json",
    "/workspace/Makefile",
    "/workspace/src/components/Button.vue",
])
def test_is_review_path_ignored_keeps_source_files(path):
    assert not _is_review_path_ignored(path)


# --- file_writes cap ---


@pytest.mark.asyncio
async def test_small_file_writes_all_shown_with_preview():
    fws = [FileWrite(path=f"/project/file{i}.ts", content=f"content_{i}") for i in range(5)]
    with patch("app.core.memory_layer._repo_snapshot", new=AsyncMock(return_value=[])):
        text = await _build_review_verification(_actions(fws), [], ROOM_ID, AsyncMock())
    assert "ALREADY saved 5 file(s)" in text
    assert text.count("Preview:") == 5
    assert "preview omitted" not in text
    assert "git_diff" not in text


@pytest.mark.asyncio
async def test_large_file_writes_capped_to_ten_with_preview():
    fws = [
        FileWrite(path=f"/project/big{i}.js", content="x" * 2000)
        for i in range(20)
    ]
    with patch("app.core.memory_layer._repo_snapshot", new=AsyncMock(return_value=[])):
        text = await _build_review_verification(_actions(fws), [], ROOM_ID, AsyncMock())
    assert "ALREADY saved 20 file(s)" in text
    assert text.count("Preview:") == _REVIEW_FILE_WRITES_MAX
    assert text.count("preview omitted") == 20 - _REVIEW_FILE_WRITES_MAX
    assert "git_diff" in text
    assert len(text) <= _REVIEW_TOTAL_MAX_CHARS


# --- _task_scoped_file_changes: ignore filter + limit ---


@pytest.mark.asyncio
async def test_task_scoped_filters_node_modules_paths():
    task_start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with _snapshot(
        *[(f"node_modules/pkg{i}/index.js", task_start + timedelta(hours=1)) for i in range(80)],
        *[(f"src/file{i}.ts", task_start + timedelta(hours=2)) for i in range(20)],
    ):
        changes = await _task_scoped_file_changes(TASK_ID, ROOM_ID, _started(task_start))

    shown = [c for c in changes if c.startswith("- `")]
    assert len(shown) == 20
    assert all("node_modules" not in c for c in shown)
    assert any("+80 generated/build files ignored" in c for c in changes)


@pytest.mark.asyncio
async def test_task_scoped_truncates_to_max_changed_files():
    task_start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with _snapshot(
        *[(f"src/file{i}.ts", task_start + timedelta(minutes=i + 1)) for i in range(60)],
    ):
        changes = await _task_scoped_file_changes(TASK_ID, ROOM_ID, _started(task_start))

    shown = [c for c in changes if c.startswith("- `")]
    assert len(shown) == _REVIEW_CHANGED_FILES_MAX
    # The most recently touched files are the ones kept.
    assert shown[0] == "- `/src/file59.ts`: changed"
    assert any("… and 20 more files" in c for c in changes)


# --- hard total cap ---


@pytest.mark.asyncio
async def test_review_block_respects_total_char_cap():
    with patch(
        "app.core.proxy_session._task_scoped_file_changes",
        new=AsyncMock(return_value=["x" * 20000]),
    ):
        text = await _build_review_verification(
            _actions(), [], ROOM_ID, AsyncMock(), task_id=TASK_ID
        )
    assert len(text) <= _REVIEW_TOTAL_MAX_CHARS
    assert "truncated" in text


# --- regression: Vue init incident (845 files, 819 in build dirs) ---


def _vue_init_files(task_start):
    return (
        [(f"node_modules/vue/src/f{i}.js", task_start + timedelta(seconds=i + 1)) for i in range(814)]
        + [(f"dist/asset{i}.css", task_start + timedelta(hours=1)) for i in range(5)]
        + [(f"src/component{i}.vue", task_start + timedelta(hours=2)) for i in range(26)]
    )


@pytest.mark.asyncio
async def test_regression_vue_init_review_block_under_token_cap():
    """Review block for the Vue-init incident must stay ≤ 12 000 chars (≈ 3 000 tokens)."""
    task_start = datetime(2026, 7, 14, tzinfo=timezone.utc)

    with _snapshot(*_vue_init_files(task_start)):
        text = await _build_review_verification(
            _actions(), [], ROOM_ID, _started(task_start), task_id=TASK_ID
        )

    assert len(text) <= _REVIEW_TOTAL_MAX_CHARS
    assert "node_modules" not in text
    assert "+819 generated/build files ignored" in text


@pytest.mark.asyncio
async def test_regression_vue_init_task_scoped_filters_all_build_dirs():
    """task_scoped_file_changes filters 819 build-dir entries leaving only 26 useful."""
    task_start = datetime(2026, 7, 14, tzinfo=timezone.utc)

    with _snapshot(*_vue_init_files(task_start)):
        changes = await _task_scoped_file_changes(TASK_ID, ROOM_ID, _started(task_start))

    shown = [c for c in changes if c.startswith("- `")]
    assert len(shown) == 26
    assert all("node_modules" not in c for c in shown)
    assert all("dist" not in c for c in shown)
    note_lines = [c for c in changes if c.startswith("- … ") or c.startswith("- +")]
    assert any("819" in c for c in note_lines)
