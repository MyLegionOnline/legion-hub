"""
Stage 6 of internal design notes — repofile removal.

These tests verify the post-removal invariants:

  1. ``RepoFile`` model, schema, and legacy router are gone (import fails).
  2. ``workspace_fs.py`` has no legacy branch — no ``_is_workspace_mode``,
     no ``_select_repofile``, no ``_legacy_read``, no ``fs_mirror_to_workspace``,
     no ``fs_sync_workspace_tree``, no ``fs_upsert_workspace_meta``, no
     ``WORKSPACE_FS_BACKEND`` constant.
  3. ``git_sync.py`` has no Contents-API functions (``sync_branch``,
     ``push_files_to_git``, ``_fetch_tree`` etc.).
  4. ``fs_read_file`` returns ``None`` for a missing file (no legacy lift
     from the DB).
  5. ``fs_write_file`` empty-overwrite guard still works (volume-based).
  6. Room/agent deletion no longer touches repofile.
  7. ``sync_files_to_workspace`` is a snapshot refresh (no RepoFile query).
"""
from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ── 1. Deleted modules / symbols ────────────────────────────────────────────

def test_repo_file_model_import_fails():
    """The model module is deleted."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.models.repo_file")


def test_repo_file_schema_import_fails():
    """The schema module is deleted."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.schemas.repo_file")


def test_repository_router_import_fails():
    """The legacy id-based router is deleted."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.api.v1.repository")


def test_repo_file_not_in_models_all():
    import app.models
    assert "RepoFile" not in app.models.__all__


def test_repository_router_not_registered():
    """main.py must not import or register the deleted router."""
    import app.main
    # If the import succeeded, the router was removed successfully
    assert not hasattr(app.main, "repository")


# ── 2. workspace_fs legacy symbols gone ─────────────────────────────────────

def test_workspace_fs_has_no_legacy_symbols():
    from app.core import workspace_fs

    gone = [
        "_is_workspace_mode",
        "_select_repofile",
        "_legacy_read",
        "_compare_snapshot_diagnostic",
        "fs_mirror_to_workspace",
        "fs_sync_workspace_tree",
        "fs_upsert_workspace_meta",
        "WORKSPACE_FS_BACKEND",
    ]
    for name in gone:
        assert not hasattr(workspace_fs, name), f"workspace_fs.{name} should be gone"


def test_workspace_fs_has_no_repo_file_import():
    """No module under app.core.workspace_fs should import RepoFile."""
    import app.core.workspace_fs as wfs
    # The module source must not reference RepoFile in a live import
    import inspect
    src = inspect.getsource(wfs)
    assert "from app.models.repo_file" not in src
    assert "RepoFile(" not in src


# ── 3. git_sync dead functions gone ─────────────────────────────────────────

def test_git_sync_has_no_contents_api_functions():
    from app.core import git_sync

    gone = [
        "sync_branch",
        "push_files_to_git",
        "_fetch_tree",
        "_fetch_blob",
        "_fetch_file_content",
        "_upsert_file",
        "_get_file_sha_github",
        "_api_base",
        "_auth_headers",
    ]
    for name in gone:
        assert not hasattr(git_sync, name), f"git_sync.{name} should be gone"


# ── 4. fs_read_file returns None for missing file ────────────────────────────

@pytest.mark.asyncio
async def test_fs_read_file_returns_none_for_missing():
    from app.core import workspace_fs

    room_id = uuid4()
    db = AsyncMock()
    with patch("app.core.workspaced_client.read_workspace_file",
               new_callable=AsyncMock, return_value=None):
        result = await workspace_fs.fs_read_file(room_id, "/nope.md", db)
    assert result is None


@pytest.mark.asyncio
async def test_fs_read_file_returns_content_when_present():
    from app.core import workspace_fs

    room_id = uuid4()
    db = AsyncMock()
    with patch("app.core.workspaced_client.read_workspace_file",
               new_callable=AsyncMock, return_value="hello"):
        result = await workspace_fs.fs_read_file(room_id, "/a.md", db)
    assert result == "hello"


# ── 5. fs_write_file empty-overwrite guard (volume-based) ────────────────────

@pytest.mark.asyncio
async def test_fs_write_file_blocks_empty_over_non_empty():
    from app.core import workspace_fs
    from app.core.workspace_fs import EmptyOverwriteBlockedError

    room_id = uuid4()
    db = AsyncMock()
    # Volume says the file currently has 10 bytes
    with patch("app.core.workspaced_client.read_workspace_file",
               new_callable=AsyncMock, return_value="1234567890"), \
         patch("app.core.workspaced_client.write_workspace_file",
               new_callable=AsyncMock) as mw:
        with pytest.raises(EmptyOverwriteBlockedError):
            await workspace_fs.fs_write_file(room_id, "/a.md", "", db)
    mw.assert_not_awaited()


@pytest.mark.asyncio
async def test_fs_write_file_allows_empty_for_new_file():
    from app.core import workspace_fs

    room_id = uuid4()
    db = AsyncMock()
    # Volume says the file doesn't exist (None)
    with patch("app.core.workspaced_client.read_workspace_file",
               new_callable=AsyncMock, return_value=None), \
         patch("app.core.workspaced_client.write_workspace_file",
               new_callable=AsyncMock) as mw:
        await workspace_fs.fs_write_file(room_id, "/new.md", "", db)
    mw.assert_awaited_once()


@pytest.mark.asyncio
async def test_fs_write_file_allow_empty_bypasses_guard():
    from app.core import workspace_fs

    room_id = uuid4()
    db = AsyncMock()
    with patch("app.core.workspaced_client.read_workspace_file",
               new_callable=AsyncMock, return_value="existing"), \
         patch("app.core.workspaced_client.write_workspace_file",
               new_callable=AsyncMock) as mw:
        await workspace_fs.fs_write_file(room_id, "/a.md", "", db, allow_empty=True)
    mw.assert_awaited_once()


# ── 6. Room/agent deletion no longer touches repofile ────────────────────────

def test_rooms_delete_has_no_repofile_cleanup():
    """rooms.py must not reference RepoFile or repofile."""
    import inspect
    from app.api.v1 import rooms
    src = inspect.getsource(rooms)
    assert "RepoFile" not in src
    assert "repofile" not in src.lower()


def test_agents_delete_has_no_repofile_cleanup():
    """agents.py must not reference RepoFile or repofile."""
    import inspect
    from app.api.v1 import agents
    src = inspect.getsource(agents)
    assert "RepoFile" not in src
    assert "repofile" not in src.lower()


# ── 7. sync_files_to_workspace is a snapshot refresh ─────────────────────────

@pytest.mark.asyncio
async def test_sync_files_to_workspace_calls_snapshot_not_repofile():
    from app.api.v1 import workspace as workspace_api
    from app.models.workspace import Workspace

    room_id = uuid4()
    ws = Workspace(room_id=room_id, status="running")
    db = AsyncMock()

    with patch.object(workspace_api, "_get_room", new_callable=AsyncMock), \
         patch.object(workspace_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_api, "_get_or_create_workspace",
                      new_callable=AsyncMock, return_value=ws), \
         patch.object(workspace_api.wsd, "get_workspace_snapshot",
                      new_callable=AsyncMock) as msnap:
        msnap.return_value = {"files": [], "generation": {}}
        result = await workspace_api.sync_files_to_workspace(
            room_id=room_id, db=db, current_user=MagicMock(),
        )

    assert result == {"ok": True, "mode": "workspace", "synced": 0}
    msnap.assert_awaited_once()
    # No DB query for repofile
    db.execute.assert_not_awaited()


# ── 8. No WORKSPACE_FS_BACKEND references in app/ ────────────────────────────

def test_no_workspace_fs_backend_env_reads_in_app():
    """No live code in app/ should read the WORKSPACE_FS_BACKEND env var.
    Mentions in docstrings/comments are fine — only actual ``os.getenv``
    / ``os.environ`` reads count as offenders."""
    import pathlib

    backend_root = pathlib.Path(__file__).resolve().parent.parent / "app"
    offenders = []
    for py in backend_root.rglob("*.py"):
        try:
            text = py.read_text()
        except Exception:
            continue
        # Only flag active env reads, not docstring mentions.
        if 'getenv("WORKSPACE_FS_BACKEND"' in text or \
            "getenv('WORKSPACE_FS_BACKEND'" in text or \
            'environ["WORKSPACE_FS_BACKEND"]' in text or \
            "environ['WORKSPACE_FS_BACKEND']" in text:
            offenders.append(str(py.relative_to(backend_root)))
    assert not offenders, f"WORKSPACE_FS_BACKEND env reads still in: {offenders}"


# ── 9. No live RepoFile references in app/ ───────────────────────────────────

def test_no_live_repofile_references_in_app():
    """No live code in app/ should reference ``RepoFile``.

    Comments and docstrings may still mention the removed model for
    historical context — only actual AST-level references count as
    offenders (imports, name loads, attribute access, type annotations,
    base classes, decorated targets). This is the test-form of the plan's
    ``grep -r RepoFile app/`` CI invariant, made robust against prose.
    """
    import ast
    import pathlib

    backend_root = pathlib.Path(__file__).resolve().parent.parent / "app"
    offenders = []
    for py in backend_root.rglob("*.py"):
        try:
            text = py.read_text()
        except Exception:
            continue
        try:
            tree = ast.parse(text, filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # import RepoFile / from x import RepoFile
            if isinstance(node, ast.alias) and node.name == "RepoFile":
                offenders.append(str(py.relative_to(backend_root)))
                break
            # any bare Name or Attribute referencing RepoFile (loads, stores,
            # annotations, base classes, decorated targets)
            if isinstance(node, ast.Name) and node.id == "RepoFile":
                offenders.append(str(py.relative_to(backend_root)))
                break
            if isinstance(node, ast.Attribute) and node.attr == "RepoFile":
                offenders.append(str(py.relative_to(backend_root)))
                break
    assert not offenders, f"live RepoFile references still in: {offenders}"
