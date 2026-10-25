"""internal design notes Stages 2 & 4 — `.gitignore` seed / reconcile plumbing.

Stage 2: ``room_git_flow._maybe_seed_shared_gitignore`` commits a shared
``.gitignore`` before Legion publishes a greenfield repo, stamps
``git_config.first_pushed_at`` so it runs once, emits ``GITIGNORE_SEEDED``, and
never touches an adopted repo.

Stage 4: ``fetch_workspace_from_git`` forwards ``gitignore_autoresolved`` from
``workspaced`` and the ``/git/sync`` endpoint emits ``GITIGNORE_RECONCILED``
when a seed ``.gitignore`` add/add conflict was resolved to the remote's file.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.core import room_git_flow
from app.core.git_sync import fetch_workspace_from_git


class _FakeDB:
    def add(self, obj):
        pass

    async def flush(self):
        pass


def _cfg():
    return room_git_flow.get_flow_config(SimpleNamespace(git_config={}))


@pytest.fixture
def emit(monkeypatch):
    fn = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", fn)
    return fn


@pytest.fixture
def ensure(monkeypatch):
    fn = AsyncMock(return_value={"ok": True, "added": True, "sha": "abc123", "branch": "dev"})
    monkeypatch.setattr("app.core.workspaced_client.git_ensure_shared_gitignore", fn)
    return fn


@pytest.fixture(autouse=True)
def _forge(monkeypatch):
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))
    monkeypatch.setattr("app.core.git_forge.remote_branch_exists", AsyncMock(return_value=False))


@pytest.mark.asyncio
async def test_seeds_and_stamps_first_pushed(emit, ensure):
    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is True
    ensure.assert_awaited_once()
    assert ensure.await_args.args[1] == "dev"  # working branch
    assert room.git_config["first_pushed_at"]
    assert emit.await_args.kwargs["event_type"] == "GITIGNORE_SEEDED"
    assert emit.await_args.kwargs["metadata"]["sha"] == "abc123"


@pytest.mark.asyncio
async def test_skips_adopted_repo(emit, ensure):
    room = SimpleNamespace(
        git_config={
            "repo_url": "https://github.com/a/b",
            "adopted_at": "2026-08-30T00:00:00+00:00",
        },
        id=uuid4(),
    )

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    ensure.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_already_published(emit, ensure):
    room = SimpleNamespace(
        git_config={
            "repo_url": "https://github.com/a/b",
            "first_pushed_at": "2026-08-30T00:00:00+00:00",
        },
        id=uuid4(),
    )

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_repo_url_is_noop(emit, ensure):
    room = SimpleNamespace(git_config={}, id=uuid4())

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_already_has_main_stamps_but_does_not_seed(monkeypatch, emit, ensure):
    monkeypatch.setattr(
        "app.core.git_forge.remote_branch_exists", AsyncMock(return_value=True)
    )
    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    ensure.assert_not_awaited()
    assert room.git_config["first_pushed_at"]  # won't re-check next promotion


@pytest.mark.asyncio
async def test_added_false_stamps_without_event(monkeypatch, emit):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_ensure_shared_gitignore",
        AsyncMock(return_value={"ok": True, "added": False, "sha": None, "branch": "dev"}),
    )
    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    emit.assert_not_awaited()
    assert room.git_config["first_pushed_at"]


@pytest.mark.asyncio
async def test_seed_transport_error_is_swallowed(monkeypatch, emit):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_ensure_shared_gitignore",
        AsyncMock(side_effect=RuntimeError("workspaced busy")),
    )
    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())

    out = await room_git_flow._maybe_seed_shared_gitignore(room.id, room, _FakeDB(), _cfg())

    assert out is False
    emit.assert_not_awaited()
    # not stamped — a transient failure should be retried on the next promotion
    assert "first_pushed_at" not in room.git_config


# ---------------------------------------------------------------------------
# Stage 4 — gitignore_autoresolved propagation + GITIGNORE_RECONCILED event
# ---------------------------------------------------------------------------

def _room():
    return MagicMock(
        git_config={"repo_url": "https://github.com/o/r", "branch": "main"},
        creator_id=uuid4(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [True, False])
async def test_fetch_forwards_gitignore_autoresolved(ok):
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {
            "ok": ok,
            "conflict": False,
            "branch": "dev",
            "gitignore_autoresolved": True,
            "stderr": "" if ok else "boom",
        }
        result = await fetch_workspace_from_git(
            room_id=uuid4(),
            git_config={"repo_url": "https://github.com/o/r", "branch": "main"},
            access_token="t",
        )

    assert result["gitignore_autoresolved"] is True


@pytest.mark.asyncio
async def test_fetch_defaults_gitignore_autoresolved_to_false():
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True, "conflict": False, "branch": "dev"}
        result = await fetch_workspace_from_git(
            room_id=uuid4(),
            git_config={"repo_url": "https://github.com/o/r", "branch": "main"},
            access_token="t",
        )

    assert result["gitignore_autoresolved"] is False


@pytest.mark.asyncio
async def test_sync_emits_gitignore_reconciled_event():
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"
    db = AsyncMock()

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf, \
         patch.object(git_api, "emit_event", new_callable=AsyncMock) as me:
        mf.return_value = {
            "synced": 1, "errors": [], "conflict": False,
            "branch": "dev", "gitignore_autoresolved": True,
        }
        result = await git_api.git_sync(
            room_id=uuid4(),
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            db=db,
            current_user=MagicMock(id=uuid4(), email="u@e"),
        )

    assert me.await_args.kwargs["event_type"] == "GITIGNORE_RECONCILED"
    assert result.gitignore_autoresolved is True
    assert result.conflict is False


@pytest.mark.asyncio
async def test_sync_no_event_when_nothing_reconciled():
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf, \
         patch.object(git_api, "emit_event", new_callable=AsyncMock) as me:
        mf.return_value = {"synced": 1, "errors": [], "conflict": False, "branch": "dev"}
        result = await git_api.git_sync(
            room_id=uuid4(),
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            db=AsyncMock(),
            current_user=MagicMock(id=uuid4(), email="u@e"),
        )

    me.assert_not_awaited()
    assert result.gitignore_autoresolved is False
