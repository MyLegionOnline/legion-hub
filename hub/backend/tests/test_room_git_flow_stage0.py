"""Stage 0 of internal design notes — branch-model foundation (hub side).

Covers ``room_git_flow``:
  * ``get_flow_config`` defaults, overrides, and clamping of bad values;
  * ``ensure_base_branches`` delegates to ``workspaced``;
  * ``backfill_dev_branches`` only touches rooms that already have a repo bound.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import room_git_flow


# ── get_flow_config ─────────────────────────────────────────────────────────

def test_defaults_on_empty_config():
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={}))
    assert cfg.working_branch == "dev"
    assert cfg.release_branch == "main"
    assert cfg.ci_provider == "workspace"
    assert cfg.promote_policy == "manual"


def test_defaults_when_git_config_missing_or_none():
    assert room_git_flow.get_flow_config(SimpleNamespace(git_config=None)).working_branch == "dev"
    assert room_git_flow.get_flow_config(SimpleNamespace()).promote_policy == "manual"


def test_overrides_are_read():
    room = SimpleNamespace(git_config={
        "working_branch": "integration",
        "ci_provider": "github_actions",
        "promote_policy": "on_each_task",
    })
    cfg = room_git_flow.get_flow_config(room)
    assert cfg.working_branch == "integration"
    assert cfg.ci_provider == "github_actions"
    assert cfg.promote_policy == "on_each_task"


def test_unknown_values_fall_back_to_default():
    room = SimpleNamespace(git_config={
        "ci_provider": "jenkins",
        "promote_policy": "yolo",
    })
    cfg = room_git_flow.get_flow_config(room)
    assert cfg.ci_provider == "workspace"
    assert cfg.promote_policy == "manual"


def test_blank_working_branch_falls_back():
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={"working_branch": "  "}))
    assert cfg.working_branch == "dev"


# ── ensure_base_branches ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_base_branches_delegates_to_workspaced():
    with patch("app.core.workspaced_client.ensure_base_branches", new=AsyncMock(
        return_value={"ok": True, "created_dev": True},
    )) as mock:
        out = await room_git_flow.ensure_base_branches("room-123")
    mock.assert_awaited_once_with("room-123")
    assert out == {"ok": True, "created_dev": True}


@pytest.mark.asyncio
async def test_backfill_local_excludes_delegates_to_workspaced():
    with patch("app.core.workspaced_client.backfill_local_excludes", new=AsyncMock(
        return_value={"ok": True, "updated": True},
    )) as mock:
        out = await room_git_flow.backfill_local_excludes("room-123")
    mock.assert_awaited_once_with("room-123")
    assert out == {"ok": True, "updated": True}


# ── backfill_dev_branches ───────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *_a, **_k):
        return _FakeResult(self._rows)


@pytest.mark.asyncio
async def test_backfill_only_touches_rooms_with_repo(monkeypatch):
    rows = [
        ("room-with-repo", {"repo_url": "https://github.com/o/r"}),
        ("room-no-repo", {}),
        ("room-null-config", None),
    ]
    monkeypatch.setattr(
        "sqlalchemy.orm.sessionmaker",
        lambda *a, **k: (lambda: _FakeSession(rows)),
    )

    seen: list[str] = []
    excludes_seen: list[str] = []

    async def _fake_ensure(room_id):
        seen.append(str(room_id))
        return {"ok": True, "created_dev": True}

    async def _fake_excludes(room_id):
        excludes_seen.append(str(room_id))
        return {"ok": True, "updated": False}

    monkeypatch.setattr(room_git_flow, "ensure_base_branches", _fake_ensure)
    monkeypatch.setattr(room_git_flow, "backfill_local_excludes", _fake_excludes)

    await room_git_flow.backfill_dev_branches(engine=object())

    assert seen == ["room-with-repo"]
    # internal design notes §3.2 — local excludes are backfilled in the same pass.
    assert excludes_seen == ["room-with-repo"]
