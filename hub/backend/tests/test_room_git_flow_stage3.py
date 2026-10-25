"""Stage 3 of internal design notes — risk classification + trivial
auto-merge revert (hub side).

Covers ``room_git_flow``:
  * ``_parse_diff_stat`` reads file / line counts and paths from ``--stat``;
  * ``classify_task_risk``:
      - small, clean diff → ``trivial`` (and persists ``task.risk``);
      - over the file / line thresholds → ``normal``;
      - a sensitive path (``migrations/`` …) → ``risky`` regardless of size;
      - a router ``hint`` may only raise the class;
      - a diff transport error is treated as ``normal`` (never auto-merged);
  * ``auto_merge_trivial_enabled`` defaults on, off only on explicit False;
  * ``revert_task_merge`` reverts ``task.merge_commit`` and emits
    ``TASK_MERGE_REVERTED`` (or ``GIT_CONFLICT`` on a dirty revert).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def __init__(self):
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass


def _task(**kw):
    base = dict(
        id=uuid4(), title="T", goal_id=None,
        branch_name=f"task/{uuid4()}", risk=None, merge_commit=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


_STAT_SMALL = " app/foo.py | 4 ++--\n 1 file changed, 2 insertions(+), 2 deletions(-)\n"
_STAT_BIG = (
    " a.py | 20 ++++\n b.py | 30 +++\n c.py | 25 ++\n d.py | 10 +\n"
    " 4 files changed, 70 insertions(+), 15 deletions(-)\n"
)
_STAT_MIGRATION = (
    " migrations/0009_add_col.py | 12 ++++++\n"
    " 1 file changed, 12 insertions(+)\n"
)


# ── _parse_diff_stat ────────────────────────────────────────────────────────

def test_parse_diff_stat_small():
    files, lines, paths = room_git_flow._parse_diff_stat(_STAT_SMALL)
    assert files == 1 and lines == 4 and paths == ["app/foo.py"]


def test_parse_diff_stat_big():
    files, lines, paths = room_git_flow._parse_diff_stat(_STAT_BIG)
    assert files == 4 and lines == 85
    assert "migrations" not in "".join(paths)


# ── classify_task_risk ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _stub_base(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_load_room", AsyncMock(return_value=SimpleNamespace(git_config={})))
    monkeypatch.setattr(room_git_flow, "_task_base_branch", AsyncMock(return_value="dev"))


def _stub_diff(monkeypatch, stat: str):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(return_value={"diff": stat}),
    )


@pytest.mark.asyncio
async def test_small_diff_is_trivial(monkeypatch):
    _stub_diff(monkeypatch, _STAT_SMALL)
    task = _task()
    out = await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB())
    assert out == "trivial"
    assert task.risk == "trivial"


@pytest.mark.asyncio
async def test_big_diff_is_normal(monkeypatch):
    _stub_diff(monkeypatch, _STAT_BIG)
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB()) == "normal"


@pytest.mark.asyncio
async def test_migration_path_is_risky(monkeypatch):
    _stub_diff(monkeypatch, _STAT_MIGRATION)
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB()) == "risky"


@pytest.mark.asyncio
async def test_router_hint_only_raises(monkeypatch):
    _stub_diff(monkeypatch, _STAT_SMALL)
    # hint raises trivial → risky
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB(), hint="risky") == "risky"
    # hint cannot lower risky → trivial
    _stub_diff(monkeypatch, _STAT_MIGRATION)
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB(), hint="trivial") == "risky"


@pytest.mark.asyncio
async def test_diff_error_defaults_to_normal(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB()) == "normal"
    assert task.risk == "normal"


@pytest.mark.asyncio
async def test_custom_thresholds_from_git_config(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"risk_thresholds": {"files": 10, "lines": 500}})),
    )
    _stub_diff(monkeypatch, _STAT_BIG)
    task = _task()
    assert await room_git_flow.classify_task_risk(str(uuid4()), task, _FakeDB()) == "trivial"


# ── auto_merge_trivial_enabled ──────────────────────────────────────────────

def test_auto_merge_default_on():
    assert room_git_flow.auto_merge_trivial_enabled(SimpleNamespace(git_config={})) is True
    assert room_git_flow.auto_merge_trivial_enabled(SimpleNamespace(git_config=None)) is True


def test_auto_merge_explicit_off():
    assert room_git_flow.auto_merge_trivial_enabled(
        SimpleNamespace(git_config={"auto_merge_trivial": False})
    ) is False


# ── revert_task_merge ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revert_task_merge_no_commit_is_noop(monkeypatch):
    out = await room_git_flow.revert_task_merge(str(uuid4()), _task(merge_commit=None), _FakeDB())
    assert out["ok"] is False and "merge commit" in out["error"]


@pytest.mark.asyncio
async def test_revert_task_merge_ok_emits_event(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_revert_commit_workspace",
        AsyncMock(return_value={"ok": True, "conflict": False, "sha": "revsha"}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task(merge_commit="mergesha")
    out = await room_git_flow.revert_task_merge(str(uuid4()), task, _FakeDB())

    assert out["ok"] is True
    assert task.merge_commit is None
    assert emit.await_args.kwargs["event_type"] == "TASK_MERGE_REVERTED"


@pytest.mark.asyncio
async def test_revert_task_merge_conflict_emits_git_conflict(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_revert_commit_workspace",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task(merge_commit="mergesha")
    out = await room_git_flow.revert_task_merge(str(uuid4()), task, _FakeDB())

    assert out["conflict"] is True
    assert task.merge_commit == "mergesha"  # not cleared
    assert emit.await_args.kwargs["event_type"] == "GIT_CONFLICT"
