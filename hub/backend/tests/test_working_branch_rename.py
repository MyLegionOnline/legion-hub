"""internal design notes Stage 3 — renaming a room's
working branch.

Covers ``app.core.working_branch_rename`` (name rules, the live-work refusals,
the remote probe and its confirmation, workspaced refusals, undo on a failed
save, the chat event), the endpoint's error mapping, the settings save no
longer writing ``working_branch``, and the flow picking the new name up.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import git as git_api
from app.core import git_sync, room_git_flow
from app.core import working_branch_rename as wbr
from app.core.room_git_flow import get_flow_config, merge_git_config

_REPO = "https://github.com/acme/app"


def _room(**cfg):
    return SimpleNamespace(id=uuid4(), git_config=dict(cfg))


def _db(room):
    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.refresh = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


def _ok(**extra):
    return {"ok": True, "renamed": True, "created": False, "head": "core", **extra}


async def _rename(room, name="core", *, blockers=(), probe=None, ws=None, confirm=False, db=None):
    db = db or _db(room)
    with patch.object(wbr, "rename_blockers", new_callable=AsyncMock, return_value=list(blockers)), \
         patch.object(wbr.wsd, "remote_branch_exists", new_callable=AsyncMock,
                      return_value=probe or {"ok": True, "exists": False}) as probe_mock, \
         patch.object(wbr.wsd, "rename_working_branch", new_callable=AsyncMock,
                      return_value=ws or _ok()) as ws_mock, \
         patch.object(wbr, "emit_event", new_callable=AsyncMock) as event_mock:
        try:
            outcome = await wbr.rename_working_branch(
                room.id, name, db, access_token="tok", confirm_remote_exists=confirm,
            )
        except wbr.RenameRefused as exc:
            outcome = exc
    return outcome, SimpleNamespace(db=db, probe=probe_mock, ws=ws_mock, event=event_mock)


# ── name rules ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["core", "team/frontend", "release-2.x", "a_b", "dev"])
def test_valid_names(name):
    assert wbr._valid_name(name)


@pytest.mark.parametrize(
    "name", ["", "main", "HEAD", "task/1", "goal/x", "session/y", "\u044f\u0434\u0440\u043e", "-x", "a..b",
             "a//b", "a/", "a.", ".a", "a.lock", "a b", "a@{1}", "x" * 101],
)
def test_invalid_names(name):
    assert not wbr._valid_name(name)


@pytest.mark.asyncio
async def test_invalid_name_is_422_and_touches_nothing():
    room = _room(repo_url=_REPO)
    outcome, m = await _rename(room, "main")
    assert isinstance(outcome, wbr.RenameRefused) and outcome.status == 422
    m.ws.assert_not_awaited()
    m.probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_name_is_a_no_op():
    room = _room(repo_url=_REPO, working_branch="core")
    outcome, m = await _rename(room, "core")
    assert outcome.ok and not outcome.changed
    m.ws.assert_not_awaited()
    m.event.assert_not_awaited()


# ── happy path ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rename_moves_branch_then_config_and_tells_the_chat():
    room = _room(repo_url=_REPO, branch="main", ci_provider="github_actions")
    outcome, m = await _rename(room, "core")

    assert outcome.ok and outcome.changed
    assert (outcome.from_branch, outcome.to_branch) == ("dev", "core")
    m.ws.assert_awaited_once_with(str(room.id), "dev", "core")
    m.probe.assert_awaited_once_with(str(room.id), _REPO, "core", "tok")
    assert room.git_config["working_branch"] == "core"
    assert room.git_config["ci_provider"] == "github_actions"
    m.db.commit.assert_awaited()
    kwargs = m.event.await_args.kwargs
    assert kwargs["event_type"] == "WORKING_BRANCH_RENAMED"
    assert kwargs["metadata"]["from_branch"] == "dev"
    assert kwargs["metadata"]["to_branch"] == "core"
    assert "dev on the remote was left" in kwargs["content"]
    assert get_flow_config(room).working_branch == "core"


@pytest.mark.asyncio
async def test_room_without_remote_skips_the_probe():
    room = _room()
    outcome, m = await _rename(room, "core", ws=_ok(renamed=False, created=True))
    assert outcome.ok and outcome.changed and not outcome.remote_checked
    m.probe.assert_not_awaited()
    assert room.git_config["working_branch"] == "core"
    assert "remote" not in m.event.await_args.kwargs["content"]


# ── refusals ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_work_is_409_with_blockers():
    room = _room(repo_url=_REPO)
    blockers = [wbr._blocker("task", 2, "a task is running")]
    outcome, m = await _rename(room, "core", blockers=blockers)
    assert isinstance(outcome, wbr.RenameRefused) and outcome.status == 409
    assert outcome.detail()["code"] == "work_in_progress"
    assert outcome.detail()["blockers"][0]["kind"] == "task"
    m.ws.assert_not_awaited()
    assert "working_branch" not in room.git_config


@pytest.mark.asyncio
async def test_existing_remote_branch_needs_confirmation():
    room = _room(repo_url=_REPO)
    outcome, m = await _rename(room, "core", probe={"ok": True, "exists": True})
    assert not outcome.ok and outcome.remote_exists and not outcome.changed
    m.ws.assert_not_awaited()
    assert "working_branch" not in room.git_config


@pytest.mark.asyncio
async def test_confirmed_remote_branch_goes_ahead():
    room = _room(repo_url=_REPO)
    outcome, m = await _rename(room, "core", probe={"ok": True, "exists": True}, confirm=True)
    assert outcome.ok and outcome.changed
    assert m.event.await_args.kwargs["metadata"]["remote_branch_exists"] is True
    assert "already had a branch core" in m.event.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_unreachable_remote_does_not_block_but_is_reported():
    room = _room(repo_url=_REPO)
    outcome, m = await _rename(room, "core", probe={"ok": False, "error": "auth"})
    assert outcome.ok and outcome.changed and not outcome.remote_checked
    assert any("could not be checked" in n for n in outcome.notes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ws, status, code",
    [
        ({"ok": False, "code": "branch_exists", "error": "exists"}, 409, "branch_exists"),
        ({"ok": False, "code": "working_branch_mismatch", "error": "x"}, 409, "working_branch_mismatch"),
        ({"ok": False, "busy": True, "stderr": "lock"}, 409, "busy"),
        ({"ok": False, "code": "invalid_name", "error": "bad"}, 422, "invalid_name"),
        ({"ok": False, "error": "boom"}, 502, "workspaced_error"),
    ],
)
async def test_workspaced_refusals_leave_the_config(ws, status, code):
    room = _room(repo_url=_REPO)
    outcome, _ = await _rename(room, "core", ws=ws)
    assert isinstance(outcome, wbr.RenameRefused)
    assert (outcome.status, outcome.code) == (status, code)
    assert "working_branch" not in room.git_config


@pytest.mark.asyncio
async def test_failed_save_renames_back():
    room = _room(repo_url=_REPO)
    db = _db(room)
    db.commit = AsyncMock(side_effect=RuntimeError("db down"))
    outcome, m = await _rename(room, "core", db=db)
    assert isinstance(outcome, wbr.RenameRefused) and outcome.status == 500
    assert [c.args for c in m.ws.await_args_list] == [
        (str(room.id), "dev", "core"), (str(room.id), "core", "dev"),
    ]
    db.rollback.assert_awaited()
    m.event.assert_not_awaited()


# ── live-work detection ──────────────────────────────────────────────────────

def _result(value=None, ids=()):
    r = MagicMock()
    r.scalar.return_value = value
    r.scalars.return_value.all.return_value = list(ids)
    return r


@pytest.mark.asyncio
async def test_blockers_name_each_kind_of_live_work():
    live, idle = uuid4(), uuid4()
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _result(1), _result(2), _result(3), _result(1), _result(ids=[live, idle]),
    ])
    with patch("app.core.user_sessions.turn_in_progress", side_effect=lambda sid: sid == live):
        blockers = await wbr.rename_blockers(uuid4(), db)
    assert [(b["kind"], b["count"]) for b in blockers] == [
        ("promotion", 1), ("goal_review", 2), ("task", 3), ("pull_conflict", 1), ("dialogue", 1),
    ]


@pytest.mark.asyncio
async def test_quiet_room_has_no_blockers():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _result(0), _result(0), _result(0), _result(0), _result(ids=[uuid4()]),
    ])
    with patch("app.core.user_sessions.turn_in_progress", return_value=False):
        assert await wbr.rename_blockers(uuid4(), db) == []


# ── endpoint ─────────────────────────────────────────────────────────────────

async def _call_endpoint(room, side_effect=None, return_value=None):
    db = _db(room)
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock,
                      return_value=MagicMock(reveal=MagicMock(return_value="tok"))), \
         patch.object(wbr, "rename_working_branch", new_callable=AsyncMock,
                      side_effect=side_effect, return_value=return_value) as core:
        resp = await git_api.rename_working_branch(
            room_id=room.id,
            body=git_api.WorkingBranchRenameBody(name="core"),
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            db=db,
            current_user=MagicMock(id=uuid4()),
        )
    return resp, core


@pytest.mark.asyncio
async def test_endpoint_maps_a_refusal_to_its_status():
    room = _room(repo_url=_REPO)
    refusal = wbr.RenameRefused(409, "work_in_progress", "busy", [{"kind": "task"}])
    with pytest.raises(HTTPException) as exc:
        await _call_endpoint(room, side_effect=refusal)
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "work_in_progress"
    assert exc.value.detail["blockers"] == [{"kind": "task"}]


@pytest.mark.asyncio
async def test_endpoint_returns_the_outcome_and_passes_the_token():
    room = _room(repo_url=_REPO)
    outcome = wbr.RenameOutcome(ok=True, changed=True, from_branch="dev", to_branch="core",
                                remote_checked=True, head="core", notes=["n"])
    resp, core = await _call_endpoint(room, return_value=outcome)
    assert resp.ok and resp.changed and resp.to_branch == "core" and resp.notes == ["n"]
    assert core.await_args.kwargs["access_token"] == "tok"


# ── the settings save and the flow ───────────────────────────────────────────

def test_settings_save_cannot_set_the_working_branch():
    merged = merge_git_config({"repo_url": _REPO, "working_branch": "core"},
                              {"repo_url": _REPO, "working_branch": "other"})
    assert merged["working_branch"] == "core"
    assert "working_branch" not in merge_git_config({}, {"working_branch": "core"})


@pytest.mark.asyncio
async def test_flow_reads_the_new_name_after_a_rename():
    room = _room(repo_url=_REPO, branch="main")
    outcome, _ = await _rename(room, "core")
    assert outcome.ok

    # Task / goal / session bases and merge targets resolve through this.
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=room)))
    assert await room_git_flow._working_branch(room.id, db) == "core"

    # Pull lands the remote main on the renamed branch.
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True}
        await git_sync.fetch_workspace_from_git(room_id=room.id, git_config=room.git_config)
    assert mf.await_args.kwargs["target_branch"] == "core"
    assert mf.await_args.kwargs["branch"] == "main"
