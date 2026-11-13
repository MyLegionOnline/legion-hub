"""Pull conflict recording against a disposable PostgreSQL database."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.api.v1 import git as git_api
from app.core import git_sync
from app.core import remote_divergence as review
from app.core.secrets.crypto import SecretValue
from app.models.agent import Agent
from app.models.message import Message
from app.models.remote_divergence import RemoteDivergence
from app.models.room import Room, RoomAgentLink, RoomUserLink
from app.models.user import User


DB_URL = os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        not DB_URL or "test" not in (make_url(DB_URL).database or ""),
        reason="requires DATABASE_URL pointing at a disposable test database",
    ),
]


@pytest_asyncio.fixture
async def pull_room(monkeypatch):
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as db:
        user = User(email=f"pull-{uuid4()}@example.com", hashed_password="x")
        db.add(user)
        await db.flush()
        room = Room(
            name="Pull test", creator_id=user.id,
            git_config={"repo_url": "https://github.com/example/project", "branch": "main"},
        )
        db.add(room)
        await db.flush()
        db.add(RoomUserLink(room_id=room.id, user_id=user.id))
        await db.commit()

    fetch = AsyncMock()
    monkeypatch.setattr(git_api.wsd, "git_fetch_workspace", fetch)
    monkeypatch.setattr(git_api.wsd, "list_workspace_branches", AsyncMock(return_value=[]))
    monkeypatch.setattr(git_api.wsd, "git_ahead_behind_remote_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(git_sync, "import_pulled_services", AsyncMock(return_value={}))

    async def pull(branch=None):
        body = git_api.GitSyncBody(branch=branch) if branch else None
        async with sessions() as db:
            return await git_api.git_sync(room.id, MagicMock(), body, db, user)

    try:
        yield sessions, room.id, pull, fetch
    finally:
        async with sessions() as db:
            await db.execute(delete(Message).where(Message.room_id == room.id))
            await db.execute(delete(RoomUserLink).where(RoomUserLink.room_id == room.id))
            await db.execute(delete(Room).where(Room.id == room.id))
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
        await engine.dispose()


def _conflict(branch="main", **changes):
    return {
        "ok": False, "stderr": "merge conflict", "conflict": True,
        "remote_ref": f"refs/legion/remote/{branch}",
        "local_sha": "a" * 40, "remote_sha": "b" * 40,
        "conflict_files": ["a.txt"],
        **changes,
    }


async def _conflicts(db, room_id):
    return (await db.scalars(select(RemoteDivergence).where(RemoteDivergence.room_id == room_id))).all()


async def _events(db, room_id):
    return (await db.scalars(select(Message).where(
        Message.room_id == room_id, Message.event_type == "GIT_CONFLICT",
    ))).all()


def _report(local_sha="a" * 40, remote_sha="b" * 40):
    return {
        "ok": True,
        "branch": "refs/legion/remote/main",
        "base": "dev",
        "branch_head": {"sha": remote_sha},
        "base_head": {"sha": local_sha},
        "conflict": True,
        "conflict_files": ["a.txt"],
        "base_files": ["a.txt", "staticfiles/a.css"],
        "branch_binary_files": ["artifact"],
    }


@pytest.mark.asyncio
async def test_review_uses_chatmaster_and_suggests_an_agent_from_the_room(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        review_id = row.id
        room = await db.get(Room, room_id)
        agent = Agent(
            owner_id=room.creator_id, nickname="Frontend", agent_uid=str(uuid4()),
            token=str(uuid4()), short_description="Maintains a.txt",
        )
        db.add(agent)
        await db.flush()
        agent_id = agent.id
        db.add(RoomAgentLink(room_id=room_id, agent_id=agent_id))
        await db.commit()

    async def completion(config, messages, schema, **kwargs):
        assert str(agent_id) in schema["properties"]["suggested_agent"]["enum"]
        assert "remote main" in messages[1]["content"] and "local dev" in messages[1]["content"]
        assert "staticfiles/a.css" in messages[1]["content"] and "artifact" in messages[1]["content"]
        analysis = {key: f"Explanation of {key}" for key in schema["required"]}
        analysis.update(recommendation="rework", suggested_agent=str(agent_id))
        return SimpleNamespace(text=json.dumps(analysis))

    monkeypatch.setattr(review.wsd, "git_divergence_workspace", AsyncMock(return_value=_report()))
    monkeypatch.setattr(
        "app.core.divergence_review._resolve_cm_llm", AsyncMock(return_value=(None, {}, None)),
    )
    model = AsyncMock(side_effect=completion)
    monkeypatch.setattr("app.core.chat_master.get_llm_json_completion", model)
    try:
        async with sessions() as db:
            result = await review.run_remote_divergence_review(room_id, review_id, db)
            assert result.state == "reviewed" and result.analysis["suggested_agent"] == str(agent_id)
            await review.run_remote_divergence_review(room_id, review_id, db)
            model.assert_awaited_once()
            events = (await db.scalars(select(Message).where(
                Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_REVIEW",
            ))).all()
            event, = events
            assert event.event_metadata["remote_divergence_id"] == str(review_id)
            assert event.event_metadata["report"]["base_flagged_files"] == ["staticfiles/a.css"]
    finally:
        async with sessions() as db:
            await db.execute(delete(RoomAgentLink).where(RoomAgentLink.agent_id == agent_id))
            await db.execute(delete(Agent).where(Agent.id == agent_id))
            await db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["take_remote", "keep_local", "reset_to_remote"])
async def test_resolve_review_fetches_before_applying_and_emits_once(pull_room, monkeypatch, decision):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", AsyncMock(return_value=_report()))
    mutation = AsyncMock(return_value={
        "ok": True, "sha": "c" * 40, "backup_ref": "refs/legion/trash/dev@time", "head_changed": True,
    })
    monkeypatch.setattr(review.wsd, "merge_workspace_branch", mutation)
    monkeypatch.setattr(review.wsd, "git_reset_branch_workspace", mutation)
    services = AsyncMock(return_value=({"added": ["web"]}, ["api"]))
    monkeypatch.setattr(review, "refresh_services_after_pull", services)

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        room = await db.get(Room, room_id)
        user = await db.get(User, room.creator_id)
        reviewed = await review.run_remote_divergence_review(room_id, row.id, db)
        assert reviewed.analysis["_fallback"]
        fetch.reset_mock(return_value=True)
        fetch.return_value = {**_conflict(), "ok": True, "conflict": False}
        result = await review.resolve_remote_divergence(
            room_id, row.id, db, decision, note="reviewed", current_user=user, access_token="test-token",
        )
        assert result["ok"] and result["services_restarted"] == ["api"]
        fetch.assert_awaited_once_with(
            str(room_id), remote_url="https://github.com/example/project", branch="main",
            target_branch="dev", token="test-token", fetch_only=True,
        )
        tips = ("a" * 40, "b" * 40)
        if decision == "reset_to_remote":
            mutation.assert_awaited_once_with(
                str(room_id), "dev", "refs/legion/remote/main", expected_tips=tips,
            )
            assert result["backup_ref"] == "refs/legion/trash/dev@time"
        else:
            mutation.assert_awaited_once_with(
                str(room_id), "refs/legion/remote/main", target="dev", expected_tips=tips,
                strategy="theirs" if decision == "take_remote" else "ours",
            )
        services.assert_awaited_once()
        assert services.await_args.kwargs["head_changed"]
        assert row.state == "resolved" and row.decision == decision and row.resolved_at
        assert (await review.resolve_remote_divergence(room_id, row.id, db, decision, current_user=user))["ok"]
        assert mutation.await_count == 1
        event, = (await db.scalars(select(Message).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_RESOLVED",
        ))).all()
        assert event.event_metadata["note"] == "reviewed"


@pytest.mark.asyncio
async def test_moved_tip_marks_review_stale_and_can_be_reviewed_again(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    report = AsyncMock(return_value=_report())
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", report)
    mutation = AsyncMock()
    monkeypatch.setattr(review.wsd, "merge_workspace_branch", mutation)

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        room = await db.get(Room, room_id)
        user = await db.get(User, room.creator_id)
        await review.run_remote_divergence_review(room_id, row.id, db)
        new_sha = "c" * 40
        fetch.return_value = {**_conflict(remote_sha=new_sha), "ok": True}
        result = await review.resolve_remote_divergence(room_id, row.id, db, "keep_local", current_user=user)
        assert not result["ok"] and result["stale"]
        assert row.state == "stale" and row.analysis is None
        mutation.assert_not_awaited()

        report.return_value = _report(remote_sha=new_sha)
        refreshed = await review.run_remote_divergence_review(room_id, row.id, db)
        assert refreshed.state == "reviewed" and refreshed.remote_sha == new_sha


@pytest.mark.asyncio
async def test_new_pull_supersedes_inflight_review(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", AsyncMock(return_value=_report()))
    model_started = asyncio.Event()
    new_pull_finished = asyncio.Event()

    async def analyse(*args):
        model_started.set()
        await new_pull_finished.wait()
        return {"summary": "Review of the old tips"}

    monkeypatch.setattr(review, "_analyse", analyse)

    async def run_review(review_id):
        async with sessions() as db:
            return await review.run_remote_divergence_review(room_id, review_id, db)

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        review_id = row.id

    job = asyncio.create_task(run_review(review_id))
    try:
        await asyncio.wait_for(model_started.wait(), timeout=5)
        new_sha = "d" * 40
        fetch.return_value = _conflict(local_sha=new_sha)
        await asyncio.wait_for(pull(), timeout=5)
    finally:
        new_pull_finished.set()
        result = await job

    assert result.state == "detected" and result.local_sha == new_sha and result.analysis is None
    async with sessions() as db:
        events = (await db.scalars(select(Message).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_REVIEW",
        ))).all()
        assert events == []


@pytest.mark.asyncio
async def test_cancelled_review_can_be_retried(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    report = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", report)

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        review_id = row.id
        with pytest.raises(asyncio.CancelledError):
            await review.run_remote_divergence_review(room_id, review_id, db)
        await db.refresh(row)
        assert row.state == "detected"

        report.side_effect = None
        report.return_value = _report()
        result = await review.run_remote_divergence_review(room_id, review_id, db)
        assert result.state == "reviewed"


@pytest.mark.asyncio
async def test_fallback_review_can_ask_chatmaster_again(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull()
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", AsyncMock(return_value=_report()))

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        review_id = row.id
        first = await review.run_remote_divergence_review(room_id, review_id, db)
        assert first.state == "reviewed" and first.analysis["_fallback"]
        assert review.can_rereview(first)

    async def completion(config, messages, schema, **kwargs):
        analysis = {key: f"Explanation of {key}" for key in schema["required"]}
        analysis["recommendation"] = "keep_local"
        return SimpleNamespace(text=json.dumps(analysis))

    monkeypatch.setattr(
        "app.core.divergence_review._resolve_cm_llm", AsyncMock(return_value=(None, {}, None)),
    )
    monkeypatch.setattr("app.core.chat_master.get_llm_json_completion", AsyncMock(side_effect=completion))
    async with sessions() as db:
        second = await review.run_remote_divergence_review(room_id, review_id, db)
        assert second.state == "reviewed" and not second.analysis.get("_fallback")
        assert second.analysis["recommendation"] == "keep_local"
        assert not review.can_rereview(second)
        events = (await db.scalars(select(Message).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_REVIEW",
        ))).all()
        assert len(events) == 2


@pytest.mark.asyncio
async def test_report_failure_still_publishes_review_for_release_branch(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    await pull("main")
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", AsyncMock(return_value={
        "ok": False, "error": "report unavailable",
    }))

    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        assert row.branch == "main"
        reviewed = await review.run_remote_divergence_review(room_id, row.id, db)
        assert reviewed.state == "reviewed"
        assert reviewed.analysis["_degraded"]
        assert reviewed.analysis["recommendation"] == "keep_local"
        event, = (await db.scalars(select(Message).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_REVIEW",
        ))).all()
        assert event.event_metadata["decisions"] == ["take_remote", "keep_local", "reset_to_remote"]


@pytest.mark.asyncio
async def test_pull_updates_one_review_and_success_closes_only_its_branch(pull_room):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    assert (await pull()).conflict
    assert fetch.await_args.kwargs["branch"] == "main"
    assert fetch.await_args.kwargs["target_branch"] == "dev"
    async with sessions() as db:
        record, = await _conflicts(db, room_id)
        event, = await _events(db, room_id)
        assert (record.branch, record.remote_ref) == ("dev", "refs/legion/remote/main")
        assert (record.local_sha, record.remote_sha) == ("a" * 40, "b" * 40)
        assert event.event_metadata["remote_divergence_id"] == str(record.id)
        assert event.event_metadata["branch"] == "dev"
        assert event.content == "Pull hit a conflict — review it"
        record_id = record.id
        record.state = "reviewed"
        record.analysis = {"recommendation": "keep_local"}
        await db.commit()

    await pull()
    async with sessions() as db:
        record = await db.get(RemoteDivergence, record_id)
        assert record.state == "reviewed" and record.analysis == {"recommendation": "keep_local"}
        assert len(await _events(db, room_id)) == 1

    fetch.return_value = _conflict(remote_sha="c" * 40, conflict_files=["b.txt"])
    await pull()
    fetch.return_value = _conflict("feature/topic")
    assert (await pull("feature/topic")).conflict
    assert fetch.await_args.kwargs["branch"] == fetch.await_args.kwargs["target_branch"] == "feature/topic"
    async with sessions() as db:
        record = await db.get(RemoteDivergence, record_id)
        assert record.remote_sha == "c" * 40
        assert record.analysis is None
        assert len(await _events(db, room_id)) == 2
        state = await git_api.build_git_state(room_id, db)
        gates = {gate.branch: gate for gate in state.gates}
        assert set(gates) == {"dev", "feature/topic"}
        assert (gates["dev"].kind, gates["dev"].id) == ("remote_divergence", str(record_id))
        assert (gates["dev"].state, gates["dev"].conflict_files) == ("detected", ["b.txt"])

    fetch.return_value = {"ok": False, "stderr": "fetch failed"}
    assert not (await pull()).ok
    fetch.return_value = {
        "ok": False, "stderr": "private files tracked", "userfiles_policy_conflict": True,
        "userfiles_tracked": [".userfiles/private.txt"],
    }
    assert (await pull()).userfiles_policy_conflict
    async with sessions() as db:
        assert (await db.get(RemoteDivergence, record_id)).state == "detected"
        assert len(await _events(db, room_id)) == 2

    fetch.return_value = {"ok": True, "head_changed": False}
    assert (await pull()).ok
    async with sessions() as db:
        record = await db.get(RemoteDivergence, record_id)
        assert record.state == "resolved" and record.resolved_at is not None
        assert [gate.branch for gate in (await git_api.build_git_state(room_id, db)).gates] == ["feature/topic"]

    fetch.return_value = _conflict()
    assert (await pull()).conflict
    async with sessions() as db:
        reopened, = [
            row for row in await _conflicts(db, room_id)
            if row.branch == "dev" and row.state == "detected"
        ]
        assert reopened.id != record_id
        assert len(await _events(db, room_id)) == 3


@pytest.mark.asyncio
async def test_concurrent_pulls_insert_one_record_and_one_chat_event(pull_room):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    results = await asyncio.gather(pull(), pull())
    assert all(result.conflict for result in results)
    async with sessions() as db:
        record, = await _conflicts(db, room_id)
        event, = await _events(db, room_id)
        assert event.event_metadata["remote_divergence_id"] == str(record.id)


@pytest.mark.asyncio
async def test_webhook_pull_conflict_opens_the_same_review(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    secret = MagicMock()
    secret.reveal.return_value = b"webhook-secret"
    monkeypatch.setattr(git_api, "binding_id_from_config", lambda *_: None)
    monkeypatch.setattr(
        git_api, "legacy_secret_value",
        lambda _config, key: secret if key == "webhook_secret" else None,
    )
    monkeypatch.setattr(git_api, "verify_git_webhook_signature", lambda *_: True)
    request = MagicMock()
    request.body = AsyncMock(return_value=b'{"ref": "refs/heads/main"}')
    request.headers = {}
    request.client = MagicMock(host="127.0.0.1")

    fetch.return_value = _conflict()
    for _ in range(2):
        async with sessions() as db:
            result = await git_api.git_webhook(room_id, request, db)
        assert result["ok"] is False
    assert (await pull()).conflict
    async with sessions() as db:
        record, = await _conflicts(db, room_id)
        event, = await _events(db, room_id)
        assert (record.branch, record.state) == ("dev", "detected")
        assert event.event_metadata["remote_divergence_id"] == str(record.id)


@pytest_asyncio.fixture
async def pull_conflict_api(pull_room, monkeypatch):
    sessions, room_id, pull, fetch = pull_room
    fetch.return_value = _conflict()
    assert (await pull()).conflict
    async with sessions() as db:
        row, = await _conflicts(db, room_id)
        did = row.id
        room = await db.get(Room, room_id)
        user = await db.get(User, room.creator_id)

    app = FastAPI()
    app.include_router(git_api.router, prefix="/api/v1")

    async def session():
        async with sessions() as db:
            yield db

    app.dependency_overrides[git_api.get_session] = session
    app.dependency_overrides[git_api.deps.get_current_user] = lambda: user
    monkeypatch.setattr(git_api, "engine", sessions.kw["bind"])
    path = f"/api/v1/rooms/{room_id}/git/remote-divergence/{did}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield app, client, path


@pytest.mark.asyncio
async def test_remote_divergence_http_review_and_resolve(pull_room, pull_conflict_api, monkeypatch):
    _, room_id, _, fetch = pull_room
    _, client, path = pull_conflict_api
    report = AsyncMock(return_value=_report())
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", report)
    merge = AsyncMock(return_value={"ok": True, "sha": "c" * 40})
    monkeypatch.setattr(review.wsd, "merge_workspace_branch", merge)
    token = AsyncMock(return_value=SecretValue("test-token"))
    monkeypatch.setattr(git_api, "_resolve_git_access_token", token)

    response = await client.get(path)
    assert response.status_code == 200
    conflict = response.json()
    assert conflict["state"] == "detected" and conflict["branch"] == "dev"
    assert conflict["analysis"] is None and conflict["report"] is None
    assert (await client.post(f"{path}/resolve", json={"decision": "unknown"})).status_code == 422
    before_review = await client.post(f"{path}/resolve", json={"decision": "keep_local"})
    assert before_review.status_code == 409
    assert before_review.json()["detail"]["code"] == "review_required"
    token.assert_not_awaited()

    accepted, repeated = await asyncio.gather(*(client.post(f"{path}/review") for _ in range(2)))
    assert accepted.status_code == repeated.status_code == 202
    assert accepted.json()["id"] == repeated.json()["id"] == conflict["id"]
    assert {accepted.json()["state"], repeated.json()["state"]} <= {"reviewing", "reviewed"}
    report.assert_awaited_once_with(str(room_id), "refs/legion/remote/main", "dev")
    token.assert_not_awaited()

    response = await client.get(path)
    reviewed = response.json()
    assert reviewed["state"] == "reviewed" and reviewed["analysis"]["summary"]
    assert reviewed["report"]["base_head"]["sha"] == reviewed["local_sha"]
    assert reviewed["report"]["branch_head"]["sha"] == reviewed["remote_sha"]

    fetch.return_value = {**_conflict(), "ok": True}
    resolution = await client.post(f"{path}/resolve", json={"decision": "keep_local", "note": "reviewed"})
    assert resolution.status_code == 200
    assert resolution.json()["decision"] == "keep_local" and resolution.json()["sha"] == "c" * 40
    assert fetch.await_args.kwargs["token"] == "test-token"
    assert fetch.await_args.kwargs["fetch_only"] is True
    assert (await client.post(f"{path}/resolve", json={"decision": "keep_local"})).status_code == 200
    assert (await client.post(f"{path}/review")).status_code == 409
    merge.assert_awaited_once()
    token.assert_awaited_once()
    resolved = (await client.get(path)).json()
    assert resolved["state"] == "resolved" and resolved["decision"] == "keep_local"
    assert resolved["report"] == reviewed["report"]


@pytest.mark.asyncio
async def test_remote_divergence_http_refreshes_stale_review(pull_room, pull_conflict_api, monkeypatch):
    _, _, pull, fetch = pull_room
    _, client, path = pull_conflict_api
    report = AsyncMock(return_value=_report())
    monkeypatch.setattr(review.wsd, "git_divergence_workspace", report)
    merge = AsyncMock()
    monkeypatch.setattr(review.wsd, "merge_workspace_branch", merge)
    monkeypatch.setattr(git_api, "_resolve_git_access_token", AsyncMock(return_value=SecretValue("test-token")))
    assert (await client.post(f"{path}/review")).status_code == 202

    fetch.return_value = {**_conflict(remote_sha="c" * 40), "ok": True}
    response = await client.post(f"{path}/resolve", json={"decision": "take_remote"})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "stale"
    stale = (await client.get(path)).json()
    assert stale["state"] == "stale" and stale["analysis"] is None and stale["report"] is None
    merge.assert_not_awaited()

    report.return_value = _report(remote_sha="c" * 40)
    assert (await client.post(f"{path}/review")).status_code == 202
    assert fetch.await_args.kwargs["token"] == "test-token"
    refreshed = (await client.get(path)).json()
    assert refreshed["state"] == "reviewed" and refreshed["remote_sha"] == "c" * 40
    assert refreshed["report"]["branch_head"]["sha"] == refreshed["remote_sha"]

    fetch.return_value = _conflict(remote_sha="d" * 40)
    assert (await pull()).conflict
    updated = (await client.get(path)).json()
    assert updated["state"] == "detected" and updated["remote_sha"] == "d" * 40
    assert updated["analysis"] is None and updated["report"] is None


@pytest.mark.asyncio
async def test_remote_divergence_http_scopes_conflict_to_room_and_member(pull_room, pull_conflict_api):
    sessions, room_id, _, _ = pull_room
    app, client, path = pull_conflict_api
    async with sessions() as db:
        conflict, = await _conflicts(db, room_id)
        room = await db.get(Room, room_id)
        other_room = Room(name="Other room", creator_id=room.creator_id)
        db.add(other_room)
        await db.flush()
        db.add(RoomUserLink(room_id=other_room.id, user_id=room.creator_id))
        await db.commit()

    try:
        other_path = f"/api/v1/rooms/{other_room.id}/git/remote-divergence/{conflict.id}"
        assert (await client.get(other_path)).status_code == 404
        assert (await client.post(f"{other_path}/review")).status_code == 404
        assert (await client.post(f"{other_path}/resolve", json={"decision": "keep_local"})).status_code == 404

        app.dependency_overrides[git_api.deps.get_current_user] = lambda: User(
            email="outsider@example.com", hashed_password="x",
        )
        assert (await client.get(path)).status_code == 403
        assert (await client.post(f"{path}/review")).status_code == 403
        assert (await client.post(f"{path}/resolve", json={"decision": "keep_local"})).status_code == 403
    finally:
        async with sessions() as db:
            await db.execute(delete(RoomUserLink).where(RoomUserLink.room_id == other_room.id))
            await db.execute(delete(Room).where(Room.id == other_room.id))
            await db.commit()



@pytest.mark.asyncio
@pytest.mark.parametrize("use_goal", [False, True])
async def test_pull_rework_dispatches_once_and_closes_only_after_merge(
    pull_room, pull_conflict_api, monkeypatch, use_goal,
):
    from app.core.room_git_flow import merge_task_branch
    from app.models.goal import RoomGoal
    from app.models.task import Task, TaskState

    sessions, room_id, pull, fetch = pull_room
    _, client, _ = pull_conflict_api
    async with sessions() as db:
        room = await db.get(Room, room_id)
        agent = Agent(owner_id=room.creator_id, nickname="Editor", agent_uid=str(uuid4()), token=str(uuid4()))
        db.add(agent)
        await db.flush()
        agent_id = agent.id
        db.add(RoomAgentLink(room_id=room_id, agent_id=agent_id))
        goal_id = None
        branch = "dev"
        if use_goal:
            goal = RoomGoal(room_id=room_id, title="Goal", branch_name=f"goal/{uuid4()}")
            db.add(goal)
            await db.flush()
            goal_id, branch = goal.id, goal.branch_name
        await db.commit()

    snapshot = _conflict(branch if use_goal else "main")
    fetch.return_value = snapshot
    await pull(branch if use_goal else None)
    async with sessions() as db:
        row = await db.scalar(select(RemoteDivergence).where(
            RemoteDivergence.room_id == room_id, RemoteDivergence.branch == branch,
        ))
        review_id = row.id
        row.state = "reviewed"
        row.analysis = {"rework_brief": "Preserve both edits."}
        await db.commit()

    fetch.return_value = {**snapshot, "ok": True, "conflict": False}
    prepare = AsyncMock(return_value={
        "ok": True, "sha": "c" * 40, "diff_base": "d" * 40,
        "remote_patches": {"a.txt": "+incoming line"},
    })
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(review.wsd, "git_prepare_pull_rework_workspace", prepare)
    monkeypatch.setattr("app.core.proxy_session.request_task_dispatch", dispatch)
    path = f"/api/v1/rooms/{room_id}/git/remote-divergence/{review_id}/resolve"
    missing = await client.post(path, json={"decision": "rework"})
    assert missing.status_code == 409 and "Choose an agent" in missing.json()["detail"]["message"]
    unrelated = await client.post(path, json={"decision": "rework", "agent_id": str(uuid4())})
    assert unrelated.status_code == 409
    prepare.assert_not_awaited()
    if not use_goal:
        async with sessions() as db:
            row = await db.get(RemoteDivergence, review_id)
            row.analysis = {**row.analysis, "suggested_agent": str(agent_id)}
            await db.commit()

    body = {"decision": "rework", "note": "Keep the existing layout."}
    if use_goal:
        body["agent_id"] = str(agent_id)
    responses = await asyncio.gather(*(client.post(path, json=body) for _ in range(2)))
    assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
    task_id = UUID(responses[0].json()["rework_task_id"])
    assert responses[1].json()["rework_task_id"] == str(task_id)
    prepare.assert_awaited_once_with(
        str(room_id), f"task/{task_id}", branch, snapshot["remote_ref"],
        expected_tips=(snapshot["local_sha"], snapshot["remote_sha"]), conflict_files=["a.txt"],
    )
    dispatch.assert_awaited_once()

    async with sessions() as db:
        task = await db.get(Task, task_id)
        row = await db.get(RemoteDivergence, review_id)
        assert task.goal_id == goal_id and task.assigned_agent_id == agent_id
        assert task.state == TaskState.ASSIGNED
        assert all(text in task.description for text in (
            "Preserve both edits.", body["note"], "+incoming line", f"git show {row.remote_sha}:a.txt",
        ))
        assert row.state == "reviewed" and row.resolved_at is None and row.rework_task_id == task_id
        assert await db.scalar(select(Message.id).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_RESOLVED",
        )) is None

        sha = "e" * 40
        merge = AsyncMock(side_effect=[{"ok": True}, {"ok": False}, *[{"ok": True, "sha": sha}] * 4])
        monkeypatch.setattr(review.wsd, "merge_workspace_branch", merge)
        monkeypatch.setattr("app.core.room_git_flow.commit_working_tree", AsyncMock())
        report = AsyncMock(return_value={"ok": True, "ahead": 0})
        monkeypatch.setattr(review.wsd, "git_divergence_workspace", report)
        assert not (await merge_task_branch(room_id, task, db))["ok"]
        report.assert_not_awaited()
        for _ in range(2):
            assert (await merge_task_branch(room_id, task, db))["target"] == branch
        report.assert_awaited_once_with(str(room_id), snapshot["remote_sha"], sha)
        assert row.state == "resolved" and row.resolved_at is not None
        resolved, = (await db.scalars(select(Message).where(
            Message.room_id == room_id, Message.event_type == "REMOTE_DIVERGENCE_RESOLVED",
        ))).all()
        assert resolved.event_metadata["rework_task_id"] == str(task_id)
        assert resolved.event_metadata["sha"] == sha

        await db.execute(delete(Message).where(Message.room_id == room_id, Message.task_id == task_id))
        await db.delete(task)
        if goal_id:
            await db.flush()
            await db.delete(await db.get(RoomGoal, goal_id))
        await db.execute(delete(RoomAgentLink).where(RoomAgentLink.agent_id == agent_id))
        await db.flush()
        await db.execute(delete(Agent).where(Agent.id == agent_id))
        await db.commit()
