"""Tests for connector mode=worker.

Covers the "patch through the same WS transport" design: a mode=worker
dispatch carries a base64 tar snapshot of the task branch
(``core/connector_session.py``'s ``_build_dispatch_context``) instead of git
access, and its ``result`` frame carries a unified diff back
(``core/connector_reverse.py``'s ``_handle_result``) instead of a push.

  * dispatch: the payload carries ``mode=worker``, ``task_branch`` and
    ``workspace_archive_b64``, and ``task_data_text`` is wrapped with the
    worker preamble — mirrors the existing operator-preamble assertion style
    in test_connector_dispatch.py.
  * dispatch fails retryably (``worker_no_archive``) when no archive
    could be built — same shape as the existing ``connector_not_granted``/
    ``connector_offline`` failure paths.
  * result: a clean patch is applied onto the task branch and, on the CM's
    approve, folded into ``dev`` via ``merge_task_branch`` — exactly the
    local loop's own verdict path, not a second copy of it.
  * result: a patch that fails to apply leaves the task in WAITING_REVIEW
    with a GIT_CONFLICT event and never reaches merge_task_branch — the
    existing "Divergence review" flow takes it from there, this stage does
    not reimplement conflict resolution.

``workspaced``/git are never actually called — ``workspaced_client.archive_
workspace_branch``/``apply_patch_workspace`` and ``room_git_flow.merge_task_
branch`` are mocked throughout, same spirit as test_connector_reverse.py
mocking out ``_cm_turn``. Gated on INIT_DB_TEST_URL, same as the sibling
connector test files.
"""

import asyncio
import base64
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_user_and_room(room_id, owner_id, *, cm_agent_id):
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=owner_id, email=f"conn-worker-{owner_id}@example.com", hashed_password="x")
        )
        room = Room(id=room_id, name="conn-worker-room", creator_id=owner_id)
        room.coordinator_config = {"master_agent_id": str(cm_agent_id)}
        session.add(room)
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=owner_id))
        await session.commit()
        break


async def _make_granted_worker_setup(owner_id, room_id):
    """Same as test_connector_reverse.py's ``_make_granted_connector_setup``,
    with two worker-mode differences: the agent's ``llm_config`` selects
    ``connector_mode=worker``, and the task's ``branch_name`` is pre-set —
    without a live ``workspaced``, ``ensure_task_branch``'s HTTP call to it
    fails and is swallowed (``proxy_session.py`` logs and moves on), so
    nothing would otherwise populate it before dispatch.
    """
    from app.core import connector_pairing
    from app.core.room_git_flow import task_branch_name
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.task import DispatchStatus, Task, TaskState

    cm_agent_id = uuid4()
    task_id = uuid4()

    await _make_user_and_room(room_id, owner_id, cm_agent_id=cm_agent_id)

    async for db in get_session():
        cm_agent = Agent(
            id=cm_agent_id,
            owner_id=owner_id,
            nickname="chat-master",
            agent_uid=f"cm-{cm_agent_id.hex[:8]}",
            token="tok-" + uuid4().hex,
        )
        db.add(cm_agent)
        await db.commit()
        break

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-worker")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        agent = Agent(
            id=uuid4(),
            owner_id=owner_id,
            nickname="remote-worker",
            agent_uid=f"remote-worker-{uuid4().hex[:8]}",
            token="tok-" + uuid4().hex,
            executor="connector",
            connector_id=paired.connector_id,
            llm_config={"connector_mode": "worker"},
        )
        db.add(agent)
        await db.flush()
        task = Task(
            id=task_id,
            room_id=room_id,
            title="add a health endpoint",
            assigned_agent_id=agent.id,
            state=TaskState.ASSIGNED,
            dispatch_status=DispatchStatus.READY,
            branch_name=task_branch_name(task_id),
        )
        db.add(task)
        await db.commit()
        break

    return task_id, agent.id, paired.connector_id


def _set_timeouts(*, timeout_minutes: str, poll_seconds: str):
    os.environ["CONNECTOR_DISPATCH_TIMEOUT_MINUTES"] = timeout_minutes
    os.environ["CONNECTOR_DISPATCH_POLL_SECONDS"] = poll_seconds


def _clear_timeouts():
    os.environ.pop("CONNECTOR_DISPATCH_TIMEOUT_MINUTES", None)
    os.environ.pop("CONNECTOR_DISPATCH_POLL_SECONDS", None)


def _one_db():
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def _dispatch_and_wait(task_id, room_id):
    """See test_connector_reverse.py's identical helper for the rationale."""
    from app.core.proxy_session import handle_task_created
    from app.db.session import get_session
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import Task
    from sqlalchemy import select

    bg_db = _one_db()
    runtime_task = asyncio.create_task(
        handle_task_created(task_id, room_id, None, None, bg_db)
    )
    await asyncio.sleep(0.4)

    async for db in get_session():
        task = await db.get(Task, task_id)
        run_id = task.run_id
        sessions = (
            await db.execute(
                select(CmAgentProxySession).where(CmAgentProxySession.task_id == task_id)
            )
        ).scalars().all()
        agent_session = next((s for s in sessions if s.side == "agent"), None)
        cm_session = next((s for s in sessions if s.side == "chat_master"), None)
        break

    return (
        runtime_task,
        bg_db,
        agent_session.id if agent_session else None,
        cm_session.id if cm_session else None,
        run_id,
    )


@pytest.mark.asyncio
async def test_worker_dispatch_carries_archive_branch_and_preamble():
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    fake_archive = base64.b64encode(b"fake-tar-bytes").decode("ascii")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_worker_setup(owner_id, room_id)

        with patch(
            "app.core.workspaced_client.archive_workspace_branch",
            AsyncMock(return_value=b"fake-tar-bytes"),
        ):
            runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
                task_id, room_id
            )
        try:
            assert agent_session_id is not None, "dispatch should have created the session pair"

            async for db in get_session():
                rows = (
                    await db.execute(
                        select(ConnectorOutbox).where(
                            ConnectorOutbox.connector_id == connector_id,
                            ConnectorOutbox.kind == "dispatch",
                        )
                    )
                ).scalars().all()
                break
            assert len(rows) == 1
            payload = rows[0].payload
            assert payload["mode"] == "worker"
            assert payload["task_branch"] == f"task/{task_id}"
            assert payload["workspace_archive_b64"] == fake_archive
            assert "worker" in payload["task_data_text"].lower()
        finally:
            runtime_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime_task
            await bg_db.close()
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_worker_dispatch_fails_retryably_without_an_archive():
    from app.db.session import get_session
    from app.models.task import DispatchStatus, Task

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, _connector_id = await _make_granted_worker_setup(owner_id, room_id)

        with patch(
            "app.core.workspaced_client.archive_workspace_branch",
            AsyncMock(return_value=None),
        ):
            from app.core.proxy_session import handle_task_created

            bg_db = _one_db()
            await asyncio.wait_for(
                handle_task_created(task_id, room_id, None, None, bg_db), timeout=5
            )
            await bg_db.close()

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.dispatch_status == DispatchStatus.READY
            break
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_worker_result_applies_patch_and_merges_on_approve():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import Task, TaskState
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_worker_setup(owner_id, room_id)

        with patch(
            "app.core.workspaced_client.archive_workspace_branch",
            AsyncMock(return_value=b"fake-tar-bytes"),
        ):
            runtime_task, bg_db, agent_session_id, cm_session_id, run_id = await _dispatch_and_wait(
                task_id, room_id
            )

        apply_mock = AsyncMock(return_value={"ok": True, "conflict": False, "sha": "deadbeef"})
        merge_mock = AsyncMock(return_value={"ok": True, "sha": "cafef00d", "branch": f"task/{task_id}"})
        fake_cm_turn = AsyncMock(
            return_value=(None, {"summary": "Looks good."}, 0, 0, "mock", 0, 0)
        )
        with patch("app.core.workspaced_client.apply_patch_workspace", apply_mock), \
             patch("app.core.room_git_flow.merge_task_branch", merge_mock), \
             patch("app.core.connector_reverse._cm_turn", fake_cm_turn):
            async for db in get_session():
                await connector_reverse.handle_inbound_frame(
                    db,
                    frame_type="result",
                    payload={
                        "summary": "Added the health endpoint and tests.",
                        "success": True,
                        "patch": "diff --git a/app.py b/app.py\n+added\n",
                    },
                    session_id=agent_session_id,
                    run_id=run_id,
                    task_id=task_id,
                    connector_id=connector_id,
                )
                break

        await asyncio.wait_for(runtime_task, timeout=5)
        await bg_db.close()

        apply_mock.assert_awaited_once()
        apply_call = apply_mock.await_args
        assert apply_call.args[1] == f"task/{task_id}"
        assert "added" in apply_call.args[2]
        merge_mock.assert_awaited_once()

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.state == TaskState.COMPLETED
            assert task.merge_commit == "cafef00d"

            cm_session = await db.get(CmAgentProxySession, cm_session_id)
            assert cm_session.close_reason == "task_done"
            break
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_worker_result_conflict_leaves_task_waiting_review_without_merging():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.models.message import Message
    from app.models.task import Task, TaskState
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_worker_setup(owner_id, room_id)

        with patch(
            "app.core.workspaced_client.archive_workspace_branch",
            AsyncMock(return_value=b"fake-tar-bytes"),
        ):
            runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
                task_id, room_id
            )

        apply_mock = AsyncMock(
            return_value={"ok": False, "conflict": True, "stderr": "patch does not apply"}
        )
        merge_mock = AsyncMock()
        with patch("app.core.workspaced_client.apply_patch_workspace", apply_mock), \
             patch("app.core.room_git_flow.merge_task_branch", merge_mock):
            async for db in get_session():
                await connector_reverse.handle_inbound_frame(
                    db,
                    frame_type="result",
                    payload={
                        "summary": "Added the health endpoint.",
                        "success": True,
                        "patch": "diff --git a/app.py b/app.py\n+added\n",
                    },
                    session_id=agent_session_id,
                    run_id=run_id,
                    task_id=task_id,
                    connector_id=connector_id,
                )
                break

        await asyncio.wait_for(runtime_task, timeout=5)
        await bg_db.close()

        merge_mock.assert_not_awaited()

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.state == TaskState.WAITING_REVIEW

            conflicts = (
                await db.execute(
                    select(Message).where(Message.task_id == task_id, Message.event_type == "GIT_CONFLICT")
                )
            ).scalars().all()
            assert len(conflicts) == 1
            break
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()
