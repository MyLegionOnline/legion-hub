"""
Unit + integration tests for task dependency tracking (Stage 6 / internal design notes).

Unit tests (no DB):
  - TestCreateTasksWithDepsUnit  — linear chain, parallel branches, cycle, edge cases,
                                   predecessors that are already finished
  - TestRefreshBlockedDispatchStatusUnit — level-triggered unblock of one task
  - TestTryUnblockTasksUnit      — partial deps (no unblock), last dep (unblock), no deps

Integration tests (require PostgreSQL via DATABASE_URL env var):
  - TestDevTestDeployDB          — full dev → test → deploy, step-by-step approve
  - TestDependencyFailureDB      — get_blocked_dependents on rejected predecessor
  - TestNoDepRegressionDB        — tasks without deps are all READY (parallel dispatch)

Run DB tests:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_task_deps.py -m db
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.core.task_deps import (
    TaskDependencyCycleError,
    create_tasks_with_deps,
    get_blocked_dependents,
    refresh_blocked_dispatch_status,
    try_unblock_tasks,
    validate_no_cycle,
)
from app.models.task import DispatchStatus, Task, TaskState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _task(title: str = "task") -> Task:
    """Construct an in-memory Task with a pre-assigned UUID (no DB required)."""
    return Task(room_id=uuid.uuid4(), title=title)


def _mock_db(*, cycle_detected: bool = False) -> AsyncMock:
    """Return a mocked AsyncSession.

    When *cycle_detected* is True, db.execute() returns a row — the cycle CTE
    in validate_no_cycle interprets a non-None first() as a cycle.
    """
    db = AsyncMock()
    db.flush = AsyncMock()
    db.add = MagicMock()

    cycle_row = MagicMock()
    cycle_row.first.return_value = object() if cycle_detected else None
    db.execute = AsyncMock(return_value=cycle_row)
    return db


# ---------------------------------------------------------------------------
# Unit tests — create_tasks_with_deps
# ---------------------------------------------------------------------------


class TestCreateTasksWithDepsUnit:
    """dispatch_status is set correctly without a real database."""

    @pytest.mark.asyncio
    async def test_checkpoint_fields_survive_graph_creation(self):
        """A human gate keeps its payload while dependency status is computed."""
        checkpoint = Task(
            room_id=uuid.uuid4(),
            title="Confirm publication",
            state=TaskState.WAITING_APPROVAL,
            assigned_agent_id=None,
            checkpoint_kind="confirm",
            checkpoint_payload={"prompt": "Publish the approved copy?"},
        )
        follow_up = _task("publish")
        db = _mock_db()

        await create_tasks_with_deps(
            [
                ("confirm", [], checkpoint),
                ("publish", ["confirm"], follow_up),
            ],
            db,
        )

        assert checkpoint.state == TaskState.WAITING_APPROVAL
        assert checkpoint.checkpoint_kind == "confirm"
        assert checkpoint.checkpoint_payload == {"prompt": "Publish the approved copy?"}
        assert checkpoint.assigned_agent_id is None
        assert checkpoint.dispatch_status == DispatchStatus.READY
        assert follow_up.dispatch_status == DispatchStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_linear_chain_first_ready_rest_blocked(self):
        """dev → test → deploy: dev=READY, test=BLOCKED, deploy=BLOCKED."""
        dev = _task("dev")
        test = _task("test")
        deploy = _task("deploy")
        db = _mock_db()

        result = await create_tasks_with_deps(
            [("dev", [], dev), ("test", ["dev"], test), ("deploy", ["test"], deploy)],
            db,
        )

        assert len(result) == 3
        assert dev.dispatch_status == DispatchStatus.READY
        assert test.dispatch_status == DispatchStatus.BLOCKED
        assert deploy.dispatch_status == DispatchStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_parallel_branches_both_ready(self):
        """fe and be are independent → both READY; integration depends on both → BLOCKED."""
        fe = _task("frontend")
        be = _task("backend")
        integration = _task("integration")
        db = _mock_db()

        await create_tasks_with_deps(
            [("fe", [], fe), ("be", [], be), ("integration", ["fe", "be"], integration)],
            db,
        )

        assert fe.dispatch_status == DispatchStatus.READY
        assert be.dispatch_status == DispatchStatus.READY
        assert integration.dispatch_status == DispatchStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_cycle_raises_error(self):
        """A → B and B → A must raise TaskDependencyCycleError."""
        a = _task("A")
        b = _task("B")
        db = _mock_db(cycle_detected=True)

        with pytest.raises(TaskDependencyCycleError):
            await create_tasks_with_deps(
                [("a", ["b"], a), ("b", ["a"], b)],
                db,
            )

    @pytest.mark.asyncio
    async def test_no_deps_all_ready(self):
        """Tasks with empty depends_on are all READY."""
        tasks = [_task(f"t{i}") for i in range(4)]
        db = _mock_db()
        triples = [(f"t{i}", [], t) for i, t in enumerate(tasks)]

        await create_tasks_with_deps(triples, db)

        for t in tasks:
            assert t.dispatch_status == DispatchStatus.READY

    @pytest.mark.asyncio
    async def test_unresolvable_ref_treated_as_no_dep(self):
        """An unresolvable depends_on ref is logged and ignored — task becomes READY."""
        a = _task("A")
        db = _mock_db()

        await create_tasks_with_deps([("a", ["ghost-ref"], a)], db)

        assert a.dispatch_status == DispatchStatus.READY

    @pytest.mark.asyncio
    async def test_self_ref_in_depends_on_is_ignored(self):
        """A task listing its own ref in depends_on must not block itself."""
        a = _task("A")
        db = _mock_db()

        await create_tasks_with_deps([("a", ["a"], a)], db)

        assert a.dispatch_status == DispatchStatus.READY

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty_without_touching_db(self):
        """Empty input returns [] and never flushes."""
        db = _mock_db()

        result = await create_tasks_with_deps([], db)

        assert result == []
        db.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_uuid_string_in_depends_on_resolves(self):
        """A UUID string (existing task reference) in depends_on resolves correctly."""
        existing = _task("existing")
        dependent = _task("dependent")
        db = _mock_db()

        # existing is already in the ref_map via str(existing.id)
        await create_tasks_with_deps(
            [(None, [], existing), (None, [str(existing.id)], dependent)],
            db,
        )

        assert existing.dispatch_status == DispatchStatus.READY
        assert dependent.dispatch_status == DispatchStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_dep_already_finished_is_ready_at_creation(self):
        """A predecessor that finished BEFORE the edge existed must not block.

        try_unblock_tasks only fires while a predecessor transitions, so a task
        blocked on an already-completed one waits for an event that can never
        happen: not dispatchable, and refused even by a manual Start.
        """
        done_id = uuid.uuid4()
        follow_up = _task("publish")
        db = _mock_db()

        with patch(
            "app.core.task_deps._finished_task_ids",
            AsyncMock(return_value={done_id}),
        ):
            await create_tasks_with_deps([(None, [str(done_id)], follow_up)], db)

        assert follow_up.dispatch_status == DispatchStatus.READY

    @pytest.mark.asyncio
    async def test_one_unfinished_dep_still_blocks(self):
        """Mixed predecessors: the unfinished one keeps the task blocked."""
        done_id = uuid.uuid4()
        running_id = uuid.uuid4()
        follow_up = _task("deploy")
        db = _mock_db()

        with patch(
            "app.core.task_deps._finished_task_ids",
            AsyncMock(return_value={done_id}),
        ):
            await create_tasks_with_deps(
                [(None, [str(done_id), str(running_id)], follow_up)], db
            )

        assert follow_up.dispatch_status == DispatchStatus.BLOCKED


# ---------------------------------------------------------------------------
# Unit tests — refresh_blocked_dispatch_status
# ---------------------------------------------------------------------------


class TestRefreshBlockedDispatchStatusUnit:
    """Level-triggered unblock for a task nobody will ever unblock by event."""

    @staticmethod
    def _db_with_deps(*dep_ids: uuid.UUID) -> AsyncMock:
        db = AsyncMock()
        db.flush = AsyncMock()
        db.add = MagicMock()
        rows = MagicMock()
        rows.all.return_value = [(dep_id,) for dep_id in dep_ids]
        db.execute = AsyncMock(return_value=rows)
        return db

    @pytest.mark.asyncio
    async def test_all_predecessors_finished_promotes_to_ready(self):
        dep_id = uuid.uuid4()
        task = _task("deploy")
        task.dispatch_status = DispatchStatus.BLOCKED
        db = self._db_with_deps(dep_id)

        with patch(
            "app.core.task_deps._finished_task_ids",
            AsyncMock(return_value={dep_id}),
        ):
            promoted = await refresh_blocked_dispatch_status(task, db)

        assert promoted is True
        assert task.dispatch_status == DispatchStatus.READY

    @pytest.mark.asyncio
    async def test_unfinished_predecessor_stays_blocked(self):
        dep_id = uuid.uuid4()
        task = _task("deploy")
        task.dispatch_status = DispatchStatus.BLOCKED
        db = self._db_with_deps(dep_id)

        with patch(
            "app.core.task_deps._finished_task_ids", AsyncMock(return_value=set())
        ):
            promoted = await refresh_blocked_dispatch_status(task, db)

        assert promoted is False
        assert task.dispatch_status == DispatchStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_ready_task_is_left_alone(self):
        task = _task("deploy")
        task.dispatch_status = DispatchStatus.READY
        db = self._db_with_deps()

        assert await refresh_blocked_dispatch_status(task, db) is False
        db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests — try_unblock_tasks
# ---------------------------------------------------------------------------


class TestTryUnblockTasksUnit:
    """try_unblock_tasks with mocked database."""

    @pytest.mark.asyncio
    async def test_two_unfinished_deps_nothing_unblocked(self):
        """Both predecessors still running → blocked task stays blocked (empty result)."""
        db = AsyncMock()
        empty = MagicMock()
        empty.fetchall.return_value = []
        db.execute = AsyncMock(return_value=empty)

        result = await try_unblock_tasks(uuid.uuid4(), db)

        assert result == []
        db.execute.assert_called_once()  # only the main SQL, no UPDATE/SELECT

    @pytest.mark.asyncio
    async def test_last_dep_completed_task_is_returned(self):
        """When the last predecessor finishes, the blocked task is returned."""
        blocked_id = uuid.uuid4()
        task_obj = MagicMock()
        task_obj.id = blocked_id

        db = AsyncMock()
        db.flush = AsyncMock()

        main_result = MagicMock()
        main_result.fetchall.return_value = [(str(blocked_id),)]

        update_result = MagicMock()

        select_result = MagicMock()
        select_result.scalars.return_value.all.return_value = [task_obj]

        db.execute = AsyncMock(side_effect=[main_result, update_result, select_result])

        result = await try_unblock_tasks(uuid.uuid4(), db)

        assert len(result) == 1
        assert result[0].id == blocked_id

    @pytest.mark.asyncio
    async def test_no_dependent_tasks_returns_empty(self):
        """A completed task that nobody depends on produces an empty result."""
        db = AsyncMock()
        empty = MagicMock()
        empty.fetchall.return_value = []
        db.execute = AsyncMock(return_value=empty)

        result = await try_unblock_tasks(uuid.uuid4(), db)

        assert result == []
        # No UPDATE or SELECT should be issued after the main query
        assert db.execute.call_count == 1


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    """Async SQLAlchemy session wired to the test PostgreSQL database."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    # Register all models so SQLModel.metadata is complete
    import app.models.session       # noqa: F401
    import app.models.goal          # noqa: F401
    import app.models.goal_plan     # noqa: F401
    import app.models.task          # noqa: F401
    import app.models.project_memory  # noqa: F401
    import app.models.agent_memory  # noqa: F401
    import app.models.message       # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


async def _insert_fixtures(db, room_id: uuid.UUID, user_id: uuid.UUID, agent_id: uuid.UUID) -> None:
    """Insert minimal room / user / agent rows for FK constraints."""
    from sqlalchemy import text

    await db.execute(
        text(
            'INSERT INTO "user" (id, email, hashed_password, is_active, two_factor_enabled) '
            "VALUES (:id, :email, 'x', true, false) ON CONFLICT DO NOTHING"
        ),
        {"id": str(user_id), "email": f"{user_id}@test.com"},
    )
    await db.execute(
        text(
            "INSERT INTO room (id, name, type, room_type, has_coordinator, creator_id, state) "
            "VALUES (:id, 'test', 'group', 'group', false, :cid, 'PLANNING') ON CONFLICT DO NOTHING"
        ),
        {"id": str(room_id), "cid": str(user_id)},
    )
    await db.execute(
        text(
            "INSERT INTO agent (id, nickname, agent_uid, agent_type, status, version, "
            "update_policy, short_description, owner_id, token) "
            "VALUES (:id, 'a', :uid, 'internal', 'offline', '1', 'manual', '', :oid, :tok) "
            "ON CONFLICT DO NOTHING"
        ),
        {
            "id": str(agent_id),
            "uid": f"test-{agent_id}",
            "oid": str(user_id),
            "tok": f"tok-{agent_id}",
        },
    )
    await db.flush()


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestDevTestDeployDB:
    """Full dev → test → deploy pipeline — integration scenarios."""

    async def _setup_pipeline(self, db):
        """Helper: create dev/test/deploy tasks with deps; return (room_id, dev, test, deploy)."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db, room_id, user_id, agent_id)

        dev = Task(room_id=room_id, title="dev")
        test = Task(room_id=room_id, title="test")
        deploy = Task(room_id=room_id, title="deploy")
        for t in (dev, test, deploy):
            db.add(t)

        await create_tasks_with_deps(
            [("dev", [], dev), ("test", ["dev"], test), ("deploy", ["test"], deploy)],
            db,
        )
        return room_id, dev, test, deploy

    async def test_initial_dispatch_statuses(self, db_session):
        """After batch creation: dev=READY, test=BLOCKED, deploy=BLOCKED."""
        _, dev, test, deploy = await self._setup_pipeline(db_session)
        await db_session.commit()

        assert dev.dispatch_status == DispatchStatus.READY
        assert test.dispatch_status == DispatchStatus.BLOCKED
        assert deploy.dispatch_status == DispatchStatus.BLOCKED

    async def test_approve_dev_unblocks_test_only(self, db_session):
        """Approving dev must unblock test; deploy stays BLOCKED."""
        _, dev, test, deploy = await self._setup_pipeline(db_session)

        dev.state = TaskState.APPROVED
        db_session.add(dev)
        await db_session.flush()

        newly_ready = await try_unblock_tasks(dev.id, db_session)
        await db_session.commit()

        ready_ids = {t.id for t in newly_ready}
        assert test.id in ready_ids, "test must be unblocked when dev is approved"
        assert deploy.id not in ready_ids, "deploy must stay blocked until test is done"

        await db_session.refresh(test)
        await db_session.refresh(deploy)
        assert test.dispatch_status == DispatchStatus.READY
        assert deploy.dispatch_status == DispatchStatus.BLOCKED

    async def test_approve_test_unblocks_deploy(self, db_session):
        """Approving test (after dev) must unblock deploy."""
        _, dev, test, deploy = await self._setup_pipeline(db_session)

        # Approve dev → unblock test
        dev.state = TaskState.APPROVED
        db_session.add(dev)
        await db_session.flush()
        await try_unblock_tasks(dev.id, db_session)

        # Approve test → unblock deploy
        test.state = TaskState.APPROVED
        db_session.add(test)
        await db_session.flush()
        newly_ready = await try_unblock_tasks(test.id, db_session)
        await db_session.commit()

        ready_ids = {t.id for t in newly_ready}
        assert deploy.id in ready_ids, "deploy must be unblocked after test is approved"

        await db_session.refresh(deploy)
        assert deploy.dispatch_status == DispatchStatus.READY

    async def test_approve_dev_does_not_affect_deploy_directly(self, db_session):
        """Approving dev must not unblock deploy (it still depends on test)."""
        _, dev, test, deploy = await self._setup_pipeline(db_session)

        dev.state = TaskState.APPROVED
        db_session.add(dev)
        await db_session.flush()
        await try_unblock_tasks(dev.id, db_session)
        await db_session.commit()

        await db_session.refresh(deploy)
        assert deploy.dispatch_status == DispatchStatus.BLOCKED


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestDependencyFailureDB:
    """get_blocked_dependents returns correct downstream tasks on predecessor failure."""

    async def test_reject_dev_exposes_test_as_direct_dependent(self, db_session):
        """After dev failure: test is a direct dependent; deploy is not (it depends on test)."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        dev = Task(room_id=room_id, title="dev")
        test = Task(room_id=room_id, title="test")
        deploy = Task(room_id=room_id, title="deploy")
        for t in (dev, test, deploy):
            db_session.add(t)

        await create_tasks_with_deps(
            [("dev", [], dev), ("test", ["dev"], test), ("deploy", ["test"], deploy)],
            db_session,
        )
        await db_session.commit()

        blocked = await get_blocked_dependents(dev.id, db_session)
        blocked_ids = {t.id for t in blocked}

        assert test.id in blocked_ids, "test is a direct dependent of dev"
        assert deploy.id not in blocked_ids, "deploy depends on test, not directly on dev"

    async def test_reject_test_exposes_deploy(self, db_session):
        """When test fails (after dev approved), deploy is returned as its blocked dependent."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        dev = Task(room_id=room_id, title="dev")
        test = Task(room_id=room_id, title="test")
        deploy = Task(room_id=room_id, title="deploy")
        for t in (dev, test, deploy):
            db_session.add(t)

        await create_tasks_with_deps(
            [("dev", [], dev), ("test", ["dev"], test), ("deploy", ["test"], deploy)],
            db_session,
        )

        # Approve dev → test becomes READY, deploy stays BLOCKED
        dev.state = TaskState.APPROVED
        db_session.add(dev)
        await db_session.flush()
        await try_unblock_tasks(dev.id, db_session)
        await db_session.commit()

        # test is now READY but not yet approved — deploy still BLOCKED, depends on test
        blocked = await get_blocked_dependents(test.id, db_session)
        blocked_ids = {t.id for t in blocked}

        assert deploy.id in blocked_ids, "deploy must be returned as blocked dependent of test"

    async def test_no_dependents_returns_empty(self, db_session):
        """A task that nothing depends on returns an empty list."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        solo = Task(room_id=room_id, title="solo")
        db_session.add(solo)
        await create_tasks_with_deps([(None, [], solo)], db_session)
        await db_session.commit()

        blocked = await get_blocked_dependents(solo.id, db_session)
        assert blocked == []


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestNoDepRegressionDB:
    """Regression: tasks without dependencies must all remain READY (parallel dispatch)."""

    async def test_independent_tasks_all_ready(self, db_session):
        """Three tasks with no deps are all READY — parallel dispatch is unaffected."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        tasks = [Task(room_id=room_id, title=f"task-{i}") for i in range(3)]
        for t in tasks:
            db_session.add(t)

        result = await create_tasks_with_deps(
            [(f"t{i}", [], t) for i, t in enumerate(tasks)],
            db_session,
        )
        await db_session.commit()

        assert len(result) == 3
        for t in tasks:
            assert t.dispatch_status == DispatchStatus.READY

    async def test_single_task_no_deps_is_ready(self, db_session):
        """A single task with no deps is READY."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        t = Task(room_id=room_id, title="solo")
        db_session.add(t)
        result = await create_tasks_with_deps([(None, [], t)], db_session)
        await db_session.commit()

        assert len(result) == 1
        assert result[0].dispatch_status == DispatchStatus.READY

    async def test_existing_tasks_unblocked_immediately_on_dep_completion(self, db_session):
        """Complete end-to-end: two independent roots both dispatch, then their shared child."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        await _insert_fixtures(db_session, room_id, user_id, agent_id)

        root_a = Task(room_id=room_id, title="root-a")
        root_b = Task(room_id=room_id, title="root-b")
        child = Task(room_id=room_id, title="child")
        for t in (root_a, root_b, child):
            db_session.add(t)

        await create_tasks_with_deps(
            [("a", [], root_a), ("b", [], root_b), ("c", ["a", "b"], child)],
            db_session,
        )

        assert root_a.dispatch_status == DispatchStatus.READY
        assert root_b.dispatch_status == DispatchStatus.READY
        assert child.dispatch_status == DispatchStatus.BLOCKED

        # Complete root_a — child still blocked (root_b not done)
        root_a.state = TaskState.APPROVED
        db_session.add(root_a)
        await db_session.flush()
        partial = await try_unblock_tasks(root_a.id, db_session)
        assert child.id not in {t.id for t in partial}

        await db_session.refresh(child)
        assert child.dispatch_status == DispatchStatus.BLOCKED

        # Complete root_b — now child should be unblocked
        root_b.state = TaskState.APPROVED
        db_session.add(root_b)
        await db_session.flush()
        newly_ready = await try_unblock_tasks(root_b.id, db_session)
        await db_session.commit()

        assert child.id in {t.id for t in newly_ready}
        await db_session.refresh(child)
        assert child.dispatch_status == DispatchStatus.READY
