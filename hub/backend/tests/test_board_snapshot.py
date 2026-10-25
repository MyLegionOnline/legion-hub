"""Stage 3 of internal design notes — Board snapshot.

The board is what the user sees on the "Board" tab; until this stage ChatMaster
could not see it at all (defect M1) and a p1 session saw ten flat tasks (M2).
These tests pin the three properties that make the brick usable as a *replacement*
for replaying the conversation:

  * it reproduces the UI grouping — goals → state rows → assigned agent;
  * blockers are named, with the same wording the board shows;
  * the rendered block never exceeds its token budget, however big the room is.

No database: the projector's queries are fed from a scripted fake.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.core.board_snapshot import (
    BOARD_SNAPSHOT_TOKEN_BUDGET,
    _counter,
    board_snapshot_enabled,
    build_board_snapshot,
    render_board_snapshot,
)
from app.models.goal import GoalStatus, RoomGoal
from app.models.goal_plan import GoalPlan, GoalPlanStatus, GoalPlanStep, PlanStepStatus
from app.models.task import DispatchStatus, Task, TaskState

_ROOM_ID = uuid.uuid4()
_FRONT = uuid.uuid4()
_BACK = uuid.uuid4()


# ---------------------------------------------------------------------------
# Scripted DB
# ---------------------------------------------------------------------------


class _Scalars:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return list(self._rows)


class _Result:
    """Either a scalars() result (ORM rows) or an iterable of tuples."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> _Scalars:
        return _Scalars(self._rows)

    def __iter__(self):
        return iter(self._rows)


def _db(*results: list[Any]) -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_Result(r) for r in results])
    return db


def _task(
    title: str,
    state: TaskState,
    *,
    goal_id: uuid.UUID | None = None,
    agent_id: uuid.UUID | None = None,
    dispatch: DispatchStatus = DispatchStatus.READY,
    retry: int = 0,
    minutes_ago: int = 0,
) -> Task:
    return Task(
        id=uuid.uuid4(),
        room_id=_ROOM_ID,
        goal_id=goal_id,
        title=title,
        state=state,
        assigned_agent_id=agent_id,
        dispatch_status=dispatch,
        retry_count=retry,
        updated_at=datetime.utcnow() - timedelta(minutes=minutes_ago),
    )


def _goal(title: str, status: GoalStatus = GoalStatus.ACTIVE, position: int = 0) -> RoomGoal:
    return RoomGoal(id=uuid.uuid4(), room_id=_ROOM_ID, title=title, status=status, position=position)


def _agent_rows(*pairs: tuple[uuid.UUID, str]) -> list[Any]:
    return [tuple(p) for p in pairs]


# ---------------------------------------------------------------------------
# A room shaped like the production room the plan was written from:
# 2 goals, 23 tasks, an ungrouped section, blockers.
# ---------------------------------------------------------------------------


def _busy_room() -> tuple[list[RoomGoal], list[Task], Task, Task]:
    devlog = _goal("DevLog blog", GoalStatus.COMPLETED, position=0)
    seo = _goal("Blog SEO optimization", GoalStatus.ACTIVE, position=1)

    needs_retry = _task(
        "Run the e2e tests", TaskState.BLOCKED, goal_id=seo.id, agent_id=_BACK, retry=1
    )
    waits_predecessor = _task(
        "Deploy to prod",
        TaskState.ASSIGNED,
        goal_id=seo.id,
        agent_id=_BACK,
        dispatch=DispatchStatus.BLOCKED,
    )

    tasks = [
        _task("Lay out the article template", TaskState.IN_PROGRESS, goal_id=seo.id, agent_id=_FRONT),
        _task("Add sitemap.xml", TaskState.ASSIGNED, goal_id=seo.id, agent_id=_BACK),
        needs_retry,
        waits_predecessor,
        _task("No goal: clean up the logs", TaskState.DRAFT),
        _task("No goal: update the README", TaskState.WAITING_REVIEW, agent_id=_FRONT),
    ]
    tasks += [
        _task(f"Finished task {i}", TaskState.COMPLETED, goal_id=devlog.id, agent_id=_FRONT)
        for i in range(14)
    ]
    tasks += [
        _task(f"Cancelled {i}", TaskState.CANCELLED, goal_id=devlog.id) for i in range(3)
    ]
    return [devlog, seo], tasks, needs_retry, waits_predecessor


async def _build_busy() -> dict[str, Any]:
    goals, tasks, needs_retry, waits = _busy_room()
    plan = GoalPlan(id=uuid.uuid4(), goal_id=goals[1].id, status=GoalPlanStatus.APPROVED)
    steps = [
        GoalPlanStep(
            id=uuid.uuid4(),
            plan_id=plan.id,
            title=f"Step {i}",
            status=PlanStepStatus.COMPLETED if i == 0 else PlanStepStatus.PENDING,
            position=i,
            assigned_agent_id=_FRONT,
        )
        for i in range(3)
    ]
    dep_rows = [(waits.id, needs_retry.id)]
    db = _db(
        goals,
        tasks,
        _agent_rows((_FRONT, "Frontender"), (_BACK, "Backender")),
        dep_rows,
        [plan],
        steps,
    )
    return await build_board_snapshot(_ROOM_ID, db)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    @pytest.mark.asyncio
    async def test_goals_keep_board_order_and_own_their_tasks(self):
        snapshot = await _build_busy()

        titles = [g["title"] for g in snapshot["goals"]]
        assert titles == ["DevLog blog", "Blog SEO optimization"]

        seo = snapshot["goals"][1]
        assert set(seo["columns"]) == {"assigned", "in_progress", "blocked"}
        assert [t["title"] for t in seo["columns"]["in_progress"]] == ["Lay out the article template"]
        assert seo["total"] == 4

    @pytest.mark.asyncio
    async def test_terminal_states_are_counted_not_listed(self):
        snapshot = await _build_busy()

        devlog = snapshot["goals"][0]
        assert devlog["columns"] == {}
        assert devlog["collapsed"] == {"completed": 14, "cancelled": 3}

    @pytest.mark.asyncio
    async def test_tasks_without_a_goal_have_their_own_section(self):
        snapshot = await _build_busy()

        without = snapshot["without_goal"]
        assert without["total"] == 2
        assert set(without["columns"]) == {"draft", "waiting_review"}

    @pytest.mark.asyncio
    async def test_assigned_agent_is_a_nickname_not_a_uuid(self):
        snapshot = await _build_busy()

        assigned = snapshot["goals"][1]["columns"]["assigned"]
        assert {t["agent"] for t in assigned} == {"Backender"}

    @pytest.mark.asyncio
    async def test_counters(self):
        snapshot = await _build_busy()

        counters = snapshot["counters"]
        assert counters["total"] == 23
        assert counters["completed"] == 14
        # blocked: state=blocked plus dispatch_status=blocked
        assert counters["blocked"] == 2
        assert counters["active"] == 23 - 17 - 2

    @pytest.mark.asyncio
    async def test_plan_of_a_goal_is_summarised(self):
        snapshot = await _build_busy()

        plan = snapshot["goals"][1]["plan"]
        assert plan["steps_total"] == 3
        assert plan["steps_done"] == 1
        assert plan["steps"][0]["agent"] == "Frontender"


# ---------------------------------------------------------------------------
# Blockers
# ---------------------------------------------------------------------------


class TestBlockers:
    @pytest.mark.asyncio
    async def test_blocked_task_says_retry(self):
        snapshot = await _build_busy()

        blocked = snapshot["goals"][1]["columns"]["blocked"][0]
        assert blocked["blocked_reason"] == "Not started — Retry"
        assert blocked["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_dependency_block_names_the_predecessor(self):
        snapshot = await _build_busy()

        waiting = next(
            t
            for t in snapshot["goals"][1]["columns"]["assigned"]
            if t["title"] == "Deploy to prod"
        )
        assert waiting["blocked_reason"] == "Waiting for predecessor"
        assert waiting["waiting_for"] == ["Run the e2e tests"]

    @pytest.mark.asyncio
    async def test_completed_predecessor_is_not_reported_as_waiting(self):
        goal = _goal("Goal")
        done = _task("Already done", TaskState.COMPLETED, goal_id=goal.id)
        follower = _task("Next", TaskState.ASSIGNED, goal_id=goal.id)
        # No assigned agents in this room, so the projector skips the agent
        # lookup: goals → tasks → dependencies → plans.
        db = _db([goal], [follower, done], [(follower.id, done.id)], [])

        snapshot = await build_board_snapshot(_ROOM_ID, db)

        assert snapshot["goals"][0]["columns"]["assigned"][0]["waiting_for"] == []

    @pytest.mark.asyncio
    async def test_blocked_summary_lists_every_blocker_once(self):
        snapshot = await _build_busy()

        reasons = {(b["title"], b["reason"]) for b in snapshot["blocked_summary"]}
        assert ("Run the e2e tests", "Not started — Retry") in reasons
        assert ("Deploy to prod", "Waiting for predecessor") in reasons


# ---------------------------------------------------------------------------
# Rendering and the token budget (risk 2 of the plan)
# ---------------------------------------------------------------------------


class TestRender:
    @pytest.mark.asyncio
    async def test_rendered_block_is_readable_and_within_budget(self):
        snapshot = await _build_busy()

        rendered = render_board_snapshot(snapshot)

        assert rendered.startswith("## Task Board")
        assert "### Goal: Blog SEO optimization" in rendered
        assert "### No goal" in rendered
        assert "### Blockers" in rendered
        assert "Not started — Retry" in rendered
        assert _counter.count(rendered) <= BOARD_SNAPSHOT_TOKEN_BUDGET

    @pytest.mark.asyncio
    async def test_a_huge_room_still_fits_and_says_what_it_hid(self):
        goal = _goal("Big goal")
        tasks = [
            _task(
                f"A very long task title number {i} with details, "
                f"which must not end up in the snapshot in full",
                TaskState.ASSIGNED,
                goal_id=goal.id,
                agent_id=_FRONT,
            )
            for i in range(200)
        ]
        db = _db([goal], tasks, _agent_rows((_FRONT, "Frontender")), [], [], [])

        snapshot = await build_board_snapshot(_ROOM_ID, db)
        rendered = render_board_snapshot(snapshot)

        assert _counter.count(rendered) <= BOARD_SNAPSHOT_TOKEN_BUDGET
        assert "showing" in rendered

    @pytest.mark.asyncio
    async def test_empty_room(self):
        db = _db([], [])

        snapshot = await build_board_snapshot(_ROOM_ID, db)
        rendered = render_board_snapshot(snapshot)

        assert snapshot["counters"]["total"] == 0
        assert "## Task Board" in rendered

    @pytest.mark.asyncio
    async def test_titles_are_clipped_at_projection_time(self):
        goal = _goal("Goal")
        long_title = "x" * 400
        db = _db([goal], [_task(long_title, TaskState.ASSIGNED, goal_id=goal.id)], [], [])

        snapshot = await build_board_snapshot(_ROOM_ID, db)

        stored = snapshot["goals"][0]["columns"]["assigned"][0]["title"]
        assert len(stored) < 100 and stored.endswith("…")
        assert long_title not in render_board_snapshot(snapshot)


# ---------------------------------------------------------------------------
# Memory-layer wiring
# ---------------------------------------------------------------------------


class TestMemoryLayerWiring:
    def test_user_cm_p2_prompt_contains_the_board(self):
        from app.core.memory_layer import build_system_prompt

        prompt = build_system_prompt(
            {
                "room_memory": "memory",
                "repo_snapshot": {"file_count": 0},
                "agents_list": [],
                "board_snapshot": {
                    "goals": [],
                    "without_goal": {"columns": {}, "collapsed": {}, "total": 0},
                    "blocked_summary": [],
                    "counters": {"total": 3, "active": 3, "blocked": 0, "completed": 0},
                    "scan_truncated": False,
                },
            },
            "user_cm_p2",
        )

        assert "## Task Board" in prompt
        # Board comes after the agents list and before any goal detail (3.3).
        assert prompt.index("## Agents in Room") < prompt.index("## Task Board")

    @pytest.mark.asyncio
    async def test_rollback_switch_removes_the_brick(self, monkeypatch):
        from app.core.memory_layer import _board_snapshot

        monkeypatch.setenv("MEMORY_BOARD_SNAPSHOT", "false")
        assert board_snapshot_enabled() is False
        assert await _board_snapshot(_ROOM_ID, AsyncMock()) is None

    @pytest.mark.asyncio
    async def test_a_broken_board_never_breaks_the_turn(self):
        from app.core.memory_layer import _board_snapshot

        with patch(
            "app.core.memory_layer.build_board_snapshot",
            new=AsyncMock(side_effect=RuntimeError("db is down")),
        ):
            assert await _board_snapshot(_ROOM_ID, AsyncMock()) is None
