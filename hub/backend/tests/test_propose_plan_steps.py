"""propose_plan `plan_steps` → GoalPlanStep.assigned_agent_id → task assignee.

Before this, `propose_plan` carried only goal_title/goal_description; the
per-step assignee ChatMaster named ("Task for @Writer…") lived only as
prose and was dropped when the plan pipeline built its single synthetic step,
so every auto-planned task came out unassigned.

All DB access is stubbed; no PostgreSQL.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.user_sessions import (
    _extract_propose_plan_from_text,
    _plan_steps_from_proposal,
    _resolve_room_agent,
    _trigger_plan_pipeline,
)


# ---------------------------------------------------------------------------
# _resolve_room_agent — @nickname / UUID → room agent id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveRoomAgent:
    def _db(self, rows):
        db = AsyncMock()
        result = MagicMock()
        result.all.return_value = rows
        db.execute = AsyncMock(return_value=result)
        return db

    async def test_empty_token_is_none_without_query(self):
        db = AsyncMock()
        db.execute = AsyncMock()
        assert await _resolve_room_agent("", uuid.uuid4(), db) is None
        assert await _resolve_room_agent("  @  ", uuid.uuid4(), db) is None
        db.execute.assert_not_called()

    async def test_matches_nickname_case_insensitively_and_strips_at(self):
        aid = uuid.uuid4()
        db = self._db([(aid, "Writer"), (uuid.uuid4(), "QA")])
        assert await _resolve_room_agent("@writer", uuid.uuid4(), db) == aid
        assert await _resolve_room_agent("Writer", uuid.uuid4(), db) == aid

    async def test_matches_uuid_only_within_room(self):
        aid = uuid.uuid4()
        db = self._db([(aid, "dev")])
        assert await _resolve_room_agent(str(aid), uuid.uuid4(), db) == aid
        assert await _resolve_room_agent(str(uuid.uuid4()), uuid.uuid4(), db) is None

    async def test_unknown_nickname_is_none(self):
        db = self._db([(uuid.uuid4(), "dev")])
        assert await _resolve_room_agent("@ghost", uuid.uuid4(), db) is None


# ---------------------------------------------------------------------------
# _plan_steps_from_proposal — normalization
# ---------------------------------------------------------------------------


class TestPlanStepsFromProposal:
    def test_tool_shape_passthrough(self):
        out = _plan_steps_from_proposal(
            {
                "plan_steps": [
                    {"title": "Write README section", "description": "d", "assignee": "@Writer"},
                    {"title": "Verify build", "assignee": ""},
                ]
            }
        )
        assert out == [
            {"title": "Write README section", "description": "d", "assignee": "@Writer"},
            {"title": "Verify build", "description": "", "assignee": ""},
        ]

    def test_missing_or_wrong_type_yields_empty(self):
        assert _plan_steps_from_proposal({}) == []
        assert _plan_steps_from_proposal({"plan_steps": "nope"}) == []
        assert _plan_steps_from_proposal({"plan_steps": [123, None]}) == []

    def test_alias_keys_and_bare_string(self):
        out = _plan_steps_from_proposal(
            {"plan_steps": ["Just a title", {"name": "Step 2", "agent": "qa"}]}
        )
        assert out[0] == {"title": "Just a title", "description": "", "assignee": ""}
        assert out[1] == {"title": "Step 2", "description": "", "assignee": "qa"}

    def test_assigned_agent_id_key_is_accepted(self):
        aid = str(uuid.uuid4())
        out = _plan_steps_from_proposal(
            {"plan_steps": [{"title": "x", "assigned_agent_id": aid}]}
        )
        assert out == [{"title": "x", "description": "", "assignee": aid}]


# ---------------------------------------------------------------------------
# _extract_propose_plan_from_text — XML <plan_steps>
# ---------------------------------------------------------------------------


class TestExtractXmlPlanSteps:
    def test_parses_steps_with_assignees(self):
        text = (
            "Here is the plan.\n"
            "<propose_plan>\n"
            "<goal_title>README git instructions</goal_title>\n"
            "<goal_description>A section about the first pull with a token.</goal_description>\n"
            "<plan_summary>Task for @Writer</plan_summary>\n"
            "<plan_steps>\n"
            "<step><title>Write the section</title><description>with a git remote -v check</description><assignee>@Writer</assignee></step>\n"
            "<step><title>Check the build</title><description>nothing broke</description><assignee></assignee></step>\n"
            "</plan_steps>\n"
            "</propose_plan>\n"
        )
        data = _extract_propose_plan_from_text(text)
        assert data is not None
        assert data["goal_title"] == "README git instructions"
        assert data["plan_steps"] == [
            {"title": "Write the section", "description": "with a git remote -v check", "assignee": "@Writer"},
            {"title": "Check the build", "description": "nothing broke", "assignee": ""},
        ]

    def test_no_plan_steps_block_is_empty_list(self):
        text = (
            "<propose_plan>\n"
            "<goal_title>T</goal_title>\n"
            "<goal_description>D</goal_description>\n"
            "</propose_plan>"
        )
        data = _extract_propose_plan_from_text(text)
        assert data is not None
        assert data["plan_steps"] == []

    def test_agent_tag_alias(self):
        text = (
            "<propose_plan><goal_title>T</goal_title><goal_description>D</goal_description>"
            "<plan_steps><step><title>A</title><agent>@dev</agent></step></plan_steps>"
            "</propose_plan>"
        )
        data = _extract_propose_plan_from_text(text)
        assert data["plan_steps"] == [{"title": "A", "description": "", "assignee": "@dev"}]


# ---------------------------------------------------------------------------
# _trigger_plan_pipeline — steps become GoalPlanStep rows with assignees
# ---------------------------------------------------------------------------


def _goal_row(title: str) -> MagicMock:
    g = MagicMock()
    g.id = uuid.uuid4()
    g.title = title
    g.description = "goal desc"
    return g


@pytest.mark.asyncio
class TestTriggerPlanPipelineSteps:
    async def _run(self, plan_steps, resolver):
        from app.core.session_close import GoalCreate, SessionCloseActions

        goal = _goal_row("README git instructions")
        actions = SessionCloseActions(
            goals_create=[GoalCreate(title=goal.title, description="")]
        )

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.commit = AsyncMock()

        goal_result = MagicMock()
        goal_result.scalars.return_value.first.return_value = goal
        names_result = MagicMock()
        names_result.all.return_value = [
            (aid, nick) for nick, aid in resolver.items()
        ]
        # The plan text also carries the goal's requirements digest
        # (internal design notes Stage 4); this goal has none.
        requirements_result = MagicMock()
        requirements_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=[goal_result, names_result, requirements_result])

        added_steps: list = []
        real_add = db.add

        def _capture(obj):
            if type(obj).__name__ == "GoalPlanStep":
                added_steps.append(obj)
            return real_add(obj)

        db.add = MagicMock(side_effect=_capture)

        async def _fake_resolver(token, room_id, _db):
            return resolver.get((token or "").lstrip("@"))

        with (
            patch("app.core.user_sessions._resolve_room_agent", side_effect=_fake_resolver),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.plan_session.dispatch_plan_event") as dispatch,
        ):
            await _trigger_plan_pipeline(
                uuid.uuid4(), actions, db, plan_steps=plan_steps
            )

        return added_steps, dispatch

    async def test_steps_carry_resolved_assignee(self):
        writer_id = uuid.uuid4()
        steps, dispatch = await self._run(
            [
                {"title": "Write the section", "description": "d", "assignee": "@Writer"},
                {"title": "Check the build", "description": "", "assignee": ""},
            ],
            {"Writer": writer_id},
        )
        assert [s.title for s in steps] == ["Write the section", "Check the build"]
        assert steps[0].assigned_agent_id == writer_id
        assert steps[1].assigned_agent_id is None
        # plan text handed to the decomposition session names the agent
        plan_text = dispatch.call_args.args[3]
        assert "@Writer" in plan_text

    async def test_unknown_assignee_leaves_step_unassigned(self):
        steps, _ = await self._run(
            [{"title": "X", "description": "", "assignee": "@ghost"}],
            {},
        )
        assert steps[0].assigned_agent_id is None

    async def test_no_steps_falls_back_to_single_step(self):
        steps, _ = await self._run(None, {})
        assert len(steps) == 1
        assert steps[0].title == "README git instructions"
        assert steps[0].assigned_agent_id is None
