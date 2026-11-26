"""A looping planner must not inflate the goal with its own repeats.

Incident 2026-11-25 (room Mylegion-Main): DeepSeek left the propose_plan
``goal_description`` argument unclosed and kept writing — reasoning, the plan
in XML form four more times, the chat history.  The goal grew to 134k chars,
the fallback plan step copied it, and the executor's first prompt was 130k
tokens against a 100k window.

No database access.
"""
from __future__ import annotations

from app.core.memory_layer import build_system_prompt
from app.core.user_sessions import (
    _extract_propose_plan_from_text,
    _repair_tool_plan,
)

CLEAN = "Rewrite HomePage.vue for the new copy. No deploy."


def _xml_plan(desc: str, *, closed: bool = True) -> str:
    tail = (
        "</goal_description><plan_steps><step><title>Rewrite</title>"
        "<description>Do it</description><assignee>@Paul</assignee></step>"
        "</plan_steps></propose_plan>"
    )
    return (
        "<propose_plan><goal_title>Landing</goal_title><goal_description>"
        + desc
        + (tail if closed else "")
    )


class TestRepairToolPlan:
    def test_clean_plan_is_untouched(self):
        plan = {"goal_title": "Landing", "goal_description": CLEAN}
        assert _repair_tool_plan(plan) is plan

    def test_leaked_reply_is_replaced_by_finished_xml_plan(self):
        leaked = (
            CLEAN + " draft one\n</think>I'll re-output the plan.\n"
            + _xml_plan("draft two", closed=False)
            + "\n</think>Once more:\n"
            + _xml_plan(CLEAN)
            + "\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"close_session\">"
        )
        repaired = _repair_tool_plan(
            {"goal_title": "Landing", "goal_description": leaked, "plan_steps": []}
        )
        assert repaired["goal_description"] == CLEAN
        assert [s["title"] for s in repaired["plan_steps"]] == ["Rewrite"]

    def test_leak_without_xml_plan_is_cut_at_first_marker(self):
        leaked = CLEAN + "\n</think>reasoning that never ends"
        repaired = _repair_tool_plan({"goal_title": "Landing", "goal_description": leaked})
        assert repaired["goal_description"] == CLEAN


class TestTextPlanTakesLastFinishedBlock:
    def test_unclosed_first_block_does_not_swallow_the_second(self):
        text = (
            _xml_plan("first draft", closed=False)
            + "\n</think>retry\n"
            + _xml_plan(CLEAN)
        )
        data = _extract_propose_plan_from_text(text)
        assert data is not None
        assert data["goal_description"] == CLEAN


class TestPlanStepCopyOfGoalIsNotRepeated:
    def _layer(self, step_desc: str) -> dict:
        return {
            "goal": {"id": "g", "title": "Landing", "description": CLEAN * 20, "status": "ACTIVE"},
            "plan_of_goal": {
                "id": "p",
                "status": "APPROVED",
                "steps": [{
                    "position": 0,
                    "title": "Landing",
                    "description": step_desc,
                    "status": "PENDING",
                    "assigned_agent_id": None,
                    "is_parallel": False,
                }],
            },
        }

    def test_fallback_step_is_rendered_by_reference(self):
        prompt = build_system_prompt(self._layer(CLEAN * 20), "cm_agent_proxy_agent")
        assert prompt.count(CLEAN * 20) == 1
        assert "same as the goal description above" in prompt

    def test_own_step_description_is_kept(self):
        prompt = build_system_prompt(self._layer("Only the hero"), "cm_agent_proxy_agent")
        assert "Only the hero" in prompt
