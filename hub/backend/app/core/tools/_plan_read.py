"""plan_read — read the current plan(s) for this room (read-only).

Available to both agent and CM roles.
Agents can read the plan to understand what is planned but cannot edit it
mid-session — plan edits go through close_session (CM only).
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class PlanReadTool(Tool):
    name = "plan_read"
    description = (
        "Return the plan(s) for this room: all goals, their plan status, "
        "and the ordered list of steps with status and assigned agent. "
        "Use this to understand what is planned and which steps are pending, "
        "in-progress, completed, or blocked. "
        "To modify the plan, use close_session (ChatMaster side only)."
    )
    input_schema = {"type": "object", "properties": {}}
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.models.goal import RoomGoal
        from app.models.goal_plan import GoalPlan, GoalPlanStep

        goals = (
            await ctx.db.execute(
                select(RoomGoal)
                .where(RoomGoal.room_id == ctx.room_id)
                .order_by(RoomGoal.created_at)
            )
        ).scalars().all()

        if not goals:
            return "No goals found for this room — no plan to read."

        result = []
        for goal in goals:
            plan = (
                await ctx.db.execute(
                    select(GoalPlan).where(GoalPlan.goal_id == goal.id)
                )
            ).scalars().first()
            if plan is None:
                continue

            steps = (
                await ctx.db.execute(
                    select(GoalPlanStep)
                    .where(GoalPlanStep.plan_id == plan.id)
                    .order_by(GoalPlanStep.position)
                )
            ).scalars().all()

            result.append({
                "goal_id": str(goal.id),
                "goal_title": goal.title,
                "plan_id": str(plan.id),
                "plan_status": plan.status.value,
                "steps": [
                    {
                        "step_id": str(s.id),
                        "title": s.title,
                        "status": s.status.value,
                        "assigned_agent_id": str(s.assigned_agent_id) if s.assigned_agent_id else None,
                        "position": s.position,
                    }
                    for s in steps
                ],
            })

        if not result:
            return "No plans found for this room's goals."

        return json.dumps(result, ensure_ascii=False, indent=2)


REGISTRY.register(PlanReadTool())
