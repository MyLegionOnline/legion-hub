"""Run one agent sub-step in an isolated, bounded LLM context.

The child loop deliberately has no parent conversation history: it reuses the
parent session row as the correlation point for observations, context metrics
and the room's tool-call event feed, so the visible chat grouping is unchanged.
It additionally gets its own ``agent_subtask`` ``Session`` row (chatflow
session-lifecycle repair plan stage 5.4) purely for bookkeeping — tokens,
cost, model, duration — so that spend stops being folded anonymously into the
parent's totals. The parent still receives the compact tool report and still
applies any close_session actions to itself; the child row never receives
its own close actions.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import UUID

from app.core.events import emit_event
from app.core.subtask_activity import completion_metadata
from app.core.tools.registry import REGISTRY, Tool, ToolContext

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 25
MAX_MAX_TOOL_CALLS = 50
DEFAULT_MAX_SUBTASKS_PER_SESSION = 10


def _read_subtask_limit() -> int:
    """Read the per-parent-session subtask limit without breaking startup.

    ``PROXY_AGENT_MAX_SUBTASKS`` is the documented name.  The longer alias is
    accepted as well because it makes the unit explicit for deployments that
    already use ``*_MAX_TOOL_CALLS`` settings.  A malformed or negative value
    falls back to the safe default rather than making the backend fail during
    module import.
    """
    raw = os.getenv("PROXY_AGENT_MAX_SUBTASKS")
    if raw is None:
        raw = os.getenv("PROXY_AGENT_MAX_SUBTASK_CALLS")
    try:
        value = int(raw) if raw is not None else DEFAULT_MAX_SUBTASKS_PER_SESSION
    except (TypeError, ValueError):
        return DEFAULT_MAX_SUBTASKS_PER_SESSION
    return value if value >= 0 else DEFAULT_MAX_SUBTASKS_PER_SESSION


# This is intentionally read once per process, like the other proxy loop
# limits.  A restart applies an environment change to new sessions.
MAX_SUBTASKS_PER_SESSION = _read_subtask_limit()

# A process-local guard is defense in depth.  The child schema is filtered
# below, but a model can still hallucinate a tool name and the registry can be
# called directly by tests or another executor.
_SUBTASK_DEPTH: dict[str, int] = {}
# Counts accepted child-session launches, not currently-running children.  A
# failed child still consumed a bounded sub-session and therefore must count
# towards the parent guardrail.
_SUBTASK_CALLS: dict[str, int] = {}


def _json_report(**fields: Any) -> str:
    # Keep the usage shape stable for both successful and failed subtasks.  A
    # parent proxy can therefore roll up the child cost without having to
    # infer it from the human-readable summary.
    fields.setdefault("tokens_in", 0)
    fields.setdefault("tokens_out", 0)
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def _compact_paths(close_actions: Any) -> list[str]:
    """Return only file paths from parsed close actions.

    The child report must stay compact and must never copy file contents.  This
    accepts the parsed ``SessionCloseActions`` model so aliases such as
    ``files`` are normalized before the report is built.
    """
    raw_writes = getattr(close_actions, "file_writes", None)
    if raw_writes is None and isinstance(close_actions, dict):
        raw_writes = close_actions.get("file_writes")
    if not isinstance(raw_writes, list):
        return []
    paths: list[str] = []
    for item in raw_writes:
        path = item.get("path") if isinstance(item, dict) else getattr(item, "path", None)
        normalized_path = path.strip() if isinstance(path, str) else ""
        if normalized_path and normalized_path not in paths:
            paths.append(normalized_path)
    return paths


def _without_run_subtask(
    schemas: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return a shallow schema list without the recursive tool itself."""
    return [
        schema
        for schema in (schemas or [])
        if schema.get("name") != "run_subtask"
        and schema.get("function", {}).get("name") != "run_subtask"
    ]


def _registry_child_schemas(agent: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = getattr(agent, "tools_allowed", None)
    if allowed is not None and not isinstance(allowed, list):
        allowed = None
    if isinstance(allowed, list):
        allowed = [str(name) for name in allowed if str(name) != "run_subtask"]
    from app.core.tools.registry import strip_dialogue_only

    ant = strip_dialogue_only(REGISTRY.ant_schemas_for("agent", allowed))
    oai = strip_dialogue_only(REGISTRY.oai_schemas_for("agent", allowed))
    return _without_run_subtask(ant), _without_run_subtask(oai)


def _unpack_llm_result(result: Any) -> tuple[
    str | None,
    dict[str, Any] | None,
    int,
    int,
    str,
    int,
    int,
    int,
]:
    """Accept the current seven-field result and a future stats extension."""
    if not isinstance(result, (tuple, list)) or len(result) < 7:
        raise TypeError("_call_llm_with_tools returned an invalid result")
    text = result[0] if isinstance(result[0], str) else None
    close_input = result[1] if isinstance(result[1], dict) else None

    def _int_value(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    tokens_in = _int_value(result[2])
    tokens_out = _int_value(result[3])
    model = str(result[4] or "")
    max_window_in = _int_value(result[5])
    last_window_in = _int_value(result[6])
    tool_calls_used = 0
    if len(result) > 7:
        try:
            tool_calls_used = max(0, int(result[7]))
        except (TypeError, ValueError):
            tool_calls_used = 0
    return (
        text,
        close_input,
        tokens_in,
        tokens_out,
        model,
        max_window_in,
        last_window_in,
        tool_calls_used,
    )


class RunSubtaskTool(Tool):
    name = "run_subtask"
    description = (
        "Run one concrete sub-step of the current agent task in a fresh, bounded "
        "agent context. The child must complete only this step and call close_session."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "step_title": {
                "type": "string",
                "description": "Short name of the sub-step for logs and the result.",
            },
            "step_instructions": {
                "type": "string",
                "description": "Concrete instructions for this one sub-step.",
            },
            "max_tool_calls": {
                "type": "integer",
                "minimum": 1,
                "default": DEFAULT_MAX_TOOL_CALLS,
                "description": "Child tool-call budget; values above the hard cap are capped.",
            },
        },
        "required": ["step_title", "step_instructions"],
        "additionalProperties": False,
    }
    roles = frozenset({"agent"})
    requires_container = False
    mutating = False

    async def _finish(self, ctx: ToolContext, report: str) -> str:
        """Persist a compact completion marker when this call has UI context."""
        subtask = completion_metadata(ctx.subtask_activity, report)
        if subtask is None:
            return report
        try:
            status = subtask["status"]
            title = subtask["title"]
            who = ctx.actor_label or "Agent"
            content = (
                f"{who} finished the sub-step: {title}"
                if status == "completed"
                else f"{who} did not finish the sub-step: {title}"
            )
            await emit_event(
                room_id=ctx.room_id,
                event_type="AGENT_SUBTASK",
                content=content,
                db=ctx.db,
                task_id=ctx.task_id,
                metadata={
                    "session_id": str(ctx.session_id),
                    "agent_id": str(ctx.actor_agent_id) if ctx.actor_agent_id else None,
                    "nickname": ctx.actor_label,
                    "subtask": subtask,
                },
            )
        except Exception:
            logger.warning("run_subtask: could not emit lifecycle event", exc_info=True)
        return report

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        step_title = tool_input.get("step_title")
        step_instructions = tool_input.get("step_instructions")
        if not isinstance(step_title, str) or not step_title.strip():
            return await self._finish(ctx, "Error: step_title must be a non-empty string.")
        if not isinstance(step_instructions, str) or not step_instructions.strip():
            return await self._finish(ctx, "Error: step_instructions must be a non-empty string.")

        raw_max_calls = tool_input.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
        if isinstance(raw_max_calls, bool) or not isinstance(raw_max_calls, int):
            return await self._finish(ctx, "Error: max_tool_calls must be an integer.")
        if raw_max_calls < 1:
            return await self._finish(ctx, "Error: max_tool_calls must be at least 1.")
        max_tool_calls = min(raw_max_calls, MAX_MAX_TOOL_CALLS)

        depth_key = str(ctx.session_id)
        depth = _SUBTASK_DEPTH.get(depth_key, 0)
        if depth >= 1:
            return await self._finish(ctx, _json_report(
                step_title=step_title.strip(),
                status="failed",
                error="Nested run_subtask is not allowed; subtask depth is limited to 1.",
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            ))

        subtask_calls = _SUBTASK_CALLS.get(depth_key, 0)
        if subtask_calls >= MAX_SUBTASKS_PER_SESSION:
            return await self._finish(ctx, _json_report(
                step_title=step_title.strip(),
                status="failed",
                error=(
                    "The parent session reached its run_subtask limit "
                    f"({MAX_SUBTASKS_PER_SESSION}). Combine the remaining steps "
                    "and execute them directly without run_subtask."
                ),
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            ))

        _SUBTASK_CALLS[depth_key] = subtask_calls + 1
        _SUBTASK_DEPTH[depth_key] = depth + 1
        try:
            report = await self._execute_isolated(
                ctx,
                step_title=step_title.strip(),
                step_instructions=step_instructions.strip(),
                max_tool_calls=max_tool_calls,
            )
            return await self._finish(ctx, report)
        except Exception as exc:
            logger.exception("run_subtask failed for session %s", ctx.session_id)
            return await self._finish(ctx, _json_report(
                step_title=step_title.strip(),
                status="failed",
                error=f"Subtask could not be started: {exc}",
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            ))
        finally:
            if depth:
                _SUBTASK_DEPTH[depth_key] = depth
            else:
                _SUBTASK_DEPTH.pop(depth_key, None)

    async def _create_subtask_session(
        self, ctx: ToolContext, *, agent_id: UUID, step_title: str
    ) -> Any:
        """Best-effort: give this child loop its own Session row (stage 5.4).

        Returns None on failure — bookkeeping must never block the subtask.
        """
        from app.repositories.session import SessionRepository
        from app.schemas.session import AgentSubtaskSessionCreate

        try:
            return await SessionRepository(ctx.db).create(
                AgentSubtaskSessionCreate(
                    room_id=ctx.room_id,
                    agent_id=agent_id,
                    task_id=ctx.task_id,
                    parent_session_id=ctx.session_id,
                    step_title=step_title[:200],
                )
            )
        except Exception:
            logger.warning(
                "run_subtask: could not create subtask session row for parent %s",
                ctx.session_id, exc_info=True,
            )
            return None

    async def _close_subtask_session(
        self,
        subtask_session: Any,
        db: Any,
        *,
        tokens_in: int,
        tokens_out: int,
        model: str,
        llm_config: dict[str, Any] | None = None,
        max_window_in: int,
        last_window_in: int,
        tool_calls_used: int,
        status: str,
        summary: str,
    ) -> None:
        """Record final usage and close the child's bookkeeping row.

        Never raises: a failure here must not affect the subtask's own result.
        """
        if subtask_session is None:
            return
        from datetime import datetime, timezone

        from app.core.model_pricing import compute_cost as _compute_cost
        from app.core.token_budget import track_llm_usage, update_context_window_metrics

        try:
            cost = _compute_cost(model, tokens_in, tokens_out, llm_config)
            track_llm_usage(subtask_session, tokens_in, tokens_out, cost, model=model)
            update_context_window_metrics(subtask_session, max_window_in, last_window_in)
            # The child's own transcript is not persisted (no parent history is
            # replayed into it either) — message_count approximates the number
            # of provider round trips: one per tool call plus the final reply.
            subtask_session.message_count = max(1, tool_calls_used + 1)
            subtask_session.turn_count = 1
            subtask_session.closed_at = datetime.now(timezone.utc)
            subtask_session.close_reason = "task_done" if status == "completed" else "error"
            subtask_session.summary = summary[:4000] if summary else None
            db.add(subtask_session)
            await db.flush()
        except Exception:
            logger.warning(
                "run_subtask: could not close subtask session %s",
                getattr(subtask_session, "id", "?"), exc_info=True,
            )

    async def _execute_isolated(
        self,
        ctx: ToolContext,
        *,
        step_title: str,
        step_instructions: str,
        max_tool_calls: int,
    ) -> str:
        from app.core.context_observations import (
            build_task_read_manifest,
            format_read_manifest_note,
        )
        from app.core.memory_layer import build_memory_layer, build_system_prompt
        from app.core.proxy_session import _call_llm_with_tools
        from app.models.agent import Agent
        from app.models.session.base import Session

        parent_session = await ctx.db.get(Session, ctx.session_id)
        if parent_session is None:
            return _json_report(
                step_title=step_title,
                status="failed",
                error="Parent agent session was not found.",
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            )

        agent_id = ctx.actor_agent_id or getattr(parent_session, "agent_id", None)
        if agent_id is None:
            return _json_report(
                step_title=step_title,
                status="failed",
                error="The parent session has no agent identity.",
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            )
        agent = await ctx.db.get(Agent, agent_id)
        if agent is None:
            return _json_report(
                step_title=step_title,
                status="failed",
                error="The parent agent was not found.",
                tokens_used=0,
                tool_calls_used=0,
                file_writes=[],
            )

        memory = await build_memory_layer(parent_session, ctx.db)
        base_system_prompt = build_system_prompt(memory, "cm_agent_proxy_agent")
        manifest: list[dict[str, Any]] = []
        if ctx.task_id is not None:
            try:
                manifest = await build_task_read_manifest(ctx.db, task_id=ctx.task_id)
            except Exception:
                logger.warning(
                    "run_subtask: durable read manifest failed for task %s",
                    ctx.task_id,
                    exc_info=True,
                )
        manifest_note = format_read_manifest_note(manifest)

        system_prompt = (
            f"{base_system_prompt}\n\n"
            "[ISOLATED SUBTASK]\n"
            "You are executing exactly one sub-step of a larger agent task.\n"
            f"Step title: {step_title}\n"
            f"Step instructions: {step_instructions}\n"
            "Work only on this step, use the workspace tools as needed, and finish "
            "with close_session containing a concise summary."
        )
        if manifest_note:
            system_prompt += f"\n\n{manifest_note}"

        # This is the complete child seed.  In particular, do not append the
        # parent's provider messages or any prior tool results here.
        history = [{
            "role": "user",
            "content": (
                f"Execute the isolated subtask '{step_title}'.\n"
                f"Instructions: {step_instructions}"
            ),
        }]

        if ctx.workspace_ant_tools is None or ctx.workspace_oai_tools is None:
            workspace_ant_tools, workspace_oai_tools = _registry_child_schemas(agent)
        else:
            workspace_ant_tools = _without_run_subtask(ctx.workspace_ant_tools)
            workspace_oai_tools = _without_run_subtask(ctx.workspace_oai_tools)
        mcp_ant_tools = _without_run_subtask(ctx.mcp_ant_tools)
        mcp_oai_tools = _without_run_subtask(ctx.mcp_oai_tools)
        mcp_bindings = {
            name: binding
            for name, binding in (ctx.mcp_bindings or {}).items()
            if name != "run_subtask"
        }

        llm_config = dict(ctx.llm_config or getattr(agent, "llm_config", {}) or {})
        secret_value = ctx.secret_value
        if secret_value is None and getattr(agent, "owner_id", None) is not None:
            # Direct callers do not carry the parent loop's already-resolved
            # secret. Resolve it only as a fallback; proxy calls reuse the
            # short-lived value already present in ToolContext.
            from app.core.secrets.runtime import resolve_agent_secret

            secret_value = await resolve_agent_secret(
                db=ctx.db,
                owner_id=agent.owner_id,
                agent_id=agent.id,
                config=llm_config,
                request_id=ctx.task_id or ctx.session_id,
            )

        subtask_session = await self._create_subtask_session(
            ctx, agent_id=agent_id, step_title=step_title
        )
        tokens_in = tokens_out = tool_calls_used = 0
        max_window = last_window = 0
        model = ""
        session_status = "failed"
        session_summary = step_title
        try:
            usage_stats: dict[str, int] = {}
            result = await _call_llm_with_tools(
                llm_config,
                system_prompt,
                history,
                room_id=ctx.room_id,
                # The correlation point for observations/tool-call events/context
                # metrics stays the parent session — only the token/cost/duration
                # bookkeeping row above is new (stage 5.4).
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                db=ctx.db,
                # Inherit the parent's generation so the child loop stops at its
                # next boundary when the task is cancelled, blocked or retried.
                # Without it the subtask ran on past a Cancel the user pressed.
                run_id=ctx.run_id,
                workspace_ant_tools=workspace_ant_tools,
                workspace_oai_tools=workspace_oai_tools,
                mcp_ant_tools=mcp_ant_tools,
                mcp_oai_tools=mcp_oai_tools,
                mcp_bindings=mcp_bindings,
                mcp_user_id=ctx.mcp_user_id or getattr(agent, "owner_id", None),
                max_tool_calls=max_tool_calls,
                or_session_id=str(ctx.session_id),
                actor_agent_id=agent_id,
                actor_label=ctx.actor_label or getattr(agent, "nickname", None),
                on_exhaust="force_close",
                budget_used=0,
                context_role="agent_subtask",
                secret_value=secret_value,
                usage_stats=usage_stats,
                subtask_activity=ctx.subtask_activity,
            )
            text, close_input, tokens_in, tokens_out, model, max_window, last_window, tool_calls_used = (
                _unpack_llm_result(result)
            )
            tool_calls_used = max(
                tool_calls_used,
                max(0, int(usage_stats.get("tool_calls_used", 0) or 0)),
            )
            tokens_used = tokens_in + tokens_out

            if close_input is not None:
                # A child loop only returns the close_session payload; the
                # actual work — memory/task/file actions — is always applied to
                # the parent agent session through the same parser/filter/
                # workspace path used by ordinary proxy sessions. The child's
                # own Session row above never receives close actions; it only
                # records that this run happened and what it cost.
                from app.core.session_close import (
                    apply_close_actions,
                    extract_close_actions_from_tool_use,
                )

                try:
                    close_actions = extract_close_actions_from_tool_use(close_input)
                except Exception as exc:
                    logger.exception(
                        "run_subtask: invalid close_session input for session %s",
                        ctx.session_id,
                    )
                    session_summary = f"The subtask returned invalid close_session actions: {exc}"
                    return _json_report(
                        step_title=step_title,
                        status="failed",
                        error=session_summary,
                        tokens_used=tokens_used,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tool_calls_used=tool_calls_used,
                        model=model,
                        file_writes=[],
                    )

                try:
                    await apply_close_actions(parent_session, close_actions, ctx.db)
                except Exception as exc:
                    logger.exception(
                        "run_subtask: failed to apply close_session actions for session %s",
                        ctx.session_id,
                    )
                    session_summary = (
                        f"The subtask completed, but its close actions could not be applied: {exc}"
                    )
                    return _json_report(
                        step_title=step_title,
                        status="failed",
                        error=session_summary,
                        tokens_used=tokens_used,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tool_calls_used=tool_calls_used,
                        model=model,
                        file_writes=[],
                    )

                summary = close_actions.summary
                if not isinstance(summary, str) or not summary.strip():
                    summary = "Subtask returned close_session without a summary."
                session_status = "completed"
                session_summary = summary.strip()
                return _json_report(
                    step_title=step_title,
                    status="completed",
                    summary=session_summary,
                    file_writes=_compact_paths(close_actions),
                    tokens_used=tokens_used,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    tool_calls_used=tool_calls_used,
                    model=model,
                )

            failure = (
                "The subtask exhausted its bounded tool budget without a usable "
                "close_session report."
                if text is None
                else "The subtask returned text without calling close_session."
            )
            session_summary = failure
            return _json_report(
                step_title=step_title,
                status="failed",
                error=failure,
                tokens_used=tokens_used,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls_used=tool_calls_used,
                model=model,
                file_writes=[],
            )
        finally:
            await self._close_subtask_session(
                subtask_session,
                ctx.db,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model,
                llm_config=llm_config,
                max_window_in=max_window,
                last_window_in=last_window,
                tool_calls_used=tool_calls_used,
                status=session_status,
                summary=session_summary,
            )


REGISTRY.register(RunSubtaskTool())
