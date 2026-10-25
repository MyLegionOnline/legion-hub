from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools import _run_subtask as run_subtask_module


class _FakeDb:
    def __init__(self, session, agent):
        self.session = session
        self.agent = agent
        self.added: list = []

    async def get(self, model, _key):
        if getattr(model, "__name__", "") == "Session":
            return self.session
        if getattr(model, "__name__", "") == "Agent":
            return self.agent
        return None

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass


def _make_context(*, ant_tools=None, oai_tools=None, secret_value=object()):
    session_id = uuid4()
    task_id = uuid4()
    agent_id = uuid4()
    session = SimpleNamespace(
        id=session_id,
        kind="cm_agent_proxy",
        room_id=uuid4(),
        task_id=task_id,
        agent_id=agent_id,
    )
    agent = SimpleNamespace(
        id=agent_id,
        nickname="Worker",
        owner_id=None,
        llm_config={"provider": "anthropic", "model": "test-model"},
        tools_allowed=None,
    )
    db = _FakeDb(session, agent)
    ctx = ToolContext(
        room_id=session.room_id,
        session_id=session_id,
        task_id=task_id,
        db=db,
        llm_config=agent.llm_config,
        secret_value=secret_value,
        actor_agent_id=agent_id,
        actor_label="Worker",
        workspace_ant_tools=ant_tools,
        workspace_oai_tools=oai_tools,
    )
    return ctx, session, agent


@pytest.mark.asyncio
async def test_run_subtask_is_registered_for_agents_only():
    tool = REGISTRY.get("run_subtask")

    assert tool is not None
    assert tool.roles == frozenset({"agent"})
    assert "run_subtask" in {schema["name"] for schema in REGISTRY.ant_schemas_for("agent")}
    assert "run_subtask" not in {schema["name"] for schema in REGISTRY.ant_schemas_for("cm")}


@pytest.mark.asyncio
async def test_run_subtask_uses_clean_seed_and_returns_compact_close_report(monkeypatch):
    tool = REGISTRY.get("run_subtask")
    assert tool is not None
    ant_tools = [
        {"name": "read_file", "description": "read", "input_schema": {}},
        {"name": "run_subtask", "description": "recursive", "input_schema": {}},
    ]
    oai_tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "run_subtask", "parameters": {}}},
    ]
    ctx, _session, _agent = _make_context(ant_tools=ant_tools, oai_tools=oai_tools)
    async def _fake_llm(*_args, **kwargs):
        kwargs["usage_stats"]["tool_calls_used"] = 3
        return (
            None,
            {
                "summary": "The isolated step is complete.",
                "file_writes": [
                    {"path": "src/app.py", "content": "must not be returned"},
                    {"path": "src/app.py", "content": "duplicate"},
                ],
            },
            12,
            8,
            "test-model",
            100,
            80,
        )
    llm_mock = AsyncMock(side_effect=_fake_llm)
    monkeypatch.setattr("app.core.proxy_session._call_llm_with_tools", llm_mock)
    monkeypatch.setattr(
        "app.core.memory_layer.build_memory_layer",
        AsyncMock(return_value={"room_memory": "memory"}),
    )
    monkeypatch.setattr(
        "app.core.context_observations.build_task_read_manifest",
        AsyncMock(return_value=[
            {"path": "src/app.py", "ranges": [[1, 8]], "observation_refs": ["obs-1"]}
        ]),
    )
    apply_close_mock = AsyncMock()
    monkeypatch.setattr("app.core.session_close.apply_close_actions", apply_close_mock)

    result = await tool.execute(ctx, {
        "step_title": "Implement parser",
        "step_instructions": "Update the parser and run its focused tests.",
        "max_tool_calls": 80,
    })
    report = json.loads(result)

    assert report == {
        "step_title": "Implement parser",
        "status": "completed",
        "summary": "The isolated step is complete.",
        "file_writes": ["src/app.py"],
        "tokens_used": 20,
        "tokens_in": 12,
        "tokens_out": 8,
        "tool_calls_used": 3,
        "model": "test-model",
    }
    assert apply_close_mock.await_count == 1
    applied_session, applied_actions, applied_db = apply_close_mock.await_args.args
    assert applied_session is _session
    assert applied_db is ctx.db
    assert [write.path for write in applied_actions.file_writes] == [
        "src/app.py",
        "src/app.py",
    ]
    args = llm_mock.await_args.args
    kwargs = llm_mock.await_args.kwargs
    assert kwargs["max_tool_calls"] == 50
    assert kwargs["on_exhaust"] == "force_close"
    assert kwargs["context_role"] == "agent_subtask"
    assert kwargs["secret_value"] is ctx.secret_value
    assert args[2] == [{
        "role": "user",
        "content": (
            "Execute the isolated subtask 'Implement parser'.\n"
            "Instructions: Update the parser and run its focused tests."
        ),
    }]
    assert all(
        schema.get("name") != "run_subtask"
        and schema.get("function", {}).get("name") != "run_subtask"
        for schema in [*kwargs["workspace_ant_tools"], *kwargs["workspace_oai_tools"]]
    )
    assert "src/app.py" in args[1]
    assert "obs-1" in args[1]
    assert "Implement parser" in args[1]


@pytest.mark.asyncio
async def test_run_subtask_rejects_recursive_execution(monkeypatch):
    tool = REGISTRY.get("run_subtask")
    assert tool is not None
    ctx, _session, _agent = _make_context()
    key = str(ctx.session_id)
    run_subtask_module._SUBTASK_DEPTH[key] = 1
    try:
        result = await tool.execute(ctx, {
            "step_title": "Nested",
            "step_instructions": "Do not run another subtask.",
        })
    finally:
        run_subtask_module._SUBTASK_DEPTH.pop(key, None)

    report = json.loads(result)
    assert report["status"] == "failed"
    assert "depth" in report["error"]


@pytest.mark.asyncio
async def test_run_subtask_enforces_parent_session_limit(monkeypatch):
    tool = REGISTRY.get("run_subtask")
    assert tool is not None
    ctx, _session, _agent = _make_context()
    key = str(ctx.session_id)
    run_subtask_module._SUBTASK_CALLS.pop(key, None)
    monkeypatch.setattr(run_subtask_module, "MAX_SUBTASKS_PER_SESSION", 1)
    isolated_mock = AsyncMock(
        return_value=json.dumps({"status": "completed", "step_title": "First"})
    )
    monkeypatch.setattr(tool, "_execute_isolated", isolated_mock)

    try:
        first = json.loads(await tool.execute(ctx, {
            "step_title": "First",
            "step_instructions": "Do the first isolated step.",
        }))
        second = json.loads(await tool.execute(ctx, {
            "step_title": "Second",
            "step_instructions": "Do not start after the limit.",
        }))
    finally:
        run_subtask_module._SUBTASK_CALLS.pop(key, None)

    assert first["status"] == "completed"
    assert second["status"] == "failed"
    assert "run_subtask limit (1)" in second["error"]
    assert "execute them directly" in second["error"]
    assert isolated_mock.await_count == 1


@pytest.mark.asyncio
async def test_run_subtask_reports_missing_close_session(monkeypatch):
    tool = REGISTRY.get("run_subtask")
    assert tool is not None
    ctx, _session, _agent = _make_context()
    monkeypatch.setattr(
        "app.core.proxy_session._call_llm_with_tools",
        AsyncMock(return_value=("I stopped early.", None, 4, 6, "test-model", 20, 20)),
    )
    monkeypatch.setattr(
        "app.core.memory_layer.build_memory_layer",
        AsyncMock(return_value={}),
    )

    report = json.loads(await tool.execute(ctx, {
        "step_title": "Early stop",
        "step_instructions": "Make the change and close the subtask.",
    }))

    assert report["status"] == "failed"
    assert "without calling close_session" in report["error"]
    assert report["tokens_used"] == 10


# ---------------------------------------------------------------------------
# Stage 5.4 — the child gets its own agent_subtask bookkeeping Session row
# ---------------------------------------------------------------------------


class TestSubtaskBookkeepingSession:
    @pytest.mark.asyncio
    async def test_success_creates_and_closes_bookkeeping_row(self, monkeypatch):
        tool = REGISTRY.get("run_subtask")
        ctx, _session, _agent = _make_context()
        bookkeeping = SimpleNamespace(
            id=uuid4(), kind="agent_subtask", tokens_in=0, tokens_out=0,
            cost_usd=0, max_context_window_tokens=0, last_context_window_tokens=0,
        )
        repo_instance = AsyncMock()
        repo_instance.create = AsyncMock(return_value=bookkeeping)

        monkeypatch.setattr(
            "app.core.proxy_session._call_llm_with_tools",
            AsyncMock(return_value=(
                None,
                {"summary": "Done."},
                12, 8, "test-model", 100, 80,
            )),
        )
        monkeypatch.setattr(
            "app.core.memory_layer.build_memory_layer", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            "app.core.session_close.apply_close_actions", AsyncMock()
        )

        with patch(
            "app.repositories.session.SessionRepository",
            return_value=repo_instance,
        ) as repo_cls:
            report = json.loads(await tool.execute(ctx, {
                "step_title": "Implement parser",
                "step_instructions": "Update the parser.",
            }))

        assert report["status"] == "completed"
        repo_cls.assert_called_once_with(ctx.db)
        create_payload = repo_instance.create.await_args.args[0]
        assert create_payload.kind == "agent_subtask"
        assert create_payload.room_id == ctx.room_id
        assert create_payload.agent_id == ctx.actor_agent_id
        assert create_payload.task_id == ctx.task_id
        assert create_payload.parent_session_id == ctx.session_id
        assert create_payload.step_title == "Implement parser"

        assert bookkeeping.tokens_in == 12
        assert bookkeeping.tokens_out == 8
        assert bookkeeping.close_reason == "task_done"
        assert bookkeeping.closed_at is not None
        assert bookkeeping.message_count >= 1
        assert bookkeeping.turn_count == 1
        assert bookkeeping.summary == "Done."
        assert bookkeeping in ctx.db.added

    @pytest.mark.asyncio
    async def test_failure_still_closes_bookkeeping_row_as_error(self, monkeypatch):
        tool = REGISTRY.get("run_subtask")
        ctx, _session, _agent = _make_context()
        bookkeeping = SimpleNamespace(
            id=uuid4(), kind="agent_subtask", tokens_in=0, tokens_out=0,
            cost_usd=0, max_context_window_tokens=0, last_context_window_tokens=0,
        )
        repo_instance = AsyncMock()
        repo_instance.create = AsyncMock(return_value=bookkeeping)

        monkeypatch.setattr(
            "app.core.proxy_session._call_llm_with_tools",
            AsyncMock(return_value=("I stopped early.", None, 4, 6, "test-model", 20, 20)),
        )
        monkeypatch.setattr(
            "app.core.memory_layer.build_memory_layer", AsyncMock(return_value={})
        )

        with patch("app.repositories.session.SessionRepository", return_value=repo_instance):
            report = json.loads(await tool.execute(ctx, {
                "step_title": "Early stop",
                "step_instructions": "Make the change and close the subtask.",
            }))

        assert report["status"] == "failed"
        assert bookkeeping.close_reason == "error"
        assert bookkeeping.closed_at is not None

    @pytest.mark.asyncio
    async def test_bookkeeping_failure_does_not_break_the_subtask(self, monkeypatch):
        """SessionRepository.create raising must not affect the subtask's own result."""
        tool = REGISTRY.get("run_subtask")
        ctx, _session, _agent = _make_context()

        monkeypatch.setattr(
            "app.core.proxy_session._call_llm_with_tools",
            AsyncMock(return_value=(
                None, {"summary": "Done anyway."}, 12, 8, "test-model", 100, 80,
            )),
        )
        monkeypatch.setattr(
            "app.core.memory_layer.build_memory_layer", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            "app.core.session_close.apply_close_actions", AsyncMock()
        )

        with patch(
            "app.repositories.session.SessionRepository",
            side_effect=RuntimeError("db unavailable"),
        ):
            report = json.loads(await tool.execute(ctx, {
                "step_title": "Implement parser",
                "step_instructions": "Update the parser.",
            }))

        assert report["status"] == "completed"
        assert report["summary"] == "Done anyway."
