"""
Tests for AGENT_INTERIM_TEXT: the text a model writes alongside its tool calls.

Regression: every tool loop kept that text only in its working history, so the
chat showed an agent's reasoning and tool calls but never its own words —
OpenRouter's log had "Timing has to match the real transients — checking." as
the completion of a step whose only trace in the Hub was a bash call.

Covers:
  1. emit_interim_text_event — event shape, bare nickname, empty/markup skip.
  2. Both proxy_session tool loops (Anthropic, OpenAI) emit the step's text,
     and a run_subtask child does not.
  3. Both user_sessions ws loops (Anthropic, OpenAI) emit the step's text.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.events import emit_interim_text_event

STEP_TEXT = "Timing has to match the real transients. Checking."


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    return db


def _anthropic_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _anthropic_client(*payloads: dict) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(side_effect=[_anthropic_response(p) for p in payloads])
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


_ANT_STEP = {
    "usage": {"input_tokens": 20, "output_tokens": 5},
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": STEP_TEXT},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
    ],
}
_ANT_FINAL = {
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "Done."}],
}


def _tool_call(call_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _openai_response(content: str | None, tool_calls=None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.reasoning = None
    message.reasoning_content = None
    message.reasoning_details = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    resp.usage.completion_tokens_details = None
    return resp


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        _openai_response(STEP_TEXT, tool_calls=[_tool_call("c1", "read_file", {"path": "a.py"})]),
        _openai_response("Done."),
    ])
    return client


def _registry() -> MagicMock:
    registry = MagicMock()
    registry.is_registered.return_value = True
    registry.get.return_value.execute = AsyncMock(return_value="print('a')")
    return registry


def _interim_texts(mock: AsyncMock) -> list[str]:
    return [c.kwargs["text"] for c in mock.await_args_list]


class TestEmitInterimTextEvent:
    @pytest.mark.asyncio
    async def test_persists_agent_words_with_bare_nickname(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        with patch("app.core.events.emit_event", new_callable=AsyncMock) as emit:
            await emit_interim_text_event(
                room_id=room_id, session_id=session_id, agent_id=agent_id,
                actor_label="@Videoagent", task_id=None, text=f"  {STEP_TEXT}\n",
                db=_make_db(),
            )
        emit.assert_awaited_once()
        kwargs = emit.await_args.kwargs
        assert kwargs["event_type"] == "AGENT_INTERIM_TEXT"
        assert kwargs["content"] == STEP_TEXT
        assert kwargs["sender_agent_id"] == agent_id
        assert kwargs["sender_type"] == "agent"
        assert kwargs["metadata"]["nickname"] == "Videoagent"
        assert kwargs["metadata"]["session_id"] == str(session_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [None, "", "   \n"])
    async def test_empty_text_emits_nothing(self, text):
        with patch("app.core.events.emit_event", new_callable=AsyncMock) as emit:
            await emit_interim_text_event(
                room_id=uuid.uuid4(), session_id=uuid.uuid4(), agent_id=None,
                actor_label=None, task_id=None, text=text, db=_make_db(),
            )
        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_leaked_tool_markup_is_not_published(self):
        leak = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">x</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        with patch("app.core.events.emit_event", new_callable=AsyncMock) as emit:
            await emit_interim_text_event(
                room_id=uuid.uuid4(), session_id=uuid.uuid4(), agent_id=None,
                actor_label=None, task_id=None, text=leak, db=_make_db(),
            )
        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_write_never_breaks_the_turn(self):
        with patch("app.core.events.emit_event", AsyncMock(side_effect=RuntimeError("db down"))):
            await emit_interim_text_event(
                room_id=uuid.uuid4(), session_id=uuid.uuid4(), agent_id=None,
                actor_label=None, task_id=None, text=STEP_TEXT, db=_make_db(),
            )


class TestProxyToolLoop:
    @pytest.mark.asyncio
    async def test_anthropic_step_text_is_emitted(self):
        from app.core.proxy_session import _call_llm_with_tools

        with (
            patch("app.core.proxy_session.httpx.AsyncClient",
                  return_value=_anthropic_client(_ANT_STEP, _ANT_FINAL)),
            patch("app.core.tools.REGISTRY", _registry()),
            patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_interim_text_event", new_callable=AsyncMock) as interim,
        ):
            text, *_ = await _call_llm_with_tools(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=_make_db(),
                workspace_ant_tools=[{"name": "read_file"}],
                actor_label="@Videoagent",
            )
        assert text == "Done."
        assert _interim_texts(interim) == [STEP_TEXT]
        assert interim.await_args.kwargs["actor_label"] == "@Videoagent"

    @pytest.mark.asyncio
    async def test_openai_step_text_is_emitted(self):
        from app.core.proxy_session import _call_llm_with_tools

        with (
            patch("app.core.proxy_session.AsyncOpenAI", return_value=_openai_client()),
            patch("app.core.tools.REGISTRY", _registry()),
            patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_interim_text_event", new_callable=AsyncMock) as interim,
        ):
            text, *_ = await _call_llm_with_tools(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=_make_db(),
                workspace_oai_tools=[
                    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
                ],
            )
        assert text == "Done."
        assert _interim_texts(interim) == [STEP_TEXT]

    @pytest.mark.asyncio
    async def test_subtask_child_keeps_its_words_out_of_the_chat(self):
        from app.core.proxy_session import _call_llm_with_tools

        with (
            patch("app.core.proxy_session.httpx.AsyncClient",
                  return_value=_anthropic_client(_ANT_STEP, _ANT_FINAL)),
            patch("app.core.tools.REGISTRY", _registry()),
            patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_interim_text_event", new_callable=AsyncMock) as interim,
        ):
            await _call_llm_with_tools(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=_make_db(),
                workspace_ant_tools=[{"name": "read_file"}],
                subtask_activity={"id": "sub-1", "title": "child"},
            )
        interim.assert_not_awaited()


class TestWsLoop:
    @pytest.mark.asyncio
    async def test_anthropic_step_text_is_emitted(self):
        from app.core.user_sessions import _call_with_ws_loop

        with (
            patch("app.core.user_sessions.httpx.AsyncClient",
                  return_value=_anthropic_client(_ANT_STEP, _ANT_FINAL)),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_status", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_interim_text_event", new_callable=AsyncMock) as interim,
            patch("app.core.tools.REGISTRY", _registry()),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                "user_agent_p2",
                uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), _make_db(),
                ant_ws_tools=[
                    {"name": "read_file", "description": "read", "input_schema": {"type": "object"}}
                ],
                oai_ws_tools=[],
            )
        assert result.text == "Done."
        assert _interim_texts(interim) == [STEP_TEXT]

    @pytest.mark.asyncio
    async def test_openai_step_text_is_emitted(self):
        from app.core.user_sessions import _call_with_ws_loop

        with (
            patch("app.core.user_sessions.AsyncOpenAI", return_value=_openai_client()),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_interim_text_event", new_callable=AsyncMock) as interim,
            patch("app.core.tools.REGISTRY", _registry()),
        ):
            result = await _call_with_ws_loop(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                "user_agent_p2",
                uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), _make_db(),
                ant_ws_tools=[],
                oai_ws_tools=[
                    {"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object"}}}
                ],
            )
        assert result.text == "Done."
        assert _interim_texts(interim) == [STEP_TEXT]
