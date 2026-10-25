"""
Tests for the tool-call-exhaustion fallback in direct user sessions.

Regression: after the workspace tool loop hit its ceiling, the forced text call
was made *without* tool schemas and with the pre-loop history.  A DeepSeek-class
model then re-ran its first exploratory tool call — and, with no schema for the
provider's parser to latch onto, emitted it as raw ``<｜DSML｜tool_calls>``
template markup, which was published into the room verbatim.

Covers:
  1. strip_tool_call_markup / clean_reply_text remove leaked template markup
     without eating ordinary prose.
  2. _forced_text_history carries this turn's tool results into the forced call.
  3. force_text keeps `tools` in the request and uses tool_choice=none instead.
  4. The whole exhaustion path end to end (Anthropic loop).
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.session_close import (
    TOOL_MARKUP_FALLBACK_TEXT,
    clean_reply_text,
    strip_tool_call_markup,
)
from app.core.user_sessions import (
    _FORCED_TEXT_INSTRUCTION,
    _TOOL_MIMIC_FALLBACK_TEXT,
    _TOOL_MIMIC_MAX_RETRIES,
    _WS_MAX_TOOL_CALLS,
    _call_llm_with_tools,
    _call_with_ws_loop,
    _forced_text_history,
    _looks_like_unexecuted_tool_call,
    _modernize_tool_notes,
    _strip_tool_call_mimic,
    _tool_note_text,
)

# The exact shape that reached a room: DeepSeek DSML with fullwidth pipes.
DSML_LEAK = (
    '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">\n'
    '<｜DSML｜parameter name="command" string="true">python3 -c "\n'
    "import pandas as pd\n"
    "df = pd.read_excel('/workspace/data.xlsx')\n"
    'print(df.dtypes)\n"</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="description" string="true">Read Excel file'
    "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
)


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
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


def _openai_response(content: str | None, tool_calls=None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


class TestStripToolCallMarkup:
    def test_strips_the_real_dsml_leak(self):
        assert strip_tool_call_markup(DSML_LEAK) == ("", True)

    def test_keeps_prose_written_before_the_leak(self):
        cleaned, leaked = strip_tool_call_markup("Portrait 1: 18% of the base.\n\n" + DSML_LEAK)
        assert leaked is True
        assert cleaned == "Portrait 1: 18% of the base."

    def test_cuts_an_unterminated_leak_at_the_tail(self):
        cleaned, leaked = strip_tool_call_markup(
            'Done. <｜DSML｜invoke name="bash">python3 -c ...'
        )
        assert (cleaned, leaked) == ("Done.", True)

    def test_strips_other_vendor_templates(self):
        assert strip_tool_call_markup("a <｜tool▁calls▁begin｜>x<｜tool▁calls▁end｜> b")[1]
        assert strip_tool_call_markup('a <tool_call>{"n":1}</tool_call> b')[1]
        assert strip_tool_call_markup("a <function_calls>x</function_calls> b")[1]

    @pytest.mark.parametrize(
        "text",
        [
            "Portrait 1: women aged 30–40 — 18% of the base, with 5 < 10 and a > b.",
            "Use `df.head()`, then <b>bold</b> text.",
            "```python\nif x < y and y > z:\n    pass\n```",
            "Comparison: 2<3, 4>1 — it all adds up.",
        ],
    )
    def test_no_false_positive_on_ordinary_text(self, text):
        assert strip_tool_call_markup(text) == (text.strip(), False)

    def test_empty_input(self):
        assert strip_tool_call_markup("") == ("", False)
        assert strip_tool_call_markup(None) == ("", False)

    def test_clean_reply_text_falls_back_when_nothing_but_markup(self):
        assert clean_reply_text(DSML_LEAK, where="t") == TOOL_MARKUP_FALLBACK_TEXT

    def test_clean_reply_text_passes_normal_replies_through(self):
        assert clean_reply_text("The final answer.", where="t") == "The final answer."


class TestForcedTextHistory:
    def _notes(self) -> list[dict[str, str]]:
        return [
            {"role": "assistant", "content": "[tool call] bash({...})"},
            {
                "role": "user",
                "content": "[tool results]\n\n### bash\nshape=(1200, 14)",
            },
        ]

    def test_carries_tool_results_into_the_forced_call(self):
        out = _forced_text_history([{"role": "user", "content": "analyze"}], self._notes())
        blob = "\n".join(m["content"] for m in out)
        assert "shape=(1200, 14)" in blob
        assert _FORCED_TEXT_INSTRUCTION in blob

    def test_roles_alternate(self):
        """Anthropic rejects two consecutive user messages."""
        out = _forced_text_history([{"role": "user", "content": "q"}], self._notes())
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert all(a["role"] != b["role"] for a, b in zip(out, out[1:]))

    def test_without_notes_still_appends_the_instruction(self):
        out = _forced_text_history([{"role": "user", "content": "q"}], [])
        assert len(out) == 1
        assert out[0]["content"].startswith("q")
        assert _FORCED_TEXT_INSTRUCTION in out[0]["content"]

    def test_does_not_mutate_caller_state(self):
        history = [{"role": "user", "content": "q"}]
        notes = self._notes()
        _forced_text_history(history, notes)
        assert history == [{"role": "user", "content": "q"}]
        assert notes[-1]["content"].endswith("shape=(1200, 14)")


class TestForceTextKeepsToolSchemas:
    """Dropping `tools` is what let raw template markup through — force_text
    must forbid tool *calls*, not remove the tool *schemas*."""

    @pytest.mark.asyncio
    async def test_openai_sends_tools_with_tool_choice_none(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_openai_response("Result."))
        with patch("app.core.user_sessions.AsyncOpenAI", return_value=client):
            result = await _call_llm_with_tools(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                matrix_key="user_cm_p2",
                force_text=True,
            )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"], "tool schemas must stay in the request"
        assert kwargs["tool_choice"] == "none"
        assert result.text == "Result."

    @pytest.mark.asyncio
    async def test_anthropic_sends_tools_with_tool_choice_none(self):
        response = _anthropic_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Result."}],
            }
        )
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx):
            result = await _call_llm_with_tools(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                matrix_key="user_cm_p2",
                force_text=True,
            )
        body = client.post.call_args.kwargs["json"]
        assert body["tools"], "tool schemas must stay in the request"
        assert body["tool_choice"] == {"type": "none"}
        assert result.text == "Result."

    @pytest.mark.asyncio
    async def test_markup_only_reply_is_retried_then_reported(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_openai_response(DSML_LEAK), _openai_response(DSML_LEAK)]
        )
        with patch("app.core.user_sessions.AsyncOpenAI", return_value=client):
            result = await _call_llm_with_tools(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                force_text=True,
            )
        assert client.chat.completions.create.await_count == 2, "must retry once"
        assert result.text == TOOL_MARKUP_FALLBACK_TEXT
        assert "DSML" not in result.text

    @pytest.mark.asyncio
    async def test_retry_succeeds_and_returns_prose(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[_openai_response(DSML_LEAK), _openai_response("The portraits are ready.")]
        )
        with patch("app.core.user_sessions.AsyncOpenAI", return_value=client):
            result = await _call_llm_with_tools(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                force_text=True,
            )
        assert result.text == "The portraits are ready."


class TestWsLoopExhaustionFallback:
    @pytest.mark.asyncio
    async def test_forced_call_sees_tool_results_and_keeps_tools(self):
        """The reproduction: every iteration calls bash, the ceiling is hit, and
        the forced text call must still carry the schemas *and* what bash found."""
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        db = _make_db()

        tool_use = _anthropic_response(
            {
                "usage": {"input_tokens": 20, "output_tokens": 5},
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "head x"}}
                ],
            }
        )
        final = _anthropic_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Portrait 1: 18% of the base."}],
            }
        )
        client = MagicMock()
        # 2 ceiling + 1 extra loop pass, then the forced text call.
        client.post = AsyncMock(side_effect=[tool_use, tool_use, tool_use, final])
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        registry = MagicMock()
        registry.is_registered.return_value = True
        registry.get.return_value.execute = AsyncMock(return_value="shape=(1200, 14)")

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions._WS_MAX_TOOL_CALLS", 2),
            patch("app.core.tools.REGISTRY", registry),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "build buyer portraits"}],
                "user_agent_p2",
                room_id, session_id, agent_id, db,
                ant_ws_tools=[
                    {"name": "bash", "description": "run", "input_schema": {"type": "object"}}
                ],
                oai_ws_tools=[],
            )

        assert client.post.await_count == 4
        forced_body = client.post.call_args_list[-1].kwargs["json"]
        assert forced_body["tools"], "forced call must keep tool schemas"
        assert forced_body["tool_choice"] == {"type": "none"}

        blob = "\n".join(
            m["content"] for m in forced_body["messages"] if isinstance(m.get("content"), str)
        )
        assert "shape=(1200, 14)" in blob, "forced call must see what the tools found"
        assert _FORCED_TEXT_INSTRUCTION in blob
        assert result.text == "Portrait 1: 18% of the base."

    @pytest.mark.asyncio
    async def test_ceiling_is_env_configurable_and_above_the_old_ten(self):
        assert _WS_MAX_TOOL_CALLS >= 30

        import importlib

        with patch.dict("os.environ", {"USER_SESSION_MAX_TOOL_CALLS": "7"}):
            module = importlib.reload(importlib.import_module("app.core.user_sessions"))
            assert module._WS_MAX_TOOL_CALLS == 7
        importlib.reload(module)


# Regression: internal design notes Stage 10 live run. A model resumed mid ws_loop
# (after an MCP-credit pause) and, instead of a real tool call, wrote out
# "[tool call] mcp__legion-vps-mcp__vps_exec({...})" as plain prose —
# the exact textual shape _note_iteration uses to compact prior tool calls
# into history. Unlike the DSML leak above, this is not template markup to
# strip; it never reached a real tool, so retrying with the wire's own
# tool_choice forced (not just asking again) is the only way to recover it.
def _tool_call(call_id: str, name: str, args: dict) -> MagicMock:
    import json as _json

    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = _json.dumps(args)
    return tc


class TestLooksLikeUnexecutedToolCall:
    def test_matches_the_observed_leak(self):
        assert _looks_like_unexecuted_tool_call(
            '[tool call] mcp__legion-vps-mcp__vps_exec({"command": '
            '["/usr/sbin/nginx","-s","reload"]})'
        )

    def test_matches_bare_mcp_call_syntax(self):
        assert _looks_like_unexecuted_tool_call('mcp__legion-vps-mcp__vps_service({"action":"reload"})')

    def test_does_not_flag_ordinary_prose(self):
        assert not _looks_like_unexecuted_tool_call("Portrait 1: 18% of the base, no tool call needed.")

    def test_empty_input(self):
        assert not _looks_like_unexecuted_tool_call("")
        assert not _looks_like_unexecuted_tool_call(None)


class TestToolCallMimicRetry:
    @pytest.mark.asyncio
    async def test_anthropic_retries_with_tool_choice_any_then_executes(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        db = _make_db()

        mimic_text = _anthropic_response(
            {
                "usage": {"input_tokens": 20, "output_tokens": 10},
                "stop_reason": "end_turn",
                "content": [{
                    "type": "text",
                    "text": '[tool call] mcp__legion-vps-mcp__vps_exec({"command":["/usr/sbin/nginx","-s","reload"]})',
                }],
            }
        )
        real_tool_use = _anthropic_response(
            {
                "usage": {"input_tokens": 15, "output_tokens": 5},
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "reload"}}
                ],
            }
        )
        final = _anthropic_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Reloaded."}],
            }
        )
        client = MagicMock()
        client.post = AsyncMock(side_effect=[mimic_text, real_tool_use, final])
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        registry = MagicMock()
        registry.is_registered.return_value = True
        registry.get.return_value.execute = AsyncMock(return_value="reloaded ok")

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.tools.REGISTRY", registry),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "reload nginx"}],
                "user_agent_p2",
                room_id, session_id, agent_id, db,
                ant_ws_tools=[
                    {"name": "bash", "description": "run", "input_schema": {"type": "object"}}
                ],
                oai_ws_tools=[],
            )

        assert client.post.await_count == 3
        retry_body = client.post.call_args_list[1].kwargs["json"]
        assert retry_body["tool_choice"] == {"type": "any"}
        assert result.text == "Reloaded."
        assert result.tool_input is None

    @pytest.mark.asyncio
    async def test_anthropic_gives_up_after_the_retry_budget(self):
        """Every retry mimics again — stop at the budget and never publish the
        mimic itself."""
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        db = _make_db()

        mimic = lambda text: _anthropic_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
            }
        )
        leak = '[tool call] mcp__legion-vps-mcp__vps_exec({"a":1})'
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[mimic(leak)] * (1 + _TOOL_MIMIC_MAX_RETRIES)
        )
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "reload nginx"}],
                "user_agent_p2",
                room_id, session_id, agent_id, db,
                ant_ws_tools=[
                    {"name": "bash", "description": "run", "input_schema": {"type": "object"}}
                ],
                oai_ws_tools=[],
            )

        assert client.post.await_count == 1 + _TOOL_MIMIC_MAX_RETRIES, "bounded, never a loop"
        assert result.text == _TOOL_MIMIC_FALLBACK_TEXT
        assert "[tool call]" not in result.text

    @pytest.mark.asyncio
    async def test_openai_retries_with_tool_choice_required_then_executes(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        db = _make_db()

        mimic_resp = _openai_response(
            '[tool call] mcp__legion-vps-mcp__vps_exec({"command":["nginx","-s","reload"]})'
        )
        real_call_resp = _openai_response(
            None, tool_calls=[_tool_call("c1", "bash", {"cmd": "reload"})]
        )
        final_resp = _openai_response("Reloaded.")

        oai_client = MagicMock()
        oai_client.chat.completions.create = AsyncMock(
            side_effect=[mimic_resp, real_call_resp, final_resp]
        )

        registry = MagicMock()
        registry.is_registered.return_value = True
        registry.get.return_value.execute = AsyncMock(return_value="reloaded ok")

        with (
            patch("app.core.user_sessions.AsyncOpenAI", return_value=oai_client),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.tools.REGISTRY", registry),
        ):
            result = await _call_with_ws_loop(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "reload nginx"}],
                "user_agent_p2",
                room_id, session_id, agent_id, db,
                ant_ws_tools=[],
                oai_ws_tools=[
                    {"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}}}
                ],
            )

        assert oai_client.chat.completions.create.await_count == 3
        retry_kwargs = oai_client.chat.completions.create.call_args_list[1].kwargs
        assert retry_kwargs["tool_choice"] == "required"
        assert result.text == "Reloaded."
        assert result.tool_input is None


def _ws_client(*responses) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    client.post = AsyncMock(side_effect=list(responses))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return client, ctx


def _read_file_registry() -> MagicMock:
    registry = MagicMock()
    registry.is_registered.return_value = True
    registry.get.return_value.execute = AsyncMock(return_value="lines")
    return registry


_READ_FILE_ANT = [{"name": "read_file", "description": "r", "input_schema": {"type": "object"}}]

_READ_FILE_USE = {
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "a.md"}}],
}


def _ant_text(text: str) -> MagicMock:
    return _anthropic_response({
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
    })


# The reported leak: an agent paging a long file with read_file ran out of the
# turn's tool budget, and the forced text reply (tools switched off, history
# full of call-shaped notes) was the next read_file call written as prose.
class TestToolNotesDoNotLookLikeCalls:
    def test_note_names_tools_without_call_syntax(self):
        note = _tool_note_text(["read_file", "grep", "read_file"])
        assert "read_file, grep" in note
        assert "(" not in note and "{" not in note

    def test_legacy_notes_are_rewritten(self):
        history = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": '[tool call] read_file({"path": "a; b(c)"}); grep({"p": "x"})',
            },
            {"role": "user", "content": "[tool results]\n\n### read_file ..."},
        ]
        out = _modernize_tool_notes(history)
        assert out is not history
        assert out[1]["content"] == _tool_note_text(["read_file", "grep"])
        assert history[1]["content"].startswith("[tool call]"), "caller untouched"
        assert out[2] is history[2]

    def test_history_without_legacy_notes_is_returned_as_is(self):
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        assert _modernize_tool_notes(history) is history

    def test_new_note_is_itself_caught_as_a_mimic(self):
        assert _looks_like_unexecuted_tool_call(_tool_note_text(["read_file"]))

    def test_strip_keeps_prose_around_the_mimic(self):
        text = 'Reading on.\n[tool call] read_file({"path": "a.md"})'
        assert _strip_tool_call_mimic(text) == "Reading on."
        assert _strip_tool_call_mimic("[tool call] read_file({})") == _TOOL_MIMIC_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_loop_sends_new_format_and_converts_legacy_history(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        client, ctx = _ws_client(_anthropic_response(_READ_FILE_USE), _ant_text("Done."))
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": '[tool call] grep({"p": "x"})'},
            {"role": "user", "content": "[tool results]\n\n### grep ..."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "read a.md"},
        ]

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.tools.REGISTRY", _read_file_registry()),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS", history, "user_agent_p2",
                room_id, session_id, agent_id, _make_db(),
                ant_ws_tools=_READ_FILE_ANT,
                oai_ws_tools=[],
            )

        sent = client.post.call_args_list[0].kwargs["json"]["messages"]
        blob = "\n".join(m["content"] for m in sent if isinstance(m.get("content"), str))
        assert "[tool call]" not in blob
        assert result.text == "Done."
        assert result.tool_notes[0]["content"] == _tool_note_text(["read_file"])
        assert '"path": "a.md"' in result.tool_notes[1]["content"], "args stay in the results"


class TestMimicRetryBudget:
    @pytest.mark.asyncio
    async def test_second_slip_in_the_same_turn_is_retried_too(self):
        """The old guard retried once per turn; a later slip leaked verbatim."""
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        mimic = '[tool call] read_file({"path": "a.md"})'
        use = _anthropic_response(_READ_FILE_USE)
        client, ctx = _ws_client(
            _ant_text(mimic), use, _ant_text(mimic), use, _ant_text("Done.")
        )
        registry = _read_file_registry()

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.tools.REGISTRY", registry),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS", [{"role": "user", "content": "read"}], "user_agent_p2",
                room_id, session_id, agent_id, _make_db(),
                ant_ws_tools=_READ_FILE_ANT,
                oai_ws_tools=[],
            )

        assert client.post.await_count == 5
        assert registry.get.return_value.execute.await_count == 2
        assert result.text == "Done."

    @pytest.mark.asyncio
    async def test_openai_failed_retries_publish_fallback_not_the_mimic(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        leak = '[tool call] read_file({"path": "a.md"})'
        oai_client = MagicMock()
        oai_client.chat.completions.create = AsyncMock(
            side_effect=[_openai_response(leak)] * (1 + _TOOL_MIMIC_MAX_RETRIES)
        )
        with (
            patch("app.core.user_sessions.AsyncOpenAI", return_value=oai_client),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            result = await _call_with_ws_loop(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS", [{"role": "user", "content": "read"}], "user_agent_p2",
                room_id, session_id, agent_id, _make_db(),
                ant_ws_tools=[],
                oai_ws_tools=[
                    {"type": "function", "function": {"name": "read_file", "description": "r", "parameters": {"type": "object"}}}
                ],
            )
        assert oai_client.chat.completions.create.await_count == 1 + _TOOL_MIMIC_MAX_RETRIES
        assert result.text == _TOOL_MIMIC_FALLBACK_TEXT


class TestForcedReplyMimicGuard:
    @pytest.mark.asyncio
    async def test_forced_reply_that_is_a_call_is_not_published(self):
        """The exact screenshot: read_file(offset 144, limit 62) as the reply."""
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        use = _anthropic_response(_READ_FILE_USE)
        client, ctx = _ws_client(
            use, use, use,
            _ant_text(
                '[tool call] read_file({"path": "bible/episode_arcs_1_20.md", '
                '"offset": 144, "limit": 62})'
            ),
        )

        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions._WS_MAX_TOOL_CALLS", 2),
            patch("app.core.tools.REGISTRY", _read_file_registry()),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS", [{"role": "user", "content": "read the bible"}], "user_agent_p2",
                room_id, session_id, agent_id, _make_db(),
                ant_ws_tools=_READ_FILE_ANT,
                oai_ws_tools=[],
            )

        assert result.forced_text_fallback is True
        assert "read_file" not in result.text
        assert "limit" in result.text and "continue" in result.text

    @pytest.mark.asyncio
    async def test_prose_before_the_mimic_is_kept(self):
        room_id, session_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        use = _anthropic_response(_READ_FILE_USE)
        client, ctx = _ws_client(
            use, use, use,
            _ant_text('Read chapters 1–5.\n[tool call] read_file({"offset": 144})'),
        )
        with (
            patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions._WS_MAX_TOOL_CALLS", 2),
            patch("app.core.tools.REGISTRY", _read_file_registry()),
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS", [{"role": "user", "content": "read"}], "user_agent_p2",
                room_id, session_id, agent_id, _make_db(),
                ant_ws_tools=_READ_FILE_ANT,
                oai_ws_tools=[],
            )
        assert result.text.startswith("Read chapters 1–5.")
        assert "read_file" not in result.text
