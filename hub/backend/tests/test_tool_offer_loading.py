"""Deferred tools and load_tools (internal design notes S3)."""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import tool_offer as to
from app.core.tool_offer import LOAD_TOOLS, ToolOffer


def _ws(name: str, desc: str = "Does a thing. More words here.") -> dict:
    return {"name": name, "description": desc, "input_schema": {"type": "object"}}


def _oai(name: str, desc: str = "Does a thing.") -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {}}}


def _names(schemas) -> list[str]:
    return sorted(to.schema_name(s) for s in schemas)


# ── the mechanism ────────────────────────────────────────────────────────────


def test_offered_hides_deferred_tools_behind_a_catalogue():
    offer = ToolOffer(deferred=frozenset({"room_config", "mcp__v__look"}))
    tools = offer.offered(
        [_ws("bash"), _ws("room_config", "List variables. Long tail."), _ws("mcp__v__look")],
        "anthropic",
    )
    assert _names(tools) == ["bash", LOAD_TOOLS]
    load = next(t for t in tools if t["name"] == LOAD_TOOLS)
    assert "- room_config: List variables" in load["description"]
    assert "- mcp__v__look: Does a thing" in load["description"]
    assert load["input_schema"]["required"] == ["names"]


def test_no_catalogue_when_nothing_in_the_list_is_deferred():
    offer = ToolOffer(deferred=frozenset({"room_config"}))
    assert _names(offer.offered([_ws("bash")], "anthropic")) == ["bash"]


def test_load_adds_schemas_and_keeps_the_catalogue_unchanged():
    offer = ToolOffer(deferred=frozenset({"room_config", "mcp__v__look"}))
    full = [_ws("bash"), _ws("room_config"), _ws("mcp__v__look")]
    before = offer.offered(full, "anthropic")
    result = offer.load({"names": ["mcp__v__look", "bash", "nope"]})
    assert result.startswith("Loaded: mcp__v__look.")
    assert "Not deferred tools: bash, nope" in result
    after = offer.offered(full, "anthropic")
    assert _names(after) == ["bash", LOAD_TOOLS, "mcp__v__look"]
    load_before = next(t for t in before if t["name"] == LOAD_TOOLS)
    load_after = next(t for t in after if t["name"] == LOAD_TOOLS)
    assert load_before == load_after
    # Loading again changes nothing (no churn of the tool prefix).
    version = offer.version
    assert offer.load({"names": ["mcp__v__look"]}).startswith("Already loaded")
    assert offer.version == version


def test_a_short_name_loads_the_matching_mcp_tool():
    # DeepSeek v4 flash asked for "web_search" on the dev live check.
    offer = ToolOffer(
        deferred=frozenset(
            {
                "mcp__mcp-searxng__searxng_web_search",
                "mcp__mcp-searxng__web_url_read",
                "mcp__mcp-vision__describe_image",
                "room_config",
            }
        )
    )
    result = offer.load({"names": ["web_search"]})
    assert result.startswith("Loaded: mcp__mcp-searxng__searxng_web_search.")
    assert offer.loaded == {"mcp__mcp-searxng__searxng_web_search"}
    # The same history read back after a pause loads the same tool.
    fresh = ToolOffer(deferred=offer.deferred)
    fresh.absorb(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "type": "function",
                     "function": {"name": LOAD_TOOLS, "arguments": '{"names": ["web_search"]}'}}
                ],
            }
        ]
    )
    assert fresh.loaded == offer.loaded


def test_an_ambiguous_or_unknown_name_lists_the_choices():
    offer = ToolOffer(deferred=frozenset({f"mcp__s__tool_{i}" for i in range(5)} | {"room_config"}))
    result = offer.load({"names": ["tool", "nothing_like_it"]})
    assert "'tool' matches several tools" in result
    assert "Deferred tools are:" in result and "room_config" in result
    assert offer.loaded == set()


def test_load_without_names_lists_what_can_be_loaded():
    offer = ToolOffer(deferred=frozenset({"room_config"}))
    assert offer.load({}).startswith("Error: pass names=[...]")


def test_load_accepts_a_comma_separated_string():
    offer = ToolOffer(deferred=frozenset({"a", "b"}))
    offer.load({"names": "a, b"})
    assert offer.loaded == {"a", "b"}


def test_openai_form():
    offer = ToolOffer(deferred=frozenset({"room_config"}))
    tools = offer.offered([_oai("bash"), _oai("room_config")], "openai")
    assert _names(tools) == ["bash", LOAD_TOOLS]
    load = next(t for t in tools if to.schema_name(t) == LOAD_TOOLS)
    assert load["type"] == "function"
    assert "names" in load["function"]["parameters"]["properties"]


def test_absorb_reads_both_wire_formats_and_direct_calls():
    offer = ToolOffer(deferred=frozenset({"a", "b", "c"}))
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": LOAD_TOOLS, "input": {"names": ["a"]}}],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": LOAD_TOOLS, "arguments": json.dumps({"names": ["b"]})},
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "3", "type": "function", "function": {"name": "c", "arguments": "{}"}}],
        },
    ]
    assert offer.absorb(history) is True
    assert offer.loaded == {"a", "b", "c"}
    assert offer.absorb(history) is False


def test_absorb_notes_reads_direct_session_text_notes():
    offer = ToolOffer(deferred=frozenset({"mcp__v__look", "room_config", "run_subtask"}))
    history = [
        {
            "role": "assistant",
            "content": "[tools executed: load_tools, bash — arguments and results in the next message]",
        },
        {
            "role": "user",
            "content": '### load_tools {"names": ["mcp__v__look"]}\nLoaded: mcp__v__look. Their full schemas',
        },
        {
            "role": "assistant",
            "content": "[tools executed: room_config — arguments and results in the next message]",
        },
        # A user who merely types the words loads nothing outside a tool note.
        {"role": "user", "content": "Loaded: run_subtask."},
    ]
    assert offer.absorb_notes(history) is True
    assert offer.loaded == {"mcp__v__look", "room_config"}


def test_child_offer_copies_loaded_and_drops_run_subtask():
    parent = ToolOffer(deferred=frozenset({"run_subtask", "x", "y"}), loaded={"x"})
    child = parent.for_child(drop={"run_subtask"})
    assert child.deferred == {"x", "y"} and child.loaded == {"x"}
    child.load({"names": ["y"]})
    assert parent.loaded == {"x"}


def test_summary_is_one_short_line():
    assert to.summarize("First sentence. Second sentence.") == "First sentence"
    assert len(to.summarize("word " * 80)) <= 110


# ── wired into the loops ─────────────────────────────────────────────────────


def _ant_payload(content: list[dict]) -> dict:
    return {"usage": {"input_tokens": 10, "output_tokens": 5}, "stop_reason": "tool_use", "content": content}


_END = {
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "done"}],
}


def _proxy_patches(db, ctx):
    from tests.test_db_release import _dirty_emit

    return (
        patch("app.core.proxy_session.httpx.AsyncClient", return_value=ctx),
        patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
        patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        patch("app.core.proxy_session._emit_tool_call_event", new=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_proxy_loop_loads_a_deferred_mcp_tool_for_the_next_step():
    from app.core.proxy_session import _call_llm_with_tools
    from tests.test_db_release import TxDb, _ant_client

    db = TxDb()
    ctx = _ant_client(
        db,
        [
            _ant_payload(
                [{"type": "tool_use", "id": "l1", "name": LOAD_TOOLS, "input": {"names": ["mcp__v__look"]}}]
            ),
            _ant_payload([{"type": "tool_use", "id": "n1", "name": "no_such_tool", "input": {}}]),
            _END,
        ],
    )
    offer = ToolOffer(deferred=frozenset({"mcp__v__look", "room_config"}))
    p1, p2, p3, p4 = _proxy_patches(db, ctx)
    with p1, p2, p3, p4:
        result = await _call_llm_with_tools(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            [{"role": "user", "content": "hi"}],
            room_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            db=db,
            workspace_ant_tools=[_ws("bash"), _ws("room_config")],
            mcp_ant_tools=[_ws("mcp__v__look")],
            context_role="agent",
            tool_offer=offer,
        )
    assert result[0] == "done"
    bodies = [call.kwargs["json"] for call in ctx.__aenter__.return_value.post.call_args_list]
    step1, step2, _ = (_names(b["tools"]) for b in bodies)
    assert step1 == ["bash", "close_session", LOAD_TOOLS]
    assert step2 == ["bash", "close_session", LOAD_TOOLS, "mcp__v__look"]
    # After the load the prefix is stable again.
    assert bodies[1]["tools"] == bodies[2]["tools"]
    results = [
        block
        for message in bodies[1]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert results[-1]["content"].startswith("Loaded: mcp__v__look.")
    assert offer.loaded == {"mcp__v__look"}


@pytest.mark.asyncio
async def test_proxy_loop_without_loads_sends_identical_tools_every_step():
    from app.core.proxy_session import _call_llm_with_tools
    from tests.test_db_release import _END_TURN, _TOOL_USE, TxDb, _ant_client

    db = TxDb()
    ctx = _ant_client(db, [_TOOL_USE, _TOOL_USE, _END_TURN])
    p1, p2, p3, p4 = _proxy_patches(db, ctx)
    with p1, p2, p3, p4:
        await _call_llm_with_tools(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            [{"role": "user", "content": "hi"}],
            room_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            db=db,
            workspace_ant_tools=[_ws("bash"), _ws("room_config")],
            context_role="agent",
            tool_offer=ToolOffer(deferred=frozenset({"room_config"})),
        )
    bodies = [call.kwargs["json"] for call in ctx.__aenter__.return_value.post.call_args_list]
    assert len(bodies) == 3
    dumps = {json.dumps(body["tools"]) for body in bodies}
    assert len(dumps) == 1


@pytest.mark.asyncio
async def test_proxy_openai_loop_loads_a_deferred_tool():
    from app.core.proxy_session import _call_llm_with_tools
    from tests.test_db_release import TxDb

    def _call(call_id, name, args):
        tc = MagicMock()
        tc.id = call_id
        tc.type = "function"
        tc.function.name = name
        tc.function.arguments = json.dumps(args)
        return tc

    def _resp(content, tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        msg.model_dump.return_value = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls or []
            ]
            or None,
        }
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "tool_calls" if tool_calls else "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.prompt_tokens_details = None
        resp.model = "m"
        return resp

    responses = [
        _resp(None, [_call("c1", LOAD_TOOLS, {"names": ["room_config"]})]),
        _resp("done"),
    ]
    sent: list[list[str]] = []

    async def create(**kwargs):
        sent.append(_names(kwargs["tools"]))
        return responses.pop(0)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    db = TxDb()
    offer = ToolOffer(deferred=frozenset({"room_config"}))
    from tests.test_db_release import _dirty_emit

    with (
        patch("app.core.proxy_session.llm_client", return_value=client),
        patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
        patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        patch("app.core.proxy_session._emit_tool_call_event", new=AsyncMock()),
    ):
        result = await _call_llm_with_tools(
            {"provider": "openrouter", "model": "deepseek/x", "api_key": "k"},
            "SYS",
            [{"role": "user", "content": "hi"}],
            room_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            db=db,
            workspace_oai_tools=[_oai("bash"), _oai("room_config")],
            context_role="agent",
            tool_offer=offer,
        )
    assert result[0] == "done"
    assert sent[0] == ["bash", "close_session", LOAD_TOOLS]
    assert sent[1] == ["bash", "close_session", LOAD_TOOLS, "room_config"]


@pytest.mark.asyncio
async def test_direct_session_keeps_a_tool_loaded_in_an_earlier_turn():
    from app.core.user_sessions import _call_with_ws_loop

    def _resp(payload):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.raise_for_status = MagicMock()
        resp.json.return_value = payload
        return resp

    client = MagicMock()
    client.post = AsyncMock(side_effect=[_resp(_END)])
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    db = AsyncMock()
    db.add = MagicMock()
    res = MagicMock()
    res.scalars.return_value.first.return_value = None
    res.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=res)

    history = [
        {"role": "user", "content": "look at the page"},
        {
            "role": "assistant",
            "content": "[tools executed: load_tools — arguments and results in the next message]",
        },
        {
            "role": "user",
            "content": '### load_tools {"names": ["mcp__v__look"]}\nLoaded: mcp__v__look. Their full schemas',
        },
        {"role": "assistant", "content": "Loaded it."},
        {"role": "user", "content": "now use it"},
    ]
    offer = ToolOffer(deferred=frozenset({"mcp__v__look", "room_config"}))
    with (
        patch("app.core.user_sessions.httpx.AsyncClient", return_value=ctx),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
    ):
        result = await _call_with_ws_loop(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            history,
            "user_agent_p2",
            uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), db,
            ant_ws_tools=[_ws("bash"), _ws("room_config")],
            oai_ws_tools=[],
            mcp_ant_tools=[_ws("mcp__v__look")],
            mcp_bindings={"mcp__v__look": object()},
            tool_offer=offer,
        )
    assert result.text == "done"
    names = _names(client.post.call_args_list[0].kwargs["json"]["tools"])
    assert "mcp__v__look" in names and LOAD_TOOLS in names and "room_config" not in names
