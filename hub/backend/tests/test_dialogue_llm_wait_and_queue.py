"""A dialogue turn waiting on the model: visible, stoppable, and not looping.

Incident 2026-11-09 (room "Legion marketing", ChatMaster on an OpenRouter
model): one model request of the ChatMaster's tool loop hung for ~14 minutes.
The chat showed nothing (no status after 60 s, so no Stop bar either), six
user messages queued behind it unseen, and once the model answered it spent
~100 ``workspace_service`` calls failing with the same error.

* the request is awaited beside a heartbeat ("waiting for the model · m:ss");
* Stop drops the request in flight instead of waiting for it;
* a queued message does not interrupt it — "Send now" does;
* five identical failures in a row put a note to the agent (not the user),
  and a second streak ends the tool loop with a text answer.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import loop_guard, session_interject, user_sessions
from app.core.events import AgentStatus


@pytest.fixture(autouse=True)
def _clean():
    def _reset():
        for store in (
            session_interject._PENDING,
            session_interject._PENDING_IDS,
            session_interject._DELIVERED_IDS,
            session_interject._DROPPED_IDS,
            session_interject._LLM_CALLS,
            session_interject._ABORT_REASON,
        ):
            store.clear()
        session_interject._STOP_REQUESTED.clear()
        session_interject._FLUSH_REQUESTED.clear()

    _reset()
    yield
    _reset()


# ---------------------------------------------------------------------------
# session_interject: the request in flight, "Send now", Stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_now_with_an_empty_queue_does_nothing():
    sid = uuid.uuid4()
    event = session_interject.begin_llm_call(sid)
    assert session_interject.request_flush(sid) == session_interject.FLUSH_NOTHING_QUEUED
    assert not event.is_set()


@pytest.mark.asyncio
async def test_send_now_interrupts_the_request_in_flight():
    sid = uuid.uuid4()
    event = session_interject.begin_llm_call(sid)
    session_interject.queue(sid, "where is the service?")
    assert not event.is_set(), "a queued message alone must not cut the request"
    assert session_interject.request_flush(sid) == session_interject.FLUSH_INTERRUPTED
    assert event.is_set()
    assert session_interject.abort_reason(sid) == session_interject.ABORT_FLUSH


@pytest.mark.asyncio
async def test_send_now_during_a_tool_applies_to_the_next_request():
    sid = uuid.uuid4()
    session_interject.queue(sid, "hello")
    assert session_interject.request_flush(sid) == session_interject.FLUSH_NEXT_STEP
    event = session_interject.begin_llm_call(sid)
    assert event.is_set()
    assert session_interject.abort_reason(sid) == session_interject.ABORT_FLUSH
    # Once the loop drained the queue the flag is gone: the re-asked request runs.
    session_interject.drain(sid)
    session_interject.end_llm_call(sid, event)
    assert not session_interject.begin_llm_call(sid).is_set()


@pytest.mark.asyncio
async def test_stop_aborts_the_request_and_wins_over_send_now():
    sid = uuid.uuid4()
    event = session_interject.begin_llm_call(sid)
    session_interject.queue(sid, "x")
    session_interject.request_flush(sid)
    session_interject.request_stop(sid)
    assert event.is_set()
    assert session_interject.abort_reason(sid) == session_interject.ABORT_STOP


def test_queued_ids_become_delivered_or_dropped():
    sid = uuid.uuid4()
    first, second, third = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session_interject.queue(sid, "a", first)
    session_interject.queue(sid, "b", second)
    assert session_interject.drain(sid) == ["a", "b"]
    session_interject.queue(sid, "c", third)
    session_interject.clear(sid)
    assert session_interject.take_marks(sid) == ([first, second], [third])
    assert session_interject.take_marks(sid) == ([], [])


def test_stop_drops_queued_ids():
    sid = uuid.uuid4()
    mid = uuid.uuid4()
    session_interject.queue(sid, "a", mid)
    session_interject.request_stop(sid)
    assert session_interject.take_marks(sid) == ([], [mid])


# ---------------------------------------------------------------------------
# loop_guard Signal D
# ---------------------------------------------------------------------------

_INCIDENT_ERROR = (
    "slug: legion-tracker\nstate: error\nhealth: error\n"
    "error: environment preparation failed (environment_manifest_missing): "
    "manifest not found: backend/requirements.txt\ndeclaration: unchanged"
)


def test_error_line_ignores_status_fields():
    assert loop_guard.tool_error_line("state: error\nhealth: error") is None
    assert loop_guard.tool_error_line("Error: action must be list, declare") == "action must be list, declare"
    assert loop_guard.tool_error_line(_INCIDENT_ERROR).startswith("environment preparation failed")


def test_five_identical_failures_note_the_agent():
    streak = loop_guard.ToolErrorStreak(limit=5, max_notes=1)
    notes = [streak.observe("workspace_service", _INCIDENT_ERROR) for _ in range(5)]
    assert notes[:4] == [None] * 4
    assert notes[4] and notes[4].startswith("[LOOP GUARD")
    assert "workspace_service" in notes[4] and "environment preparation failed" in notes[4]
    assert "tell the user that the plan cannot be completed" in notes[4]
    assert not streak.exhausted


def test_a_success_in_between_resets_the_streak():
    streak = loop_guard.ToolErrorStreak(limit=5)
    for _ in range(4):
        assert streak.observe("workspace_service", _INCIDENT_ERROR) is None
    assert streak.observe("glob", "backend/requirements.txt") is None
    for _ in range(4):
        assert streak.observe("workspace_service", _INCIDENT_ERROR) is None


def test_numbers_do_not_make_errors_different():
    streak = loop_guard.ToolErrorStreak(limit=3)
    assert streak.observe("bash", "Error: exit code 1 after 12s") is None
    assert streak.observe("bash", "Error: exit code 2 after 40s") is None
    assert streak.observe("bash", "Error: exit code 1 after 3s") is not None


def test_second_streak_after_a_note_exhausts_the_loop():
    streak = loop_guard.ToolErrorStreak(limit=5, max_notes=1)
    for _ in range(5):
        streak.observe("workspace_service", _INCIDENT_ERROR)
    assert not streak.exhausted
    last = None
    for _ in range(5):
        last = streak.observe("workspace_service", _INCIDENT_ERROR)
    assert streak.exhausted
    assert "the tool loop is now stopped" in last


# ---------------------------------------------------------------------------
# The ChatMaster / direct-agent tool loop (OpenAI-compatible branch)
# ---------------------------------------------------------------------------


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    return db


def _tool_call(call_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _response(content: str | None, tool_calls=None) -> MagicMock:
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


_OAI_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object"}}}
]


def _registry(result: str = "print('a')") -> MagicMock:
    registry = MagicMock()
    registry.is_registered.return_value = True
    registry.get.return_value.execute = AsyncMock(return_value=result)
    return registry


def _run_loop(client: MagicMock, session_id: uuid.UUID, registry: MagicMock | None = None):
    return user_sessions._call_with_ws_loop(
        {"provider": "openai", "model": "m", "api_key": "k"},
        "SYS",
        [{"role": "user", "content": "start the service"}],
        "user_cm_p2",
        uuid.uuid4(), session_id, uuid.uuid4(), _make_db(),
        ant_ws_tools=[],
        oai_ws_tools=_OAI_TOOLS,
    )


async def _until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _hanging_then(*later):
    """``create`` that hangs on its first call, then returns ``later`` in order."""
    calls: list[dict] = []
    replies = list(later)

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await asyncio.Event().wait()
        return replies.pop(0)

    client = MagicMock()
    client.chat.completions.create = create
    return client, calls


@pytest.mark.asyncio
async def test_hung_request_reports_waiting_for_the_model():
    sid = uuid.uuid4()
    client, _ = _hanging_then(_response("Done."))
    with (
        patch("app.core.user_sessions.AsyncOpenAI", return_value=client),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_interim_text_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_status", new_callable=AsyncMock) as status,
        patch("app.core.user_sessions._LLM_HEARTBEAT_S", 0.02),
        patch("app.core.tools.REGISTRY", _registry()),
    ):
        turn = asyncio.ensure_future(_run_loop(client, sid))
        await _until(lambda: status.await_count >= 3)
        session_interject.request_stop(sid)
        result = await asyncio.wait_for(turn, 3)

    beats = [c for c in status.await_args_list if c.args[1] == AgentStatus.LLM_HEARTBEAT]
    assert len(beats) >= 3
    extra = beats[-1].kwargs["extra"]
    assert extra["phase"] == "llm" and extra["attempt_elapsed_s"] >= 0
    assert result.externally_stopped is True
    assert not session_interject.in_llm_call(sid)


@pytest.mark.asyncio
async def test_queued_message_waits_and_send_now_re_asks_with_it():
    sid = uuid.uuid4()
    client, calls = _hanging_then(_response("Here it is."))
    with (
        patch("app.core.user_sessions.AsyncOpenAI", return_value=client),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_interim_text_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_status", new_callable=AsyncMock),
        patch("app.core.user_sessions._LLM_HEARTBEAT_S", 0.02),
        patch("app.core.tools.REGISTRY", _registry()),
    ):
        turn = asyncio.ensure_future(_run_loop(client, sid))
        await _until(lambda: session_interject.in_llm_call(sid))
        session_interject.queue(sid, "Where is the service?")
        await asyncio.sleep(0.1)
        assert not turn.done(), "a queued message must not cut the request by itself"
        assert len(calls) == 1
        assert session_interject.request_flush(sid) == session_interject.FLUSH_INTERRUPTED
        result = await asyncio.wait_for(turn, 3)

    assert result.text == "Here it is."
    assert len(calls) == 2
    second = json.dumps(calls[1]["messages"], ensure_ascii=False)
    assert "Where is the service?" in second
    assert result.injected_user_messages == ["Where is the service?"]


@pytest.mark.asyncio
async def test_repeated_identical_failures_reach_the_agent_then_end_the_loop():
    sid = uuid.uuid4()
    seen: list[list[dict]] = []

    async def create(**kwargs):
        seen.append(kwargs["messages"])
        return _response(None, [_tool_call(f"c{len(seen)}", "read_file", {"path": f"a{len(seen)}.py"})])

    client = MagicMock()
    client.chat.completions.create = create
    forced = user_sessions._LLMTurnResult("I could not start it: the manifest is missing.", None, 0, 0, "m")
    with (
        patch("app.core.user_sessions.AsyncOpenAI", return_value=client),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_interim_text_event", new_callable=AsyncMock),
        patch("app.core.user_sessions.emit_status", new_callable=AsyncMock),
        patch("app.core.user_sessions._call_llm_with_tools", new=AsyncMock(return_value=forced)) as forced_call,
        patch("app.core.tools.REGISTRY", _registry(_INCIDENT_ERROR)),
    ):
        result = await asyncio.wait_for(_run_loop(client, sid), 5)

    # 10 failing calls: a note after the 5th, the loop ended after the 10th.
    assert len(seen) == 10
    tool_msgs = [m for m in seen[5] if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"].startswith("[LOOP GUARD")
    forced_call.assert_awaited_once()
    assert forced_call.await_args.kwargs.get("force_text") is True
    assert result.text.startswith("I could not start it")
