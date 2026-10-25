"""Stage 0.4 of internal design notes — the load generator's mock LLM.

The mock stands in for the provider during the load ladder, so a bug in it
would be measured as a Hub failure and land in `load_before.md` as a wrong
`N_max` with a wrong named cause. These tests pin the two things that could go
wrong silently: the shape of the responses both wires expect, and the rule
that picks the next scenario turn without server-side state.

The mock is not importable as a package (it lives in hub/scripts/loadgen), so
it is loaded by path.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

def _find_mock() -> Path | None:
    """The mock lives outside the backend tree; find it wherever we are run.

    In the container the loadgen directory is mounted read-only at /loadgen
    (see docker-compose.dev.yml); from a full checkout it sits next to the
    backend at hub/scripts/loadgen.
    """
    candidates = [
        Path("/loadgen/mock_llm.py"),
        Path(__file__).resolve().parents[2] / "scripts" / "loadgen" / "mock_llm.py",
    ]
    env = os.getenv("LEGION_LOADGEN_DIR")
    if env:
        candidates.insert(0, Path(env) / "mock_llm.py")
    return next((c for c in candidates if c.exists()), None)


_MOCK_PATH = _find_mock()


@pytest.fixture(scope="module")
def mock():
    if _MOCK_PATH is None:  # pragma: no cover
        pytest.skip("mock_llm.py not reachable — mount hub/scripts/loadgen at /loadgen")
    spec = importlib.util.spec_from_file_location("loadgen_mock_llm", _MOCK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loadgen_mock_llm"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(mock):
    from fastapi.testclient import TestClient

    mock.DELAY_SEC = 0.0  # the ladder's 3 s delay would only slow the tests
    mock.JITTER_SEC = 0.0
    return TestClient(mock.app)


# --- tools the Hub actually offers -----------------------------------------

_CHAT_TOOLS = [
    {"type": "function", "function": {"name": "glob", "parameters": {}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {}}},
    {"type": "function", "function": {"name": "repo_map", "parameters": {}}},
    {"type": "function", "function": {"name": "write_file", "parameters": {}}},
    {"type": "function", "function": {"name": "close_session", "parameters": {}}},
]

_ANT_TOOLS = [
    {"name": "glob", "input_schema": {}},
    {"name": "read_file", "input_schema": {}},
    {"name": "repo_map", "input_schema": {}},
    {"name": "write_file", "input_schema": {}},
    {"name": "close_session", "input_schema": {}},
]


def chat_body(tool_results: int, **extra):
    """An OpenAI-wire request with `tool_results` tools already executed."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(tool_results):
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "glob", "arguments": "{}"}}]}
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    return {"model": "mock-model", "messages": messages, "tools": _CHAT_TOOLS, **extra}


def ant_body(tool_results: int, **extra):
    """An Anthropic-wire request with `tool_results` tools already executed."""
    messages = [{"role": "user", "content": "go"}]
    for i in range(tool_results):
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "glob", "input": {}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]})
    return {"model": "mock-model", "system": "sys", "messages": messages,
            "tools": _ANT_TOOLS, "max_tokens": 4096, **extra}


# --- response shape ---------------------------------------------------------


def test_chat_wire_returns_a_tool_call_in_openai_shape(client):
    data = client.post("/v1/chat/completions", json=chat_body(0)).json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "glob"
    json.loads(call["function"]["arguments"])  # must be a JSON string, not a dict
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


def test_messages_wire_returns_a_tool_use_block_in_anthropic_shape(client):
    data = client.post("/v1/messages", json=ant_body(0)).json()
    assert data["stop_reason"] == "tool_use"
    block = data["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "glob"
    assert isinstance(block["input"], dict)  # a dict here, unlike the chat wire
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("path,body_fn", [
    ("/v1/chat/completions", chat_body),
    ("/v1/messages", ant_body),
])
def test_streaming_is_refused_loudly(client, path, body_fn):
    """A silently non-streamed answer would surface as a parse error mid-ladder."""
    resp = client.post(path, json=body_fn(0, stream=True))
    assert resp.status_code == 400


# --- turn selection ---------------------------------------------------------


def test_turns_advance_with_the_conversation_not_with_server_state(client):
    """Two interleaved conversations must not consume each other's turns."""
    first = client.post("/v1/chat/completions", json=chat_body(0)).json()
    other = client.post("/v1/chat/completions", json=chat_body(2)).json()
    first_again = client.post("/v1/chat/completions", json=chat_body(0)).json()

    name = lambda d: d["choices"][0]["message"]["tool_calls"][0]["function"]["name"]  # noqa: E731
    assert name(first) == "glob"
    assert name(other) == "repo_map"
    assert name(first_again) == "glob"  # unchanged by the other conversation


def test_both_wires_walk_the_same_scenario(client, mock):
    chat_names, ant_names = [], []
    for i in range(len(mock.SCENARIO)):
        chat = client.post("/v1/chat/completions", json=chat_body(i)).json()
        chat_names.append(chat["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
        ant = client.post("/v1/messages", json=ant_body(i)).json()
        ant_names.append(ant["content"][0]["name"])
    assert chat_names == ant_names
    assert chat_names[-1] == "close_session"


def test_running_past_the_scenario_still_closes_the_session(client, mock):
    """Otherwise the loop runs to its turn cap and that is recorded as a Hub failure."""
    beyond = len(mock.SCENARIO) + 5
    data = client.post("/v1/chat/completions", json=chat_body(beyond)).json()
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "close_session"


def test_forced_tool_choice_wins_over_the_scenario(client):
    """The Hub's forced_close / forced_answer paths name the tool they require."""
    chat = client.post("/v1/chat/completions", json=chat_body(
        0, tool_choice={"type": "function", "function": {"name": "close_session"}})).json()
    assert chat["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "close_session"

    ant = client.post("/v1/messages", json=ant_body(
        0, tool_choice={"type": "tool", "name": "close_session"})).json()
    assert ant["content"][0]["name"] == "close_session"


def test_tool_choice_none_yields_plain_text(client):
    chat = client.post("/v1/chat/completions", json=chat_body(0, tool_choice="none")).json()
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["choices"][0]["message"]["content"]

    ant = client.post("/v1/messages", json=ant_body(0, tool_choice={"type": "none"})).json()
    assert ant["stop_reason"] == "end_turn"
    assert ant["content"][0]["type"] == "text"


def test_tool_absent_from_the_offered_set_falls_back_to_close(client):
    """ChatMaster is offered no write_file; calling it would measure error handling."""
    body = chat_body(3)  # scenario turn 3 is write_file
    body["tools"] = [{"type": "function", "function": {"name": "close_session", "parameters": {}}}]
    data = client.post("/v1/chat/completions", json=body).json()
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "close_session"


# --- catalog and bookkeeping ------------------------------------------------


def test_models_catalog_is_served_for_provider_validation(client):
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert any(m["id"] == "mock-model" for m in data["data"])


def test_stats_counts_both_wires(client):
    before = client.get("/stats").json()
    client.post("/v1/chat/completions", json=chat_body(0))
    client.post("/v1/messages", json=ant_body(0))
    after = client.get("/stats").json()
    assert after["chat_completions"] == before["chat_completions"] + 1
    assert after["messages"] == before["messages"] + 1


def test_broken_scenario_path_does_not_take_the_mock_down(mock, tmp_path):
    """The mock boots with the load stack; a bad path must not look like a Hub failure."""
    missing = tmp_path / "nope.json"
    turns = mock.load_scenario(str(missing))
    assert turns and turns[0]["tool"] == "close_session"


def test_default_scenario_ends_by_closing_the_session(mock):
    assert mock.SCENARIO[-1]["tool"] == "close_session"
