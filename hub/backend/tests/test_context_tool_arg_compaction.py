"""Past write_file/edit_file arguments are compacted before fresh tool results.

A direct video-agent room (2026-10-24): a direct agent writing Remotion shots sat
at the 96k window for 110 steps.  Only tool results were evictable, so whole
TSX bodies in past ``write_file`` calls grew the conversation to 63k and every
new ``bash`` output was turned into a receipt the moment it arrived.
"""
from __future__ import annotations

import copy
import json

from app.core.context_assembler import ContextAssembler
from app.core.context_policy import ContextPolicy
from app.core.token_counter import TokenCounter

_COUNTER = TokenCounter()


def _policy(**overrides) -> ContextPolicy:
    values = {
        "enabled": True,
        "input_cap": 16_000,
        "active_tool_result_cap": 6_000,
        "keep_raw_results": 3,
        "output_reserve": 4_000,
        "model_context_limit": 200_000,
        "checkpoint_cap": 600,
        "eviction_mode": "on_demand",
    }
    values.update(overrides)
    return ContextPolicy(**values)


def _text(tokens: int, tag: str) -> str:
    """A distinct string of roughly ``tokens`` tokens."""
    line = f"const {tag} = interpolate(frame, [0, 30], [0, 1]);\n"
    text = line
    while _COUNTER.count(text) < tokens:
        text += line
    return text


def _ant(call_id: str, tool: str, result: str, **args):
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call_id, "name": tool, "input": args}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result}],
        },
    ]


def _oai(call_id: str, tool: str, result: str, **args):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _build(messages, **policy):
    return ContextAssembler(_policy(**policy)).build(
        system_prompt="system", tool_schemas=[], messages=messages
    )


def test_old_write_body_goes_before_fresh_result():
    fresh_log = "exit_code=2\n" + _text(3_000, "log")
    messages = [
        {"role": "user", "content": "render the promo"},
        *_ant("w1", "write_file", "ok", path="src/Shot1.tsx", content=_text(7_000, "a")),
        *_ant("w2", "write_file", "ok", path="src/Shot2.tsx", content=_text(7_000, "b")),
        *_ant("b1", "bash", fresh_log, command="npx remotion still"),
    ]

    assembly = _build(messages)

    assert assembly.allowed
    assert assembly.evicted_results == 0
    assert assembly.compacted_args == 1
    first = assembly.messages[1]["content"][0]["input"]
    assert first["path"] == "src/Shot1.tsx"
    assert first["content"].startswith("[context receipt: argument content of write_file")
    # Only as much as needed: the newer body stays.
    assert assembly.messages[3]["content"][0]["input"]["content"] == messages[3]["content"][0]["input"]["content"]
    assert assembly.messages[-1]["content"][0]["content"] == fresh_log


def test_openai_edit_args_stay_valid_json():
    messages = [
        {"role": "user", "content": "fix the shot"},
        *_oai(
            "e1",
            "edit_file",
            "ok",
            path="src/Shot1.tsx",
            old_string=_text(6_000, "old"),
            new_string=_text(6_000, "new"),
        ),
        *_oai("b1", "bash", "exit_code=0\n" + _text(5_000, "out"), command="npm test"),
    ]

    assembly = _build(messages)

    assert assembly.evicted_results == 0
    call = assembly.messages[1]["tool_calls"][0]
    args = json.loads(call["function"]["arguments"])
    assert args["path"] == "src/Shot1.tsx"
    assert args["old_string"].startswith("[context receipt: argument old_string of edit_file")
    assert args["new_string"].startswith("[context receipt: argument new_string of edit_file")
    assert call["id"] == "e1"


def test_nothing_compacted_under_budget():
    messages = [
        {"role": "user", "content": "write it"},
        *_ant("w1", "write_file", "ok", path="a.tsx", content=_text(2_000, "a")),
        *_ant("b1", "bash", "exit_code=0", command="ls"),
    ]

    assembly = _build(messages)

    assert assembly.compacted_args == 0
    assert assembly.messages == messages


def test_newest_call_group_keeps_its_arguments():
    body = _text(18_000, "big")
    messages = [
        {"role": "user", "content": "write it"},
        *_ant("w1", "write_file", "ok", path="big.tsx", content=body),
    ]

    assembly = _build(messages)

    assert assembly.compacted_args == 0
    assert assembly.messages[1]["content"][0]["input"]["content"] == body


def test_source_history_is_not_mutated():
    messages = [
        {"role": "user", "content": "render"},
        *_ant("w1", "write_file", "ok", path="a.tsx", content=_text(8_000, "a")),
        *_oai("w2", "write_file", "ok", path="b.tsx", content=_text(8_000, "b")),
        *_ant("b1", "bash", "exit_code=0\n" + _text(2_000, "c"), command="ls"),
    ]
    original = copy.deepcopy(messages)

    assembly = _build(messages)

    assert assembly.compacted_args >= 1
    assert messages == original
