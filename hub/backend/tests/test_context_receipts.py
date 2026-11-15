"""An evicted tool result keeps what the agent needs to go on.

Prod, 2026-11-14: receipts carried only a sha256, so an agent forgot its own
measurements and which browser a script had launched, re-ran heavy commands
and reasoned from a wrong memory.
"""
from __future__ import annotations

import json

from app.core.context_assembler import ContextAssembler
from app.core.context_policy import ContextPolicy

_REF = "0f8e2c1a-1b2c-4d5e-8f90-a1b2c3d4e5f6"


def _policy() -> ContextPolicy:
    return ContextPolicy(
        enabled=True,
        input_cap=16_000,
        active_tool_result_cap=500,
        keep_raw_results=1,
        output_reserve=4_000,
        model_context_limit=200_000,
        checkpoint_cap=600,
        eviction_mode="eager",
    )


def _turn(call_id: str, tool: str, content: str, **args) -> list[dict]:
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
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def _assemble(messages: list[dict]) -> list[dict]:
    return ContextAssembler(_policy()).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    ).messages


def _bash_output() -> str:
    lines = ["exit_code: 0", "stdout:", "engine=chromium"]
    lines += [f"noise line {i} " + "x" * 80 for i in range(200)]
    lines += ["after blockH=869 gapAbove=16", "img 302x302"]
    return "\n".join(lines) + f"\n[observation_ref={_REF}]"


def test_bash_receipt_keeps_head_tail_and_observation_ref() -> None:
    messages = [{"role": "user", "content": "Fix the layout"}]
    messages += _turn("old", "bash", _bash_output(), cmd="python measure.py")
    messages += _turn("new", "bash", "exit_code: 0", cmd="true")

    receipt = _assemble(messages)[2]["content"]

    assert receipt.startswith("[context receipt: tool=bash")
    assert "engine=chromium" in receipt
    assert "gapAbove=16" in receipt
    assert "img 302x302" in receipt
    assert "noise line 100 " not in receipt
    assert "lines omitted" in receipt
    assert f'observation_read(ref="{_REF}")' in receipt
    assert receipt.count(_REF) == 1
    assert len(receipt) < 2_500


def test_large_output_receipt_points_at_saved_log() -> None:
    output = (
        "[Full output (90000 chars) saved to /workspace/.legion/logs/abc-123.log]\n"
        + "\n".join(f"line {i} " + "y" * 80 for i in range(150))
    )
    messages = [{"role": "user", "content": "Build"}]
    messages += _turn("old", "bash", output, cmd="make")
    messages += _turn("new", "bash", "exit_code: 0", cmd="true")

    receipt = _assemble(messages)[2]["content"]

    assert "/workspace/.legion/logs/abc-123.log" in receipt
    assert "read_file" in receipt


def test_read_receipt_has_no_excerpt() -> None:
    messages = [{"role": "user", "content": "Read"}]
    messages += _turn("old", "read_file", "z" * 20_000, path="app/a.py")
    messages += _turn("new", "read_file", "latest", path="app/b.py")

    receipt = _assemble(messages)[2]["content"]

    assert receipt.startswith("[context receipt: tool=read_file")
    assert "Excerpt" not in receipt


def test_short_output_is_not_replaced_by_a_larger_receipt() -> None:
    messages = [{"role": "user", "content": "Check"}]
    messages += _turn("old", "bash", "exit_code: 0\nstdout:\nok", cmd="echo ok")
    messages += _turn("new", "bash", "exit_code: 0", cmd="true")

    assembled = _assemble(messages)

    assert assembled[2]["content"] == "exit_code: 0\nstdout:\nok"
