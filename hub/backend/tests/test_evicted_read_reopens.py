"""Compaction must reopen the reads it takes away.

``read_ledger`` answers "what can the agent currently see", and the read tools
refuse to resend anything it covers.  Compaction happens outside the tools, so
unless it reports what it removed, the ledger goes on claiming coverage for
text that is no longer in the prompt and every re-read comes back as a bare
receipt for content the agent has no way to look at.
"""
from __future__ import annotations

import json

import pytest

from app.core.context_assembler import ContextAssembler
from app.core.context_observations import (
    intersect_interval,
    mark_observations_evicted,
    mark_reads_evicted,
    record_read_interval,
    subtract_interval_list,
    take_evicted_observation,
)
from app.core.context_policy import ContextPolicy
from app.models.session import Session


class _FakeDB:
    """Just enough of AsyncSession for the context_state helpers.

    ``load_context_state`` type-checks the row, so this hands back a real
    (unsaved) Session whose ``context_state`` survives between calls.
    """

    def __init__(self):
        self.row = Session(context_state={})

    async def get(self, _model, _pk):
        return self.row

    def add(self, _value):
        return None

    async def flush(self):
        return None


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


def _tool_turn(call_id: str, tool: str, content: str, **args):
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


async def _read(db, session, path, start, end, ref="ref"):
    return await record_read_interval(
        db,
        session_id=session,
        path=path,
        source_fingerprint="fp",
        requested_start=start,
        requested_end=end,
        observation_ref=ref,
    )


def test_interval_helpers_are_exact():
    assert intersect_interval(5, 15, [[1, 10], [20, 30]]) == [[5, 10]]
    assert intersect_interval(40, 50, [[1, 10]]) == []
    assert subtract_interval_list([[1, 100]], [[20, 30]]) == [[1, 19], [31, 100]]
    assert subtract_interval_list([[1, 10]], [[1, 10]]) == []


@pytest.mark.asyncio
async def test_a_re_read_of_an_evicted_range_is_served_again():
    db, session = _FakeDB(), "11111111-1111-1111-1111-111111111111"

    assert await _read(db, session, "app/a.py", 1, 100) == [[1, 100]]
    # While the text is still in context, asking again is refused.
    assert await _read(db, session, "app/a.py", 1, 100) == []

    await mark_reads_evicted(
        db, session_id=session, reads=[{"path": "app/a.py", "start": 1, "end": 100}]
    )

    assert await _read(db, session, "app/a.py", 1, 100) == [[1, 100]]
    # Serving it put it back in context, so the next ask is refused once more.
    assert await _read(db, session, "app/a.py", 1, 100) == []


@pytest.mark.asyncio
async def test_only_the_evicted_part_of_a_file_reopens():
    db, session = _FakeDB(), "22222222-2222-2222-2222-222222222222"
    await _read(db, session, "app/a.py", 1, 100)
    await _read(db, session, "app/a.py", 101, 200)

    await mark_reads_evicted(
        db, session_id=session, reads=[{"path": "app/a.py", "start": 1, "end": 100}]
    )

    assert await _read(db, session, "app/a.py", 1, 200) == [[1, 100]]


@pytest.mark.asyncio
async def test_whole_path_eviction_reopens_every_recorded_range():
    db, session = _FakeDB(), "33333333-3333-3333-3333-333333333333"
    await _read(db, session, "app/a.py", 1, 40)

    await mark_reads_evicted(
        db,
        session_id=session,
        reads=[{"path": "app/a.py", "start": None, "end": None}],
    )

    assert await _read(db, session, "app/a.py", 1, 40) == [[1, 40]]


@pytest.mark.asyncio
async def test_an_untouched_path_is_not_reopened():
    db, session = _FakeDB(), "55555555-5555-5555-5555-555555555555"
    await _read(db, session, "app/a.py", 1, 40)
    await _read(db, session, "app/b.py", 1, 40)

    await mark_reads_evicted(
        db, session_id=session, reads=[{"path": "app/a.py", "start": 1, "end": 40}]
    )

    assert await _read(db, session, "app/b.py", 1, 40) == []


@pytest.mark.asyncio
async def test_an_observation_eviction_mark_is_one_shot():
    db, session = _FakeDB(), "44444444-4444-4444-4444-444444444444"

    assert await take_evicted_observation(db, session_id=session, ref="ref-9") is False

    await mark_observations_evicted(db, session_id=session, refs=["ref-9"])
    assert await take_evicted_observation(db, session_id=session, ref="ref-9") is True
    # Consumed: the payload is back in context, so suppression applies again.
    assert await take_evicted_observation(db, session_id=session, ref="ref-9") is False


def test_the_assembler_reports_the_reads_it_evicts():
    messages = [{"role": "user", "content": "Study the repo"}]
    for index in range(6):
        messages += _tool_turn(
            f"call-{index}",
            "read_file",
            str(index) * 4_000,
            path=f"app/{index}.go",
            offset=1,
            limit=200,
        )

    assembly = ContextAssembler(_policy(input_cap=4_000, output_reserve=256)).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    assert assembly.evicted_reads, "an over-budget prompt must report its evictions"
    assert all(item["start"] == 1 and item["end"] == 200 for item in assembly.evicted_reads)
    evicted_paths = [item["path"] for item in assembly.evicted_reads]
    # Oldest first, and whatever stayed raw is not reported as gone.
    assert evicted_paths == [f"app/{i}.go" for i in range(len(evicted_paths))]
    assert "app/5.go" not in evicted_paths


def test_nothing_is_reported_evicted_while_the_prompt_fits():
    messages = [{"role": "user", "content": "Study the repo"}]
    for index in range(4):
        messages += _tool_turn(
            f"call-{index}", "read_file", str(index) * 500, path=f"app/{index}.go"
        )

    assembly = ContextAssembler(_policy()).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    assert assembly.evicted_reads == []
    assert assembly.evicted_observations == []


def test_the_assembler_reports_evicted_observation_refs():
    ref = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    messages = [{"role": "user", "content": "Search"}]
    messages += _tool_turn(
        "grep-1",
        "grep",
        "x" * 4_000 + f"\n[observation_ref={ref} pattern='HandleFunc']",
        pattern="HandleFunc",
    )
    messages += _tool_turn("call-late", "read_file", "z" * 4_000, path="app/z.go")

    assembly = ContextAssembler(_policy(input_cap=2_000, output_reserve=256)).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    assert assembly.evicted_observations == [ref]
