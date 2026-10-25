"""Stage 5 observability for assembled prompts and preflight rejections."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.core.context_assembler import (
    ContextAssembly,
    log_preflight_observation,
    log_preflight_rejection,
    persist_context_assembly,
    preflight_rejection_details,
    preflight_observation_details,
)
from app.core.context_policy import ContextPolicy
from tests._hub_paths import hub_script


def _assembly(tokens: int, reason: str | None = None) -> ContextAssembly:
    return ContextAssembly(
        messages=[],
        estimated_input_tokens=tokens,
        active_tool_tokens=0,
        active_tool_tokens_by_name={},
        evicted_tokens=0,
        evicted_results=0,
        dropped_messages=0,
        dropped_tokens=0,
        checkpoint={},
        rejected_reason=reason,
    )


def _policy(
    *, input_cap: int = 16_000, model_context_limit: int = 100_000
) -> ContextPolicy:
    return ContextPolicy(
        enabled=True,
        input_cap=input_cap,
        active_tool_result_cap=8_000,
        keep_raw_results=2,
        output_reserve=4_000,
        model_context_limit=model_context_limit,
        checkpoint_cap=2_000,
        context_window_limit=100_000,
    )


class _Row:
    context_snapshot = None


class _Db:
    def __init__(self, row):
        self.row = row
        self.added = []

    async def get(self, _model, _session_id):
        return self.row

    def add(self, row):
        self.added.append(row)


@pytest.mark.asyncio
async def test_successful_assemblies_are_persisted_for_size_report():
    row = _Row()
    db = _Db(row)
    session_id = uuid4()

    await persist_context_assembly(
        _assembly(12_345),
        session_id=session_id,
        db=db,
        role="agent",
        request_kind="tool_loop/openai",
    )

    gateway = row.context_snapshot["context_gateway"]
    assert gateway["last_assembled_input_tokens"] == 12_345
    assert gateway["last_successful_assembled_input_tokens"] == 12_345
    assert gateway["max_successful_assembled_input_tokens"] == 12_345
    assert gateway["last_rejected_reason"] is None
    assert gateway["successful_assemblies"][-1]["tokens"] == 12_345
    assert gateway["successful_assemblies"][-1]["role"] == "agent"
    assert gateway["successful_assemblies"][-1]["kind"] == "tool_loop/openai"
    assert db.added == [row]


@pytest.mark.asyncio
async def test_rejection_does_not_pollute_successful_prompt_history():
    row = _Row()
    db = _Db(row)
    session_id = uuid4()
    await persist_context_assembly(_assembly(900), session_id=session_id, db=db)
    await persist_context_assembly(
        _assembly(1_200, "prompt_cap"), session_id=session_id, db=db
    )

    gateway = row.context_snapshot["context_gateway"]
    assert len(gateway["successful_assemblies"]) == 1
    assert gateway["last_successful_assembled_input_tokens"] == 900
    assert gateway["last_rejected_reason"] == "prompt_cap"


@pytest.mark.parametrize(
    ("reason", "input_cap", "model_cap", "remaining", "cap", "source"),
    [
        ("prompt_cap", 16_000, 100_000, None, 16_000, "input_cap"),
        ("prompt_cap", 100_000, 8_000, None, 8_000, "model_context_limit"),
        ("model_context_limit", 16_000, 8_000, None, 8_000, "model_context_limit"),
        ("context_window_limit", 16_000, 100_000, None, 100_000, "context_window_limit"),
        ("session_budget", 16_000, 100_000, 123, 123, "session_remaining_tokens"),
    ],
)
def test_preflight_rejection_details_report_effective_cap(
    reason, input_cap, model_cap, remaining, cap, source
):
    policy = _policy(input_cap=input_cap, model_context_limit=model_cap)
    details = preflight_rejection_details(
        _assembly(16_001, reason),
        session_id=uuid4(),
        role="agent",
        request_kind="tool_loop",
        policy=policy,
        session_remaining_tokens=remaining,
    )

    assert details["event"] == "context_preflight_rejected"
    assert details["tokens"] == 16_001
    assert details["cap"] == cap
    assert details["cap_source"] == source
    assert details["role"] == "agent"
    assert details["kind"] == "tool_loop"
    assert details["mode"] == "window"
    assert details["assembled_input_tokens"] == 16_001
    assert details["output_reserve"] == 4_000
    assert details["app_cap"] == 100_000
    assert details["model_cap"] == model_cap
    assert details["cumulative_tokens"] == 0


def test_preflight_rejection_is_a_json_log_record(caplog):
    session_id = uuid4()
    with caplog.at_level("WARNING"):
        log_preflight_rejection(
            _assembly(16_001, "prompt_cap"),
            session_id=session_id,
            role="agent",
            request_kind="tool_loop",
            policy=_policy(),
        )

    record = next(r for r in caplog.records if r.message.startswith("context_preflight_rejected "))
    payload = json.loads(record.message.removeprefix("context_preflight_rejected "))
    assert payload == {
        "event": "context_preflight_rejected",
        "session_id": str(session_id),
        "role": "agent",
        "kind": "tool_loop",
        "reason": "prompt_cap",
        "tokens": 16_001,
        "cap": 16_000,
        "cap_source": "input_cap",
        "mode": "window",
        "assembled_input_tokens": 16_001,
        "output_reserve": 4_000,
        "app_cap": 100_000,
        "model_cap": 100_000,
        "cumulative_tokens": 0,
    }


def test_every_preflight_emits_window_and_cumulative_diagnostics(caplog):
    session_id = uuid4()
    with caplog.at_level("INFO"):
        log_preflight_observation(
            _assembly(12_345),
            session_id=session_id,
            role="agent",
            request_kind="tool_loop/openai",
            policy=_policy(),
            cumulative_tokens=110_000,
        )

    record = next(r for r in caplog.records if r.message.startswith("context_preflight "))
    payload = json.loads(record.message.removeprefix("context_preflight "))
    assert payload == {
        "event": "context_preflight",
        "session_id": str(session_id),
        "role": "agent",
        "kind": "tool_loop/openai",
        "reason": "allowed",
        "allowed": True,
        "tokens": 12_345,
        "cap": 100_000,
        "cap_source": "context_window_limit",
        "mode": "window",
        "assembled_input_tokens": 12_345,
        "output_reserve": 4_000,
        "app_cap": 100_000,
        "model_cap": 100_000,
        "cumulative_tokens": 110_000,
    }


def test_preflight_observation_keeps_specific_rejection_reason():
    details = preflight_observation_details(
        _assembly(100_001, "context_window_limit"),
        session_id=uuid4(),
        role="agent",
        request_kind="tool_loop",
        policy=_policy(),
    )
    assert details["allowed"] is False
    assert details["reason"] == "context_window_limit"
    assert details["cap_source"] == "context_window_limit"


def test_session_budget_rejection_marks_legacy_guard_with_effective_mode():
    from app.core.metrics import record_context_assembly

    with (
        patch("app.core.metrics.context_preflight_rejections_total") as rejections,
        patch("app.core.metrics.record_legacy_cumulative_guard_hit") as legacy_hits,
    ):
        record_context_assembly(_assembly(1_000, "session_budget"))

    rejections.labels.assert_called_once_with(mode="window", reason="session_budget")
    legacy_hits.assert_called_once_with(mode="window")


def test_top_prompts_sql_report_uses_successful_assemblies_and_24h_window():
    script = hub_script("top_prompt_sizes_24h.sql")
    text = script.read_text(encoding="utf-8")

    assert "successful_assemblies" in text
    assert "INTERVAL '24 hours'" in text
    assert "DISTINCT ON (session_id)" in text
