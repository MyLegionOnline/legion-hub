from __future__ import annotations

import uuid

from app.core.memory_layer import (
    _HANDOFF_RECENT_CHARS_PER_TURN,
    _HANDOFF_RECENT_TURNS_MAX,
    _HANDOFF_RENDER_MAX_CHARS,
    _render_task_handoff,
)
from app.models.task import Task, TaskHandoff, TaskState
from app.schemas.task import TaskOut


def test_handoff_renderer_includes_scoped_checkpoint():
    rendered = _render_task_handoff(
        {
            "summary": "Continuation after the limit.",
            "completed_work": ["read schema.prisma"],
            "artifacts": ["apps/web/prisma/schema.prisma"],
            "files_read": ["apps/web/package.json"],
            "remaining_work": "Run the tests.",
        }
    )

    assert "Task Continuation Context" in rendered
    assert "schema.prisma" in rendered
    assert "Files already read" in rendered
    assert "DO NOT" in rendered


def test_handoff_renderer_caps_recent_context_turns_and_chars():
    """A bloated snapshot must not overflow the render budget.

    Regression: continuation tasks used to fail preflight with prompt_cap
    because a 20-turn × 4000-char recent_context alone busted input_cap.
    """
    huge_turn_content = "X" * 20_000
    handoff = {
        "summary": "s",
        "recent_context": [
            {"role": "user", "content": f"turn-{i}-{huge_turn_content}"}
            for i in range(30)
        ],
    }

    rendered = _render_task_handoff(handoff)

    assert len(rendered) <= _HANDOFF_RENDER_MAX_CHARS
    # Only the last N turns survive; earlier ones must be dropped.
    assert "turn-29" in rendered
    assert "turn-0-" not in rendered
    kept_turn_lines = [line for line in rendered.splitlines() if line.startswith("**[user]**")]
    assert len(kept_turn_lines) <= _HANDOFF_RECENT_TURNS_MAX
    # Each surviving turn is truncated to the per-turn char cap plus the
    # role marker and the truncation suffix — never the full 20 000 chars.
    for line in kept_turn_lines:
        assert len(line) < _HANDOFF_RECENT_CHARS_PER_TURN + 200


def test_handoff_renderer_hard_cap_when_all_sections_are_bloated():
    """Even without recent_context, all other sections combined stay bounded."""
    handoff = {
        "summary": "S" * 100_000,
        "completed_work": ["done " + "A" * 2_000 for _ in range(200)],
        "artifacts": [f"path/file-{i}.bin" for i in range(500)],
        "decisions": "D" * 100_000,
        "tests_and_results": ["pytest " + "T" * 2_000 for _ in range(200)],
        "remaining_work": "R" * 100_000,
        "files_read": [f"read/file-{i}.py" for i in range(500)],
        "read_manifest": [
            {"path": f"m/{i}.py", "fingerprint": "abc", "ranges": [[1, 10]]}
            for i in range(500)
        ],
    }

    rendered = _render_task_handoff(handoff)

    assert len(rendered) <= _HANDOFF_RENDER_MAX_CHARS
    # The critical trailing rules block must survive truncation so the agent
    # still sees the operating rules for continuation tasks.
    assert "CRITICAL — token budget rules" in rendered or "handoff truncated" in rendered


def test_handoff_renderer_respects_explicit_max_chars_override():
    handoff = {"summary": "s", "recent_context": [{"role": "user", "content": "x" * 10_000}]}
    rendered = _render_task_handoff(handoff, max_chars=1_000)
    assert len(rendered) <= 1_000


def test_prompt_cap_exception_is_distinct_from_token_budget():
    """Regression guard for the misleading '0 of 100 000 tokens' notice.

    Preflight rejection due to an oversized non-compactable prompt (system
    prompt + tool schemas + first user turn) must NOT be reported as a
    cumulative-token-budget exhaustion, because tokens_used is 0.
    """
    from app.core.proxy_session import (
        ContextWindowLimitReached,
        PromptCapLimitReached,
        TokenBudgetLimitReached,
        ToolLimitReached,
    )

    exc = PromptCapLimitReached(
        [],
        0,
        0,
        "claude-haiku-4-5-20251001",
        tool_calls_used=0,
        tool_calls_limit=60,
        tokens_used=0,
        rejected_reason="prompt_cap",
    )

    assert exc.reason == "prompt_cap"
    assert exc.rejected_reason == "prompt_cap"
    assert isinstance(exc, ToolLimitReached)
    assert not isinstance(exc, TokenBudgetLimitReached)
    assert not isinstance(exc, ContextWindowLimitReached)


def test_continuation_task_and_handoff_model_contract():
    source_id = uuid.uuid4()
    target = Task(
        room_id=uuid.uuid4(),
        title="Continue work",
        continuation_of_task_id=source_id,
        state=TaskState.ASSIGNED,
    )
    handoff = TaskHandoff(
        source_task_id=source_id,
        target_task_id=target.id,
        summary="Checkpoint",
        context_snapshot={"artifacts": ["report.md"]},
    )

    assert handoff.context_snapshot["artifacts"] == ["report.md"]
    assert TaskOut.model_validate(target).continuation_of_task_id == source_id
    assert TaskState.CONTINUED.value == "continued"
