"""
Stage 2 tests: context-window enforcement (window mode is now the default).

No DB required — pure unit tests on ContextAssembler and proxy_session helpers.

Covers:
  - TestAssemblerWindowModeRejection        — assembler rejects with "context_window_limit"
  - TestContextWindowLimitReachedException  — new exception class fields
"""
from __future__ import annotations

import importlib
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.core.context_policy as policy_mod
from app.core.context_assembler import ContextAssembler
from app.core.context_policy import ContextPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy(
    *,
    context_window_limit: int = 100_000,
    output_reserve: int = 1_000,
    input_cap: int = 200_000,
    model_context_limit: int = 200_000,
    active_tool_result_cap: int = 10_000,
    keep_raw_results: int = 3,
    checkpoint_cap: int = 600,
    enabled: bool = True,
) -> ContextPolicy:
    return ContextPolicy(
        enabled=enabled,
        input_cap=input_cap,
        active_tool_result_cap=active_tool_result_cap,
        keep_raw_results=keep_raw_results,
        output_reserve=output_reserve,
        model_context_limit=model_context_limit,
        checkpoint_cap=checkpoint_cap,
        context_window_limit=context_window_limit,
    )


def _assembler(policy: ContextPolicy) -> ContextAssembler:
    from app.core.token_counter import TokenCounter

    class _FixedCounter(TokenCounter):
        """Returns len(str(obj)) // 4 as token count — predictable in tests."""
        def count(self, obj: Any) -> int:
            import json
            if isinstance(obj, str):
                return max(1, len(obj) // 4)
            return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)

        def truncate(self, text: str, max_tokens: int) -> str:
            return text[: max_tokens * 4]

    return ContextAssembler(policy, counter=_FixedCounter())


def _big_message(char_count: int) -> dict:
    """Single user message whose content is approximately char_count characters."""
    return {"role": "user", "content": "x" * char_count}


# ---------------------------------------------------------------------------
# TestAssemblerWindowModeRejection
# ---------------------------------------------------------------------------

class TestAssemblerWindowModeRejection:
    """Assembler emits rejected_reason='context_window_limit' in window mode
    when total + output_reserve > context_window_limit."""

    def test_rejects_when_prompt_exceeds_window_limit(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        # context_window_limit = 5_000 tokens; output_reserve = 1_000
        # → hard budget for prompt = 4_000 tokens
        # Message is ~6_000 tokens (24_000 chars / 4) → should reject
        p = _policy(context_window_limit=5_000, output_reserve=1_000,
                    input_cap=200_000, model_context_limit=200_000)
        assembler = asm_mod.ContextAssembler(p)
        messages = [_big_message(24_000)]  # ~6_000 tokens

        assembly = assembler.build(
            system_prompt="",
            tool_schemas=[],
            messages=messages,
        )
        assert assembly.rejected_reason == "context_window_limit"
        assert not assembly.allowed

    def test_does_not_reject_when_within_window_limit(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        # context_window_limit = 50_000; output_reserve = 1_000
        # → budget = 49_000 tokens
        # Message is ~100 tokens (400 chars / 4) → well within limit
        p = _policy(context_window_limit=50_000, output_reserve=1_000,
                    input_cap=200_000, model_context_limit=200_000)
        assembler = asm_mod.ContextAssembler(p)
        messages = [_big_message(400)]  # ~100 tokens

        assembly = assembler.build(
            system_prompt="",
            tool_schemas=[],
            messages=messages,
        )
        assert assembly.allowed
        assert assembly.rejected_reason is None

    def test_rejected_reason_exact_boundary(self, monkeypatch):
        """total + output_reserve == context_window_limit + 1 → reject."""
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        from app.core.token_counter import TokenCounter

        class _ExactCounter(TokenCounter):
            """Returns fixed value regardless of input."""
            def __init__(self, value: int):
                super().__init__()
                self._value = value

            def count(self, obj: Any) -> int:
                return self._value

            def truncate(self, text: str, max_tokens: int) -> str:
                return text[:max_tokens]

        # window_limit=10_000, output_reserve=1_000 → prompt budget = 9_000
        # counter returns 9_001 → total + output_reserve = 10_001 > 10_000 → reject
        p = _policy(context_window_limit=10_000, output_reserve=1_000,
                    input_cap=200_000, model_context_limit=200_000)
        assembler = asm_mod.ContextAssembler(p, counter=_ExactCounter(9_001))
        assembly = assembler.build(
            system_prompt="",
            tool_schemas=[],
            messages=[{"role": "user", "content": "hi"}],
        )
        assert assembly.rejected_reason == "context_window_limit"

    def test_no_session_remaining_needed_in_window_mode(self, monkeypatch):
        """In window mode, passing session_remaining_tokens=None still enforces
        context_window_limit (no session_budget fallback needed)."""
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        p = _policy(context_window_limit=5_000, output_reserve=1_000,
                    input_cap=200_000, model_context_limit=200_000)
        assembler = asm_mod.ContextAssembler(p)
        messages = [_big_message(24_000)]  # ~6_000 tokens

        assembly = assembler.build(
            system_prompt="",
            tool_schemas=[],
            messages=messages,
            session_remaining_tokens=None,
        )
        assert assembly.rejected_reason == "context_window_limit"

    def test_window_mode_ignores_legacy_session_budget(self, monkeypatch):
        """A low cumulative remainder cannot reject an otherwise valid window."""
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        assembly = asm_mod.ContextAssembler(_policy()).build(
            system_prompt="",
            tool_schemas=[],
            messages=[{"role": "user", "content": "continue"}],
            session_remaining_tokens=1,
        )

        assert assembly.allowed
        assert assembly.rejected_reason is None

    def test_cumulative_mode_keeps_legacy_session_budget_rollback(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "cumulative")
        importlib.reload(policy_mod)
        import app.core.context_assembler as asm_mod
        importlib.reload(asm_mod)

        assembly = asm_mod.ContextAssembler(_policy()).build(
            system_prompt="",
            tool_schemas=[],
            messages=[{"role": "user", "content": "continue"}],
            session_remaining_tokens=1,
        )

        assert not assembly.allowed
        assert assembly.rejected_reason == "session_budget"


# ---------------------------------------------------------------------------
# TestContextWindowLimitReachedException
# ---------------------------------------------------------------------------

class TestContextWindowLimitReachedException:
    def test_reason_attribute(self):
        from app.core.proxy_session import ContextWindowLimitReached
        exc = ContextWindowLimitReached(
            working=[],
            tokens_in=100,
            tokens_out=50,
            model="test-model",
            context_window_limit=30_000,
        )
        assert exc.reason == "context_window"
        assert exc.context_window_limit == 30_000

    def test_tokens_limit_matches_context_window_limit(self):
        from app.core.proxy_session import ContextWindowLimitReached
        exc = ContextWindowLimitReached(
            working=[],
            tokens_in=0,
            tokens_out=0,
            model="m",
            context_window_limit=42_000,
        )
        assert exc.tokens_limit == 42_000

    def test_is_subclass_of_tool_limit_reached(self):
        from app.core.proxy_session import ContextWindowLimitReached, ToolLimitReached
        assert issubclass(ContextWindowLimitReached, ToolLimitReached)

    def test_caught_by_tool_limit_reached_handler(self):
        from app.core.proxy_session import ContextWindowLimitReached, ToolLimitReached
        exc = ContextWindowLimitReached([], 0, 0, "m", context_window_limit=10_000)
        caught = False
        try:
            raise exc
        except ToolLimitReached as e:
            caught = True
            assert e.reason == "context_window"
        assert caught

    def test_default_context_window_limit_uses_token_limit(self):
        from app.core.proxy_session import ContextWindowLimitReached
        from app.core.token_budget import TOKEN_LIMIT
        exc = ContextWindowLimitReached([], 0, 0, "m")
        assert exc.context_window_limit == TOKEN_LIMIT

    def test_model_context_rejection_keeps_its_diagnostic_reason(self):
        from app.core.proxy_session import PromptCapLimitReached

        exc = PromptCapLimitReached(
            [],
            0,
            0,
            "m",
            rejected_reason="model_context_limit",
            estimated_input_tokens=128_000,
            cap=120_000,
        )

        assert exc.reason == "model_context_limit"
        assert exc.rejected_reason == "model_context_limit"


# ---------------------------------------------------------------------------
# TestProxyLoopWindowMode
# ---------------------------------------------------------------------------

class TestProxyLoopWindowMode:
    @pytest.mark.asyncio
    async def test_cumulative_usage_does_not_block_next_fitting_request(self, monkeypatch):
        """The real proxy-loop must not turn 110k historic usage into a cap.

        The provider call is the regression boundary: old code calculated a
        zero ``session_remaining_tokens`` and rejected before reaching it.
        """
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        from app.core import proxy_session

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="continued", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=250, completion_tokens=10),
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        policy = _policy(
            context_window_limit=100_000,
            input_cap=100_000,
            model_context_limit=200_000,
            output_reserve=1_000,
        )

        with (
            patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
            patch.object(proxy_session, "AsyncOpenAI", return_value=client),
            patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
            patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
            patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        ):
            result = await proxy_session._call_llm_with_tools(
                {"provider": "openai", "api_key": "test", "model": "test-model"},
                "system",
                [{"role": "user", "content": "A small next request."}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=MagicMock(),
                budget_used=110_000,
                on_exhaust="raise",
                context_role="agent",
            )

        assert result[0] == "continued"
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_real_window_overflow_never_calls_provider(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        from app.core import proxy_session

        create = AsyncMock()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        policy = _policy(
            context_window_limit=1_024,
            input_cap=100_000,
            model_context_limit=200_000,
            output_reserve=256,
        )

        with (
            patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
            patch.object(proxy_session, "AsyncOpenAI", return_value=client),
            patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
            patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
            patch.object(proxy_session, "record_context_measurement", AsyncMock()),
            pytest.raises(proxy_session.ContextWindowLimitReached),
        ):
            await proxy_session._call_llm_with_tools(
                {"provider": "openai", "api_key": "test", "model": "test-model"},
                "system " * 3_000,
                [{"role": "user", "content": "continue"}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=MagicMock(),
                on_exhaust="raise",
                context_role="agent",
            )

        create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_four_large_turns_continue_after_cumulative_usage_reaches_window(self, monkeypatch):
        """Four independently fitting requests are not a 100k session budget.

        ``prompt_tokens`` deliberately reports 25k on every fake provider
        response.  The fourth request is therefore the regression boundary:
        the session has consumed 100k in total, but no individual request is
        larger than the configured 100k context window.
        """
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        from app.core import proxy_session

        def response(turn: int) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=f"turn {turn}", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=25_000, completion_tokens=10),
            )

        create = AsyncMock(side_effect=[response(turn) for turn in range(4)])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        policy = _policy(
            context_window_limit=100_000,
            input_cap=100_000,
            model_context_limit=200_000,
            output_reserve=1_000,
        )
        # A realistically large request; every call still fits one window.
        system_prompt = "turn-context " * 20_000

        with (
            patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
            patch.object(proxy_session, "AsyncOpenAI", return_value=client),
            patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
            patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
            patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        ):
            cumulative = 0
            for turn in range(4):
                result = await proxy_session._call_llm_with_tools(
                    {"provider": "openai", "api_key": "test", "model": "test-model"},
                    system_prompt,
                    [{"role": "user", "content": f"continue turn {turn}"}],
                    room_id=uuid.uuid4(),
                    session_id=uuid.uuid4(),
                    task_id=None,
                    db=MagicMock(),
                    budget_used=cumulative,
                    on_exhaust="raise",
                    context_role="agent",
                )
                assert result[0] == f"turn {turn}"
                cumulative += result[2] + result[3]

        assert cumulative >= 100_000
        assert create.await_count == 4

    @pytest.mark.asyncio
    async def test_proxy_compacts_old_tool_results_before_provider_request(self, monkeypatch):
        """The actual proxy assembly must replace old tool output with receipts."""
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        from app.core import proxy_session

        history: list[dict[str, Any]] = [{"role": "user", "content": "inspect repository"}]
        for index in range(4):
            call_id = f"tool-{index}"
            history.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"large-file.txt"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "old tool output " * 6_000,
                },
            ])

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="compacted and continued", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2_000, completion_tokens=10),
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        policy = _policy(
            context_window_limit=10_000,
            input_cap=10_000,
            model_context_limit=20_000,
            output_reserve=1_000,
            active_tool_result_cap=300,
            keep_raw_results=1,
        )
        persisted = AsyncMock()

        with (
            patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
            patch.object(proxy_session, "AsyncOpenAI", return_value=client),
            patch.object(proxy_session, "persist_context_assembly", persisted),
            patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
            patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        ):
            result = await proxy_session._call_llm_with_tools(
                {"provider": "openai", "api_key": "test", "model": "test-model"},
                "system",
                history,
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=MagicMock(),
                on_exhaust="raise",
                context_role="agent",
            )

        assembly = persisted.await_args.args[0]
        assert result[0] == "compacted and continued"
        assert create.await_count == 1
        assert assembly.allowed
        assert assembly.evicted_results == 4
        assert assembly.estimated_input_tokens + policy.output_reserve <= policy.context_window_limit
        assert any(
            message.get("role") == "tool"
            and str(message.get("content", "")).startswith("[context receipt:")
            for message in assembly.messages
        )

    @pytest.mark.asyncio
    async def test_provider_ceiling_is_not_mapped_to_token_limit(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        from app.core import proxy_session

        create = AsyncMock()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        policy = _policy(
            context_window_limit=100_000,
            input_cap=100_000,
            model_context_limit=2_000,
            output_reserve=256,
        )

        with (
            patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
            patch.object(proxy_session, "AsyncOpenAI", return_value=client),
            patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
            patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
            patch.object(proxy_session, "record_context_measurement", AsyncMock()),
            pytest.raises(proxy_session.PromptCapLimitReached) as caught,
        ):
            await proxy_session._call_llm_with_tools(
                {"provider": "openai", "api_key": "test", "model": "test-model"},
                "provider-ceiling " * 3_000,
                [{"role": "user", "content": "continue"}],
                room_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                task_id=None,
                db=MagicMock(),
                on_exhaust="raise",
                context_role="agent",
            )

        assert caught.value.reason == "model_context_limit"
        assert caught.value.rejected_reason == "model_context_limit"
        create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_over_budget_session_differs_by_explicit_mode(self, monkeypatch):
        """``cumulative`` remains a deliberate rollback-only behaviour."""
        from app.core import proxy_session

        policy = _policy(
            context_window_limit=100_000,
            input_cap=100_000,
            model_context_limit=200_000,
            output_reserve=1_000,
        )

        async def run(mode: str) -> AsyncMock:
            monkeypatch.setenv("CONTEXT_LIMIT_MODE", mode)
            importlib.reload(policy_mod)
            create = AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="continued", tool_calls=None),
                )],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
            ))
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            with (
                patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=policy)),
                patch.object(proxy_session, "AsyncOpenAI", return_value=client),
                patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
                patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
                patch.object(proxy_session, "record_context_measurement", AsyncMock()),
            ):
                if mode == "cumulative":
                    with pytest.raises(proxy_session.TokenBudgetLimitReached):
                        await proxy_session._call_llm_with_tools(
                            {"provider": "openai", "api_key": "test", "model": "test-model"},
                            "system",
                            [{"role": "user", "content": "small next request"}],
                            room_id=uuid.uuid4(),
                            session_id=uuid.uuid4(),
                            task_id=None,
                            db=MagicMock(),
                            budget_used=110_000,
                            on_exhaust="raise",
                            context_role="agent",
                        )
                else:
                    result = await proxy_session._call_llm_with_tools(
                        {"provider": "openai", "api_key": "test", "model": "test-model"},
                        "system",
                        [{"role": "user", "content": "small next request"}],
                        room_id=uuid.uuid4(),
                        session_id=uuid.uuid4(),
                        task_id=None,
                        db=MagicMock(),
                        budget_used=110_000,
                        on_exhaust="raise",
                        context_role="agent",
                    )
                    assert result[0] == "continued"
            return create

        window_create = await run("window")
        cumulative_create = await run("cumulative")

        assert window_create.await_count == 1
        cumulative_create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Teardown — restore module state after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_modules():
    """Reload modules after each test so monkeypatched env doesn't leak."""
    yield
    importlib.reload(policy_mod)
    import app.core.context_assembler as asm_mod
    importlib.reload(asm_mod)
