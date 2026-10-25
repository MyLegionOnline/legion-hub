import pytest

from app.core.model_pricing import cache_cost_adjustment, cache_prices, compute_cost, model_prices
from app.core.provider_cache import extract_cache_usage


def test_agent_prices_win_over_the_known_table():
    cfg = {"input_price": 0.065, "output_price": 0.18}
    cost = compute_cost("deepseek/deepseek-v4-flash", 1_000_000, 1_000_000, cfg)
    assert cost == pytest.approx(0.245)
    # Even for a model we know: the agent's own price is the source of truth.
    assert model_prices("claude-sonnet-4-6", cfg) == (0.065, 0.18)


def test_explicit_zero_price_means_free():
    cfg = {"input_price": 0, "output_price": 0}
    assert compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, cfg) == 0


def test_price_strings_from_the_form_are_accepted():
    cfg = {"input_price": "3", "output_price": "15"}
    assert compute_cost("x", 1_000_000, 0, cfg) == pytest.approx(3.0)


def test_no_agent_price_falls_back_to_known_models_only():
    assert compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0, {}) == pytest.approx(0.8)
    assert compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0, {"input_price": None}) == pytest.approx(0.8)
    assert compute_cost("z-ai/glm-5.3-flash", 36_720_000, 443_500, {"model": "z-ai/glm-5.3-flash"}) == 0
    assert compute_cost("z-ai/glm-5.3-flash", 36_720_000, 443_500) == 0


def test_garbage_price_is_ignored():
    assert compute_cost("unknown", 1_000_000, 0, {"input_price": "abc", "output_price": -1}) == 0


# ── provider-cached prompt tokens ────────────────────────────────────────────


def test_openrouter_cache_read_is_discounted_from_input():
    # prompt_tokens already contains the cached tokens: only the difference.
    cfg = {"input_price": 0.065, "output_price": 0.18, "cache_read_price": 0.013}
    delta = cache_cost_adjustment(
        1_000_000, 0, included_in_input=True, model="deepseek/x", llm_config=cfg
    )
    assert delta == pytest.approx(0.013 - 0.065)
    full = compute_cost("deepseek/x", 1_000_000, 0, cfg)
    assert full + delta == pytest.approx(0.013)


def test_anthropic_cache_tokens_are_added_at_cache_price():
    # input_tokens excludes cache reads/writes: they are charged on top.
    delta = cache_cost_adjustment(
        1_000_000, 1_000_000, included_in_input=False, model="claude-sonnet-4-6"
    )
    assert delta == pytest.approx(0.3 + 3.75)


def test_unset_cache_price_costs_like_input():
    cfg = {"input_price": 2.0, "output_price": 8.0}
    assert cache_prices("x", cfg) == (2.0, 2.0)
    assert cache_cost_adjustment(500_000, 0, included_in_input=True, model="x", llm_config=cfg) == 0
    assert cache_cost_adjustment(
        500_000, 0, included_in_input=False, model="x", llm_config=cfg
    ) == pytest.approx(1.0)


def test_free_model_cache_is_free():
    cfg = {"input_price": 0, "output_price": 0}
    assert cache_cost_adjustment(10**6, 10**6, included_in_input=False, model="x", llm_config=cfg) == 0


def test_usage_shape_decides_whether_cache_is_inside_input():
    anthropic = extract_cache_usage(
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 900}
    )
    assert anthropic.read_tokens == 900 and anthropic.included_in_input is False

    openrouter = extract_cache_usage(
        {"prompt_tokens": 1000, "completion_tokens": 5,
         "prompt_tokens_details": {"cached_tokens": 900, "cache_write_tokens": 50}}
    )
    assert (openrouter.read_tokens, openrouter.write_tokens) == (900, 50)
    assert openrouter.included_in_input is True
