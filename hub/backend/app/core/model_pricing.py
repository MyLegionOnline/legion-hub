"""Session cost from token counts — the one place prices are looked up.

The source of truth is the agent's own ``llm_config``: ``input_price`` /
``output_price`` (and ``cache_read_price`` / ``cache_write_price`` for
provider-cached prompt tokens) in USD per 1M tokens, auto-filled from the
provider catalog in the agent form and editable there. An explicit ``0``
means a free model.

Only when the agent has no price at all do we fall back to the small table of
well-known models below. An unknown model with no configured price costs 0 —
never a guessed default, which used to bill every OpenRouter/OpenAI-compatible
model as Claude Sonnet ($3 / $15).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# (input_price_per_1M_tokens, output_price_per_1M_tokens)
_KNOWN_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
}


def _price(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


def model_prices(
    model: Optional[str], llm_config: Optional[Mapping[str, Any]] = None
) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for this call."""
    if llm_config:
        in_p = _price(llm_config.get("input_price"))
        out_p = _price(llm_config.get("output_price"))
        if in_p is not None or out_p is not None:
            return in_p or 0.0, out_p or 0.0
    return _KNOWN_MODEL_PRICING.get(model or "", (0.0, 0.0))


def compute_cost(
    model: Optional[str],
    tokens_in: int,
    tokens_out: int,
    llm_config: Optional[Mapping[str, Any]] = None,
) -> float:
    in_p, out_p = model_prices(model, llm_config)
    return round(((tokens_in or 0) * in_p + (tokens_out or 0) * out_p) / 1_000_000, 8)


# Anthropic's published cache multipliers relative to the input price
# (5-minute write). Used only for known Claude models without agent prices.
_CLAUDE_CACHE_READ_FACTOR = 0.1
_CLAUDE_CACHE_WRITE_FACTOR = 1.25


def cache_prices(
    model: Optional[str], llm_config: Optional[Mapping[str, Any]] = None
) -> tuple[float, float]:
    """(cache read, cache write) USD per 1M tokens.

    ``cache_read_price`` / ``cache_write_price`` from the agent form win; an
    unset one costs the same as a regular input token, so an unpriced cache
    is never cheaper than it would be without caching.
    """
    in_p, _ = model_prices(model, llm_config)
    read_p = _price((llm_config or {}).get("cache_read_price"))
    write_p = _price((llm_config or {}).get("cache_write_price"))
    agent_priced = llm_config and (
        _price(llm_config.get("input_price")) is not None
        or _price(llm_config.get("output_price")) is not None
    )
    if not agent_priced and (model or "") in _KNOWN_MODEL_PRICING and (model or "").startswith("claude-"):
        read_p = in_p * _CLAUDE_CACHE_READ_FACTOR if read_p is None else read_p
        write_p = in_p * _CLAUDE_CACHE_WRITE_FACTOR if write_p is None else write_p
    return (in_p if read_p is None else read_p, in_p if write_p is None else write_p)


def cache_cost_adjustment(
    read_tokens: int,
    write_tokens: int,
    *,
    included_in_input: bool,
    model: Optional[str],
    llm_config: Optional[Mapping[str, Any]] = None,
) -> float:
    """Cost correction for provider-cached prompt tokens of one request.

    ``compute_cost`` prices every reported input token at the input price.
    OpenAI-compatible APIs (OpenRouter included) count cached tokens inside
    ``prompt_tokens``, so the correction is the price difference; Anthropic
    reports them outside ``input_tokens``, so they are added at full cache
    price. The result may be negative (a cache read is cheaper than input).
    """
    if not read_tokens and not write_tokens:
        return 0.0
    in_p, _ = model_prices(model, llm_config)
    read_p, write_p = cache_prices(model, llm_config)
    base = in_p if included_in_input else 0.0
    delta = (read_tokens or 0) * (read_p - base) + (write_tokens or 0) * (write_p - base)
    return round(delta / 1_000_000, 8)
