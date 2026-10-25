"""Stage 5 — durable vision output via context_observations.

Tests for _vision_obs_note (user_sessions.py) and the record_observation
dedup contract for mcp-vision payloads.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_observations import ObservationResult
from app.core.user_sessions import _VISION_OBS_HEAD_CHARS, _vision_obs_note


# ── helpers ──────────────────────────────────────────────────────────────────


def _vision_binding(tool_name: str = "describe_image") -> MagicMock:
    b = MagicMock()
    b.market_slug = "mcp-vision"
    b.tool_name = tool_name
    return b


def _non_vision_binding() -> MagicMock:
    b = MagicMock()
    b.market_slug = "mcp-searxng"
    b.tool_name = "search"
    return b


def _fake_obs(*, duplicate: bool = False, stale: bool = False) -> ObservationResult:
    artifact = MagicMock()
    artifact.id = uuid4()
    artifact.tool_name = "mcp__mcp-vision__describe_image"
    artifact.token_count = 42
    artifact.stale = stale
    return ObservationResult(artifact=artifact, duplicate=duplicate)


# ── note content ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vision_binding_note_contains_receipt_and_head():
    obs = _fake_obs()
    with patch("app.core.context_observations.record_observation", AsyncMock(return_value=obs)):
        note = await _vision_obs_note(
            _vision_binding(),
            "mcp__mcp-vision__describe_image",
            {"image_path": "images/ref.png"},
            "A 3-column card grid with blue accent",
            object(),
            uuid4(),
            uuid4(),
        )
    assert f"observation_read(ref=" in note
    assert str(obs.artifact.id) in note
    assert "A 3-column card grid" in note


@pytest.mark.asyncio
async def test_long_result_head_is_clamped_with_ellipsis():
    long_result = "d" * (_VISION_OBS_HEAD_CHARS + 200)
    obs = _fake_obs()
    with patch("app.core.context_observations.record_observation", AsyncMock(return_value=obs)):
        note = await _vision_obs_note(
            _vision_binding(),
            "mcp__mcp-vision__describe_image",
            {"image_path": "ref.png"},
            long_result,
            object(),
            uuid4(),
            uuid4(),
        )
    assert "…" in note
    # Head portion (after the receipt line) must be clamped
    head_part = note.split("\n", 1)[1] if "\n" in note else note
    assert len(head_part) <= _VISION_OBS_HEAD_CHARS + 1  # +1 for the ellipsis char


@pytest.mark.asyncio
async def test_short_result_has_no_ellipsis():
    short = "Just a short description."
    obs = _fake_obs()
    with patch("app.core.context_observations.record_observation", AsyncMock(return_value=obs)):
        note = await _vision_obs_note(
            _vision_binding(), "mcp__mcp-vision__describe_image",
            {"image_path": "ref.png"}, short, object(), uuid4(), uuid4(),
        )
    assert "…" not in note
    assert short in note


@pytest.mark.asyncio
async def test_dedup_hit_still_returns_receipt():
    """Duplicate observation → note still carries the observation_read receipt."""
    obs = _fake_obs(duplicate=True)
    with patch("app.core.context_observations.record_observation", AsyncMock(return_value=obs)):
        note = await _vision_obs_note(
            _vision_binding(), "mcp__mcp-vision__describe_image",
            {"image_path": "images/ref.png"}, "Same description",
            object(), uuid4(), uuid4(),
        )
    assert "observation_read(ref=" in note


# ── non-vision fall-through ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_vision_market_slug_falls_back_to_truncate():
    note = await _vision_obs_note(
        _non_vision_binding(),
        "mcp__mcp-searxng__search",
        {"query": "test"},
        "search results here",
        object(), uuid4(), uuid4(),
    )
    assert "observation_read" not in note
    assert "search results here" in note


@pytest.mark.asyncio
async def test_unknown_vision_tool_name_falls_back_to_truncate():
    binding = MagicMock()
    binding.market_slug = "mcp-vision"
    binding.tool_name = "some_future_tool"  # not in _MCP_VISION_TOOL_NAMES
    note = await _vision_obs_note(
        binding, "mcp__mcp-vision__some_future_tool",
        {}, "some result", object(), uuid4(), uuid4(),
    )
    assert "observation_read" not in note


@pytest.mark.asyncio
async def test_none_db_falls_back_to_truncate():
    note = await _vision_obs_note(
        _vision_binding(), "mcp__mcp-vision__describe_image",
        {"image_path": "ref.png"}, "A description",
        None,  # db=None
        uuid4(), uuid4(),
    )
    assert "observation_read" not in note
    assert "A description" in note


# ── resilience ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_error_falls_back_to_truncate():
    with patch("app.core.context_observations.record_observation", AsyncMock(side_effect=RuntimeError("DB down"))):
        note = await _vision_obs_note(
            _vision_binding(), "mcp__mcp-vision__describe_image",
            {"image_path": "ref.png"}, "A description",
            object(), uuid4(), uuid4(),
        )
    assert "observation_read" not in note
    assert "A description" in note


# ── record_observation dedup contract ────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_call_same_image_is_dedup_hit_note_still_has_receipt():
    """Two identical _vision_obs_note calls → both notes carry the receipt.

    The second call returns duplicate=True from record_observation (same ref).
    _vision_obs_note must include the receipt in both cases.
    """
    obs_first = _fake_obs(duplicate=False)
    obs_second = _fake_obs(duplicate=True)
    # Same artifact id so both receipts point to the same observation.
    obs_second.artifact.id = obs_first.artifact.id

    call_count = 0

    async def fake_record(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return obs_first if call_count == 1 else obs_second

    room_id = uuid4()
    session_id = uuid4()
    with patch("app.core.context_observations.record_observation", fake_record):
        note1 = await _vision_obs_note(
            _vision_binding(), "mcp__mcp-vision__describe_image",
            {"image_path": "images/ref.png"}, "A 3-column card grid",
            object(), room_id, session_id,
        )
        note2 = await _vision_obs_note(
            _vision_binding(), "mcp__mcp-vision__describe_image",
            {"image_path": "images/ref.png"}, "A 3-column card grid",
            object(), room_id, session_id,
        )

    assert call_count == 2, "record_observation should be called for each vision note"
    # Second ObservationResult must be marked duplicate by our fake
    assert obs_second.duplicate
    # Both notes must include the observation_read receipt
    assert "observation_read(ref=" in note1
    assert "observation_read(ref=" in note2
    # Both notes point to the same ref
    ref = str(obs_first.artifact.id)
    assert ref in note1
    assert ref in note2
