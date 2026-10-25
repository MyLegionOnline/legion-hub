"""Unit tests for stage 7.5: llm_config.media validation in AgentBase / AgentUpdate.

No DB required — pure Pydantic validation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.agent import AgentCreate, AgentUpdate


def _create(media: dict) -> None:
    AgentCreate(nickname="test", llm_config={"media": media})


def _update(media: dict) -> None:
    AgentUpdate(llm_config={"media": media})


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_image_generation_capability():
    _create({"capability": "image_generation"})


def test_valid_aspect_ratio():
    _create({"capability": "image_generation", "aspect_ratio": "16:9"})


def test_valid_aspect_ratio_various():
    for ratio in ("1:1", "3:2", "4:3", "9:16", "21:9"):
        _create({"aspect_ratio": ratio})


def test_valid_default_size_exact_pixels():
    _create({"capability": "image_generation", "default_size": "1000x100"})


def test_valid_default_size_tier():
    _create({"capability": "image_generation", "default_size": "2K"})


def test_valid_fit_modes():
    for fit in ("cover", "contain", "stretch"):
        _create({"capability": "image_generation", "fit": fit})


def test_valid_reference_vision():
    for rv in ("auto", "off"):
        _create({"capability": "image_generation", "reference_vision": rv})


def test_valid_model_non_empty():
    _create({"capability": "image_generation", "model": "openai/dall-e-3"})


def test_no_media_key_is_ok():
    AgentCreate(nickname="test", llm_config={"model": "gpt-4"})


def test_media_none_is_ok():
    AgentCreate(nickname="test", llm_config={"media": None})


def test_update_valid_media():
    _update({"capability": "image_generation", "default_size": "800x600", "fit": "contain"})


# ---------------------------------------------------------------------------
# capability validation
# ---------------------------------------------------------------------------

def test_invalid_capability_raises_422():
    with pytest.raises(ValidationError, match="media.capability"):
        _create({"capability": "text_generation"})


def test_unknown_capability_raises_422():
    with pytest.raises(ValidationError, match="media.capability"):
        _create({"capability": "UNKNOWN"})


# ---------------------------------------------------------------------------
# aspect_ratio validation
# ---------------------------------------------------------------------------

def test_invalid_aspect_ratio_plain_string_raises():
    with pytest.raises(ValidationError, match="media.aspect_ratio"):
        _create({"aspect_ratio": "wide"})


def test_invalid_aspect_ratio_wxh_format_raises():
    with pytest.raises(ValidationError, match="media.aspect_ratio"):
        _create({"aspect_ratio": "1920x1080"})


# ---------------------------------------------------------------------------
# default_size / size validation
# ---------------------------------------------------------------------------

def test_invalid_default_size_raises():
    with pytest.raises(ValidationError, match="media.default_size"):
        _create({"default_size": "1920:1080"})


def test_default_size_zero_side_raises():
    with pytest.raises(ValidationError, match="media.default_size"):
        _create({"default_size": "0x100"})


def test_size_field_also_validated():
    with pytest.raises(ValidationError, match="media.size"):
        _create({"size": "invalid"})


# ---------------------------------------------------------------------------
# fit validation
# ---------------------------------------------------------------------------

def test_invalid_fit_raises():
    with pytest.raises(ValidationError, match="media.fit"):
        _create({"capability": "image_generation", "fit": "zoom"})


# ---------------------------------------------------------------------------
# reference_vision validation
# ---------------------------------------------------------------------------

def test_invalid_reference_vision_raises():
    with pytest.raises(ValidationError, match="media.reference_vision"):
        _create({"capability": "image_generation", "reference_vision": "maybe"})


# ---------------------------------------------------------------------------
# model validation
# ---------------------------------------------------------------------------

def test_empty_model_raises():
    with pytest.raises(ValidationError, match="media.model"):
        _create({"capability": "image_generation", "model": ""})


def test_whitespace_model_raises():
    with pytest.raises(ValidationError, match="media.model"):
        _create({"capability": "image_generation", "model": "   "})


# ---------------------------------------------------------------------------
# AgentUpdate mirrors the same rules
# ---------------------------------------------------------------------------

def test_update_invalid_capability_raises():
    with pytest.raises(ValidationError, match="media.capability"):
        _update({"capability": "bad_cap"})


def test_update_invalid_fit_raises():
    with pytest.raises(ValidationError, match="media.fit"):
        _update({"fit": "crop"})
