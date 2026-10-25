"""Unit tests for Stage 4/5: Phase A — image prompt authoring.

No real LLM calls — _call_llm is mocked.
Tests cover:
  - Tool schema structure (Anthropic + OpenAI formats)
  - _validate_image_prompts: empty list, duplicate filenames, short prompt
  - _generate_image_prompts uses the TEXT model (top-level llm_config, not media sub-dict)
  - _generate_image_prompts passes ANTHROPIC_IMAGE_PROMPTS_TOOL / OPENAI_IMAGE_PROMPTS_TOOL
  - Happy path: valid images list returned
Stage 5 additions:
  - schema: aspect_ratio enum lifted, size/fit/reference_paths added
  - _validate_image_prompts: unknown reference_paths / invalid size dropped,
    excess reference_paths truncated, known_reference_paths=None skips the check
  - _generate_image_prompts: reference manifest text, multimodal vision turn
    (Anthropic vs OpenAI-compatible blocks), reference_vision "off", and
    unresolved references surfacing as UNAVAILABLE instead of vanishing
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.media_refs import MEDIA_REF_MAX_PER_IMAGE, ReferenceManifestEntry, ResolvedRef
from app.core.proxy_session import (
    ANTHROPIC_IMAGE_PROMPTS_TOOL,
    MEDIA_PROMPTGEN_SUFFIX,
    OPENAI_IMAGE_PROMPTS_TOOL,
    ImagePromptValidationError,
    _generate_image_prompts,
    _validate_image_prompts,
)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def test_anthropic_tool_schema_structure():
    assert ANTHROPIC_IMAGE_PROMPTS_TOOL["name"] == "submit_image_prompts"
    schema = ANTHROPIC_IMAGE_PROMPTS_TOOL["input_schema"]
    assert schema["type"] == "object"
    assert "images" in schema["properties"]
    item_props = schema["properties"]["images"]["items"]["properties"]
    assert "filename" in item_props
    assert "prompt" in item_props
    assert "aspect_ratio" in item_props
    assert schema["properties"]["images"]["items"]["required"] == ["filename", "prompt"]


def test_openai_tool_schema_structure():
    assert OPENAI_IMAGE_PROMPTS_TOOL["type"] == "function"
    fn = OPENAI_IMAGE_PROMPTS_TOOL["function"]
    assert fn["name"] == "submit_image_prompts"
    assert "parameters" in fn
    assert fn["parameters"]["type"] == "object"


def test_media_promptgen_suffix_contains_key_rules():
    assert "submit_image_prompts" in MEDIA_PROMPTGEN_SUFFIX
    assert "kebab-case" in MEDIA_PROMPTGEN_SUFFIX
    assert "English" in MEDIA_PROMPTGEN_SUFFIX


def test_media_promptgen_suffix_mentions_references_and_size():
    assert "reference_paths" in MEDIA_PROMPTGEN_SUFFIX
    assert "size" in MEDIA_PROMPTGEN_SUFFIX
    assert "never invent a reference path" in MEDIA_PROMPTGEN_SUFFIX


def test_schema_aspect_ratio_has_no_enum():
    """Stage 5.4: the hard 3-value enum is lifted — checking moves to _validate_image_prompts."""
    item_props = ANTHROPIC_IMAGE_PROMPTS_TOOL["input_schema"]["properties"]["images"]["items"]["properties"]
    assert "enum" not in item_props["aspect_ratio"]


def test_schema_has_size_fit_reference_paths():
    for tool in (
        ANTHROPIC_IMAGE_PROMPTS_TOOL["input_schema"],
        OPENAI_IMAGE_PROMPTS_TOOL["function"]["parameters"],
    ):
        item_props = tool["properties"]["images"]["items"]["properties"]
        assert item_props["size"]["type"] == "string"
        assert item_props["fit"]["enum"] == ["cover", "contain", "stretch"]
        assert item_props["reference_paths"]["type"] == "array"
        assert item_props["reference_paths"]["maxItems"] == MEDIA_REF_MAX_PER_IMAGE


# ---------------------------------------------------------------------------
# _validate_image_prompts
# ---------------------------------------------------------------------------


def test_validate_empty_list_raises():
    with pytest.raises(ImagePromptValidationError, match="empty"):
        _validate_image_prompts([])


def test_validate_duplicate_filenames_raises():
    images = [
        {"filename": "cafe-interior", "prompt": "A bright cafe with warm lighting"},
        {"filename": "cafe-interior", "prompt": "A different cafe photo"},
    ]
    with pytest.raises(ImagePromptValidationError, match="Duplicate"):
        _validate_image_prompts(images)


def test_validate_short_prompt_raises():
    images = [{"filename": "cafe", "prompt": "short"}]
    with pytest.raises(ImagePromptValidationError, match="too short"):
        _validate_image_prompts(images)


def test_validate_valid_images_passes():
    images = [
        {"filename": "cafe-interior", "prompt": "A bright cafe with warm lighting and wooden tables"},
        {"filename": "eggs-benedict", "prompt": "Close-up of eggs benedict on a white plate"},
    ]
    _validate_image_prompts(images)  # must not raise


def test_validate_whitespace_only_prompt_raises():
    images = [{"filename": "cafe", "prompt": "   "}]
    with pytest.raises(ImagePromptValidationError, match="too short"):
        _validate_image_prompts(images)


def test_validate_none_known_paths_skips_reference_check():
    """Backward compat: no manifest was built → reference_paths is not policed."""
    images = [{
        "filename": "cafe", "prompt": "A bright cafe with warm lighting",
        "reference_paths": ["images/made-up.png"],
    }]
    result = _validate_image_prompts(images, known_reference_paths=None)
    assert result == images


def test_validate_drops_image_with_unknown_reference_path():
    images = [
        {
            "filename": "cafe", "prompt": "A bright cafe with warm lighting",
            "reference_paths": ["images/logo.png"],
        },
        {
            "filename": "menu", "prompt": "A printed menu on a wooden table",
            "reference_paths": ["images/invented.png"],
        },
    ]
    result = _validate_image_prompts(images, known_reference_paths=frozenset({"images/logo.png"}))
    assert [img["filename"] for img in result] == ["cafe"]


def test_validate_raises_when_every_image_has_unknown_reference_path():
    images = [{
        "filename": "cafe", "prompt": "A bright cafe with warm lighting",
        "reference_paths": ["images/invented.png"],
    }]
    with pytest.raises(ImagePromptValidationError, match="no valid images"):
        _validate_image_prompts(images, known_reference_paths=frozenset({"images/logo.png"}))


def test_validate_drops_image_with_invalid_size():
    images = [
        {"filename": "cafe", "prompt": "A bright cafe with warm lighting", "size": "not-a-size"},
        {"filename": "menu", "prompt": "A printed menu on a wooden table"},
    ]
    result = _validate_image_prompts(images)
    assert [img["filename"] for img in result] == ["menu"]


def test_validate_truncates_excess_reference_paths():
    paths = [f"images/ref-{i}.png" for i in range(MEDIA_REF_MAX_PER_IMAGE + 2)]
    images = [{
        "filename": "cafe", "prompt": "A bright cafe with warm lighting",
        "reference_paths": paths,
    }]
    result = _validate_image_prompts(images, known_reference_paths=frozenset(paths))
    assert len(result[0]["reference_paths"]) == MEDIA_REF_MAX_PER_IMAGE


# ---------------------------------------------------------------------------
# _generate_image_prompts
# ---------------------------------------------------------------------------

_VALID_IMAGES = [
    {
        "filename": "cafe-interior",
        "prompt": "A warm, naturally lit cafe interior with wooden furniture",
        "aspect_ratio": "4:3",
    },
    {
        "filename": "eggs-benedict",
        "prompt": "Close-up of eggs benedict on a white plate, food photography",
        "aspect_ratio": "1:1",
    },
]


def _make_agent(llm_config: dict) -> MagicMock:
    agent = MagicMock()
    agent.llm_config = llm_config
    agent.id = uuid.uuid4()
    return agent


def _make_agent_session() -> MagicMock:
    session = MagicMock()
    session.id = uuid.uuid4()
    session.tokens_in = 0
    session.tokens_out = 0
    session.cost_usd = 0.0
    session.context_snapshot = {}
    session.max_context_window_tokens = 0
    session.last_context_window_tokens = 0
    return session


def _make_task(media_spec: dict | None = None) -> MagicMock:
    task = MagicMock()
    task.title = "Generate menu images"
    task.description = "Generate 2 images for the menu page"
    task.goal_id = uuid.uuid4()
    task.assigned_agent_id = uuid.uuid4()
    task.state = "in_progress"
    task.id = uuid.uuid4()
    task.room_id = uuid.uuid4()
    task.media_spec = media_spec if media_spec is not None else {}
    return task


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_generate_image_prompts_happy_path():
    """_generate_image_prompts returns validated images from tool_input."""
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock) as mock_mem,
        patch("app.core.proxy_session.build_system_prompt", return_value="sys prompt") as mock_sys,
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
    ):
        mock_mem.return_value = {}
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")

        result = await _generate_image_prompts(session, agent, task, db)

    assert result == _VALID_IMAGES
    mock_mem.assert_awaited_once_with(session, db)
    mock_sys.assert_called_once()


@pytest.mark.asyncio
async def test_generate_image_prompts_uses_text_model_not_media_model():
    """_call_llm must receive the top-level llm_config, not the media sub-dict."""
    text_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "api_key": "sk-text-key",
        "media": {
            "capability": "image_generation",
            "model": "google/gemini-3.1-flash-lite-image",
            "api_key": "sk-image-key",
        },
    }
    agent = _make_agent(text_config)
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    # First positional arg to _call_llm must be the full top-level config
    call_args = mock_llm.call_args
    passed_config = call_args[0][0]
    assert passed_config is text_config
    assert passed_config.get("model") == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_generate_image_prompts_passes_image_tools():
    """_call_llm must be called with IMAGE_PROMPTS_TOOL, not the close_session tool."""
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    call_kwargs = mock_llm.call_args[1]
    assert call_kwargs["anthropic_tool"] is ANTHROPIC_IMAGE_PROMPTS_TOOL
    assert call_kwargs["openai_tool"] is OPENAI_IMAGE_PROMPTS_TOOL
    assert call_kwargs["tool_required"] is True


@pytest.mark.asyncio
async def test_generate_image_prompts_raises_on_empty_result():
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
    ):
        mock_llm.return_value = (None, {"images": []}, 100, 50, "claude-sonnet-5")
        with pytest.raises(ImagePromptValidationError, match="empty"):
            await _generate_image_prompts(session, agent, task, db)


@pytest.mark.asyncio
async def test_generate_image_prompts_raises_on_duplicate_filenames():
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    dup_images = [
        {"filename": "same-name", "prompt": "A well-lit scene with wooden furniture"},
        {"filename": "same-name", "prompt": "Another well-lit scene with warm colors"},
    ]
    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
    ):
        mock_llm.return_value = (None, {"images": dup_images}, 100, 50, "claude-sonnet-5")
        with pytest.raises(ImagePromptValidationError, match="Duplicate"):
            await _generate_image_prompts(session, agent, task, db)


# ---------------------------------------------------------------------------
# _generate_image_prompts — stage 5: reference manifest + vision turn
# ---------------------------------------------------------------------------


def _make_resolved_ref(path="images/logo.png", role="logo", width=256, height=128) -> ResolvedRef:
    return ResolvedRef(
        path=path, mime="image/png", data=b"\x89PNG\r\n" + b"x" * 16,
        width=width, height=height, role=role, rasterized_from=None,
    )


@pytest.mark.asyncio
async def test_generate_image_prompts_no_media_spec_uses_plain_text_content():
    """No references declared → history content stays a plain string, unchanged from stage 4."""
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task()
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock) as mock_resolve,
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    mock_resolve.assert_not_awaited()
    history = mock_llm.call_args[0][2]
    assert isinstance(history[0]["content"], str)
    assert "Reference files" not in history[0]["content"]


@pytest.mark.asyncio
async def test_generate_image_prompts_with_resolved_reference_builds_anthropic_vision_turn():
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/logo.png", "role": "logo"}]})
    db = _make_db()

    entry = ReferenceManifestEntry(
        path="images/logo.png", role="logo", note="", ref=_make_resolved_ref(), error=None,
    )

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock, return_value=[entry]),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    history = mock_llm.call_args[0][2]
    content = history[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "images/logo.png" in content[0]["text"]
    assert "256x128px" in content[0]["text"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_generate_image_prompts_openai_compatible_provider_uses_image_url_block():
    agent = _make_agent({"provider": "openrouter", "model": "some/model"})
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/logo.png", "role": "logo"}]})
    db = _make_db()

    entry = ReferenceManifestEntry(
        path="images/logo.png", role="logo", note="", ref=_make_resolved_ref(), error=None,
    )

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock, return_value=[entry]),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    content = mock_llm.call_args[0][2][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_generate_image_prompts_reference_vision_off_keeps_text_only():
    agent = _make_agent({
        "provider": "anthropic", "model": "claude-sonnet-5",
        "media": {"reference_vision": "off"},
    })
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/logo.png", "role": "logo"}]})
    db = _make_db()

    entry = ReferenceManifestEntry(
        path="images/logo.png", role="logo", note="", ref=_make_resolved_ref(), error=None,
    )

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock, return_value=[entry]),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    content = mock_llm.call_args[0][2][0]["content"]
    assert isinstance(content, str)
    assert "images/logo.png" in content


@pytest.mark.asyncio
async def test_generate_image_prompts_unresolved_reference_shown_as_unavailable():
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/missing.png", "role": "logo"}]})
    db = _make_db()

    entry = ReferenceManifestEntry(
        path="images/missing.png", role="logo", note="", ref=None,
        error="reference not found: images/missing.png",
    )

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock, return_value=[entry]),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    content = mock_llm.call_args[0][2][0]["content"]
    assert isinstance(content, str)  # no resolved ref → no vision blocks
    assert "UNAVAILABLE" in content
    assert "images/missing.png" in content


@pytest.mark.asyncio
async def test_generate_image_prompts_resolution_exception_does_not_crash():
    """A resolver bug/outage must not take down prompt authoring for the whole task."""
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/logo.png", "role": "logo"}]})
    db = _make_db()

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch(
            "app.core.proxy_session.resolve_task_references",
            new_callable=AsyncMock, side_effect=RuntimeError("boom"),
        ),
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        result = await _generate_image_prompts(session, agent, task, db)

    assert result == _VALID_IMAGES


@pytest.mark.asyncio
async def test_generate_image_prompts_passes_known_reference_paths_to_validate():
    agent = _make_agent({"provider": "anthropic", "model": "claude-sonnet-5"})
    session = _make_agent_session()
    task = _make_task(media_spec={"references": [{"path": "images/logo.png", "role": "logo"}]})
    db = _make_db()

    entry = ReferenceManifestEntry(
        path="images/logo.png", role="logo", note="", ref=_make_resolved_ref(), error=None,
    )

    with (
        patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
        patch("app.core.proxy_session._call_llm", new_callable=AsyncMock) as mock_llm,
        patch("app.core.proxy_session.track_llm_usage"),
        patch("app.core.proxy_session.resolve_task_references", new_callable=AsyncMock, return_value=[entry]),
        patch("app.core.proxy_session._validate_image_prompts", wraps=_validate_image_prompts) as mock_validate,
    ):
        mock_llm.return_value = (None, {"images": _VALID_IMAGES}, 100, 50, "claude-sonnet-5")
        await _generate_image_prompts(session, agent, task, db)

    mock_validate.assert_called_once_with(_VALID_IMAGES, frozenset({"images/logo.png"}))
