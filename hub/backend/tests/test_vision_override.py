"""Vision plan stage 4: per-agent BYOK dispatch contracts."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.mcp_gateway import McpToolBinding
from app.core.context_policy import ContextPolicy
from app.core.secrets.crypto import SecretValue
from app.core.secrets.store import (
    BindingTarget,
    PostgresEncryptedSecretStore,
    SecretAccessDenied,
)
from app.core.user_sessions import _call_with_ws_loop
from app.core.vision_dispatch import (
    VISION_PUBLIC_TOOL_NAME,
    execute_vision_override,
    should_dispatch_vision_override,
)
from app.models.secret import SecretKind, SecretPurpose, SecretResourceType
from app.schemas.agent import AgentCreate


def _png_b64() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\nvision-test").decode()


def _agent_vision_target(injection_config: dict | None = None) -> BindingTarget:
    return BindingTarget(
        resource_type=SecretResourceType.AGENT,
        resource_id=uuid4(),
        purpose=SecretPurpose.LLM_VISION,
        injection_config=injection_config,
    )


def test_agent_llm_vision_binding_is_a_permitted_llm_api_key_slot():
    """Regression: (AGENT, LLM_VISION) must validate like LLM_PRIMARY/LLM_MEDIA.

    The binding allow-list in ``_validate_target`` gained ``llm_primary`` and
    ``llm_media`` but not ``llm_vision``, so saving a per-agent vision key —
    the very same secret already bound in the primary slot — was rejected with
    ``SecretAccessDenied`` ("Secret operation is not permitted" from the API).
    """

    target = PostgresEncryptedSecretStore._validate_target(_agent_vision_target())

    assert target.purpose == SecretPurpose.LLM_VISION
    assert target.expected_kind == SecretKind.LLM_API_KEY


def test_agent_llm_vision_binding_rejects_injection_config_like_other_llm_slots():
    with pytest.raises(SecretAccessDenied):
        PostgresEncryptedSecretStore._validate_target(
            _agent_vision_target(injection_config={"mode": "bearer"})
        )


def _vision_config() -> dict:
    return {
        "provider": "openrouter",
        "model": "private-vision-model",
        "base_url": "https://override.example/v1",
        "secret_binding_id": str(uuid4()),
        "enabled": True,
    }


def _vision_binding() -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug="mcp-vision",
        tool_name="describe_image",
        custom_config=None,
        isolation="stateless",
    )


def _wide_context_policy() -> ContextPolicy:
    return ContextPolicy(
        enabled=True,
        input_cap=100_000,
        active_tool_result_cap=10_000,
        keep_raw_results=3,
        output_reserve=1_000,
        model_context_limit=200_000,
        checkpoint_cap=600,
        context_window_limit=100_000,
    )


@pytest.mark.asyncio
async def test_byok_vision_uses_agent_secret_and_openai_image_content_without_gateway():
    owner_id, agent_id, room_id, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    secret = SecretValue("byok-test-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "description": "OpenRouter private-vision-model has byok-test-key",
            "ocr_text": "Ready",
            "answer": "Yes",
        })}}]
    }
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    config = {"vision": _vision_config()}

    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=secret) as resolve,
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
    ):
        text = await execute_vision_override(
            llm_config=config,
            arguments={"image_base64": _png_b64(), "question": "Is it ready?"},
            room_id=room_id,
            db=MagicMock(),
            owner_id=owner_id,
            agent_id=agent_id,
            request_id=request_id,
        )

    result = json.loads(text)
    assert result == {
        "description": "[vision backend] [vision backend] has [vision backend]",
        "ocr_text": "Ready",
        "answer": "Yes",
    }
    assert "byok-test-key" not in text
    resolve.assert_awaited_once()
    assert resolve.await_args.kwargs["purpose"] == SecretPurpose.LLM_VISION
    post = client.post.await_args
    assert post.args[0] == "https://override.example/v1/chat/completions"
    content = post.kwargs["json"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{_png_b64()}"


@pytest.mark.asyncio
async def test_byok_vision_rejects_bad_input_before_decrypting_key():
    with patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock) as resolve:
        text = await execute_vision_override(
            llm_config={"vision": _vision_config()},
            arguments={"image_base64": "not-an-image"},
            room_id=uuid4(),
            db=MagicMock(),
            owner_id=uuid4(),
            agent_id=uuid4(),
            request_id=uuid4(),
        )

    assert text == "Error: [not_an_image] invalid image data (not valid base64)"
    resolve.assert_not_awaited()


async def _run_byok(config: dict, *, tool_name: str = "describe_image") -> str:
    return await execute_vision_override(
        llm_config=config,
        arguments={"image_base64": _png_b64()},
        room_id=uuid4(),
        db=MagicMock(),
        owner_id=uuid4(),
        agent_id=uuid4(),
        request_id=uuid4(),
        tool_name=tool_name,
    )


@pytest.mark.asyncio
async def test_byok_vision_disabled_reports_not_configured():
    text = await _run_byok({"vision": {**_vision_config(), "enabled": False}})
    assert text == "Error: [not_configured] vision override is not enabled for this agent"


@pytest.mark.asyncio
async def test_byok_vision_key_resolution_failure_reports_no_key():
    with patch(
        "app.core.vision_dispatch.resolve_for_use",
        new_callable=AsyncMock,
        side_effect=RuntimeError("secret store down"),
    ):
        text = await _run_byok({"vision": _vision_config()})
    assert text == "Error: [no_key] the agent's vision provider key could not be resolved"
    assert "secret store" not in text


@pytest.mark.asyncio
async def test_byok_vision_provider_401_reports_auth_error():
    response = MagicMock(status_code=401)
    response.json.return_value = {"error": "bad key"}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=SecretValue("k")),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
    ):
        text = await _run_byok({"vision": {**_vision_config(), "provider": "openai"}})
    assert text.startswith("Error: [auth_error] ")


@pytest.mark.asyncio
async def test_byok_vision_provider_unreachable_reports_provider_error():
    import httpx as _httpx

    context = MagicMock()
    context.__aenter__ = AsyncMock(side_effect=_httpx.ConnectError("no route"))
    context.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=SecretValue("k")),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
    ):
        text = await _run_byok({"vision": {**_vision_config(), "provider": "openai"}})
    assert text.startswith("Error: [provider_error] ")


def test_vision_config_requires_safe_shape_and_plaintext_key_is_rejected():
    valid = AgentCreate(
        nickname="vision-agent",
        llm_config={"vision": _vision_config()},
    )
    assert valid.llm_config["vision"]["enabled"] is True
    with pytest.raises(ValueError, match="must reference a SecretStore binding"):
        AgentCreate(
            nickname="vision-agent",
            llm_config={"vision": {**_vision_config(), "api_key": "do-not-store-me"}},
        )


@pytest.mark.asyncio
async def test_direct_agent_tool_loop_bypasses_gateway_for_enabled_vision_override():
    """The direct-chat loop must select BYOK before execute_mcp_tool/credits."""

    room_id, session_id, agent_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    binding = _vision_binding()
    assert should_dispatch_vision_override({"vision": _vision_config()}, binding)

    tool_use = MagicMock()
    tool_use.raise_for_status = MagicMock()
    tool_use.json.return_value = {
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "stop_reason": "tool_use",
        "content": [{
            "type": "tool_use",
            "id": "vision-call",
            "name": VISION_PUBLIC_TOOL_NAME,
            "input": {"image_base64": _png_b64()},
        }],
    }
    final = MagicMock()
    final.raise_for_status = MagicMock()
    final.json.return_value = {
        "usage": {"input_tokens": 8, "output_tokens": 3},
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "The image is ready."}],
    }
    client = MagicMock()
    client.post = AsyncMock(side_effect=[tool_use, final])
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    query_result = MagicMock()
    query_result.scalars.return_value.first.return_value = None
    query_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=query_result)
    config = {
        "provider": "anthropic",
        "model": "text-model",
        "api_key": "legacy-test-key",
        "vision": _vision_config(),
    }

    with (
        patch("app.core.user_sessions.httpx.AsyncClient", return_value=context),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.vision_dispatch.execute_vision_override",
            new_callable=AsyncMock,
            return_value='{"description":"done","ocr_text":""}',
        ) as byok,
        patch("app.core.mcp_gateway.execute_mcp_tool", new_callable=AsyncMock) as gateway,
    ):
        result = await _call_with_ws_loop(
            config,
            "SYS",
            [{"role": "user", "content": "inspect the screenshot"}],
            "user_agent_p2",
            room_id,
            session_id,
            agent_id,
            db,
            ant_ws_tools=[],
            oai_ws_tools=[],
            mcp_ant_tools=[{"name": VISION_PUBLIC_TOOL_NAME, "description": "vision", "input_schema": {"type": "object"}}],
            mcp_oai_tools=[],
            mcp_bindings={VISION_PUBLIC_TOOL_NAME: binding},
            mcp_user_id=owner_id,
        )

    assert result.text == "The image is ready."
    byok.assert_awaited_once_with(
        llm_config=config,
        arguments={"image_base64": _png_b64()},
        room_id=room_id,
        db=db,
        owner_id=owner_id,
        agent_id=agent_id,
        request_id=session_id,
        tool_name="describe_image",
    )
    gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_agent_tool_loop_bypasses_gateway_for_enabled_vision_override():
    """The task-proxy loop has the same no-credit routing guarantee."""

    from types import SimpleNamespace

    from app.core import proxy_session

    room_id, session_id, agent_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    binding = _vision_binding()
    tool_call = SimpleNamespace(
        id="vision-call",
        function=SimpleNamespace(
            name=VISION_PUBLIC_TOOL_NAME,
            arguments=json.dumps({"image_base64": _png_b64()}),
        ),
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="Proxy vision result accepted.", tool_calls=[]),
            )],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
        ),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    )
    config = {
        "provider": "openai",
        "model": "text-model",
        "api_key": "legacy-test-key",
        "vision": _vision_config(),
    }
    byok = AsyncMock(return_value='{"description":"done","ocr_text":""}')
    gateway = AsyncMock()
    db = MagicMock()

    with (
        patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=_wide_context_policy())),
        patch.object(proxy_session, "AsyncOpenAI", return_value=client),
        patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
        patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
        patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        patch.object(proxy_session, "emit_status", AsyncMock()),
        patch.object(proxy_session, "_emit_tool_call_event", AsyncMock()),
        patch("app.core.vision_dispatch.execute_vision_override", byok),
        patch("app.core.mcp_gateway.execute_mcp_tool", gateway),
    ):
        result = await proxy_session._call_llm_with_tools(
            config,
            "SYS",
            [{"role": "user", "content": "inspect the screenshot"}],
            room_id=room_id,
            session_id=session_id,
            task_id=None,
            db=db,
            mcp_ant_tools=[],
            mcp_oai_tools=[{
                "type": "function",
                "function": {"name": VISION_PUBLIC_TOOL_NAME, "parameters": {"type": "object"}},
            }],
            mcp_bindings={VISION_PUBLIC_TOOL_NAME: binding},
            mcp_user_id=owner_id,
            actor_agent_id=agent_id,
            max_tool_calls=5,
            on_exhaust="raise",
            context_role="agent",
        )

    assert result[0] == "Proxy vision result accepted."
    byok.assert_awaited_once_with(
        llm_config=config,
        arguments={"image_base64": _png_b64()},
        room_id=room_id,
        db=db,
        owner_id=owner_id,
        agent_id=agent_id,
        request_id=session_id,
        tool_name="describe_image",
    )
    gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_agent_tool_loop_forwards_db_to_standard_mcp():
    """Normal Marketplace MCPs retain their route while Vision can resolve paths."""

    from types import SimpleNamespace

    from app.core import proxy_session

    room_id, session_id, agent_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    binding = _vision_binding()
    tool_call = SimpleNamespace(
        id="vision-call",
        function=SimpleNamespace(
            name=VISION_PUBLIC_TOOL_NAME,
            arguments=json.dumps({"image_path": "1111.png"}),
        ),
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="Vision result accepted.", tool_calls=[]),
            )],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
        ),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    )
    db = MagicMock()
    gateway = AsyncMock(return_value='{"description":"boot","ocr_text":""}')

    with (
        patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=_wide_context_policy())),
        patch.object(proxy_session, "AsyncOpenAI", return_value=client),
        patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
        patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
        patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        patch.object(proxy_session, "emit_status", AsyncMock()),
        patch.object(proxy_session, "_emit_tool_call_event", AsyncMock()),
        patch("app.core.mcp_gateway.execute_mcp_tool", gateway),
    ):
        result = await proxy_session._call_llm_with_tools(
            {"provider": "openai", "model": "text-model", "api_key": "test-key"},
            "SYS",
            [{"role": "user", "content": "inspect the image"}],
            room_id=room_id,
            session_id=session_id,
            task_id=None,
            db=db,
            mcp_ant_tools=[],
            mcp_oai_tools=[{
                "type": "function",
                "function": {"name": VISION_PUBLIC_TOOL_NAME, "parameters": {"type": "object"}},
            }],
            mcp_bindings={VISION_PUBLIC_TOOL_NAME: binding},
            mcp_user_id=owner_id,
            actor_agent_id=agent_id,
            max_tool_calls=5,
            on_exhaust="raise",
            context_role="agent",
        )

    assert result[0] == "Vision result accepted."
    gateway.assert_awaited_once_with(
        binding,
        owner_id,
        room_id,
        {"image_path": "1111.png"},
        db=db,
        request_id=session_id,
    )


# ---------------------------------------------------------------------------
# Design Vision Toolkit Stage 5 — extract_design_spec / critique_design BYOK
# ---------------------------------------------------------------------------


def _binding(tool_name: str) -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug="mcp-vision",
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
    )


def test_should_dispatch_covers_all_three_vision_tools():
    config = {"vision": _vision_config()}
    assert should_dispatch_vision_override(config, _binding("describe_image"))
    assert should_dispatch_vision_override(config, _binding("extract_design_spec"))
    assert should_dispatch_vision_override(config, _binding("critique_design"))
    assert not should_dispatch_vision_override(config, _binding("some_other_tool"))


def _openrouter_json_response(payload: dict) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return response


@pytest.mark.asyncio
async def test_byok_extract_design_spec_merges_cv_and_persists_the_spec():
    owner_id, agent_id, room_id, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    secret = SecretValue("byok-test-key")
    response = _openrouter_json_response({
        "regions": [{"id": "r1", "role": "card"}],
        "texts": [],
        "tokens": {},
        "uncertain": ["hover state via private-vision-model"],
    })
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    config = {"vision": _vision_config()}
    cv_tree = {"regions": [{"id": "r1", "parent_id": None, "box": {"x": 0, "y": 0, "w": 10, "h": 10}}], "text": []}

    fake_file = type("F", (), {"id": uuid4()})()
    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=secret),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(return_value=fake_file)),
        patch("app.core.events.emit_event", AsyncMock()),
    ):
        text = await execute_vision_override(
            llm_config=config,
            arguments={"image_base64": _png_b64(), "cv": cv_tree, "name": "dashboard"},
            room_id=room_id,
            db=MagicMock(),
            owner_id=owner_id,
            agent_id=agent_id,
            request_id=request_id,
            tool_name="extract_design_spec",
        )

    result = json.loads(text)
    assert result["spec_path"] == "design/dashboard.spec.json"
    assert result["summary"]["roles"] == ["card"]
    # The full spec (with layout boxes) never reaches the agent inline — same
    # S2 rule the Market route gets from vision_artifacts.persist_design_spec.
    assert "layout" not in result
    assert "private-vision-model" not in text


@pytest.mark.asyncio
async def test_byok_critique_design_computes_fidelity_and_wcag_deterministically():
    owner_id, agent_id, room_id, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    secret = SecretValue("byok-test-key")
    response = _openrouter_json_response({"scores": {"hierarchy": 70}, "findings": [], "summary": "fine"})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    config = {"vision": _vision_config()}
    cv_tree = {"regions": [], "text": [{"id": "t1", "box": {"x": 0, "y": 0, "w": 1, "h": 1}, "contrast_ratio": 2}]}
    diff_tree = {"diff_ratio": 0.1, "diff_regions": []}

    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=secret),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
    ):
        text = await execute_vision_override(
            llm_config=config,
            arguments={"image_base64": _png_b64(), "cv": cv_tree, "diff": diff_tree, "aspects": ["hierarchy", "fidelity"]},
            room_id=room_id,
            db=MagicMock(),
            owner_id=owner_id,
            agent_id=agent_id,
            request_id=request_id,
            tool_name="critique_design",
        )

    result = json.loads(text)
    assert result["scores"]["fidelity"] == 90
    assert result["wcag"][0]["aa"] is False  # ratio 2 never clears AA
    assert result["scores"]["overall"] == 80


@pytest.mark.asyncio
async def test_byok_rejects_a_field_outside_the_tool_contract():
    with patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock) as resolve:
        text = await execute_vision_override(
            llm_config={"vision": _vision_config()},
            arguments={"image_base64": _png_b64(), "not_a_real_field": True},
            room_id=uuid4(),
            db=MagicMock(),
            owner_id=uuid4(),
            agent_id=uuid4(),
            request_id=uuid4(),
            tool_name="critique_design",
        )
    assert text == "Error: [input_invalid] Vision input is invalid"
    resolve.assert_not_awaited()


# ---------------------------------------------------------------------------
# Design Vision Toolkit Stage 6 — R7 design_system injection on the BYOK route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_byok_injects_design_system_when_agent_did_not_supply_one():
    owner_id, agent_id, room_id, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    secret = SecretValue("byok-test-key")
    response = _openrouter_json_response({"scores": {"hierarchy": 70}, "findings": [], "summary": "fine"})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    tokens = {"colors": {"--color-primary": "#3B82F6"}}

    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=secret),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
        patch("app.core.design_tokens.get_design_system", new_callable=AsyncMock, return_value=tokens) as getter,
    ):
        await execute_vision_override(
            llm_config={"vision": _vision_config()},
            arguments={"image_base64": _png_b64()},
            room_id=room_id,
            db=MagicMock(),
            owner_id=owner_id,
            agent_id=agent_id,
            request_id=request_id,
            tool_name="critique_design",
        )

    getter.assert_awaited_once()
    assert getter.await_args.args[0] == room_id
    sent_prompt = client.post.await_args.kwargs["json"]["messages"][0]["content"][0]["text"]
    assert "--color-primary" in sent_prompt


@pytest.mark.asyncio
async def test_byok_does_not_overwrite_agent_supplied_design_system():
    owner_id, agent_id, room_id, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    secret = SecretValue("byok-test-key")
    response = _openrouter_json_response({"scores": {"hierarchy": 70}, "findings": [], "summary": "fine"})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.core.vision_dispatch.resolve_for_use", new_callable=AsyncMock, return_value=secret),
        patch("app.core.vision_dispatch.httpx.AsyncClient", return_value=context),
        patch("app.core.design_tokens.get_design_system", new_callable=AsyncMock) as getter,
    ):
        await execute_vision_override(
            llm_config={"vision": _vision_config()},
            arguments={"image_base64": _png_b64(), "design_system": {"colors": {"--brand": "#000"}}},
            room_id=room_id,
            db=MagicMock(),
            owner_id=owner_id,
            agent_id=agent_id,
            request_id=request_id,
            tool_name="critique_design",
        )

    getter.assert_not_awaited()
    sent_prompt = client.post.await_args.kwargs["json"]["messages"][0]["content"][0]["text"]
    assert "--brand" in sent_prompt
