from app.core.provider_cache import (
    anthropic_system_payload,
    extract_cache_usage,
    provider_cache_key,
    stable_tool_schemas,
    tool_schema_version,
)


def test_tool_schema_order_and_version_are_stable():
    first = [{"name": "z", "input_schema": {}}, {"name": "a", "input_schema": {}}]
    second = list(reversed(first))

    assert [item["name"] for item in stable_tool_schemas(first)] == ["a", "z"]
    assert tool_schema_version(first) == tool_schema_version(second)


def test_provider_cache_usage_normalizes_anthropic_and_openai_shapes():
    anthropic = extract_cache_usage(
        {
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 7,
        },
        "anthropic",
    )
    openai = extract_cache_usage(
        {"prompt_tokens_details": {"cached_tokens": 13}},
        "openai",
    )

    assert anthropic.read_tokens == 11
    assert anthropic.write_tokens == 7
    assert openai.read_tokens == 13


def test_provider_cache_key_is_opaque_and_system_cache_is_opt_in():
    key = provider_cache_key(
        room_id="room",
        agent_role="agent",
        system_version="v1",
        tool_schema_version_value="tools1",
    )
    assert key.startswith("legion-")
    assert isinstance(anthropic_system_payload("prompt", enabled=False), str)
    assert isinstance(anthropic_system_payload("prompt", enabled=True), list)
