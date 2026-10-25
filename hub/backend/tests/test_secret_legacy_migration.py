from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.secrets.migration import discover_legacy_sources
from app.core.secrets.redaction import assert_no_secret_fields
from app.models.agent import Agent
from app.models.market import UserInstalledItem
from app.models.room import Room


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, *rows):
        self._rows = iter(rows)

    async def execute(self, _statement):
        return _Result(next(self._rows))


@pytest.mark.asyncio
async def test_legacy_discovery_rewrites_all_supported_paths_without_exposing_values():
    owner_id = uuid4()
    agent = Agent(
        id=uuid4(), owner_id=owner_id, token="agent-token", nickname="A", agent_uid="a",
        llm_config={"provider": "openai", "api_key": "primary-value", "media": {"api_key": "media-value"}},
    )
    room = Room(
        id=uuid4(), creator_id=owner_id, name="R",
        git_config={"provider": "github", "access_token": "git-value", "webhook_secret": "hook-value"},
    )
    item = UserInstalledItem(
        id=uuid4(), user_id=owner_id, type="mcp", name="M", is_custom=True,
        custom_config={
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer mcp-value"},
            "env": {"MCP_TOKEN": "env-value", "UNSAFE": "manual-value"},
            "secret_env_migration_allowlist": ["MCP_TOKEN"],
        },
    )
    candidates, issues = await discover_legacy_sources(_Session([agent], [room], [item]))

    assert {candidate.json_path for candidate in candidates} == {
        "llm_config.api_key", "llm_config.media.api_key", "git_config.access_token",
        "git_config.webhook_secret", "custom_config.headers.Authorization", "custom_config.env.MCP_TOKEN",
    }
    assert [(issue.json_path, issue.error_code) for issue in issues] == [
        ("custom_config.env.UNSAFE", "mcp_env_requires_manual_opt_in")
    ]
    assert "primary-value" not in repr(candidates)
    assert "env-value" not in repr(issues)

    rewritten = {candidate.json_path: candidate.rewritten_config(uuid4()) for candidate in candidates}
    assert "api_key" not in rewritten["llm_config.api_key"]
    assert "api_key" not in rewritten["llm_config.media.api_key"]["media"]
    assert "access_token" not in rewritten["git_config.access_token"]
    assert "webhook_secret" not in rewritten["git_config.webhook_secret"]
    assert "Authorization" not in rewritten["custom_config.headers.Authorization"].get("headers", {})
    assert "MCP_TOKEN" not in rewritten["custom_config.env.MCP_TOKEN"].get("env", {})
    auth_candidate = next(candidate for candidate in candidates if candidate.json_path.endswith("Authorization"))
    assert auth_candidate.value.reveal() == "mcp-value"

    # Candidate rewrites operate in one transaction, so several fields of one
    # MCP configuration are applied in discovery order to the current JSON.
    for candidate in (candidate for candidate in candidates if candidate.resource is item):
        item.custom_config = candidate.rewritten_config(uuid4())
    assert_no_secret_fields(item.custom_config)
