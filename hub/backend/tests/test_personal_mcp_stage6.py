from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.personal_mcp import PersonalMcpError, feature_enabled, require_feature
from app.core.personal_mcp_rollout import rollout_capacity, rollout_policy


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class _Db:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _statement):
        return _Result(self.rows)


def _clear_rollout_env(monkeypatch):
    for name in (
        "PERSONAL_MCP_MANAGED_STDIO_ENABLED",
        "PERSONAL_MCP_ROLLOUT_STAGE",
        "PERSONAL_MCP_ROLLOUT_USER_IDS",
        "PERSONAL_MCP_INTERNAL_USER_IDS",
        "PERSONAL_MCP_CANARY_USER_IDS",
        "PERSONAL_MCP_LIMITED_USER_IDS",
        "PERSONAL_MCP_ROLLOUT_MAX_USERS",
        "PERSONAL_MCP_ROLLOUT_MAX_PACKAGES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_rollout_defaults_off_but_rollback_keeps_read_and_delete_access(monkeypatch):
    _clear_rollout_env(monkeypatch)
    owner_id = uuid4()
    assert rollout_policy().stage == "off"
    assert feature_enabled(owner_id) is False
    with pytest.raises(PersonalMcpError) as raised:
        require_feature(owner_id)
    assert raised.value.code == "personal_mcp_feature_disabled"
    require_feature(owner_id, operation="read")


def test_internal_and_canary_require_explicit_allowlist(monkeypatch):
    _clear_rollout_env(monkeypatch)
    owner_id = uuid4()
    other_id = uuid4()
    monkeypatch.setenv("PERSONAL_MCP_MANAGED_STDIO_ENABLED", "1")
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_STAGE", "internal")
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_USER_IDS", str(owner_id))
    assert feature_enabled(owner_id)
    assert not feature_enabled(other_id)


@pytest.mark.asyncio
async def test_rollout_user_and_package_caps_are_admission_boundaries(monkeypatch):
    _clear_rollout_env(monkeypatch)
    owner_id = uuid4()
    existing_owner = uuid4()
    monkeypatch.setenv("PERSONAL_MCP_MANAGED_STDIO_ENABLED", "1")
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_STAGE", "limited")
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_USER_IDS", str(owner_id))
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_MAX_USERS", "1")
    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_MAX_PACKAGES", "1")

    rows = [(existing_owner, {"package": "already-installed"})]
    denial = await rollout_capacity(owner_id, {"package": "new-package"}, _Db(rows))
    assert denial["reason"] == "user_cap"

    monkeypatch.setenv("PERSONAL_MCP_ROLLOUT_MAX_USERS", "10")
    denial = await rollout_capacity(owner_id, {"package": "new-package"}, _Db(rows))
    assert denial["reason"] == "package_cap"
