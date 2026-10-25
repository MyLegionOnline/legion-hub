"""Starter agents for a newly registered user (core/agent_defaults.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import agent_defaults, default_catalog, market_client
from app.models.agent import Agent
from app.models.market import AgentSkill, UserInstalledItem


@pytest.fixture(autouse=True)
def _provisioning_enabled(monkeypatch):
    monkeypatch.setattr(agent_defaults, "AGENT_DEFAULTS_PROVISION", True)
    default_catalog.reset()


@pytest.fixture(autouse=True)
def _market_must_not_be_called(monkeypatch):
    # internal design notes stage 3: registration never talks to Market.
    for name in ("list_catalog", "get_skill", "fetch_package"):
        monkeypatch.setattr(market_client, name, AsyncMock(side_effect=AssertionError(f"Market {name} called")))


def _db(existing_nicknames, installed_by_slug):
    """AsyncMock session: first execute() lists the user's agent nicknames,
    later ones look up an installed skill by slug."""
    db = AsyncMock()
    db.add = MagicMock()
    lookups = []

    async def execute(stmt):
        result = MagicMock()
        if not lookups:
            lookups.append("agents")
            result.scalars.return_value.all.return_value = list(existing_nicknames)
            return result
        slug = stmt.compile().params["market_slug_1"]
        lookups.append(slug)
        result.scalars.return_value.first.return_value = installed_by_slug.get(slug)
        return result

    db.execute = execute

    async def flush():
        for call in db.add.call_args_list:
            obj = call.args[0]
            if isinstance(obj, Agent) and obj.id is None:
                obj.id = uuid4()

    db.flush = flush
    return db, lookups


def _installed(slug):
    return UserInstalledItem(id=uuid4(), user_id=uuid4(), type="skill", market_slug=slug, name=slug)


@pytest.mark.asyncio
async def test_creates_five_agents_with_their_skills(monkeypatch):
    user_id = uuid4()
    installed = {s: _installed(s) for s in ("skill-arch", "skill-coder", "skill-writer")}
    vps = _installed("skill-vps-deploy")
    install = AsyncMock(return_value=vps)
    monkeypatch.setattr(agent_defaults, "install_shared_skill_for_user", install)

    db, lookups = _db([], installed)
    await agent_defaults.provision_default_agents(user_id, db)

    added = [c.args[0] for c in db.add.call_args_list]
    agents = [o for o in added if isinstance(o, Agent)]
    links = {o.agent_id: o.user_installed_item_id for o in added if isinstance(o, AgentSkill)}

    assert [a.nickname for a in agents] == ["Lily", "Sara", "Paul", "Edgar", "Kevin"]
    assert all(a.agent_type == "internal" and a.llm_config == {} and a.short_description for a in agents)
    assert all(a.owner_id == user_id for a in agents)
    by_name = {a.nickname: links[a.id] for a in agents}
    assert by_name["Lily"] == installed["skill-arch"].id
    assert by_name["Sara"] == by_name["Paul"] == installed["skill-coder"].id
    assert by_name["Edgar"] == installed["skill-writer"].id
    assert by_name["Kevin"] == vps.id
    # skill-coder looked up once for both coders; only the non-default skill
    # is linked, from the shared row shipped with the Hub.
    assert lookups.count("skill-coder") == 1
    install.assert_awaited_once()
    assert install.await_args.args[1] == "skill-vps-deploy"


@pytest.mark.asyncio
async def test_skips_taken_nicknames_and_survives_a_missing_skill(monkeypatch):
    monkeypatch.setattr(agent_defaults, "install_shared_skill_for_user", AsyncMock(return_value=None))
    db, _ = _db(["Lily", "Sara", "Paul", "Edgar"], {})
    await agent_defaults.provision_default_agents(uuid4(), db)

    added = [c.args[0] for c in db.add.call_args_list]
    assert [o.nickname for o in added if isinstance(o, Agent)] == ["Kevin"]
    assert not any(isinstance(o, AgentSkill) for o in added)


@pytest.mark.asyncio
async def test_disabled_flag_does_nothing(monkeypatch):
    monkeypatch.setattr(agent_defaults, "AGENT_DEFAULTS_PROVISION", False)
    db = AsyncMock()
    await agent_defaults.provision_default_agents(uuid4(), db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_every_starter_agent_gets_its_bundled_avatar(monkeypatch):
    from app.models.agent_avatar import AgentAvatar

    monkeypatch.setattr(agent_defaults, "install_shared_skill_for_user", AsyncMock(return_value=None))
    db, _ = _db([], {})
    db.get = AsyncMock(return_value=None)
    await agent_defaults.provision_default_agents(uuid4(), db)

    added = [c.args[0] for c in db.add.call_args_list]
    agents = [o for o in added if isinstance(o, Agent)]
    avatars = {o.agent_id: o for o in added if isinstance(o, AgentAvatar)}

    assert len(agents) == len(agent_defaults.DEFAULT_AGENTS)
    for a in agents:
        assert a.has_avatar and a.avatar_updated_at is not None
        assert avatars[a.id].mime == "image/png" and avatars[a.id].size_bytes > 0


def test_a_bundled_avatar_file_exists_for_every_starter_agent():
    for nickname, _role, _slug in agent_defaults.DEFAULT_AGENTS:
        assert (agent_defaults.BUILTIN_AVATARS_DIR / f"{nickname.lower()}.jpg").is_file()
