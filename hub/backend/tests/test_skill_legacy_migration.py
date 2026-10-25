"""Retirement of the legacy agent skill fields (skill_ids / skills → AgentSkill)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import skill_legacy_migration as mig
from app.models.agent import Agent
from app.models.market import AgentSkill, UserInstalledItem
from app.models.skill import Skill


def _scalars(values):
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    result.scalars.return_value.first.return_value = values[0] if values else None
    return result


def _agent(**cfg):
    return Agent(
        id=uuid4(),
        owner_id=uuid4(),
        nickname="a",
        agent_uid=f"agent-{uuid4()}",
        token=f"tok-{uuid4()}",
        llm_config=cfg,
    )


def _db(*results):
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=list(results))
    db.add = MagicMock()
    return db


def _item(owner_id, skill_id=None, status="active"):
    return UserInstalledItem(
        id=uuid4(), user_id=owner_id, type="skill", name="x", skill_id=skill_id or uuid4(), status=status
    )


def _added(db, cls):
    return [call.args[0] for call in db.add.call_args_list if isinstance(call.args[0], cls)]


# ── _migrate_agent ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_without_legacy_keys_is_untouched():
    agent = _agent(model="m", system_prompt="p")
    db = _db()

    assert await mig._migrate_agent(db, agent) is False
    db.execute.assert_not_awaited()
    assert agent.llm_config == {"model": "m", "system_prompt": "p"}


@pytest.mark.asyncio
async def test_skill_ids_become_agent_skill_rows_and_text_joins_prompt(monkeypatch):
    agent = _agent(model="m", system_prompt="Be nice.", skill_ids=["a", "b"], skills="  extra rules ")
    s1, s2 = uuid4(), uuid4()
    agent.llm_config["skill_ids"] = [str(s1), "not-a-uuid", None, str(s2), str(s1)]
    items = {s1: _item(agent.owner_id, s1), s2: _item(agent.owner_id, s2)}
    monkeypatch.setattr(mig, "_registry_item_for", AsyncMock(side_effect=lambda db, owner, sid: items[sid]))
    db = _db(_scalars([]))  # no explicit AgentSkill rows yet

    assert await mig._migrate_agent(db, agent) is True

    rows = _added(db, AgentSkill)
    assert [row.user_installed_item_id for row in rows] == [items[s1].id, items[s2].id]
    assert all(row.agent_id == agent.id for row in rows)
    assert agent.llm_config == {"model": "m", "system_prompt": "Be nice.\n\nextra rules"}


@pytest.mark.asyncio
async def test_text_becomes_prompt_when_agent_had_none():
    agent = _agent(skills="only rules")
    db = _db(_scalars([]))

    await mig._migrate_agent(db, agent)

    assert agent.llm_config == {"system_prompt": "only rules"}


@pytest.mark.asyncio
async def test_agent_with_explicit_rows_just_drops_ignored_legacy_keys(monkeypatch):
    agent = _agent(system_prompt="p", skill_ids=[str(uuid4())], skills="ignored")
    resolver = AsyncMock()
    monkeypatch.setattr(mig, "_registry_item_for", resolver)
    db = _db(_scalars([uuid4()]))  # an AgentSkill row already exists

    assert await mig._migrate_agent(db, agent) is True

    resolver.assert_not_awaited()
    assert _added(db, AgentSkill) == []
    assert agent.llm_config == {"system_prompt": "p"}


@pytest.mark.asyncio
async def test_unresolvable_skill_is_dropped(monkeypatch):
    agent = _agent(skill_ids=[str(uuid4())])
    monkeypatch.setattr(mig, "_registry_item_for", AsyncMock(return_value=None))
    db = _db(_scalars([]))

    await mig._migrate_agent(db, agent)

    assert _added(db, AgentSkill) == []
    assert agent.llm_config == {}


# ── _registry_item_for ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exact_registry_item_prefers_active():
    owner, sid = uuid4(), uuid4()
    disabled, active = _item(owner, sid, "disabled"), _item(owner, sid)
    db = _db(_scalars([disabled, active]))

    assert await mig._registry_item_for(db, owner, sid) is active
    db.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_falls_back_to_same_named_registry_item():
    owner = uuid4()
    builtin = Skill(id=uuid4(), owner_id=None, name="skill-arch", source="builtin")
    store_item = _item(owner)
    db = _db(_scalars([]), _scalars([store_item]))
    db.get = AsyncMock(return_value=builtin)

    assert await mig._registry_item_for(db, owner, builtin.id) is store_item
    assert "skill.name" in str(db.execute.await_args_list[1].args[0])


@pytest.mark.asyncio
async def test_foreign_or_missing_skill_resolves_to_none():
    owner = uuid4()
    foreign = Skill(id=uuid4(), owner_id=uuid4(), name="theirs", source="upload")
    db = _db(_scalars([]))
    db.get = AsyncMock(return_value=foreign)
    assert await mig._registry_item_for(db, owner, foreign.id) is None

    db = _db(_scalars([]))
    db.get = AsyncMock(return_value=None)
    assert await mig._registry_item_for(db, owner, uuid4()) is None


@pytest.mark.asyncio
async def test_creates_registry_item_when_owner_has_none():
    owner = uuid4()
    skill = Skill(id=uuid4(), owner_id=owner, name="mine", display_name="Mine", source="upload")
    db = _db(_scalars([]), _scalars([]))
    db.get = AsyncMock(return_value=skill)

    item = await mig._registry_item_for(db, owner, skill.id)

    assert item.user_id == owner and item.skill_id == skill.id
    assert item.type == "skill" and item.is_custom is True and item.name == "Mine"
    db.add.assert_called_once_with(item)
    db.flush.assert_awaited_once()


# ── migrate_legacy_skill_ids ─────────────────────────────────────────────────


def _maker(*sessions):
    it = iter(sessions)

    def factory():
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=lambda: next(it))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    return MagicMock(return_value=factory)


@pytest.mark.asyncio
async def test_migrate_never_raises_when_listing_fails(monkeypatch):
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(mig, "sessionmaker", _maker(db))

    await mig.migrate_legacy_skill_ids(MagicMock())


@pytest.mark.asyncio
async def test_migrate_commits_each_agent_and_isolates_failures(monkeypatch):
    good, bad = _agent(skills="x"), _agent(skills="y")
    listing = MagicMock()
    listing.all.return_value = [
        (good.id, good.llm_config),
        (bad.id, bad.llm_config),
        (uuid4(), {"model": "untouched"}),
    ]
    list_db = AsyncMock()
    list_db.execute = AsyncMock(return_value=listing)
    good_db = AsyncMock()
    good_db.get = AsyncMock(return_value=good)
    bad_db = AsyncMock()
    bad_db.get = AsyncMock(return_value=bad)
    monkeypatch.setattr(mig, "sessionmaker", _maker(list_db, good_db, bad_db))

    async def fake_migrate(db, agent):
        if agent is bad:
            raise RuntimeError("broken row")
        return True

    monkeypatch.setattr(mig, "_migrate_agent", fake_migrate)

    await mig.migrate_legacy_skill_ids(MagicMock())

    good_db.commit.assert_awaited_once()
    bad_db.commit.assert_not_awaited()
    bad_db.rollback.assert_awaited_once()
