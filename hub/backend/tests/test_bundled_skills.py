"""Starter skills shipped with the Hub (internal design notes stage 3).

Unit part: the manifest and version ordering. DB part (PostgreSQL, marked
`db`): seeding from the bundle, registration-time provisioning with Market
switched off, and the background sync that only pulls updates. The DB tests
create and change *shared* skill rows, so run them against a scratch
database, never the live one:

    DATABASE_URL=postgresql+asyncpg://.../legion_bundled_test pytest tests/test_bundled_skills.py

They skip unless the database name contains "test".
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from app.core import bundled_skills, default_catalog, market_client
from app.core.skill_package import build_package


# ---------- unit ---------------------------------------------------------------

def test_manifest_lists_every_shipped_skill_and_the_three_defaults():
    manifest = bundled_skills.load_manifest()
    assert [s.slug for s in manifest] == [
        "skill-arch", "skill-coder", "skill-writer",
        "skill-vps-deploy",
    ]
    assert bundled_skills.bundled_default_slugs() == [
        "skill-arch", "skill-coder", "skill-writer",
    ]


@pytest.mark.parametrize("slug", bundled_skills.bundled_slugs())
def test_every_shipped_skill_is_a_valid_package_named_after_its_folder(slug):
    package = build_package(bundled_skills._read_skill_dir(bundled_skills.BUILTIN_SKILLS_DIR / slug))
    assert package.name == slug


def test_starter_agents_only_use_shipped_skills():
    from app.core.agent_defaults import DEFAULT_AGENTS

    assert {slug for _, _, slug in DEFAULT_AGENTS} <= set(bundled_skills.bundled_slugs())


def test_version_key_orders_numerically():
    key = bundled_skills.version_key
    assert key("1.10.0") > key("1.9.0")
    assert key("2.0.0") > key("1.99.99")
    assert key("1.0.0") == key("1.0.0")
    assert key("1.0.0") < key("1.0.1")


# ---------- DB integration (requires PostgreSQL) -------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")


def _is_scratch_db(url: str) -> bool:
    # These tests wipe shared skill rows and non-custom installs: refuse any
    # database whose name does not say it is a test one (e.g. the live
    # `somemaru` of the dev stack).
    from sqlalchemy.engine import make_url

    try:
        return "test" in (make_url(url).database or "")
    except Exception:
        return False


pytestmark_db = pytest.mark.skipif(
    not _is_scratch_db(DB_URL), reason="needs DATABASE_URL pointing at a *test* database"
)


@pytest_asyncio.fixture
async def engine() -> AsyncGenerator:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel

    import app.models  # noqa: F401  (populate metadata)

    eng = create_async_engine(DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        # Each test starts from an empty set of shared rows and installs.
        await conn.execute(text("DELETE FROM user_default_provision"))
        await conn.execute(text("DELETE FROM user_installed_item WHERE NOT is_custom"))
        await conn.execute(text("DELETE FROM skill WHERE owner_id IS NULL"))
    default_catalog.reset()
    yield eng
    default_catalog.reset()
    await eng.dispose()


@pytest.fixture
def market_down(monkeypatch):
    calls = AsyncMock(side_effect=market_client.MarketClientError("Market is down"))
    for name in ("list_catalog", "get_skill", "get_catalog_item", "fetch_package"):
        monkeypatch.setattr(market_client, name, calls)
    return calls


def _session(engine):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()


async def _shared_rows(engine):
    from sqlalchemy import text

    async with engine.connect() as conn:
        res = await conn.execute(
            text("SELECT name, id, version, source, checksum FROM skill WHERE owner_id IS NULL ORDER BY name")
        )
        return {row.name: row for row in res}


async def _make_user(engine):
    from sqlalchemy import text

    user_id = uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO \"user\" (id, email, hashed_password, is_active, two_factor_enabled, is_owner) "
                 "VALUES (:id, :email, 'x', true, false, false)"),
            {"id": user_id, "email": f"{user_id}@bundled.test"},
        )
    return user_id


def _zip_of(slug: str, version: str | None = None, extra: str = "") -> bytes:
    """Package of a shipped skill, optionally re-versioned / with changed body."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, data in bundled_skills._read_skill_dir(bundled_skills.BUILTIN_SKILLS_DIR / slug):
            if path == "SKILL.md":
                text_ = data.decode("utf-8")
                if version:
                    old = build_package([("SKILL.md", data)]).version
                    text_ = text_.replace(f"version: {old}", f"version: {version}", 1)
                data = (text_ + extra).encode("utf-8")
            zf.writestr(path, data)
    return buf.getvalue()


class FakeMarket:
    """Market's public API as it was while it still republished the shipped
    skills: every shipped skill, at the shipped version unless overridden,
    with the three shipped defaults."""

    def __init__(self, versions: dict[str, str] | None = None, extra: dict[str, str] | None = None):
        self.ids = {slug: uuid4() for slug in bundled_skills.bundled_slugs()}
        self.packages: dict[str, tuple[str, bytes]] = {}
        for slug in bundled_skills.bundled_slugs():
            shipped = build_package(
                bundled_skills._read_skill_dir(bundled_skills.BUILTIN_SKILLS_DIR / slug)
            ).version
            version = (versions or {}).get(slug, shipped)
            data = _zip_of(slug, version if version != shipped else None, (extra or {}).get(slug, ""))
            self.packages[slug] = (version, data)
        self.fetched: list[str] = []
        self.asked: list[str] = []

    async def list_catalog(self, *, type=None, is_default=None, **_):
        if type == "mcp":
            return {"items": []}
        return {"items": [
            {"id": str(self.ids[s]), "slug": s, "version": self.packages[s][0]}
            for s in bundled_skills.bundled_default_slugs()
        ]}

    async def get_skill(self, slug):
        self.asked.append(slug)
        version, data = self.packages[slug]
        return {
            "id": str(self.ids[slug]), "slug": slug, "version": version,
            "versions": [{"version": version, "checksum": hashlib.sha256(data).hexdigest()}],
        }

    async def fetch_package(self, slug, version):
        self.fetched.append(slug)
        _, data = self.packages[slug]
        return data, hashlib.sha256(data).hexdigest()

    def install(self, monkeypatch):
        for name in ("list_catalog", "get_skill", "fetch_package"):
            monkeypatch.setattr(market_client, name, getattr(self, name))
        return self


@pytestmark_db
@pytest.mark.asyncio
@pytest.mark.db
class TestBundledSkillsDB:
    async def test_seed_creates_unverified_store_rows_and_is_idempotent(self, engine, market_down):
        await bundled_skills.seed_bundled_skills(engine)
        first = await _shared_rows(engine)
        await bundled_skills.seed_bundled_skills(engine)
        second = await _shared_rows(engine)

        assert set(first) == set(bundled_skills.bundled_slugs())
        assert all(r.source == "store" and r.checksum is None for r in first.values())
        assert {k: r.id for k, r in first.items()} == {k: r.id for k, r in second.items()}
        market_down.assert_not_awaited()

    async def test_seed_upgrades_an_older_row_in_place_and_never_downgrades(self, engine, market_down):
        from sqlalchemy import text

        await bundled_skills.seed_bundled_skills(engine)
        before = await _shared_rows(engine)
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE skill SET version = '0.1.0' WHERE owner_id IS NULL AND name = 'skill-arch'"))
            await conn.execute(text("UPDATE skill SET version = '9.0.0', checksum = 'm' WHERE owner_id IS NULL AND name = 'skill-coder'"))

        await bundled_skills.seed_bundled_skills(engine)
        after = await _shared_rows(engine)

        assert after["skill-arch"].id == before["skill-arch"].id
        assert after["skill-arch"].version == before["skill-arch"].version
        assert after["skill-coder"].version == "9.0.0" and after["skill-coder"].checksum == "m"

    async def test_seed_converts_a_legacy_builtin_row(self, engine, market_down):
        from sqlalchemy import text

        await bundled_skills.seed_bundled_skills(engine)
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE skill SET source = 'builtin', store_ref = NULL WHERE owner_id IS NULL AND name = 'skill-writer'"))

        await bundled_skills.seed_bundled_skills(engine)
        rows = await _shared_rows(engine)

        assert rows["skill-writer"].source == "store"

    async def test_registration_with_market_down_gets_skills_and_starter_agents(self, engine, market_down):
        from sqlalchemy import text

        from app.core.agent_defaults import provision_default_agents
        from app.core.skill_defaults import reconcile_default_mcps, reconcile_default_skills

        await bundled_skills.seed_bundled_skills(engine)
        user_id = await _make_user(engine)
        async with _session(engine) as db:
            await reconcile_default_skills(user_id, db)
            await reconcile_default_mcps(user_id, db)
            await provision_default_agents(user_id, db)
            await reconcile_default_skills(user_id, db)  # repeat is a no-op

        async with engine.connect() as conn:
            installed = (await conn.execute(text(
                "SELECT market_slug FROM user_installed_item WHERE user_id = :u ORDER BY market_slug"
            ), {"u": user_id})).scalars().all()
            ledger = (await conn.execute(text(
                "SELECT count(*) FROM user_default_provision WHERE user_id = :u"
            ), {"u": user_id})).scalar()
            agent_skills = dict((await conn.execute(text(
                "SELECT a.nickname, i.market_slug FROM agent a "
                "JOIN agent_skill s ON s.agent_id = a.id "
                "JOIN user_installed_item i ON i.id = s.user_installed_item_id "
                "WHERE a.owner_id = :u"
            ), {"u": user_id})).all())

        assert installed == sorted(bundled_skills.bundled_default_slugs() + ["skill-vps-deploy"])
        assert ledger == len(bundled_skills.bundled_default_slugs())
        assert agent_skills == {
            "Lily": "skill-arch", "Sara": "skill-coder", "Paul": "skill-coder",
            "Edgar": "skill-writer", "Kevin": "skill-vps-deploy",
        }
        market_down.assert_not_awaited()

    async def test_sync_never_takes_the_shipped_skills_from_market(self, engine, market_down, monkeypatch):
        """Market no longer republishes the shipped skills. One that still
        lists them, even at a newer version, is not asked about them: the
        Hub's own copy stays the only source and its installs stay unlinked."""
        from sqlalchemy import text

        from app.core.skill_defaults import reconcile_default_skills

        await bundled_skills.seed_bundled_skills(engine)
        user_id = await _make_user(engine)
        async with _session(engine) as db:
            await reconcile_default_skills(user_id, db)
        before = await _shared_rows(engine)

        market = FakeMarket(
            versions={"skill-arch": "99.0.0"}, extra={"skill-arch": "\nUpdated in Market.\n"},
        ).install(monkeypatch)
        assert await default_catalog.sync_once(engine) is True

        after = await _shared_rows(engine)
        async with engine.connect() as conn:
            links = dict((await conn.execute(text(
                "SELECT market_slug, market_item_id FROM user_installed_item WHERE user_id = :u"
            ), {"u": user_id})).all())

        assert market.asked == [] and market.fetched == []
        assert {s: (r.id, r.version, r.checksum) for s, r in after.items()} == {
            s: (r.id, r.version, r.checksum) for s, r in before.items()
        }
        assert set(links.values()) == {None}
        assert default_catalog.market_item_id("skill-arch") is None
        assert default_catalog.default_skill_slugs() == bundled_skills.bundled_default_slugs()

    async def test_sync_with_market_down_keeps_the_bundle(self, engine, market_down):
        await bundled_skills.seed_bundled_skills(engine)

        assert await default_catalog.sync_once(engine) is False
        assert default_catalog.is_synced() is False
        assert default_catalog.default_skill_slugs() == bundled_skills.bundled_default_slugs()

    async def test_market_install_adopts_the_bundled_install(self, engine, market_down, monkeypatch):
        from sqlalchemy import text

        from app.core.skill_defaults import reconcile_default_skills
        from app.core.skill_market_install import install_market_skill_for_user

        await bundled_skills.seed_bundled_skills(engine)
        user_id = await _make_user(engine)
        async with _session(engine) as db:
            await reconcile_default_skills(user_id, db)
        market = FakeMarket().install(monkeypatch)

        async with _session(engine) as db:
            await install_market_skill_for_user(
                user_id, {"id": market.ids["skill-arch"], "slug": "skill-arch", "version": "1.1.0"}, db
            )
        async with engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT market_item_id FROM user_installed_item WHERE user_id = :u AND market_slug = 'skill-arch'"
            ), {"u": user_id})).scalars().all()

        assert rows == [market.ids["skill-arch"]]
