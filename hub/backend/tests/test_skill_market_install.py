"""Hub -> Market skill install (internal design notes stage 2)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import market_client
from app.core.skill_market_install import (
    MarketInstallError,
    ensure_shared_skill,
    install_market_skill_for_user,
)
from app.models.market import UserInstalledItem
from app.models.skill import Skill


def _zip_bytes(name: str = "skill-arch", version: str = "1.0.0", bundled: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "SKILL.md",
            f"---\nname: {name}\ndescription: test skill\nversion: {version}\n---\nBody text\n",
        )
        if bundled:
            zf.writestr("scripts/run.py", "print('run')\n")
            zf.writestr("scripts/lib/helper.py", "def helper():\n    pass\n")
    return buf.getvalue()


def _scalar_first(value):
    result = MagicMock()
    result.scalars.return_value.first.return_value = value
    result.scalars.return_value.all.return_value = [value] if value is not None else []
    return result


def _scalar_one_or_none(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    result.scalars.return_value.first.return_value = value
    return result


@pytest.mark.asyncio
async def test_ensure_shared_skill_downloads_and_creates_when_missing(monkeypatch):
    data = _zip_bytes()
    checksum = hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(market_client, "fetch_package", AsyncMock(return_value=(data, checksum)))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))
    db.add = MagicMock()

    skill = await ensure_shared_skill("skill-arch", db, version="1.0.0")

    assert skill.owner_id is None
    assert skill.name == "skill-arch"
    assert skill.version == "1.0.0"
    assert skill.source == "store"
    assert skill.store_ref == "skill-arch@1.0.0"
    assert skill.checksum == checksum
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_ensure_shared_skill_skips_download_when_version_matches(monkeypatch):
    existing = Skill(
        id=uuid4(), owner_id=None, name="skill-arch", version="1.0.0",
        source="store", checksum="same-checksum",
    )
    monkeypatch.setattr(
        market_client,
        "get_skill",
        AsyncMock(return_value={"versions": [{"version": "1.0.0", "checksum": "same-checksum"}]}),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(market_client, "fetch_package", fetch)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(existing))

    skill = await ensure_shared_skill("skill-arch", db, version="1.0.0")

    assert skill is existing
    db.commit.assert_not_awaited()
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_shared_skill_resyncs_when_same_version_checksum_drifts(monkeypatch):
    """A skill republished under the same version string (e.g. an operator
    adding bundled files that were missing the first time) must not be
    silently served stale forever — internal design notes Stage 8
    incident: this is exactly what happened to a real 'xlsx' install."""
    existing = Skill(
        id=uuid4(), owner_id=None, name="skill-arch", version="1.0.0",
        source="store", checksum="old-checksum", file_count=0, is_executable=False,
    )
    data = _zip_bytes(bundled=True)
    new_checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        market_client,
        "get_skill",
        AsyncMock(return_value={"versions": [{"version": "1.0.0", "checksum": new_checksum}]}),
    )
    monkeypatch.setattr(
        market_client, "fetch_package", AsyncMock(return_value=(data, new_checksum))
    )
    monkeypatch.setattr(
        "app.core.skill_market_install.delete_stored_files", AsyncMock()
    )

    async def _fake_store(skill, package, db):
        skill.file_count = len(package.files)
        skill.is_executable = package.is_executable

    monkeypatch.setattr("app.core.skill_market_install.store_package_files", _fake_store)

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(existing))
    db.add = MagicMock()

    skill = await ensure_shared_skill("skill-arch", db, version="1.0.0")

    assert skill is existing
    assert skill.checksum == new_checksum
    assert skill.file_count == 2
    assert skill.is_executable is True
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_ensure_shared_skill_falls_back_to_existing_when_market_unreachable(monkeypatch):
    existing = Skill(
        id=uuid4(), owner_id=None, name="skill-arch", version="1.0.0",
        source="store", checksum="whatever",
    )
    monkeypatch.setattr(
        market_client,
        "get_skill",
        AsyncMock(side_effect=market_client.MarketClientError("down")),
    )
    fetch = AsyncMock()
    monkeypatch.setattr(market_client, "fetch_package", fetch)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(existing))

    skill = await ensure_shared_skill("skill-arch", db, version="1.0.0")

    assert skill is existing
    db.commit.assert_not_awaited()
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_shared_skill_rejects_checksum_mismatch(monkeypatch):
    data = _zip_bytes()
    monkeypatch.setattr(
        market_client, "fetch_package", AsyncMock(return_value=(data, "0" * 64))
    )

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))

    with pytest.raises(MarketInstallError, match="Checksum mismatch"):
        await ensure_shared_skill("skill-arch", db, version="1.0.0")
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_shared_skill_rejects_slug_package_name_mismatch(monkeypatch):
    data = _zip_bytes(name="skill-coder")
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(market_client, "fetch_package", AsyncMock(return_value=(data, checksum)))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))

    with pytest.raises(MarketInstallError, match="does not match slug"):
        await ensure_shared_skill("skill-arch", db, version="1.0.0")


@pytest.mark.asyncio
async def test_ensure_shared_skill_resolves_latest_version_when_omitted(monkeypatch):
    data = _zip_bytes(version="2.0.0")
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        market_client,
        "get_skill",
        AsyncMock(return_value={"versions": [{"version": "2.0.0"}, {"version": "1.0.0"}]}),
    )
    monkeypatch.setattr(market_client, "fetch_package", AsyncMock(return_value=(data, checksum)))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))
    db.add = MagicMock()

    skill = await ensure_shared_skill("skill-arch", db)

    assert skill.version == "2.0.0"
    market_client.fetch_package.assert_awaited_with("skill-arch", "2.0.0")


@pytest.mark.asyncio
async def test_ensure_shared_skill_version_update_replaces_bundled_files_without_orphans(monkeypatch):
    """A version bump on a bundled skill must drop the old file tree before
    writing the new one, so no SkillFile rows from the previous version are
    left orphaned (internal design notes Stage 2 acceptance)."""
    existing = Skill(
        id=uuid4(), owner_id=None, name="skill-arch", version="1.0.0",
        source="store", file_count=5, is_executable=True,
    )
    data = _zip_bytes(version="2.0.0", bundled=True)
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(market_client, "fetch_package", AsyncMock(return_value=(data, checksum)))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(existing))
    db.add = MagicMock()

    calls: list[str] = []
    real_delete = AsyncMock(side_effect=lambda *a, **k: calls.append("delete"))
    monkeypatch.setattr("app.core.skill_market_install.delete_stored_files", real_delete)

    async def _fake_store(skill, package, db):
        calls.append("store")
        skill.file_count = len(package.files)
        skill.is_executable = package.is_executable

    monkeypatch.setattr("app.core.skill_market_install.store_package_files", _fake_store)

    skill = await ensure_shared_skill("skill-arch", db, version="2.0.0")

    assert skill is existing
    assert skill.version == "2.0.0"
    assert calls == ["delete", "store"]  # old tree gone before the new one is written
    real_delete.assert_awaited_once_with(existing, db)
    assert skill.file_count == 2  # only the new version's files, none orphaned
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_install_market_skill_for_user_is_idempotent(monkeypatch):
    user_id = uuid4()
    market_item_id = uuid4()
    existing = UserInstalledItem(
        id=uuid4(), user_id=user_id, type="skill", market_item_id=market_item_id,
        name="skill-arch",
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_one_or_none(existing))

    called = AsyncMock()
    monkeypatch.setattr("app.core.skill_market_install.ensure_shared_skill", called)

    result = await install_market_skill_for_user(
        user_id, {"id": market_item_id, "slug": "skill-arch", "version": "1.0.0", "name": "Skill Arch"}, db
    )

    assert result is existing
    called.assert_not_awaited()
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_install_market_skill_for_user_creates_installed_item(monkeypatch):
    user_id = uuid4()
    market_item_id = uuid4()
    shared_skill = Skill(id=uuid4(), owner_id=None, name="skill-arch", version="1.0.0", source="store")

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_one_or_none(None))
    db.add = MagicMock()

    monkeypatch.setattr(
        "app.core.skill_market_install.ensure_shared_skill", AsyncMock(return_value=shared_skill)
    )

    result = await install_market_skill_for_user(
        user_id,
        {"id": market_item_id, "slug": "skill-arch", "version": "1.0.0", "name": "Skill Arch"},
        db,
    )

    created = db.add.call_args.args[0]
    assert created.user_id == user_id
    assert created.type == "skill"
    assert created.is_custom is False
    assert created.market_item_id == market_item_id
    assert created.market_slug == "skill-arch"
    assert created.skill_id == shared_skill.id
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_install_market_skill_for_user_adopts_unlinked_bundled_install(monkeypatch):
    """A starter skill linked from the Hub's bundle before Market's id was
    known gets the id instead of a duplicate install (internal design notes
    stage 3)."""
    user_id = uuid4()
    market_item_id = uuid4()
    bundled = UserInstalledItem(
        id=uuid4(), user_id=user_id, type="skill", market_slug="skill-arch", name="Skill Arch",
    )
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(side_effect=[_scalar_one_or_none(None), _scalar_first(bundled)])
    ensure = AsyncMock()
    monkeypatch.setattr("app.core.skill_market_install.ensure_shared_skill", ensure)

    result = await install_market_skill_for_user(
        user_id, {"id": market_item_id, "slug": "skill-arch", "version": "1.0.0"}, db
    )

    assert result is bundled
    assert bundled.market_item_id == market_item_id
    ensure.assert_not_awaited()
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_install_shared_skill_for_user_links_the_shared_row_without_market(monkeypatch):
    from app.core.skill_market_install import install_shared_skill_for_user

    for name in ("get_skill", "fetch_package"):
        monkeypatch.setattr(market_client, name, AsyncMock(side_effect=AssertionError(name)))
    user_id = uuid4()
    shared = Skill(id=uuid4(), owner_id=None, name="skill-arch", display_name="Skill Arch", source="store")
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(side_effect=[_scalar_first(None), _scalar_first(shared)])

    item = await install_shared_skill_for_user(user_id, "skill-arch", db)

    assert item.skill_id == shared.id
    assert item.market_slug == "skill-arch"
    assert item.market_item_id is None
    assert item.name == "Skill Arch"
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_install_shared_skill_for_user_returns_none_without_a_shared_row():
    from app.core.skill_market_install import install_shared_skill_for_user

    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(side_effect=[_scalar_first(None), _scalar_first(None)])

    assert await install_shared_skill_for_user(uuid4(), "xlsx", db) is None
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_install_shared_skill_for_user_fills_a_missing_market_id():
    from app.core.skill_market_install import install_shared_skill_for_user

    market_item_id = uuid4()
    existing = UserInstalledItem(id=uuid4(), user_id=uuid4(), type="skill", market_slug="skill-arch", name="a")
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(return_value=_scalar_first(existing))

    result = await install_shared_skill_for_user(existing.user_id, "skill-arch", db, market_item_id=market_item_id)

    assert result is existing
    assert existing.market_item_id == market_item_id
