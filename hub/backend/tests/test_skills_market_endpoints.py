"""GET/POST /api/v1/skills/market/* (internal design notes stage 2)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import skills as skills_api
from app.core import market_client
from app.core.skill_market_install import MarketInstallError


def _rows(values):
    """(market_item_id, market_slug) rows of the user's installs."""
    result = MagicMock()
    result.all.return_value = [(v, None) for v in values]
    return result


@pytest.mark.asyncio
async def test_browse_market_skills_marks_installed_items(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    installed_id = uuid4()
    not_installed_id = uuid4()

    catalog = {
        "items": [
            {
                "id": str(installed_id), "slug": "skill-arch", "name": "Arch",
                "description": "d", "category": "builtin", "is_paid": False,
                "price_tier": None, "version": "1.0.0", "is_default": True,
                "file_count": 12, "is_executable": True,
            },
            {
                "id": str(not_installed_id), "slug": "skill-coder", "name": "Coder",
                "description": "d", "category": "builtin", "is_paid": False,
                "price_tier": None, "version": "1.0.0", "is_default": True,
            },
        ],
        "total": 2, "page": 1, "page_size": 12,
    }
    monkeypatch.setattr(market_client, "list_catalog", AsyncMock(return_value=catalog))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_rows([installed_id]))

    result = await skills_api.browse_market_skills(
        q=None, page=1, page_size=12, db=db, current_user=user
    )

    by_slug = {item.slug: item for item in result.items}
    assert by_slug["skill-arch"].installed is True
    assert by_slug["skill-coder"].installed is False
    assert by_slug["skill-arch"].file_count == 12
    assert by_slug["skill-arch"].is_executable is True
    assert by_slug["skill-coder"].file_count == 0
    assert by_slug["skill-coder"].is_executable is False


@pytest.mark.asyncio
async def test_browse_market_skills_marks_bundled_install_by_slug(monkeypatch):
    """A starter skill linked from the Hub's bundle before Market's id was
    known has market_item_id NULL — it must still show as installed, or Add
    would offer a duplicate (internal design notes stage 3)."""
    catalog = {
        "items": [
            {
                "id": str(uuid4()), "slug": "skill-arch", "name": "Arch",
                "description": "d", "category": "builtin", "is_paid": False,
                "price_tier": None, "version": "1.0.0", "is_default": True,
            },
        ],
        "total": 1, "page": 1, "page_size": 12,
    }
    monkeypatch.setattr(market_client, "list_catalog", AsyncMock(return_value=catalog))
    rows = MagicMock()
    rows.all.return_value = [(None, "skill-arch")]
    db = AsyncMock()
    db.execute = AsyncMock(return_value=rows)

    result = await skills_api.browse_market_skills(
        q=None, page=1, page_size=12, db=db, current_user=SimpleNamespace(id=uuid4())
    )

    assert result.items[0].installed is True


@pytest.mark.asyncio
async def test_browse_market_skills_flags_items_without_a_published_package(monkeypatch):
    """A catalog item created but never published (no package_ref yet) must
    surface has_package=False so the frontend can disable Add — installing it
    would 404 against Market's package endpoint (skill_market_install.py)."""
    with_package_id = uuid4()
    no_package_id = uuid4()

    catalog = {
        "items": [
            {
                "id": str(with_package_id), "slug": "skill-arch", "name": "Arch",
                "description": "d", "category": "builtin", "is_paid": False,
                "price_tier": None, "version": "1.0.0", "is_default": True,
                "package_ref": "skill-arch@1.0.0",
            },
            {
                "id": str(no_package_id), "slug": "skill-xlsx", "name": "Xlsx",
                "description": "d", "category": None, "is_paid": False,
                "price_tier": None, "version": "1.0.0", "is_default": False,
                "package_ref": None,
            },
        ],
        "total": 2, "page": 1, "page_size": 12,
    }
    monkeypatch.setattr(market_client, "list_catalog", AsyncMock(return_value=catalog))

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_rows([]))

    result = await skills_api.browse_market_skills(
        q=None, page=1, page_size=12, db=db, current_user=SimpleNamespace(id=uuid4())
    )

    by_slug = {item.slug: item for item in result.items}
    assert by_slug["skill-arch"].has_package is True
    assert by_slug["skill-xlsx"].has_package is False


@pytest.mark.asyncio
async def test_browse_market_skills_502s_when_marketplace_unavailable(monkeypatch):
    monkeypatch.setattr(
        market_client, "list_catalog", AsyncMock(side_effect=market_client.MarketClientError("down"))
    )
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await skills_api.browse_market_skills(
            q=None, page=1, page_size=12, db=db, current_user=SimpleNamespace(id=uuid4())
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_install_market_skill_404s_for_unknown_slug(monkeypatch):
    monkeypatch.setattr(
        market_client, "get_skill", AsyncMock(side_effect=market_client.MarketNotFoundError("nope"))
    )
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await skills_api.install_market_skill(
            skills_api.MarketInstallIn(slug="does-not-exist"), db=db, current_user=SimpleNamespace(id=uuid4())
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_install_market_skill_propagates_install_error(monkeypatch):
    monkeypatch.setattr(
        market_client, "get_skill",
        AsyncMock(return_value={"id": str(uuid4()), "slug": "skill-arch", "version": "1.0.0", "name": "Arch"}),
    )
    monkeypatch.setattr(
        "app.api.v1.skills.install_market_skill_for_user",
        AsyncMock(side_effect=MarketInstallError("checksum mismatch", status_code=502)),
    )
    db = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await skills_api.install_market_skill(
            skills_api.MarketInstallIn(slug="skill-arch"), db=db, current_user=SimpleNamespace(id=uuid4())
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_install_market_skill_returns_skill_detail(monkeypatch):
    from app.models.skill import Skill
    from app.models.market import UserInstalledItem

    market_item_id = uuid4()
    shared_skill = Skill(
        id=uuid4(), owner_id=None, name="skill-arch", display_name="Skill Arch",
        description="d", version="1.0.0", source="store", skill_md="body",
    )
    installed = UserInstalledItem(
        id=uuid4(), user_id=uuid4(), type="skill", market_item_id=market_item_id,
        market_slug="skill-arch", name="Skill Arch", skill_id=shared_skill.id,
    )

    monkeypatch.setattr(
        market_client, "get_skill",
        AsyncMock(return_value={"id": str(market_item_id), "slug": "skill-arch", "version": "1.0.0", "name": "Arch", "is_default": True}),
    )
    monkeypatch.setattr(
        "app.api.v1.skills.install_market_skill_for_user", AsyncMock(return_value=installed)
    )

    db = AsyncMock()
    db.get = AsyncMock(return_value=shared_skill)
    db.execute = AsyncMock(return_value=_rows([]))  # SkillFile listing in _detail()

    result = await skills_api.install_market_skill(
        skills_api.MarketInstallIn(slug="skill-arch"), db=db, current_user=SimpleNamespace(id=uuid4())
    )

    assert result.installed_item_id == installed.id
    assert result.market_item_id == market_item_id
    assert result.is_default is True
    assert result.name == "skill-arch"


@pytest.mark.asyncio
async def test_get_skill_detail_surfaces_declared_dependencies(monkeypatch):
    """Stage 6 (internal design notes §8): the Skills tab detail
    view shows a bundled skill's declared pip/apt deps, so GET /skills/{id}
    must round-trip Skill.dependencies through SkillDetailOut."""
    from app.models.skill import Skill

    user = SimpleNamespace(id=uuid4())
    skill = Skill(
        id=uuid4(), owner_id=None, name="xlsx", display_name="Xlsx",
        description="d", version="1.0.0", source="store", skill_md="body",
        file_count=54, is_executable=True,
        dependencies={"pip": ["openpyxl", "pandas"]},
    )

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_rows([]))
    monkeypatch.setattr(skills_api, "_resolve_skill", AsyncMock(return_value=skill))

    result = await skills_api.get_skill(skill_id=skill.id, db=db, current_user=user)

    assert result["dependencies"] == {"pip": ["openpyxl", "pandas"]}
    assert result["file_count"] == 54
    assert result["is_executable"] is True
