"""Skills tab as the installed-item model (internal design notes stage 4)."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import skills as skills_api
from app.models.market import UserInstalledItem
from app.models.skill import Skill


@pytest.fixture(autouse=True)
def _plan_skill_limits(monkeypatch):
    """The fake DBs below script every query; the tariff's skill limits (T6,
    own tests in test_tariff_enforcement.py) would take one of them."""
    monkeypatch.setattr(
        skills_api, "get_skill_limits",
        AsyncMock(return_value=skills_api.SkillLimits("hobby", 25 << 20, 50 << 20)),
    )


def _zip_bytes(name: str = "skill-mine", version: str = "1.0.0") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "SKILL.md",
            f"---\nname: {name}\ndescription: test skill\nversion: {version}\n---\nBody text\n",
        )
    return buf.getvalue()


class _FakeUpload:
    """Minimal stand-in for FastAPI's UploadFile: one async .read()."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.filename = "skill.zip"

    async def read(self, n: int = -1) -> bytes:
        data, self._data = self._data, b""
        return data


def _first(value):
    result = MagicMock()
    result.scalars.return_value.first.return_value = value
    return result


def _rowcount(n: int):
    result = MagicMock()
    result.rowcount = n
    return result


def _all_rows(values):
    result = MagicMock()
    result.all.return_value = values
    return result


# ── GET /skills/ ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_skills_returns_installed_items_with_is_default(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    default_market_item_id = uuid4()
    other_market_item_id = uuid4()

    default_item = UserInstalledItem(
        id=uuid4(), user_id=user.id, type="skill", is_custom=False,
        market_item_id=default_market_item_id, market_slug="skill-arch",
        name="Skill Arch", skill_id=uuid4(),
    )
    default_skill = Skill(
        id=default_item.skill_id, owner_id=None, name="skill-arch",
        display_name="Skill Arch", description="d", version="1.0.0", source="store",
    )

    market_item = UserInstalledItem(
        id=uuid4(), user_id=user.id, type="skill", is_custom=False,
        market_item_id=other_market_item_id, market_slug="skill-extra",
        name="Extra", skill_id=uuid4(),
    )
    market_skill = Skill(
        id=market_item.skill_id, owner_id=None, name="skill-extra",
        display_name="Extra", description="d", version="1.0.0", source="store",
        file_count=3, is_executable=True,
    )

    upload_item = UserInstalledItem(
        id=uuid4(), user_id=user.id, type="skill", is_custom=True,
        skill_id=uuid4(), name="Mine",
    )
    upload_skill = Skill(
        id=upload_item.skill_id, owner_id=user.id, name="mine",
        display_name="Mine", description="d", version="1.0.0", source="upload", size_bytes=42,
    )

    rows = _all_rows([
        (default_item, default_skill),
        (market_item, market_skill),
        (upload_item, upload_skill),
    ])
    db = AsyncMock()
    db.execute = AsyncMock(return_value=rows)

    monkeypatch.setattr(skills_api, "reconcile_default_skills", AsyncMock())
    monkeypatch.setattr(skills_api, "default_skill_slugs", lambda: {"skill-arch"})

    result = await skills_api.list_skills(db=db, current_user=user)

    by_name = {r.name: r for r in result}
    assert by_name["skill-arch"].is_default is True
    assert by_name["skill-arch"].installed_item_id == default_item.id
    assert by_name["skill-extra"].is_default is False
    assert by_name["skill-extra"].file_count == 3
    assert by_name["skill-extra"].is_executable is True
    assert by_name["mine"].source == "upload"
    assert by_name["mine"].is_default is False
    assert by_name["mine"].skill_id == upload_skill.id


@pytest.mark.asyncio
async def test_list_skills_skips_reconcile_when_safety_net_disabled(monkeypatch):
    monkeypatch.setattr(skills_api, "SKILL_DEFAULTS_ON_LIST", False)
    reconcile = AsyncMock()
    monkeypatch.setattr(skills_api, "reconcile_default_skills", reconcile)
    monkeypatch.setattr(skills_api, "default_skill_slugs", lambda: set())

    db = AsyncMock()
    db.execute = AsyncMock(return_value=_all_rows([]))

    result = await skills_api.list_skills(db=db, current_user=SimpleNamespace(id=uuid4()))

    reconcile.assert_not_awaited()
    assert result == []


# ── POST /skills/upload ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_skill_registers_installed_item_for_new_skill():
    user = SimpleNamespace(id=uuid4())
    data = _zip_bytes(name="skill-mine")

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _first(MagicMock()),   # _get_or_create_quota: quota row already exists
        _first(None),          # existing owned skill lookup: none
        _rowcount(1),          # _reserve_quota UPDATE succeeds
        _first(None),          # installed item lookup: none yet
        _all_rows([]),         # _detail SkillFile listing
    ])
    db.add = MagicMock()

    result = await skills_api.upload_skill(
        archive=_FakeUpload(data), files=None, paths=None, overwrite=False,
        db=db, current_user=user,
    )

    added = [c.args[0] for c in db.add.call_args_list]
    skills_added = [a for a in added if isinstance(a, Skill)]
    installed_added = [a for a in added if isinstance(a, UserInstalledItem)]
    assert len(skills_added) == 1
    assert len(installed_added) == 1
    assert installed_added[0].user_id == user.id
    assert installed_added[0].is_custom is True
    assert installed_added[0].skill_id == skills_added[0].id
    assert result["name"] == "skill-mine"
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_upload_skill_reuses_existing_installed_item_on_overwrite(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    data = _zip_bytes(name="skill-mine", version="2.0.0")

    existing_skill = Skill(
        id=uuid4(), owner_id=user.id, name="skill-mine", version="1.0.0",
        source="upload", size_bytes=10, skill_md="old",
    )
    existing_installed = UserInstalledItem(
        id=uuid4(), user_id=user.id, type="skill", is_custom=True,
        skill_id=existing_skill.id, name="Old Name",
    )

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _first(MagicMock()),        # _get_or_create_quota
        _first(existing_skill),     # existing owned skill lookup: found
        _rowcount(1),                # _reserve_quota UPDATE
        _first(existing_installed),  # installed item lookup: found
        _all_rows([]),                # _detail SkillFile listing
    ])
    db.add = MagicMock()
    monkeypatch.setattr(skills_api, "delete_stored_files", AsyncMock())

    await skills_api.upload_skill(
        archive=_FakeUpload(data), files=None, paths=None, overwrite=True,
        db=db, current_user=user,
    )

    installed_added = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], UserInstalledItem)]
    assert installed_added == [existing_installed]
    assert existing_installed.name == "Skill Mine"


# ── DELETE /skills/{installed_item_id} ───────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_installed_skill_market_removes_only_registration():
    user = SimpleNamespace(id=uuid4())
    item = UserInstalledItem(
        id=uuid4(), user_id=user.id, type="skill", is_custom=False,
        skill_id=uuid4(), market_item_id=uuid4(), market_slug="skill-arch",
    )
    db = AsyncMock()
    db.get = AsyncMock(return_value=item)

    await skills_api.delete_installed_skill(item.id, db=db, current_user=user)

    db.get.assert_awaited_once_with(UserInstalledItem, item.id)
    db.delete.assert_awaited_once_with(item)
    db.execute.assert_not_awaited()  # no quota touched for a market/default skill
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_installed_skill_upload_frees_quota_and_removes_content(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    skill = Skill(id=uuid4(), owner_id=user.id, name="mine", source="upload", size_bytes=999)
    item = UserInstalledItem(id=uuid4(), user_id=user.id, type="skill", is_custom=True, skill_id=skill.id)

    db = AsyncMock()
    db.get = AsyncMock(side_effect=[item, skill])
    monkeypatch.setattr(skills_api, "delete_stored_files", AsyncMock())

    await skills_api.delete_installed_skill(item.id, db=db, current_user=user)

    # item must be deleted before skill: its skill_id FK would otherwise
    # block the skill delete (no ORM relationship() to reorder these).
    assert db.delete.await_args_list == [call(item), call(skill)]
    db.execute.assert_awaited_once()  # quota release UPDATE
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_installed_skill_404_when_missing():
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc:
        await skills_api.delete_installed_skill(uuid4(), db=db, current_user=SimpleNamespace(id=uuid4()))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_installed_skill_404_when_owned_by_other_user():
    item = UserInstalledItem(id=uuid4(), user_id=uuid4(), type="skill", is_custom=False, skill_id=uuid4())
    db = AsyncMock()
    db.get = AsyncMock(return_value=item)

    with pytest.raises(HTTPException) as exc:
        await skills_api.delete_installed_skill(item.id, db=db, current_user=SimpleNamespace(id=uuid4()))
    assert exc.value.status_code == 404
