"""store_package_files stamps bundled-package metadata onto the Skill row
(internal design notes Stage 0) — exercised by every install path
(upload, market install, builtin seed) since they all call through here."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.skill_install import store_package_files
from app.core.skill_package import build_package
from app.models.skill import Skill, SkillFile


def _skill_md(name: str = "demo") -> str:
    return f"---\nname: {name}\ndescription: A demo skill.\n---\n\nBody.\n"


@pytest.mark.asyncio
async def test_store_package_files_stamps_executable_bundle_metadata():
    entries = [
        ("SKILL.md", _skill_md().encode()),
        ("scripts/recalc.py", b"print('recalc')"),
        ("scripts/schemas/sml.xsd", b"<xsd/>"),
    ]
    package = build_package(entries)
    skill = Skill(id=uuid4(), owner_id=None, name="demo", source="upload")
    db = MagicMock()
    db.add = MagicMock()

    await store_package_files(skill, package, db)

    assert skill.file_count == 2
    assert skill.is_executable is True
    assert skill.dependencies is None
    assert db.add.call_count == 2


@pytest.mark.asyncio
async def test_store_package_files_stamps_markdown_only_skill():
    package = build_package([("SKILL.md", _skill_md().encode())])
    skill = Skill(id=uuid4(), owner_id=None, name="demo", source="upload")
    db = MagicMock()
    db.add = MagicMock()

    await store_package_files(skill, package, db)

    assert skill.file_count == 0
    assert skill.is_executable is False
    assert skill.dependencies is None
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_store_package_files_stamps_declared_dependencies():
    raw = (
        "---\nname: demo\ndescription: A demo skill.\n"
        "dependencies:\n  pip:\n    - openpyxl\n"
        "---\n\nBody.\n"
    )
    entries = [("SKILL.md", raw.encode()), ("scripts/run.py", b"pass")]
    package = build_package(entries)
    skill = Skill(id=uuid4(), owner_id=None, name="demo", source="upload")
    db = MagicMock()
    db.add = MagicMock()

    await store_package_files(skill, package, db)

    assert skill.dependencies == {"pip": ["openpyxl"]}


@pytest.mark.asyncio
async def test_store_package_files_base64_round_trips_binary_content():
    """A bundled template (e.g. an .xlsx) must survive db-backend storage —
    stored as base64 text and decodable back to the exact original bytes
    (internal design notes Stage 2)."""
    png_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
    entries = [
        ("SKILL.md", _skill_md().encode()),
        ("assets/logo.png", png_bytes),
    ]
    package = build_package(entries)
    skill = Skill(id=uuid4(), owner_id=None, name="demo", source="upload")
    db = MagicMock()
    added: list[SkillFile] = []
    db.add = MagicMock(side_effect=lambda obj: added.append(obj))

    await store_package_files(skill, package, db)

    logo = next(f for f in added if f.path == "assets/logo.png")
    assert logo.is_binary is True
    assert base64.b64decode(logo.content) == png_bytes
