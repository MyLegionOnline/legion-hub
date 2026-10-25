"""Unit tests for core.skills.resolve_agent_skills_text (Skill Storage, stage 4).

No DB required — the AsyncSession is mocked; the query result is controlled
per-test to simulate builtin/own/foreign/deleted skills.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.skill_materialize import SKILLS_ROOT
from app.core.skills import _collapse_file_tree, resolve_agent_skills_text
from app.models.agent import Agent
from app.models.market import UserInstalledItem
from app.models.skill import Skill, SkillFile


def _agent(llm_config: dict) -> Agent:
    return Agent(
        owner_id=uuid.uuid4(),
        token=uuid.uuid4().hex,
        nickname="tester",
        agent_uid="tester",
        llm_config=llm_config,
    )


def _skill(
    name: str,
    body: str,
    description: str = "",
    display_name: str = "",
    file_count: int = 0,
    dependencies: dict | None = None,
    source: str = "upload",
) -> Skill:
    return Skill(
        name=name,
        display_name=display_name,
        description=description,
        skill_md=body,
        file_count=file_count,
        dependencies=dependencies,
        source=source,
    )


def _file(skill: Skill, path: str) -> SkillFile:
    return SkillFile(skill_id=skill.id, path=path)


def _result(items: list[object]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = items
    return result


def _item(skill: Skill) -> UserInstalledItem:
    return UserInstalledItem(user_id=uuid.uuid4(), type="skill", name=skill.name, skill_id=skill.id)


def _db_returning(
    skills: list[Skill],
    explicit_items: list[UserInstalledItem] | None = None,
    files: list[SkillFile] | None = None,
) -> AsyncMock:
    """Mock db.execute for the (AgentSkill, Skill[, SkillFile]) query sequence.

    `explicit_items` defaults to one assignment per skill, in list order.  The
    SkillFile query only fires when at least one resolved skill has
    file_count>0 — pass `files` in that case, omit it otherwise.
    """
    if explicit_items is None:
        explicit_items = [_item(skill) for skill in skills]
    side_effect = [_result(explicit_items), _result(skills)]
    if files is not None:
        side_effect.append(_result(files))
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=side_effect)
    return db


@pytest.mark.asyncio
async def test_no_assignment_returns_empty_string():
    agent = _agent({})
    db = _db_returning([])

    text = await resolve_agent_skills_text(agent, db)

    assert text == ""
    assert db.execute.await_count == 1  # only the AgentSkill lookup runs


@pytest.mark.asyncio
async def test_retired_legacy_fields_are_ignored():
    agent = _agent({"skill_ids": [str(uuid.uuid4())], "skills": "legacy text"})
    db = _db_returning([])

    assert await resolve_agent_skills_text(agent, db) == ""


@pytest.mark.asyncio
async def test_skill_sections_follow_assignment_order():
    s1 = _skill("skill-coder", "Write good code.", description="Coding rules")
    s2 = _skill("skill-writer", "Write well.", display_name="Writer")
    agent = _agent({})
    db = _db_returning([s2, s1], [_item(s1), _item(s2)])  # DB order differs from assignment order

    text = await resolve_agent_skills_text(agent, db)

    assert text == (
        "## Skill: skill-coder\nCoding rules\n\nWrite good code."
        "\n\n## Skill: Writer\n\nWrite well."
    )


@pytest.mark.asyncio
async def test_deleted_skills_are_skipped():
    s1 = _skill("alive", "Body")
    gone = _skill("gone", "Body")
    agent = _agent({})
    db = _db_returning([s1], [_item(gone), _item(s1)])  # deleted/foreign skills are absent

    text = await resolve_agent_skills_text(agent, db)

    assert text == "## Skill: alive\n\nBody"


@pytest.mark.asyncio
async def test_query_failure_returns_empty_string():
    agent = _agent({})
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))

    assert await resolve_agent_skills_text(agent, db) == ""


@pytest.mark.asyncio
async def test_ownership_filter_in_query():
    """The query must restrict to builtin (owner IS NULL) or the agent owner's skills."""
    s1 = _skill("mine", "Body")
    agent = _agent({})
    db = _db_returning([s1])

    await resolve_agent_skills_text(agent, db)

    query_sql = str(db.execute.await_args_list[1].args[0])
    assert "owner_id IS NULL" in query_sql
    assert "owner_id =" in query_sql


# ---------------------------------------------------------------------------
# Bundled-files manifest (internal design notes Stage 4, §6.1/6.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_only_skill_has_no_manifest_section():
    """file_count==0 must render exactly as before Stage 4 — no extra query, no section."""
    s = _skill("plain", "Just markdown.")
    agent = _agent({})
    db = _db_returning([s])  # no `files` kwarg — third query must not happen

    text = await resolve_agent_skills_text(agent, db)

    assert text == "## Skill: plain\n\nJust markdown."
    assert db.execute.await_count == 2


@pytest.mark.asyncio
async def test_bundled_skill_gets_file_manifest_with_root_and_run_hint():
    s = _skill("xlsx", "Use the scripts.", file_count=2)
    files = [_file(s, "scripts/recalc.py"), _file(s, "README.md")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert db.execute.await_count == 3  # AgentSkill + Skill + SkillFile
    root = f"/workspace/{SKILLS_ROOT}/xlsx"
    assert f"Skill root: {root}/" in text
    assert f"python {root}/<script path>" in text
    assert "Files (2):" in text
    assert "  README.md" in text
    assert "  scripts/recalc.py" in text
    # No file content, only paths.
    assert "recalc.py" in text and "def " not in text


@pytest.mark.asyncio
async def test_bundled_manifest_collapses_large_directories():
    s = _skill("xlsx", "Body", file_count=8)
    files = [_file(s, f"scripts/schemas/f{i}.xsd") for i in range(8)]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert "scripts/schemas/…  (8 more under scripts/schemas/)" in text
    for i in range(8):
        assert f"f{i}.xsd" not in text


def test_collapse_file_tree_lists_small_dirs_individually():
    paths = ["scripts/recalc.py", "scripts/office/unpack.py", "README.md"]
    lines = _collapse_file_tree(paths)
    assert lines == paths


def test_collapse_file_tree_caps_total_lines():
    # 41 distinct single-file top-level dirs -> 41 lines, over the 40-line cap.
    paths = [f"dir{i}/file.py" for i in range(41)]
    lines = _collapse_file_tree(paths)
    assert len(lines) == 40
    assert lines[-1].startswith("…")


@pytest.mark.asyncio
async def test_manifest_includes_declared_dependencies_line():
    """Stage 5 §7.2: declared pip/apt deps get one informational line."""
    s = _skill(
        "xlsx", "Body", file_count=1, dependencies={"pip": ["openpyxl"], "apt": ["libreoffice-calc"]}
    )
    files = [_file(s, "scripts/recalc.py")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert "Dependencies — pip: openpyxl" in text
    assert "apt: libreoffice-calc" in text


@pytest.mark.asyncio
async def test_manifest_omits_dependencies_line_when_none_declared():
    s = _skill("xlsx", "Body", file_count=1)
    files = [_file(s, "scripts/recalc.py")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert "Dependencies —" not in text


# ---------------------------------------------------------------------------
# Provenance tag in the manifest (internal design notes Stage 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_tags_market_skill_as_verified():
    s = _skill("xlsx", "Body", file_count=1, source="store")
    files = [_file(s, "scripts/recalc.py")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert "Source: verified (installed from Legion Market)" in text


@pytest.mark.asyncio
async def test_manifest_tags_uploaded_skill_as_user_provided():
    s = _skill("custom", "Body", file_count=1, source="upload")
    files = [_file(s, "scripts/run.py")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    text = await resolve_agent_skills_text(agent, db)

    assert "Source: user-provided (uploaded directly, not Market-reviewed)" in text


@pytest.mark.asyncio
async def test_manifest_present_for_direct_chat_kinds_too():
    """Stage 4 §6.0 wired 'skills' into user_agent_p1/p2 — the manifest renderer
    (§6.1) is shared code, so it must show up there without extra plumbing."""
    from app.core.memory_layer import build_system_prompt

    s = _skill("xlsx", "Body", file_count=1)
    files = [_file(s, "scripts/recalc.py")]
    agent = _agent({})
    db = _db_returning([s], files=files)

    skills_text = await resolve_agent_skills_text(agent, db)
    prompt = build_system_prompt({"skills": skills_text}, "user_agent_p2")

    assert "### Bundled files" in prompt
