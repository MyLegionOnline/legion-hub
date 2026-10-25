"""Unit tests for core.skill_materialize (internal design notes Stage 3).

No DB and no workspace volume required — the AsyncSession and workspaced_client
are mocked, per [[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]]
(DB-backed tests don't run natively on this box).
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import skill_materialize as sm
from app.models.agent import Agent
from app.models.skill import Skill, SkillFile

ROOM_ID = uuid.uuid4()


def _agent() -> Agent:
    return Agent(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        token=uuid.uuid4().hex,
        nickname="tester",
        agent_uid="tester",
    )


def _skill(
    name: str = "xlsx",
    version: str = "1.0.0",
    checksum: str = "abc123",
    dependencies: dict | None = None,
) -> Skill:
    return Skill(
        id=uuid.uuid4(),
        owner_id=None,
        name=name,
        version=version,
        checksum=checksum,
        source="store",
        file_count=1,
        is_executable=True,
        dependencies=dependencies,
    )


def _scalars_result(items: list[object]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = items
    return result


def _db(skills: list[Skill], files_by_skill: dict[uuid.UUID, list[SkillFile]]) -> AsyncMock:
    """db.execute is called once for the skill lookup, then once per skill for its files."""
    side_effects = [_scalars_result(skills)]
    for skill in skills:
        side_effects.append(_scalars_result(files_by_skill.get(skill.id, [])))
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=side_effects)
    return db


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("SKILL_MATERIALIZE_ENABLED", "1")
    sm._room_locks.clear()


@pytest.mark.asyncio
async def test_no_assigned_skills_returns_empty():
    agent = _agent()
    db = _db([], {})

    with patch.object(sm, "wsd") as wsd_mock:
        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert result == []
    wsd_mock.read_workspace_file.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_flag_skips_without_touching_db_or_workspace(monkeypatch):
    monkeypatch.setenv("SKILL_MATERIALIZE_ENABLED", "0")
    agent = _agent()
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=AssertionError("must not query when disabled"))

    with patch.object(sm, "wsd") as wsd_mock:
        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert result == []
    wsd_mock.read_workspace_file.assert_not_called()


@pytest.mark.asyncio
async def test_materializes_new_skill_and_writes_marker():
    agent = _agent()
    skill = _skill()
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert len(result) == 1
    materialized = result[0]
    assert materialized.name == "xlsx"
    assert materialized.root_abs == "/workspace/.legion/skills/xlsx"
    assert materialized.files == ["scripts/recalc.py"]

    wsd_mock.import_to_workspace.assert_awaited_once()
    (room_arg, payload), _ = wsd_mock.import_to_workspace.await_args
    assert room_arg == str(ROOM_ID)
    assert payload == [{"path": ".legion/skills/xlsx/scripts/recalc.py", "content": "print(1)"}]

    wsd_mock.write_workspace_file.assert_awaited_once()
    (marker_room, marker_path, marker_json), _ = wsd_mock.write_workspace_file.await_args
    assert marker_path == ".legion/skills/.installed.json"
    marker = json.loads(marker_json)
    assert marker == {
        "xlsx": {"version": "1.0.0", "checksum": "abc123", "files": ["scripts/recalc.py"]}
    }


@pytest.mark.asyncio
async def test_idempotent_when_marker_matches_version_and_checksum():
    agent = _agent()
    skill = _skill()
    db = _db([skill], {})  # would raise if the files query ran

    existing_marker = json.dumps(
        {"xlsx": {"version": "1.0.0", "checksum": "abc123", "files": ["scripts/recalc.py"]}}
    )

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=existing_marker)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert len(result) == 1
    assert result[0].files == ["scripts/recalc.py"]
    wsd_mock.import_to_workspace.assert_not_called()
    wsd_mock.write_workspace_file.assert_not_called()
    # Only the skill-lookup query ran — no per-skill SkillFile query, confirming
    # the up-to-date branch short-circuits before touching storage.
    assert db.execute.await_count == 1


@pytest.mark.asyncio
async def test_rewrites_when_checksum_changes_and_drops_stale_files():
    agent = _agent()
    skill = _skill(checksum="new-checksum")
    sfile = SkillFile(skill_id=skill.id, path="scripts/new.py", content="print(2)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    old_marker = json.dumps(
        {"xlsx": {"version": "1.0.0", "checksum": "old-checksum", "files": ["scripts/old.py"]}}
    )

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=old_marker)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()
        wsd_mock.delete_workspace_file = AsyncMock()

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert result[0].files == ["scripts/new.py"]
    wsd_mock.import_to_workspace.assert_awaited_once()
    wsd_mock.delete_workspace_file.assert_awaited_once_with(
        str(ROOM_ID), ".legion/skills/xlsx/scripts/old.py"
    )


@pytest.mark.asyncio
async def test_binary_content_written_as_data_uri():
    agent = _agent()
    skill = _skill()
    sfile = SkillFile(skill_id=skill.id, path="assets/logo.png", content="aGVsbG8=", is_binary=True)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()

        await sm.materialize_agent_skills(ROOM_ID, agent, db)

    (_, payload), _ = wsd_mock.import_to_workspace.await_args
    assert payload == [
        {
            "path": ".legion/skills/xlsx/assets/logo.png",
            "content": "data:application/octet-stream;base64,aGVsbG8=",
        }
    ]


@pytest.mark.asyncio
async def test_skill_lookup_failure_returns_empty_without_raising():
    agent = _agent()
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))

    with patch.object(sm, "wsd") as wsd_mock:
        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert result == []
    wsd_mock.read_workspace_file.assert_not_called()


@pytest.mark.asyncio
async def test_write_failure_for_one_skill_does_not_abort_others():
    agent = _agent()
    broken = _skill(name="broken", checksum="c1")
    ok = _skill(name="ok", checksum="c2")
    broken_file = SkillFile(skill_id=broken.id, path="scripts/boom.py", content="pass", is_binary=False)
    ok_file = SkillFile(skill_id=ok.id, path="scripts/run.py", content="pass", is_binary=False)
    db = _db([broken, ok], {broken.id: [broken_file], ok.id: [ok_file]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.write_workspace_file = AsyncMock()

        async def _import_side_effect(room_id, payload, *args, **kwargs):
            if any("broken" in item["path"] for item in payload):
                raise RuntimeError("workspaced unreachable")
            return {"imported": len(payload), "errors": []}

        wsd_mock.import_to_workspace = AsyncMock(side_effect=_import_side_effect)

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    names = {m.name for m in result}
    assert names == {"ok"}


# ---------------------------------------------------------------------------
# Stage 7 — path-safety (defense-in-depth) and audit logging
# ---------------------------------------------------------------------------


def test_is_safe_relpath_rejects_traversal_and_absolute():
    assert sm._is_safe_relpath("scripts/recalc.py") is True
    assert sm._is_safe_relpath("../../etc/passwd") is False
    assert sm._is_safe_relpath("/etc/passwd") is False
    assert sm._is_safe_relpath("scripts/../../secret") is False
    assert sm._is_safe_relpath("") is False


@pytest.mark.asyncio
async def test_unsafe_path_is_skipped_but_other_files_still_written():
    agent = _agent()
    skill = _skill()
    good = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    bad = SkillFile(skill_id=skill.id, path="../escape.py", content="pwn", is_binary=False)
    db = _db([skill], {skill.id: [good, bad]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert result[0].files == ["scripts/recalc.py"]
    (_, payload), _ = wsd_mock.import_to_workspace.await_args
    assert payload == [{"path": ".legion/skills/xlsx/scripts/recalc.py", "content": "print(1)"}]


@pytest.mark.asyncio
async def test_successful_materialize_logs_audit_line(caplog):
    agent = _agent()
    skill = _skill()
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()

        with caplog.at_level("INFO", logger="app.core.skill_materialize"):
            await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert any(
        "wrote skill=xlsx" in r.message and "version=1.0.0" in r.message and "bytes=" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Stage 5 §7.2 — best-effort pip auto-install of declared dependencies.pip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pip_deps_installed_and_recorded_in_marker():
    agent = _agent()
    skill = _skill(dependencies={"pip": ["openpyxl", "pandas"]})
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()
        wsd_mock.exec_in_workspace = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})

        await sm.materialize_agent_skills(ROOM_ID, agent, db)

    wsd_mock.exec_in_workspace.assert_awaited_once()
    (room_arg, cmd), kwargs = wsd_mock.exec_in_workspace.await_args
    assert room_arg == str(ROOM_ID)
    assert cmd[:2] == ["pip3", "install"]
    assert "openpyxl" in cmd and "pandas" in cmd

    (_, _, marker_json), _ = wsd_mock.write_workspace_file.await_args
    marker = json.loads(marker_json)
    assert marker["xlsx"]["pip_installed"] is True


@pytest.mark.asyncio
async def test_pip_install_disabled_flag_skips_exec(monkeypatch):
    monkeypatch.setenv("SKILL_AUTO_PIP_INSTALL", "0")
    agent = _agent()
    skill = _skill(dependencies={"pip": ["openpyxl"]})
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()
        wsd_mock.exec_in_workspace = AsyncMock()

        await sm.materialize_agent_skills(ROOM_ID, agent, db)

    wsd_mock.exec_in_workspace.assert_not_called()
    (_, _, marker_json), _ = wsd_mock.write_workspace_file.await_args
    marker = json.loads(marker_json)
    assert "pip_installed" not in marker["xlsx"]


@pytest.mark.asyncio
async def test_pip_install_failure_recorded_without_aborting_materialization():
    agent = _agent()
    skill = _skill(dependencies={"pip": ["openpyxl"]})
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()
        wsd_mock.exec_in_workspace = AsyncMock(
            return_value={"exit_code": 1, "stdout": "", "stderr": "network unreachable"}
        )

        result = await sm.materialize_agent_skills(ROOM_ID, agent, db)

    assert len(result) == 1  # a failed pip install does not drop the skill
    (_, _, marker_json), _ = wsd_mock.write_workspace_file.await_args
    marker = json.loads(marker_json)
    assert marker["xlsx"]["pip_installed"] is False


@pytest.mark.asyncio
async def test_no_pip_deps_marker_has_no_pip_installed_key():
    agent = _agent()
    skill = _skill()  # dependencies=None
    sfile = SkillFile(skill_id=skill.id, path="scripts/recalc.py", content="print(1)", is_binary=False)
    db = _db([skill], {skill.id: [sfile]})

    with patch.object(sm, "wsd") as wsd_mock:
        wsd_mock.read_workspace_file = AsyncMock(return_value=None)
        wsd_mock.import_to_workspace = AsyncMock()
        wsd_mock.write_workspace_file = AsyncMock()
        wsd_mock.exec_in_workspace = AsyncMock()

        await sm.materialize_agent_skills(ROOM_ID, agent, db)

    wsd_mock.exec_in_workspace.assert_not_called()
    (_, _, marker_json), _ = wsd_mock.write_workspace_file.await_args
    marker = json.loads(marker_json)
    assert marker == {
        "xlsx": {"version": "1.0.0", "checksum": "abc123", "files": ["scripts/recalc.py"]}
    }
