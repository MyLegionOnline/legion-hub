"""Materialize bundled skill files onto the workspace volume for a dispatched agent.

internal design notes Stage 3. Files live in
``/workspace/.legion/skills/<skill-name>/`` — outside git/sync/snapshot
(``.legion`` is already excluded everywhere, see plan §1.1) — so a scripted
skill like ``xlsx`` can be run with ``bash`` (``python
.legion/skills/xlsx/scripts/recalc.py``) without polluting the user's repo.

Only skills explicitly assigned to the dispatched agent (via AgentSkill) are
written, and only once per (name, version, checksum) — a marker file
(``.legion/skills/.installed.json``) makes repeat session starts a no-op.
"""

import base64
import json
import logging
import os
from asyncio import Lock
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import workspaced_client as wsd
from app.core.skill_storage import (
    DatabaseSkillBackend,
    get_skill_storage_backend,
    skill_file_key,
)
from app.models.agent import Agent
from app.models.market import AgentSkill, UserInstalledItem
from app.models.skill import Skill, SkillFile

logger = logging.getLogger(__name__)

SKILLS_ROOT = ".legion/skills"
_MARKER_PATH = f"{SKILLS_ROOT}/.installed.json"

# One lock per room: several agents in the same room must not race writing
# the shared marker file (mirrors the bash-tool room lock in core/tools/_bash.py).
_room_locks: dict[str, Lock] = {}


def _room_lock(room_id: str) -> Lock:
    if room_id not in _room_locks:
        _room_locks[room_id] = Lock()
    return _room_locks[room_id]


def materialize_enabled() -> bool:
    return os.getenv("SKILL_MATERIALIZE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def auto_pip_install_enabled() -> bool:
    return os.getenv("SKILL_AUTO_PIP_INSTALL", "true").strip().lower() in {"1", "true", "yes", "on"}


def _pip_dependencies(skill: Skill) -> list[str]:
    deps = skill.dependencies
    raw = deps.get("pip") if isinstance(deps, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if isinstance(p, str) and p.strip()]


async def _install_pip_deps(room_id: str, skill: Skill, packages: list[str]) -> bool:
    """Best-effort ``pip install`` of a skill's declared ``dependencies.pip``.

    Runs once per (skill, version) — the caller records the outcome in the
    marker so a later session doesn't retry it every turn (Stage 5 §7.2).
    Never raises: a missing network or a failing install must not block
    materialization or the agent session.
    """
    try:
        result = await wsd.exec_in_workspace(
            room_id,
            ["pip3", "install", "--user", "--break-system-packages", "--quiet", *packages],
            cwd="/workspace",
            timeout_sec=300,
        )
    except Exception:
        logger.warning(
            "skill_materialize: pip install errored for skill %s (room %s)",
            skill.name, room_id, exc_info=True,
        )
        return False
    if result.get("exit_code") != 0:
        logger.warning(
            "skill_materialize: pip install failed for skill %s (room %s): %s",
            skill.name, room_id, (result.get("stderr") or "")[:500],
        )
        return False
    return True


@dataclass
class MaterializedSkill:
    """What got (or already was) written for one skill — input to Stage 4's prompt manifest."""

    name: str
    version: str
    checksum: Optional[str]
    root_abs: str                      # e.g. "/workspace/.legion/skills/xlsx"
    files: list[str] = field(default_factory=list)   # paths relative to root_abs


async def _agent_skills_with_files(agent: Agent, db: AsyncSession) -> list[Skill]:
    """Skills explicitly assigned to `agent` that have bundled files.

    Same resolution path as core.skills.resolve_agent_skills_text (AgentSkill →
    UserInstalledItem → Skill), restricted to bundled skills.
    """
    result = await db.execute(
        select(Skill)
        .join(UserInstalledItem, UserInstalledItem.skill_id == Skill.id)
        .join(AgentSkill, AgentSkill.user_installed_item_id == UserInstalledItem.id)
        .where(
            AgentSkill.agent_id == agent.id,
            UserInstalledItem.user_id == agent.owner_id,
            UserInstalledItem.type == "skill",
            UserInstalledItem.status == "active",
            Skill.file_count > 0,
            or_(Skill.owner_id.is_(None), Skill.owner_id == agent.owner_id),
        )
        .order_by(AgentSkill.created_at)
    )
    return list(result.scalars().all())


async def _load_marker(room_id: str) -> dict:
    raw = await wsd.read_workspace_file(room_id, _MARKER_PATH)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("skill_materialize: marker for room %s is corrupt — ignoring", room_id)
        return {}
    return data if isinstance(data, dict) else {}


async def _save_marker(room_id: str, marker: dict) -> None:
    await wsd.write_workspace_file(room_id, _MARKER_PATH, json.dumps(marker, sort_keys=True))


async def _fetch_skill_files(skill: Skill, db: AsyncSession) -> list[tuple[str, str, bool]]:
    """Return [(relative_path, content, is_binary)] for every SkillFile of `skill`.

    Reads from whichever backend is configured (db: content column; s3: object
    store) — mirrors core.skill_install.store_package_files's write-side logic.
    """
    result = await db.execute(select(SkillFile).where(SkillFile.skill_id == skill.id))
    rows = list(result.scalars().all())
    if not rows:
        return []

    backend = get_skill_storage_backend()
    use_db = isinstance(backend, DatabaseSkillBackend)

    out: list[tuple[str, str, bool]] = []
    for row in rows:
        if use_db:
            content = row.content
        else:
            raw = await backend.get(skill_file_key(skill.owner_id, skill.id, row.path))
            if raw is None:
                logger.warning(
                    "skill_materialize: missing object for skill %s path %s — skipping",
                    skill.name, row.path,
                )
                continue
            content = base64.b64encode(raw).decode("ascii") if row.is_binary else raw.decode("utf-8")
        if content is None:
            continue
        out.append((row.path, content, row.is_binary))
    return out


def _is_safe_relpath(rel_path: str) -> bool:
    """Defense-in-depth (Stage 7 §9): SkillFile.path already passed
    skill_package._validate_path at upload time (no "..", no absolute, no
    empty segments), so this should never trip. Re-checked here anyway so a
    future bulk-import or migration path can't silently write outside
    ``.legion/skills/<name>/`` even if it skips the parser."""
    if not rel_path or rel_path.startswith("/"):
        return False
    parts = rel_path.replace("\\", "/").split("/")
    return all(part not in ("", ".", "..") for part in parts)


async def _materialize_one(room_id: str, skill: Skill, db: AsyncSession) -> list[str]:
    """Write every SkillFile of `skill` onto the volume. Returns the relative paths written."""
    files = await _fetch_skill_files(skill, db)
    if not files:
        return []

    payload = []
    paths: list[str] = []
    total_bytes = 0
    for rel_path, content, is_binary in files:
        if not _is_safe_relpath(rel_path):
            logger.error(
                "skill_materialize: refusing unsafe path %r for skill %s — skipping file",
                rel_path, skill.name,
            )
            continue
        file_content = f"data:application/octet-stream;base64,{content}" if is_binary else content
        payload.append({"path": f"{SKILLS_ROOT}/{skill.name}/{rel_path}", "content": file_content})
        paths.append(rel_path)
        total_bytes += len(content)

    if not payload:
        return []

    await wsd.import_to_workspace(room_id, payload)
    logger.info(
        "skill_materialize: wrote skill=%s version=%s files=%d bytes=%d room=%s",
        skill.name, skill.version, len(paths), total_bytes, room_id,
    )
    return paths


async def materialize_agent_skills(
    room_id: UUID, agent: Agent, db: AsyncSession
) -> list[MaterializedSkill]:
    """Materialize every bundled skill assigned to `agent` into this room's workspace.

    Idempotent: a skill already on disk at the same (version, checksum) is not
    rewritten. Never raises — a failure here must not block the agent session
    (same contract as core.skill_defaults.reconcile_default_skills).
    """
    if not materialize_enabled():
        return []

    room_key = str(room_id)
    try:
        skills = await _agent_skills_with_files(agent, db)
    except Exception:
        logger.exception("skill_materialize: failed to resolve skills for agent %s", agent.id)
        return []
    if not skills:
        return []

    async with _room_lock(room_key):
        try:
            marker = await _load_marker(room_key)
        except Exception:
            logger.exception("skill_materialize: failed to read marker for room %s", room_key)
            marker = {}

        materialized: list[MaterializedSkill] = []
        marker_changed = False

        for skill in skills:
            root_abs = f"/workspace/{SKILLS_ROOT}/{skill.name}"
            entry = marker.get(skill.name) or {}
            up_to_date = (
                entry.get("version") == skill.version and entry.get("checksum") == skill.checksum
            )
            if up_to_date:
                materialized.append(
                    MaterializedSkill(
                        name=skill.name,
                        version=skill.version,
                        checksum=skill.checksum,
                        root_abs=root_abs,
                        files=list(entry.get("files") or []),
                    )
                )
                continue

            try:
                written = await _materialize_one(room_key, skill, db)
            except Exception:
                logger.exception(
                    "skill_materialize: failed to write files for skill %s (room %s)",
                    skill.name, room_key,
                )
                continue

            # Drop files that belonged to a now-superseded version.
            stale = set(entry.get("files") or []) - set(written)
            for stale_path in stale:
                try:
                    await wsd.delete_workspace_file(
                        room_key, f"{SKILLS_ROOT}/{skill.name}/{stale_path}"
                    )
                except Exception:
                    logger.warning(
                        "skill_materialize: failed to delete stale file %s for skill %s",
                        stale_path, skill.name,
                    )

            marker_entry = {
                "version": skill.version,
                "checksum": skill.checksum,
                "files": written,
            }
            pip_deps = _pip_dependencies(skill)
            if pip_deps and auto_pip_install_enabled():
                marker_entry["pip_installed"] = await _install_pip_deps(room_key, skill, pip_deps)
            marker[skill.name] = marker_entry
            marker_changed = True
            materialized.append(
                MaterializedSkill(
                    name=skill.name,
                    version=skill.version,
                    checksum=skill.checksum,
                    root_abs=root_abs,
                    files=written,
                )
            )

        if marker_changed:
            try:
                await _save_marker(room_key, marker)
            except Exception:
                logger.exception("skill_materialize: failed to persist marker for room %s", room_key)

        return materialized
