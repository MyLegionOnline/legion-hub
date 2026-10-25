"""Resolve an agent's skills into text for system prompt injection.

An agent's skill selection lives in AgentSkill, which links to a Hub-owned
UserInstalledItem.  The former llm_config ``skill_ids``/``skills`` fallback is
gone: skill_legacy_migration moves those fields into AgentSkill rows (and the
free text into the system prompt) at startup.

SKILL.md bodies are always injected. Skills with bundled files (file_count>0,
internal design notes Stage 4) also get a compact manifest of file
paths appended — never file content. The manifest is rendered from SkillFile
rows directly (not from an actual materialize_agent_skills() call), so it
shows up in the prompt even before Stage 3's lazy materialization has run for
this session — the file tree is the same either way.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.skill_materialize import SKILLS_ROOT
from app.models.agent import Agent
from app.models.market import AgentSkill, UserInstalledItem
from app.models.skill import Skill, SkillFile

logger = logging.getLogger(__name__)

# Manifest rendering limits (internal design notes §6.1): keep the
# bundled-files section compact regardless of how large the skill package is.
_MANIFEST_DIR_COLLAPSE_THRESHOLD = 5   # a dir with more files than this collapses to one line
_MANIFEST_MAX_LINES = 40               # hard cap on rendered file lines

# Provenance tag rendered in the manifest (internal design notes
# Stage 7): lets the agent (and, by extension, the user reading its
# reasoning) tell verified Market code from a skill someone uploaded directly.
_SOURCE_LABELS = {
    "store": "verified (installed from Legion Market)",
    "upload": "user-provided (uploaded directly, not Market-reviewed)",
    "builtin": "builtin (bundled with Legion)",
}


async def resolve_agent_skills_text(agent: Agent, db: AsyncSession) -> str:
    """Build the skills section of an agent's system prompt.

    Only AgentSkill rows count.  Each row must point to an active, user-owned
    ``type=skill`` registry item and a skill owned by the agent's owner (or a
    shared one).
    """
    try:
        assigned = await db.execute(
            select(UserInstalledItem)
            .join(AgentSkill, AgentSkill.user_installed_item_id == UserInstalledItem.id)
            .where(
                AgentSkill.agent_id == agent.id,
                UserInstalledItem.user_id == agent.owner_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.status == "active",
            )
            .order_by(AgentSkill.created_at)
        )
        explicit_items = list(assigned.scalars().all())
    except Exception:
        logger.exception("Failed to load skills for agent %s", agent.id)
        return ""

    skill_ids = [
        skill_id
        for item in explicit_items
        if (skill_id := getattr(item, "skill_id", None)) is not None
    ]
    if not skill_ids:
        return ""
    return await _render_skill_ids(skill_ids, agent.owner_id, db)


async def _render_skill_ids(skill_ids: List[UUID], owner_id: UUID, db: AsyncSession) -> str:
    """Load skill bodies in assignment order, enforcing the skill owner scope."""
    try:
        result = await db.execute(
            select(Skill).where(
                Skill.id.in_(skill_ids),
                or_(Skill.owner_id.is_(None), Skill.owner_id == owner_id),
            )
        )
        by_id = {skill.id: skill for skill in result.scalars().all()}
    except Exception:
        logger.exception("Failed to load skill bodies")
        return ""

    # Bundled-files manifest (Stage 4): only queried when at least one resolved
    # skill actually has files, so markdown-only skills cost no extra query.
    bundled_ids = [skill.id for skill in by_id.values() if skill.file_count > 0]
    files_by_skill: dict[UUID, list[SkillFile]] = {}
    if bundled_ids:
        try:
            file_result = await db.execute(
                select(SkillFile)
                .where(SkillFile.skill_id.in_(bundled_ids))
                .order_by(SkillFile.path)
            )
            for row in file_result.scalars().all():
                files_by_skill.setdefault(row.skill_id, []).append(row)
        except Exception:
            logger.exception("Failed to load bundled file manifest")

    sections: List[str] = []
    for skill_id in skill_ids:
        skill = by_id.get(skill_id)
        if not skill:
            continue
        body = (skill.skill_md or "").strip()
        section = f"## Skill: {skill.display_name or skill.name}"
        if skill.description.strip():
            section += f"\n{skill.description.strip()}"
        if body:
            section += f"\n\n{body}"
        manifest = _render_bundled_files_manifest(skill, files_by_skill.get(skill.id) or [])
        if manifest:
            section += f"\n\n{manifest}"
        sections.append(section)
    return "\n\n".join(sections)


def _render_bundled_files_manifest(skill: Skill, files: List[SkillFile]) -> str:
    """L3 progressive-disclosure manifest: file paths only, never content.

    Returns "" for markdown-only skills (file_count==0) so old skills render
    exactly as before Stage 4.
    """
    if skill.file_count <= 0 or not files:
        return ""

    root_abs = f"/workspace/{SKILLS_ROOT}/{skill.name}"
    lines = [
        "### Bundled files (available on disk — DO NOT read unless needed)",
        f"Source: {_SOURCE_LABELS.get(skill.source, skill.source)}",
        f"Skill root: {root_abs}/  (paths in SKILL.md above are relative to this root)",
        f"Run scripts via bash, e.g.  python {root_abs}/<script path> [args]",
    ]
    dep_line = _render_dependencies_line(skill.dependencies)
    if dep_line:
        lines.append(dep_line)
    lines.append(f"Files ({len(files)}):")
    for entry in _collapse_file_tree([f.path for f in files]):
        lines.append(f"  {entry}")
    return "\n".join(lines)


def _render_dependencies_line(dependencies: Optional[dict]) -> str:
    """One-line note about declared runtime deps (Stage 5 §7.2) — informational
    only; the actual install/office-image decisions happen elsewhere
    (core/skill_materialize.py, core/workspace_image.py)."""
    if not isinstance(dependencies, dict):
        return ""
    pip_deps = dependencies.get("pip") or []
    apt_deps = dependencies.get("apt") or []
    parts = []
    if pip_deps:
        parts.append(f"pip: {', '.join(str(p) for p in pip_deps)} (auto-installed if network is enabled)")
    if apt_deps:
        parts.append(f"apt: {', '.join(str(p) for p in apt_deps)} (needs the office workspace image)")
    return "Dependencies — " + "; ".join(parts) if parts else ""


def _collapse_file_tree(paths: List[str]) -> List[str]:
    """Group file paths by their immediate parent directory.

    Directories with more than `_MANIFEST_DIR_COLLAPSE_THRESHOLD` files collapse
    to a single summary line so a package like xlsx's 49-file schemas/ dir does
    not blow up the prompt. `paths` must already be sorted (the SkillFile query
    orders by path), so files under the same directory are contiguous.
    """
    groups: List[tuple[str, List[str]]] = []
    for path in paths:
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if groups and groups[-1][0] == parent:
            groups[-1][1].append(path)
        else:
            groups.append((parent, [path]))

    lines: List[str] = []
    for parent, group in groups:
        if len(group) > _MANIFEST_DIR_COLLAPSE_THRESHOLD:
            label = f"{parent}/" if parent else "./"
            lines.append(f"{label}…  ({len(group)} more under {label})")
        else:
            lines.extend(group)

    if len(lines) > _MANIFEST_MAX_LINES:
        lines = lines[: _MANIFEST_MAX_LINES - 1]
        lines.append("…  (manifest truncated — see the Files count above for the total)")
    return lines
