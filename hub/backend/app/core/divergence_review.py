"""Divergence review.

internal design notes. When ``merge_task_branch`` cannot refresh a task
branch against its moved base (overlapping edits — a real 3-way conflict), the
task is parked with ``divergence_state="detected"``. This module powers the
button that gets the room out of that dead end:

* ``build_divergence_context`` — pull a read-only report from ``workspaced``
  (merge-base, per-side commits, conflict preview via ``git merge-tree``) and
  render a compact digest for the model.
* ``run_divergence_review`` — one forced JSON turn on the ChatMaster's model
  (not a session): it explains the divergence and spells out the consequence of
  each of the three resolutions. Emits ``DIVERGENCE_REVIEW`` and moves the task
  to ``divergence_state="reviewed"``.
* ``resolve_divergence`` — execute the user's choice:
  ``take_branch`` (fold the branch in with ``-X theirs``, base loses the
  overlap), ``rework`` (spin a fresh task for the responsible agent, branch
  kept), ``delete_branch`` (tag-backup then delete, task cancelled).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import workspaced_client as wsd
from app.db.release import release_db
from app.core.events import emit_event
from app.models.task import DispatchStatus, Task, TaskState

logger = logging.getLogger(__name__)

DECISIONS = ("take_branch", "rework", "delete_branch")

_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "1-2 sentences: the essence of the divergence between the task branch and dev.",
        },
        "branch_side": {
            "type": "string",
            "description": "What the task branch did after the fork (by commits/diff).",
        },
        "base_side": {
            "type": "string",
            "description": "What arrived in the base (dev/goal) after the fork and why it overlaps.",
        },
        "conflict_explanation": {
            "type": "string",
            "description": "Where exactly the conflict is — files, regions — and its substantive cause.",
        },
        "timing": {
            "type": "string",
            "description": "Fork earlier/later; which base commits arrived after the branch was cut.",
        },
        "recommendation": {"type": "string", "enum": list(DECISIONS)},
        "recommendation_reason": {"type": "string"},
        "take_branch_consequence": {
            "type": "string",
            "description": "What of the base's work would be overwritten if the branch version is accepted in full.",
        },
        "rework_consequence": {
            "type": "string",
            "description": "What the new task must keep from the branch without breaking the base.",
        },
        "delete_branch_consequence": {
            "type": "string",
            "description": "What would be lost if the branch is deleted (the backup tag stays anyway).",
        },
        "rework_brief": {
            "type": "string",
            "description": (
                "A self-contained brief for the responsible agent if rework is chosen: "
                "what to carry over from the branch, a mandatory requirement to rebase onto dev, "
                "which files not to touch. Be sure to name the SPECIFIC input files "
                "the agent must open (mockup/reference, design spec, "
                "affected code — with exact paths from the workspace root), and which "
                "skill process to apply."
            ),
        },
    },
    "required": [
        "summary",
        "branch_side",
        "base_side",
        "conflict_explanation",
        "recommendation",
        "recommendation_reason",
        "take_branch_consequence",
        "rework_consequence",
        "delete_branch_consequence",
        "rework_brief",
    ],
}

_SYSTEM_PROMPT = (
    "You are ChatMaster. A task branch has diverged from its base (dev or a goal branch): "
    "a flat 3-way merge failed because of overlapping edits. The user cannot "
    "sort it out alone. Analyze the divergence data package and explain in plain "
    "language: what the branch did, what arrived in the base, where and why the conflict is. Then for "
    "each of the three resolution options (take_branch — accept the branch version in full; "
    "rework — rework as a separate task for the responsible agent; delete_branch — delete "
    "the branch) state the concrete consequence. Give a recommendation. If you recommend rework, "
    "list the concrete source files (mockup, reference, spec, code) in rework_brief "
    "that the executor must open, and state that the process of its "
    "skill must be followed. Write in the user's language, to the point, no filler. Return strictly the structured_output structure."
)

# Path-like tokens in a task's title/description — mockups, design specs and the
# code the task touches. The rework brief loses these when it is a paraphrase,
# so the fresh task carries them through verbatim (see ``_source_context_block``).
_ARTIFACT_EXT = (
    r"png|jpe?g|webp|gif|svg|pdf|json|md|html?|css|scss|sass|less|tsx?|jsx?|vue|py|ya?ml|toml"
)
_ARTIFACT_PATH_RE = re.compile(
    rf"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:{_ARTIFACT_EXT}))(?![\w/])",
    re.IGNORECASE,
)


def _clip(text: Any, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _mentioned_paths(*texts: Any, limit: int = 20) -> list[str]:
    seen: dict[str, None] = {}
    for text in texts:
        if not isinstance(text, str):
            continue
        for match in _ARTIFACT_PATH_RE.finditer(text):
            seen.setdefault(match.group(1), None)
            if len(seen) >= limit:
                return list(seen)
    return list(seen)


async def _assigned_skill_names(agent_id: UUID | None, db: AsyncSession) -> list[str]:
    """Display names of the skills explicitly assigned to the rework's executor."""
    if agent_id is None:
        return []
    from app.models.market import AgentSkill, UserInstalledItem
    from app.models.skill import Skill

    try:
        rows = (await db.execute(
            select(Skill.display_name, Skill.name)
            .join(UserInstalledItem, UserInstalledItem.skill_id == Skill.id)
            .join(AgentSkill, AgentSkill.user_installed_item_id == UserInstalledItem.id)
            .where(
                AgentSkill.agent_id == agent_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.status == "active",
            )
            .order_by(AgentSkill.created_at)
        )).all()
    except Exception:  # noqa: BLE001 — context is a bonus, never a hard failure
        logger.warning("divergence_review: could not resolve agent skills", exc_info=True)
        return []
    return [chosen for display, name in rows if (chosen := (display or name))]


async def _source_context_block(task: Task, db: AsyncSession) -> str:
    """A deterministic block appended to the rework task's description.

    The CM-authored ``rework_brief`` is a paraphrase and routinely drops the
    concrete pointers a "build from the mockup" executor needs — the reference
    image path, the design spec, the skill to follow. This carries them
    through verbatim so the new task does not start blind (the Frontend developer
    incident: a rework agent that never found the mockup and improvised).
    """
    original = _clip(task.description, 4000)
    paths = _mentioned_paths(task.title or "", task.description or "")
    skills = await _assigned_skill_names(task.assigned_agent_id, db)

    parts = [
        "\n\n## Context of the original task",
        "This is rework after a branch divergence. Below is the full original brief, "
        "so that nothing is lost in the transfer.",
    ]
    if original:
        parts.append(f"\n### Original description\n{original}")
    if paths:
        parts.append(
            "\n### Files mentioned in the task (mockups, references, specs, code)\n"
            "Open those of them that are needed for the work; the path is from the workspace root:\n"
            + "\n".join(f"- {p}" for p in paths)
        )
    if skills:
        parts.append(
            "\n### Executor skills\n"
            f"You have these skills connected: {', '.join(skills)}. Follow their process "
            "(the Skills section of the system prompt) — do not improvise around it."
        )
    return "\n".join(parts)


class DivergenceReviewError(RuntimeError):
    """Something went wrong building or running the review."""


# ── ChatMaster model resolution ─────────────────────────────────────────────

async def _resolve_cm_llm(room, db: AsyncSession):
    """Return ``(agent, llm_config, secret_value)`` for the room's ChatMaster.

    Raises ``DivergenceReviewError`` when the room has no usable coordinator —
    the caller falls back to a deterministic (no-LLM) analysis.
    """
    from app.core.room_coordinator import get_cm_agent_id
    from app.core.secrets.runtime import resolve_agent_secret
    from app.models.agent import Agent

    cm_id = get_cm_agent_id(room)
    if cm_id is None:
        raise DivergenceReviewError("room has no ChatMaster configured")

    agent = (await db.execute(select(Agent).where(Agent.id == cm_id))).scalars().first()
    if agent is None:
        raise DivergenceReviewError("ChatMaster agent not found")

    config = dict(agent.llm_config or {})
    try:
        secret_value = await resolve_agent_secret(
            db=db,
            owner_id=agent.owner_id,
            agent_id=agent.id,
            config=config,
            request_id=agent.id,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to no-key path
        logger.warning("divergence_review: secret resolve failed: %s", exc)
        secret_value = None
    return agent, config, secret_value


# ── Context assembly ────────────────────────────────────────────────────────

def _fmt_commits(commits: list[dict], limit: int = 20) -> str:
    if not commits:
        return "  (none)"
    rows = [
        f"  {c.get('short_sha', '')}  {c.get('timestamp', '')}  {c.get('message', '')}"
        for c in commits[:limit]
    ]
    if len(commits) > limit:
        rows.append(f"  … {len(commits) - limit} more")
    return "\n".join(rows)


def _build_digest(task: Task, report: dict) -> str:
    mb = report.get("merge_base") or {}
    bh = report.get("base_head") or {}
    nh = report.get("branch_head") or {}
    lines: list[str] = [f"Task: {task.title}"]
    if (task.description or "").strip():
        lines.append(f"Brief: {_clip(task.description, 2000)}")
    lines += [
        f"Task branch: {report.get('branch')}",
        f"Base: {report.get('base')}",
        "",
        f"Fork point (merge-base): {mb.get('sha', '')[:8]}  {mb.get('timestamp', '')}  {mb.get('subject', '')}",
        f"Base HEAD:   {bh.get('sha', '')[:8]}  {bh.get('timestamp', '')}  {bh.get('subject', '')}",
        f"Branch HEAD: {nh.get('sha', '')[:8]}  {nh.get('timestamp', '')}  {nh.get('subject', '')}",
        f"The branch is ahead of the base by {report.get('ahead', 0)} commit(s), behind by {report.get('behind', 0)}.",
        "",
        "Commits ONLY in the task branch (after the fork):",
        _fmt_commits(report.get("branch_commits") or []),
        "",
        "Commits that arrived in the BASE after the fork:",
        _fmt_commits(report.get("base_commits") or []),
        "",
        f"Files changed by the branch: {', '.join(report.get('branch_files') or []) or '(none)'}",
        f"Files changed by the base:   {', '.join(report.get('base_files') or []) or '(none)'}",
        f"Overlapping files:           {', '.join(report.get('overlap_files') or []) or '(none)'}",
        f"Real merge conflict: {'yes' if report.get('conflict') else 'no (overlap without a textual conflict)'}",
        f"Conflicting files: {', '.join(report.get('conflict_files') or []) or '(none)'}",
    ]
    for d in report.get("conflict_details") or []:
        lines += [
            "",
            f"── {d.get('path')} — BASE edit (since the fork):",
            (d.get("base_diff") or "(none)").strip()[:6000],
            f"── {d.get('path')} — BRANCH edit (since the fork):",
            (d.get("branch_diff") or "(none)").strip()[:6000],
        ]
    return "\n".join(lines)


def _redundant_analysis(base: str | None, branch: str | None) -> dict:
    """The branch no longer carries anything the base lacks (a rework already
    landed its work). Nothing to salvage — recommend deleting it."""
    return {
        "summary": (
            f"The changes of branch {branch or '?'} are already fully present in {base or 'the base'} — "
            "most likely the rework was already merged by a separate task. The branch can be safely deleted."
        ),
        "branch_side": "The branch adds nothing on top of the base.",
        "base_side": f"{base or 'The base'} already contains all edits of this branch.",
        "conflict_explanation": "There is essentially no conflict — the branch content is a subset of the base.",
        "timing": "",
        "recommendation": "delete_branch",
        "recommendation_reason": "The branch's work is already in the base, there is no reason to keep it.",
        "take_branch_consequence": "Nothing will change — the branch version is already in the base.",
        "rework_consequence": "Will create an empty task — the rework is already done.",
        "delete_branch_consequence": "The branch goes to a backup tag; nothing of value is lost.",
        "rework_brief": "Not required — the changes are already in the base.",
        "_redundant": True,
    }


async def build_divergence_context(room_id: "UUID | str", task: Task, db: AsyncSession) -> dict:
    """Fetch the workspaced divergence report and render the model digest."""
    from app.core.room_git_flow import _task_base_branch, task_branch_name

    branch = task.branch_name or task_branch_name(task.id)
    base = await _task_base_branch(room_id, task, db)
    report = await wsd.git_divergence_workspace(str(room_id), branch, base)
    if not report.get("ok"):
        raise DivergenceReviewError(report.get("error") or "divergence report failed")
    return {"branch": branch, "base": base, "report": report, "digest": _build_digest(task, report)}


# ── Deterministic fallback (no LLM) ─────────────────────────────────────────

def _fallback_analysis(report: dict) -> dict:
    overlap = report.get("overlap_files") or report.get("conflict_files") or []
    return {
        "summary": (
            f"The task branch and {report.get('base')} edit the same files "
            f"({', '.join(overlap) or 'see below'}) and do not merge automatically."
        ),
        "branch_side": f"{len(report.get('branch_commits') or [])} commit(s) in the branch after the fork.",
        "base_side": f"{len(report.get('base_commits') or [])} commit(s) arrived in the base after the fork.",
        "conflict_explanation": (
            report.get("merge_messages")
            or f"Overlapping files: {', '.join(overlap) or 'none'}."
        ),
        "timing": (
            f"Fork: {(report.get('merge_base') or {}).get('timestamp', '?')}. "
            f"Base HEAD: {(report.get('base_head') or {}).get('timestamp', '?')}."
        ),
        "recommendation": "rework",
        "recommendation_reason": (
            "Automatic analysis is unavailable — it is safer to hand the rework to a separate "
            "task than to overwrite the base."
        ),
        "take_branch_consequence": (
            f"The base's changes in files {', '.join(overlap) or '(overlap)'} will be replaced by the branch version."
        ),
        "rework_consequence": "A new task will carry the needed parts of the branch over the current dev.",
        "delete_branch_consequence": (
            f"The branch's work ({len(report.get('branch_commits') or [])} commits) will go to a backup tag and out of active development."
        ),
        "rework_brief": (
            f"Carry the edits of branch {report.get('branch')} over to the current {report.get('base')}: "
            f"{', '.join(report.get('branch_files') or []) or 'see the branch diff'}. "
            f"You must rebase onto {report.get('base')} before working. "
            f"Do not roll back the base's changes in files: {', '.join(overlap) or '(overlap)'}."
        ),
        "_fallback": True,
    }


async def _analyse(room, task: Task, ctx: dict, db: AsyncSession) -> dict:
    # No divergence left to reason about — a rework already folded the work in.
    if (ctx.get("report") or {}).get("redundant"):
        return _redundant_analysis(ctx.get("base"), ctx.get("branch"))
    try:
        agent, config, secret_value = await _resolve_cm_llm(room, db)
    except DivergenceReviewError as exc:
        logger.info("divergence_review: no LLM (%s) — deterministic analysis", exc)
        return _fallback_analysis(ctx["report"])

    from app.core.chat_master import get_llm_json_completion

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": ctx["digest"]},
    ]
    try:
        await release_db(db, reason="divergence_review/analyse")
        result = await get_llm_json_completion(
            config, messages, _ANALYSIS_SCHEMA, secret_value=secret_value
        )
        data = json.loads(result.text)
        if data.get("recommendation") not in DECISIONS:
            data["recommendation"] = "rework"
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("divergence_review: LLM analysis failed: %s", exc)
        return _fallback_analysis(ctx["report"])
    finally:
        secret_value = None


# ── Public: run the review ─────────────────────────────────────────────────

def _report_summary(report: dict) -> dict:
    """The slice of the raw report the card needs (kept small for the event row)."""
    return {
        "branch": report.get("branch"),
        "base": report.get("base"),
        "ahead": report.get("ahead"),
        "behind": report.get("behind"),
        "merge_base": report.get("merge_base"),
        "base_head": report.get("base_head"),
        "branch_head": report.get("branch_head"),
        "branch_commits": (report.get("branch_commits") or [])[:20],
        "base_commits": (report.get("base_commits") or [])[:20],
        "overlap_files": report.get("overlap_files"),
        "conflict": report.get("conflict"),
        "conflict_files": report.get("conflict_files"),
        "redundant": report.get("redundant"),
    }


def _degraded_analysis(reason: str, branch: str | None, base: str | None) -> dict:
    """Analysis card shown when the git report or the LLM turn could not run —
    the user still gets the explanation and the three actions."""
    return {
        "summary": (
            "Could not assemble a detailed divergence analysis "
            f"({reason}). Branch {branch or '?'} still diverges from {base or 'the base'}: "
            "a plain merge will not go through. Choose an action below."
        ),
        "branch_side": "Details unavailable — open the Workspace → Diff tab for the task branch.",
        "base_side": "Details unavailable.",
        "conflict_explanation": reason,
        "timing": "",
        "recommendation": "rework",
        "recommendation_reason": "Without a detailed analysis it is safer to hand the rework to a separate task.",
        "take_branch_consequence": "The base's changes in overlapping files will be replaced by the branch version — check the Diff before choosing.",
        "rework_consequence": "A new task will carry the needed parts of the branch over the current dev.",
        "delete_branch_consequence": "The branch's work will go to a backup tag and out of active development.",
        "rework_brief": (
            f"Carry the edits of branch {branch} over to the current {base or 'dev'}, "
            "without rolling back the base's changes. You must rebase before working."
        ),
        "_fallback": True,
        "_degraded": True,
    }


async def run_divergence_review(room_id: "UUID | str", task_id: "UUID | str", db: AsyncSession) -> Task:
    """Build context, analyse, emit ``DIVERGENCE_REVIEW``, mark the task reviewed.

    Never leaves the task stuck: on *any* failure it still emits a (degraded)
    ``DIVERGENCE_REVIEW`` card so the user is not left staring at a silent UI,
    and ``divergence_state`` always lands on ``reviewed``.
    """
    from app.core.room_git_flow import _load_room

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    task = (await db.execute(select(Task).where(Task.id == task_id))).scalars().first()
    if task is None:
        raise DivergenceReviewError("task not found")

    task.divergence_state = "reviewing"
    db.add(task)
    await db.commit()

    branch = task.branch_name
    base = None
    analysis: dict
    report_summary: dict = {}

    async def _do_analysis() -> tuple[dict, dict, str | None, str | None]:
        room = await _load_room(rid, db)
        ctx = await build_divergence_context(rid, task, db)
        return (
            await _analyse(room, task, ctx, db),
            _report_summary(ctx["report"]),
            ctx["branch"], ctx["base"],
        )

    try:
        # Hard ceiling — a hung workspaced call or LLM turn must not leave the
        # task stuck on "reviewing" forever (the button would stay disabled).
        analysis, report_summary, branch, base = await asyncio.wait_for(
            _do_analysis(), timeout=180.0
        )
    except Exception as exc:  # noqa: BLE001 — a card must always follow the request
        logger.exception("divergence_review: analysis failed for task %s", task_id)
        reason = "the analysis timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)[:200]
        analysis = _degraded_analysis(reason, branch, base)

    task.divergence_state = "reviewed"
    db.add(task)
    await db.flush()

    # Belt and braces: the metadata must be plain JSON or ``send_json`` throws
    # server-side and the offending websocket is dropped (no card, and the
    # client goes deaf until it reloads). Round-trip through json to guarantee it.
    safe_meta = json.loads(json.dumps({
        "task_id": str(task.id),
        "task_title": task.title,
        "branch": branch,
        "base": base,
        "analysis": analysis,
        "report": report_summary,
    }, default=str))

    await emit_event(
        room_id=rid,
        event_type="DIVERGENCE_REVIEW",
        content=analysis.get("summary") or "The branch divergence analysis is ready",
        db=db,
        task_id=task.id,
        metadata=safe_meta,
    )
    await db.commit()
    await db.refresh(task)
    logger.info(
        "divergence_review: emitted DIVERGENCE_REVIEW for task %s (degraded=%s, fallback=%s)",
        task_id, analysis.get("_degraded", False), analysis.get("_fallback", False),
    )
    return task


# ── Public: resolve the review ────────────────────────────────────────────

async def _emit_resolved(rid: UUID, task: Task, decision: str, detail: str, db: AsyncSession, **meta) -> None:
    await emit_event(
        room_id=rid,
        event_type="DIVERGENCE_RESOLVED",
        content=detail,
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "decision": decision, **meta},
    )


async def resolve_divergence(
    room_id: "UUID | str",
    task: Task,
    db: AsyncSession,
    decision: str,
    note: str | None = None,
) -> dict:
    """Execute the user's choice. Returns ``{ok, decision, detail, ...}``.

    Re-checks the divergence right before acting: a live session may have
    committed more onto the branch since the review, and a stale decision must
    not run against a moved target.
    """
    from app.core.room_git_flow import _task_base_branch, task_branch_name

    if decision not in DECISIONS:
        return {"ok": False, "detail": f"unknown decision {decision!r}"}

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    branch = task.branch_name or task_branch_name(task.id)
    base = await _task_base_branch(rid, task, db)

    if decision in ("take_branch", "rework"):
        # Re-check: still diverged the way the review described?
        try:
            fresh = await wsd.git_divergence_workspace(str(rid), branch, base)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": f"could not re-check the divergence: {exc}"}
        if not fresh.get("ok"):
            return {"ok": False, "detail": fresh.get("error") or "the divergence cannot be read"}

    if decision == "take_branch":
        merge = await wsd.merge_workspace_branch(
            str(rid), branch, target=base, theirs=True,
            message=f"divergence: take {branch} over {base} — {task.title}",
        )
        if merge.get("busy"):
            return {"ok": False, "detail": "the working tree is busy, try again"}
        if merge.get("conflict") or not merge.get("ok"):
            return {
                "ok": False,
                "detail": merge.get("stderr") or "the merge failed (modify/delete?)",
            }
        task.merge_commit = merge.get("sha")
        task.divergence_state = "resolved"
        if task.state in (TaskState.WAITING_REVIEW, TaskState.BLOCKED):
            task.state = TaskState.COMPLETED
        db.add(task)
        await db.flush()
        await _emit_resolved(
            rid, task, decision,
            f"The version of branch {branch} was accepted into {base} in full.",
            db, branch=branch, target=base, merge_commit=task.merge_commit,
        )
        await emit_event(
            room_id=rid, event_type="TASK_MERGED",
            content=f"Task '{task.title}' merged into {base} (divergence resolution).",
            db=db, task_id=task.id,
            metadata={"task_id": str(task.id), "branch": branch, "target": base,
                      "merge_commit": task.merge_commit, "via": "divergence_take_branch"},
        )
        await db.commit()
        return {"ok": True, "decision": decision, "detail": "merged", "merge_commit": task.merge_commit}

    if decision == "delete_branch":
        try:
            res = await wsd.git_discard_branch_workspace(str(rid), branch)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": f"could not delete the branch: {exc}"}
        if res.get("busy"):
            return {"ok": False, "detail": "the working tree is busy, try again"}
        if not res.get("ok"):
            return {"ok": False, "detail": res.get("stderr") or "the deletion failed"}
        task.divergence_state = "resolved"
        if task.state not in (TaskState.COMPLETED, TaskState.CANCELLED, TaskState.CONTINUED):
            task.state = TaskState.CANCELLED
        db.add(task)
        await db.flush()
        await _emit_resolved(
            rid, task, decision,
            f"Branch {branch} deleted (backup: {res.get('backup_ref')}). The task was cancelled.",
            db, branch=branch, backup_ref=res.get("backup_ref"),
        )
        await db.commit()
        return {"ok": True, "decision": decision, "detail": "deleted",
                "backup_ref": res.get("backup_ref")}

    # decision == "rework"
    return await _spawn_rework_task(rid, task, base, note, db)


async def _spawn_rework_task(rid: UUID, task: Task, base: str, note: str | None, db: AsyncSession) -> dict:
    """Create + dispatch a fresh task for the responsible agent, then retire the
    stale diverged branch (tag-backup + delete).

    The rework task's own branch is the new source of truth; leaving the old
    ``task/<id>`` branch alive just clutters the tree with a permanent "needs
    review" flag. Its commits stay reachable via ``refs/legion/trash/`` — the
    same recoverable backup ``delete_branch`` makes.
    """
    from app.core.proxy_session import request_task_dispatch, TaskDispatchPreflightError

    if task.assigned_agent_id is None:
        return {"ok": False, "detail": "the task has no assignee — nobody to hand the rework to"}

    stale_branch = task.branch_name

    # Pull the rework brief off the most recent DIVERGENCE_REVIEW for this task.
    from app.models.message import Message

    ev = (await db.execute(
        select(Message)
        .where(Message.task_id == task.id, Message.event_type == "DIVERGENCE_REVIEW")
        .order_by(Message.timestamp.desc())
        .limit(1)
    )).scalars().first()
    analysis = (ev.event_metadata or {}).get("analysis", {}) if ev else {}
    brief = analysis.get("rework_brief") or (
        f"Carry the needed edits of branch {task.branch_name} over to the current {base}, "
        f"without rolling back the base's changes. You must rebase onto {base}."
    )
    description = brief
    if note and note.strip():
        description += f"\n\nUser comment: {note.strip()}"
    description += (
        f"\n\n[Source: divergence of branch {stale_branch}. Its commits are in the backup tag "
        f"refs/legion/trash/, the branch itself is deleted — work from the current {base}.]"
    )
    description += await _source_context_block(task, db)

    rework = Task(
        room_id=rid,
        goal_id=task.goal_id,
        title=f"Divergence rework: {task.title}"[:200],
        description=description,
        state=TaskState.ASSIGNED,
        dispatch_status=DispatchStatus.READY,
        assigned_agent_id=task.assigned_agent_id,
        created_by_agent_id=task.assigned_agent_id,
        origin=task.origin,
        rework_count=(task.rework_count or 0) + 1,
    )
    db.add(rework)
    await db.flush()

    # internal design notes Stage 5: the rework task picks up the
    # original task's requirement coverage — it's finishing the same
    # requirements, not starting fresh — then gets the same deterministic
    # block (requirements table + non-goals) the original had.
    if task.goal_id:
        from app.core.requirements import render_task_requirements_block
        from app.models.requirement import TaskRequirement

        orig_links = (await db.execute(
            select(TaskRequirement).where(TaskRequirement.task_id == task.id)
        )).scalars().all()
        for link in orig_links:
            db.add(TaskRequirement(task_id=rework.id, requirement_id=link.requirement_id))
        if orig_links:
            await db.flush()
        block = await render_task_requirements_block(rework, db)
        if block:
            rework.description = (rework.description or "") + block
            db.add(rework)

    task.divergence_state = "resolved"
    if task.state not in (TaskState.COMPLETED, TaskState.CANCELLED, TaskState.CONTINUED):
        task.state = TaskState.CONTINUED
    db.add(task)
    await db.flush()

    # Retire the stale branch — best effort. A failure here (busy tree, branch
    # checked out) is not fatal: the task is already "resolved" so the tree no
    # longer flags it, it just lingers until the next cleanup.
    backup_ref = None
    if stale_branch:
        try:
            res = await wsd.git_discard_branch_workspace(str(rid), stale_branch)
            if res.get("ok"):
                backup_ref = res.get("backup_ref")
            else:
                logger.warning(
                    "divergence_review: could not retire stale branch %s: %s",
                    stale_branch, res.get("stderr") or res,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("divergence_review: discard of %s failed: %s", stale_branch, exc)

    await emit_event(
        room_id=rid, event_type="TASK_CREATED",
        content=f"Divergence rework: {rework.title}",
        db=db, task_id=rework.id,
        metadata={"task_title": rework.title, "task_state": rework.state.value,
                  "origin_divergence_task_id": str(task.id)},
    )
    _retired = (
        f" Branch {stale_branch} was moved to backup tag {backup_ref}." if backup_ref
        else f" Branch {stale_branch} was kept (it could not be deleted)." if stale_branch
        else ""
    )
    await _emit_resolved(
        rid, task, "rework",
        f"A rework task was created — '{rework.title}'.{_retired}",
        db, branch=stale_branch, rework_task_id=str(rework.id), backup_ref=backup_ref,
    )
    await db.commit()
    await db.refresh(rework)

    try:
        outcome = await request_task_dispatch(rework, rid, db, source="divergence_rework")
        await db.commit()
    except TaskDispatchPreflightError as exc:
        logger.warning("divergence_review: rework dispatch preflight failed: %s", exc)
        outcome = f"not_dispatched: {exc}"

    return {"ok": True, "decision": "rework", "detail": "rework_task_created",
            "rework_task_id": str(rework.id), "dispatch": str(outcome)}
