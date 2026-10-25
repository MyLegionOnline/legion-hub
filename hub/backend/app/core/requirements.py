"""internal design notes Stage 3 — verify quotes against the
plan dialogue journal and persist the requirements/non_goals/open_questions
an accepted propose_plan carried.

``verify_quotes`` is the only defence against a model inventing a quote: it
never collapses "journal unreadable" into "quote is false" — see
``GoalRequirement.verified`` for why that distinction has to survive to the
database column too.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.goal import RoomGoal
from app.models.requirement import (
    GoalRequirement,
    RequirementKind,
    RequirementPriority,
    RequirementStatus,
)

if TYPE_CHECKING:
    from app.models.room import Room
    from app.models.task import Task
    from app.models.user import User

__all__ = [
    "verify_quotes",
    "persist_requirements",
    "load_goal_requirements",
    "render_requirements_digest",
    "render_task_requirements_block",
    "check_coverage",
    "emit_requirements_gap",
    "resolve_requirements_gap",
    "task_requirements_payload",
    "goal_coverage_matrix",
    "list_requirements_view",
    "apply_gate1_requirement_decisions",
    "flag_requirement_conflict",
]

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

_VALID_PRIORITIES = {p.value for p in RequirementPriority}


def _normalize_for_match(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip().casefold()


async def verify_quotes(
    session: Any,
    quotes: list[str],
    db: Any = None,
) -> list[Optional[bool]]:
    """Check each quote against *session*'s plan-dialogue journal.

    Returns a list parallel to *quotes*: ``True`` when the quote is found
    verbatim (after whitespace-collapse + casefold) in some turn, ``False``
    when the journal was read but the quote isn't in it, and ``None`` when
    the journal itself couldn't be read (Redis expired, no snapshot fallback
    either) — absence of proof is not proof the model invented the quote.
    """
    if not quotes:
        return []
    if session is None:
        return [None] * len(quotes)

    try:
        from app.core import plan_dialogue

        turns = await plan_dialogue.journal_read(session, db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "requirements.verify_quotes: journal_read failed for session %s: %s",
            getattr(session, "id", "?"), exc,
        )
        return [None] * len(quotes)

    if not turns:
        return [None] * len(quotes)

    haystacks = [_normalize_for_match(t.get("content")) for t in turns]
    results: list[Optional[bool]] = []
    for quote in quotes:
        needle = _normalize_for_match(quote)
        if not needle:
            results.append(None)
            continue
        results.append(any(needle in h for h in haystacks))
    return results


async def persist_requirements(
    goal: RoomGoal,
    payload: dict[str, list[dict[str, Any]]],
    db: AsyncSession,
    *,
    session: Any = None,
) -> list[GoalRequirement]:
    """Number and write ``GoalRequirement`` rows for one goal's accepted plan.

    *payload* is the normalized ``{requirements, non_goals, open_questions}``
    dict produced by ``user_sessions._requirements_from_proposal``. Codes are
    assigned in encounter order: ``R-001…``, ``NG-001…``, ``Q-001…``. Quotes
    of requirements and non_goals are checked against *session*'s dialogue
    journal (open_questions carry an assumption, not a user quote, so they
    are never "verified"). Returns the created rows; a caller with nothing to
    persist gets an empty list without touching the session.
    """
    requirements = payload.get("requirements") or []
    non_goals = payload.get("non_goals") or []
    open_questions = payload.get("open_questions") or []

    if not requirements and not non_goals and not open_questions:
        return []

    quoted = [it.get("quote", "") for it in requirements] + [
        it.get("quote", "") for it in non_goals
    ]
    verified = await verify_quotes(session, quoted, db)

    rows: list[GoalRequirement] = []
    position = 0

    for i, item in enumerate(requirements):
        priority_raw = str(item.get("priority") or "").strip().lower()
        priority = RequirementPriority(priority_raw) if priority_raw in _VALID_PRIORITIES else None
        rows.append(GoalRequirement(
            goal_id=goal.id,
            code=f"R-{i + 1:03d}",
            kind=RequirementKind.REQUIREMENT,
            quote=item.get("quote", ""),
            turn_ref=item.get("turn") or None,
            statement=item.get("statement", ""),
            priority=priority,
            acceptance=item.get("acceptance", ""),
            verified=verified[i] if i < len(verified) else None,
            position=position,
        ))
        position += 1

    ng_offset = len(requirements)
    for i, item in enumerate(non_goals):
        idx = ng_offset + i
        rows.append(GoalRequirement(
            goal_id=goal.id,
            code=f"NG-{i + 1:03d}",
            kind=RequirementKind.NON_GOAL,
            quote=item.get("quote", ""),
            statement=item.get("statement", ""),
            verified=verified[idx] if idx < len(verified) else None,
            position=position,
        ))
        position += 1

    for i, item in enumerate(open_questions):
        rows.append(GoalRequirement(
            goal_id=goal.id,
            code=f"Q-{i + 1:03d}",
            kind=RequirementKind.QUESTION,
            statement=item.get("question", ""),
            acceptance=item.get("assumption", ""),
            verified=None,
            position=position,
        ))
        position += 1

    for row in rows:
        db.add(row)
    await db.flush()
    logger.info(
        "requirements.persist_requirements: goal %s — %d requirement(s), %d non_goal(s), %d question(s)",
        goal.id, len(requirements), len(non_goals), len(open_questions),
    )
    return rows


async def load_goal_requirements(goal_id: Any, db: AsyncSession) -> list[GoalRequirement]:
    """Fetch a goal's requirement rows in display order.

    Shared by ``plan_session.render_plan_text_with_agents`` (Stage 4) and the
    Stage 9 archive renderer — one query, one ordering, instead of each
    caller re-writing the same ``select``.
    """
    if goal_id is None:
        return []
    return (await db.execute(
        select(GoalRequirement)
        .where(GoalRequirement.goal_id == goal_id)
        .order_by(GoalRequirement.position)
    )).scalars().all()


def _verified_marker(verified: Optional[bool]) -> str:
    return " ⚠️unverified" if verified is False else ""


def render_requirements_digest(rows: list[GoalRequirement], *, for_prompt: bool = True) -> str:
    """Markdown digest of a goal's requirements/non_goals/open_questions.

    Fed as part of the ``cm_self_plan`` first message (Stage 4) so the
    decomposition model knows the exact codes to put in each task's
    ``covers`` list — the codes here (``R-NNN``/``NG-NNN``/``Q-NNN``) are the
    same ones ``TaskCreate.covers`` and ``session_close`` match against.

    ``for_prompt=False`` (Stage 9 — the plan archive) drops the trailing
    ``covers`` instruction line, which only makes sense addressed to the
    decomposition model, not to a human reading the archived spec.
    """
    reqs = [r for r in rows if r.kind == RequirementKind.REQUIREMENT]
    non_goals = [r for r in rows if r.kind == RequirementKind.NON_GOAL]
    questions = [r for r in rows if r.kind == RequirementKind.QUESTION]
    if not reqs and not non_goals and not questions:
        return ""

    lines = ["## Requirements"]
    if reqs:
        for r in reqs:
            prio = f" [{r.priority.value}]" if r.priority else ""
            acc = f" — how to verify: {r.acceptance}" if r.acceptance else ""
            lines.append(f"- **{r.code}**{prio}: “{r.quote}”{_verified_marker(r.verified)}{acc}")
    if non_goals:
        lines.append("\n### Out of scope")
        for ng in non_goals:
            lines.append(f"- **{ng.code}**: “{ng.quote}”{_verified_marker(ng.verified)}")
    if questions:
        lines.append("\n### Open questions")
        for q in questions:
            lines.append(f"- **{q.code}**: {q.statement} (assumption: {q.acceptance})")
    if for_prompt:
        lines.append(
            "\nWhen creating a task, set its `covers` field to the requirement codes "
            "it satisfies, e.g. `covers: [\"R-001\", \"R-003\"]`. Leave `covers` empty "
            "for a task that doesn't map to a specific requirement above."
        )
    return "\n".join(lines)


def _escape_cell(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


async def render_task_requirements_block(task: "Task", db: AsyncSession) -> str:
    """Deterministic markdown appended to ``Task.description`` (Stage 5).

    Code assembles this from the ``goal_requirement`` table, the model never
    paraphrases it — paraphrase is exactly the step where a requirement
    quietly erodes (see the plan's rationale). Two sections:

    - "## Requirements" — a table of the requirements *this task* covers
      (``TaskRequirement`` rows), verbatim quote + how to verify.
    - "## Out of scope" — every non-goal of the task's goal, in full; non-goals
      are shared by the whole goal, not per-task, so every task of that goal
      gets the same list regardless of what it covers.

    Returns ``""`` when the task has no goal or the goal has no requirements/
    non-goals worth writing — callers must not append an empty block.
    """
    if not task.goal_id:
        return ""

    from app.models.requirement import TaskRequirement

    covered = (await db.execute(
        select(GoalRequirement)
        .join(TaskRequirement, TaskRequirement.requirement_id == GoalRequirement.id)
        .where(
            TaskRequirement.task_id == task.id,
            GoalRequirement.kind == RequirementKind.REQUIREMENT,
        )
        .order_by(GoalRequirement.position)
    )).scalars().all()
    non_goals = (await db.execute(
        select(GoalRequirement).where(
            GoalRequirement.goal_id == task.goal_id,
            GoalRequirement.kind == RequirementKind.NON_GOAL,
        ).order_by(GoalRequirement.position)
    )).scalars().all()

    if not covered and not non_goals:
        return ""

    lines: list[str] = []
    if covered:
        lines.append("\n\n## Requirements")
        lines.append("| code | quote | how to verify |")
        lines.append("|---|---|---|")
        for r in covered:
            lines.append(f"| {r.code} | “{_escape_cell(r.quote)}” | {_escape_cell(r.acceptance)} |")
    if non_goals:
        lines.append("\n## Out of scope")
        for ng in non_goals:
            lines.append(f"- **{ng.code}**: “{_escape_cell(ng.quote)}”")
    return "\n".join(lines)


async def check_coverage(goal_id: UUID, db: AsyncSession) -> dict[str, list[dict[str, Any]]]:
    """internal design notes Stage 6 — the soft coverage gate.

    Run right after decomposition. Returns three buckets, each a list of
    plain dicts (JSON-safe, fed straight into the ``REQUIREMENTS_GAP`` event
    metadata):

    - ``uncovered_must`` — MUST requirements no task claimed via ``covers``
      (``status`` is still the ``UNCOVERED`` default Stage 4 leaves it at).
    - ``unverified`` — requirements/non-goals whose quote Stage 3 checked
      against the dialogue journal and could NOT find (``verified is
      False`` — deliberately not ``is None``, since ``None`` means the
      journal itself couldn't be read, not that the model invented it).
    - ``open_questions`` — still-open questions, each with its assumption.

    All three empty means nothing to raise; the caller (``emit_requirements_gap``)
    treats that as "no card".
    """
    rows = (await db.execute(
        select(GoalRequirement)
        .where(GoalRequirement.goal_id == goal_id)
        .order_by(GoalRequirement.position)
    )).scalars().all()

    uncovered_must = [
        {
            "code": r.code,
            "quote": r.quote,
            "statement": r.statement,
            "acceptance": r.acceptance,
        }
        for r in rows
        if r.kind == RequirementKind.REQUIREMENT
        and r.priority == RequirementPriority.MUST
        and r.status == RequirementStatus.UNCOVERED
    ]
    unverified = [
        {"code": r.code, "quote": r.quote, "kind": r.kind.value}
        for r in rows
        if r.kind in (RequirementKind.REQUIREMENT, RequirementKind.NON_GOAL)
        and r.verified is False
    ]
    open_questions = [
        {"code": r.code, "question": r.statement, "assumption": r.acceptance}
        for r in rows
        if r.kind == RequirementKind.QUESTION and r.status != RequirementStatus.CLOSED
    ]
    return {
        "uncovered_must": uncovered_must,
        "unverified": unverified,
        "open_questions": open_questions,
    }


async def task_requirements_payload(task: "Task", db: AsyncSession) -> list[dict[str, Any]]:
    """internal design notes Stage 7 — what the Gate 1 review
    card shows: this task's covered requirements with their ``acceptance``,
    so the reviewer sees "what was promised, how to verify" next to the CI
    status instead of just green/red. JSON-safe, fed straight into the
    ``TASK_WAITING_REVIEW`` event metadata. Empty when the task has no goal
    or covers nothing (a caller must not add an empty key for those).
    """
    if not getattr(task, "goal_id", None):
        return []

    from app.models.requirement import TaskRequirement

    rows = (await db.execute(
        select(GoalRequirement)
        .join(TaskRequirement, TaskRequirement.requirement_id == GoalRequirement.id)
        .where(TaskRequirement.task_id == task.id)
        .order_by(GoalRequirement.position)
    )).scalars().all()
    return [
        {
            "code": r.code,
            "quote": r.quote,
            "acceptance": r.acceptance,
            "status": r.status.value if hasattr(r.status, "value") else r.status,
        }
        for r in rows
    ]


async def goal_coverage_matrix(goal_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """internal design notes Stage 7 — the Gate 2 review card's
    requirement × task × status matrix. One row per ``GoalRequirement``
    (requirements and non-goals; open questions carry no coverage), each
    with the tasks that claim to cover it. JSON-safe, fed into the
    ``CHECKPOINT_REQUEST`` event metadata for the goal review.
    """
    from app.models.requirement import TaskRequirement
    from app.models.task import Task

    rows = (await db.execute(
        select(GoalRequirement)
        .where(
            GoalRequirement.goal_id == goal_id,
            GoalRequirement.kind != RequirementKind.QUESTION,
        )
        .order_by(GoalRequirement.position)
    )).scalars().all()
    if not rows:
        return []

    task_rows = (await db.execute(
        select(TaskRequirement.requirement_id, Task.id, Task.title, Task.state)
        .join(Task, Task.id == TaskRequirement.task_id)
        .where(TaskRequirement.requirement_id.in_([r.id for r in rows]))
    )).all()
    tasks_by_req: dict[UUID, list[dict[str, Any]]] = {}
    for req_id, task_id, task_title, task_state in task_rows:
        tasks_by_req.setdefault(req_id, []).append({
            "id": str(task_id),
            "title": task_title,
            "state": task_state.value if hasattr(task_state, "value") else task_state,
        })

    return [
        {
            "code": r.code,
            "kind": r.kind.value,
            "quote": r.quote,
            "statement": r.statement,
            "priority": r.priority.value if r.priority else None,
            "status": r.status.value if hasattr(r.status, "value") else r.status,
            "verified": r.verified,
            "tasks": tasks_by_req.get(r.id, []),
        }
        for r in rows
    ]


async def list_requirements_view(goal_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """internal design notes Stage 9 — the goal panel's full
    requirements list: requirements, non-goals AND open questions together
    (unlike ``goal_coverage_matrix``, which the Gate 2 card restricts to
    requirements/non-goals since a question carries no coverage). Same
    JSON-safe shape, ``tasks: []`` for questions.
    """
    from app.models.requirement import TaskRequirement
    from app.models.task import Task

    rows = (await db.execute(
        select(GoalRequirement)
        .where(GoalRequirement.goal_id == goal_id)
        .order_by(GoalRequirement.position)
    )).scalars().all()
    if not rows:
        return []

    req_ids = [r.id for r in rows if r.kind != RequirementKind.QUESTION]
    tasks_by_req: dict[UUID, list[dict[str, Any]]] = {}
    if req_ids:
        task_rows = (await db.execute(
            select(TaskRequirement.requirement_id, Task.id, Task.title, Task.state)
            .join(Task, Task.id == TaskRequirement.task_id)
            .where(TaskRequirement.requirement_id.in_(req_ids))
        )).all()
        for req_id, task_id, task_title, task_state in task_rows:
            tasks_by_req.setdefault(req_id, []).append({
                "id": str(task_id),
                "title": task_title,
                "state": task_state.value if hasattr(task_state, "value") else task_state,
            })

    return [
        {
            "code": r.code,
            "kind": r.kind.value,
            "quote": r.quote,
            "statement": r.statement,
            "priority": r.priority.value if r.priority else None,
            "acceptance": r.acceptance,
            "status": r.status.value if hasattr(r.status, "value") else r.status,
            "verified": r.verified,
            "tasks": tasks_by_req.get(r.id, []),
        }
        for r in rows
    ]


async def apply_gate1_requirement_decisions(
    task: "Task",
    decisions: dict[str, bool],
    db: AsyncSession,
) -> list[str]:
    """internal design notes Stage 7 — the reviewer's checklist
    on the Gate 1 card. *decisions* maps a requirement ``code`` (as shown on
    the card, from ``task_requirements_payload``) to whether the reviewer
    checked it off. Checked → ``covered``, unchecked → ``partial``; only
    requirements this task actually covers are touched, everything else on
    the card is ignored rather than raising — a stale/unknown code must not
    fail the approval. Returns the codes actually updated.
    """
    if not decisions:
        return []

    from app.models.requirement import TaskRequirement

    rows = (await db.execute(
        select(GoalRequirement)
        .join(TaskRequirement, TaskRequirement.requirement_id == GoalRequirement.id)
        .where(TaskRequirement.task_id == task.id)
    )).scalars().all()

    updated: list[str] = []
    for r in rows:
        if r.code not in decisions:
            continue
        r.status = RequirementStatus.COVERED if decisions[r.code] else RequirementStatus.PARTIAL
        r.updated_at = datetime.utcnow()
        db.add(r)
        updated.append(r.code)
    if updated:
        await db.flush()
        logger.info(
            "requirements.apply_gate1_requirement_decisions: task %s — %d requirement(s) updated",
            task.id, len(updated),
        )
    return updated


async def flag_requirement_conflict(
    task: "Task",
    codes: list[str],
    note: str,
    db: AsyncSession,
    *,
    agent_label: Optional[str] = None,
) -> list[str]:
    """internal design notes Stage 8 — the mid-task dialogue's
    ``requirement_conflict`` label.

    ChatMaster calls this (via the ``flag_requirement_conflict`` tool) from
    the SAME mid-task turn where it answers the agent, when the agent's
    question is not a wording bug but a real contradiction between
    requirements, or a requirement pointing at something that does not exist.
    No separate CR-NNN mechanism: this is the existing proxy dialogue with a
    label attached, exactly as the plan calls for.

    Sets ``status=conflict`` on every code that resolves to a requirement of
    *task*'s goal (unknown codes are ignored with a log line — a stale or
    invented code must not blow up the answering turn) and raises a
    ``REQUIREMENT_CONFLICT`` event/card for the user. Returns the codes
    actually updated; an empty list still emits the card when *codes* named
    nothing resolvable, since the conflict itself is real even if the model
    got the code wrong.
    """
    from app.core.events import emit_event

    updated: list[str] = []
    if task.goal_id and codes:
        rows = (await db.execute(
            select(GoalRequirement).where(
                GoalRequirement.goal_id == task.goal_id,
                GoalRequirement.code.in_(codes),
            )
        )).scalars().all()
        found_codes = {r.code for r in rows}
        for r in rows:
            r.status = RequirementStatus.CONFLICT
            r.updated_at = datetime.utcnow()
            db.add(r)
            updated.append(r.code)
        for code in codes:
            if code not in found_codes:
                logger.info(
                    "requirements.flag_requirement_conflict: task %s — code %r not found "
                    "for goal %s, ignoring", task.id, code, task.goal_id,
                )
        if updated:
            await db.flush()

    codes_label = ", ".join(codes) if codes else "not specified"
    content = f"Requirement conflict in task '{task.title}' ({codes_label}): {note}".strip()

    await emit_event(
        room_id=task.room_id,
        event_type="REQUIREMENT_CONFLICT",
        content=content,
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "goal_id": str(task.goal_id) if task.goal_id else None,
            "codes": codes,
            "updated_codes": updated,
            "note": note,
            "agent_label": agent_label,
        },
    )
    logger.info(
        "requirements.flag_requirement_conflict: task %s — %d requirement(s) marked conflict (%s)",
        task.id, len(updated), codes_label,
    )
    return updated


def _gap_is_empty(gap: dict[str, list[Any]]) -> bool:
    return not gap.get("uncovered_must") and not gap.get("unverified") and not gap.get("open_questions")


async def emit_requirements_gap(
    room_id: UUID,
    goal_id: UUID,
    goal_title: str,
    db: AsyncSession,
) -> Optional[dict[str, list[dict[str, Any]]]]:
    """Check coverage after decomposition and raise ``REQUIREMENTS_GAP`` if
    there's anything to show. Soft gate — never raises, never blocks the
    pipeline; a caller ignoring the return value is exactly the point.
    """
    gap = await check_coverage(goal_id, db)
    if _gap_is_empty(gap):
        return None

    parts = []
    if gap["uncovered_must"]:
        parts.append(f"{len(gap['uncovered_must'])} mandatory requirements without a task")
    if gap["unverified"]:
        parts.append(f"{len(gap['unverified'])} unverified quotes")
    if gap["open_questions"]:
        parts.append(f"{len(gap['open_questions'])} open questions")
    content = f"“{goal_title}”: " + ", ".join(parts) + "."

    from app.core.events import emit_event

    await emit_event(
        room_id=room_id,
        event_type="REQUIREMENTS_GAP",
        content=content,
        db=db,
        metadata={"goal_id": str(goal_id), "goal_title": goal_title, **gap},
    )
    logger.info(
        "requirements.emit_requirements_gap: goal %s — %d uncovered must, "
        "%d unverified, %d open questions",
        goal_id, len(gap["uncovered_must"]), len(gap["unverified"]), len(gap["open_questions"]),
    )
    return gap


def _render_gap_rework_message(goal: RoomGoal, gap: dict[str, list[dict[str, Any]]], note: Optional[str]) -> str:
    """First message of the reopened ``user_cm_p2`` session — the gap list
    verbatim, so the model works from the same codes the user saw on the
    card rather than a fresh paraphrase of "please refine the plan"."""
    lines = [f"Rework of the plan for goal '{goal.title}' — requirements were gathered incompletely:"]
    if gap["uncovered_must"]:
        lines.append("\nUncovered mandatory requirements (no task takes them):")
        for r in gap["uncovered_must"]:
            acc = f" — how to verify: {r['acceptance']}" if r.get("acceptance") else ""
            lines.append(f"- {r['code']}: “{r['quote']}”{acc}")
    if gap["unverified"]:
        lines.append("\nQuotes not found in the dialogue journal (check whether they were made up):")
        for r in gap["unverified"]:
            lines.append(f"- {r['code']}: “{r['quote']}”")
    if gap["open_questions"]:
        lines.append("\nOpen questions with provisional assumptions:")
        for r in gap["open_questions"]:
            lines.append(f"- {r['code']}: {r['question']} (assumption: {r['assumption']})")
    if note and note.strip():
        lines.append(f"\nUser comment: {note.strip()}")
    lines.append(
        "\nAsk the user for the missing details and update the requirements/tasks via "
        "propose_plan and decomposition — do not paraphrase, ask for specifics."
    )
    return "\n".join(lines)


async def resolve_requirements_gap(
    room: "Room",
    goal: RoomGoal,
    user: "User",
    decision: str,
    note: Optional[str],
    db: AsyncSession,
) -> dict[str, Any]:
    """Handle the card's two buttons.

    ``accept`` just records the decision (a follow-up ``REQUIREMENTS_GAP_RESOLVED``
    event the frontend uses to stop showing the card as open) — nothing else
    changes, the gap is a soft gate. ``rework`` reopens ``user_cm_p2`` with the
    gap list as the first message, mirroring the "reopen a session with a
    message" idiom ``waiting_reply._continue_direct_session_task`` already
    uses for direct-session tasks.
    """
    from app.core.events import emit_event

    gap = await check_coverage(goal.id, db)

    if decision == "rework":
        from app.core.room_coordinator import get_cm_agent_id
        from app.core.router import RouteDecision
        from app.core.user_sessions import process_user_turn, start_user_session

        cm_agent_id = get_cm_agent_id(room)
        if cm_agent_id is None:
            raise ValueError("room has no ChatMaster configured in coordinator_config")

        session = await start_user_session(
            room, user,
            RouteDecision(
                route="chatmaster_p2",
                mentioned_agent_ids=[],
                confidence=1.0,
                reasoning="internal design notes Stage 6: user asked to refine the plan",
            ),
            db,
            cm_agent_id=cm_agent_id,
        )
        await db.flush()
        message = _render_gap_rework_message(goal, gap, note)

        await emit_event(
            room_id=room.id,
            event_type="REQUIREMENTS_GAP_RESOLVED",
            content=f"'{goal.title}': the user asks to rework the plan against the requirements.",
            db=db,
            metadata={"goal_id": str(goal.id), "decision": "rework", "session_id": str(session.id)},
        )
        await process_user_turn(session, message, db)
    else:
        await emit_event(
            room_id=room.id,
            event_type="REQUIREMENTS_GAP_RESOLVED",
            content=f"'{goal.title}': requirements accepted as they are.",
            db=db,
            metadata={"goal_id": str(goal.id), "decision": "accept"},
        )

    return gap
