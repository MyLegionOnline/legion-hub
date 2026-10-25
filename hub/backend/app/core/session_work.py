"""Work done inside a direct user ↔ agent dialogue — branch, then task.

internal design notes. A dialogue (`@nickname`, session kinds
``user_agent_p1`` / ``user_agent_p2``) never went through task dispatch, so
until this module its file writes auto-committed onto the room's
``working_branch``: no branch of their own, no Gate 1, no record on the board,
and an empty ``0 tasks`` bundle at the publication gate.

The shape of the fix (§3.1 of the plan): **the branch appears before the
task**. Naming a branch ``task/<id>`` needs a task id, and at the moment of the
first write nobody knows yet whether this turn will produce work worth
recording — an all-reading turn is just as common. So the work accumulates on
``session/<session_id>``, created just-in-time before the first tool that could
touch the working tree, and a ``Task`` is written later, only if that branch
actually has commits (Stage 2), at which point the branch is renamed to the
canonical ``task/<id>``.

Stage 1 (this module, for now) covers the branch half:

``tool_touches_worktree``  — default-deny classifier over tool names;
``ensure_worktree_branch`` — the just-in-time pin, called from the workspace
                             tool loop before a tool runs;
``open_dialogue_task`` / ``recorded_work_branch`` — what the dialogue is
                             currently building on, read back by
                             ``room_git_flow.ensure_session_branch``.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping
from uuid import UUID

logger = logging.getLogger(__name__)

# ── session.context_snapshot keys ────────────────────────────────────────────

# The dialogue's own branch, written when it is created and cleared when the
# branch is adopted as a task (Stage 2).
SNAPSHOT_WORK_BRANCH = "work_branch"
# The task this dialogue opened out of its own work, while it is still open.
SNAPSHOT_TASK_ID = "adhoc_task_id"
# The user message that started the current burst of work — the title and
# description of the task are derived from it, with no LLM call (Stage 2).
SNAPSHOT_WORK_REQUEST = "work_request"
# Set whenever a tool that could touch the volume ran in this dialogue, cleared
# once the burst has been finalized. Without it every later turn — including
# the reading ones — would re-run Gate 1 on a task that has not changed.
SNAPSHOT_WORK_TOUCHED = "work_touched"

# Task states in which the dialogue's task is still "the current work": the
# next writing turn keeps building on its branch instead of opening a new one.
# COMPLETED / CANCELLED close the burst; the next write starts a fresh branch.
_OPEN_TASK_STATES = frozenset({"in_progress", "waiting_review", "blocked"})

# Tools that cannot touch the working tree. Everything *not* listed here pins
# the branch before it runs — default-deny, so a tool added later lands on the
# safe side (it gets a branch) instead of silently writing on `dev`. This is
# deliberately not ``Tool.mutating``: that flag means "changes state" in the
# broad sense (``task_create``, ``todo_write`` write to the database), while
# what matters here is only the volume.
_WORKTREE_SAFE_TOOLS = frozenset({
    "ask_user",
    "bash_output",
    "bash_wait",
    "find_file",
    "get_outline",
    "git_diff",
    "glob",
    "grep",
    "observation_read",
    "plan_read",
    "preview_fetch",
    "read_file",
    "read_files",
    "repo_map",
    "request_file_read",
    "task_create",
    "task_list",
    "task_update",
    "todo_read",
    "todo_write",
    "web_fetch",
    "web_search",
})


def tool_touches_worktree(tool_name: str) -> bool:
    """True when *tool_name* may write into the room's volume."""
    return str(tool_name or "") not in _WORKTREE_SAFE_TOOLS


# ── snapshot access ──────────────────────────────────────────────────────────

def _snapshot(session) -> dict[str, Any]:
    snap = getattr(session, "context_snapshot", None)
    return dict(snap) if isinstance(snap, Mapping) else {}


async def _remember(session, db, **values: Any) -> None:
    """Persist snapshot keys without disturbing the conversation history.

    ``_save_history`` copies ``context_snapshot`` before replacing
    ``messages``, so keys written here survive the rest of the turn.
    """
    snap = _snapshot(session)
    for key, value in values.items():
        if value is None:
            snap.pop(key, None)
        else:
            snap[key] = value
    session.context_snapshot = snap
    # context_snapshot is a plain JSONB column: an in-place-aliased `messages`
    # list can make SQLAlchemy's flush-time value comparison conclude "unchanged"
    # and drop the column from the UPDATE (see _save_history in user_sessions).
    try:
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(session, "context_snapshot")
    except (AttributeError, KeyError):
        pass  # non-ORM stand-in in unit tests
    db.add(session)
    await db.flush()


def recorded_work_branch(session) -> str | None:
    """The dialogue's own branch, if it ever created one."""
    value = _snapshot(session).get(SNAPSHOT_WORK_BRANCH)
    return str(value) if value else None


def recorded_work_request(session) -> str | None:
    value = _snapshot(session).get(SNAPSHOT_WORK_REQUEST)
    return str(value) if value else None


async def open_dialogue_task(session, db):
    """The task this dialogue opened, while it is still the current work.

    Returns ``None`` when the dialogue never opened one, when the row is gone,
    or when the task is finished (merged / completed / cancelled) — the next
    burst of work then starts on a fresh branch.
    """
    raw = _snapshot(session).get(SNAPSHOT_TASK_ID)
    if not raw:
        return None
    try:
        task_id = UUID(str(raw))
    except ValueError:
        return None

    from sqlalchemy.future import select

    from app.models.task import Task

    task = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
    if task is None:
        return None
    if getattr(task, "merge_commit", None):
        return None
    state = getattr(task.state, "value", task.state)
    return task if str(state) in _OPEN_TASK_STATES else None


# ── the just-in-time pin ─────────────────────────────────────────────────────

def _is_dialogue_session(session) -> bool:
    """Only user ↔ agent dialogues. ChatMaster sessions keep the old pin: the
    ``cm`` role has no file-writing tools, so there is no work to record."""
    return str(getattr(session, "kind", "") or "").startswith("user_agent")


async def _park_before_switch(room_id, current: str | None, target: str) -> None:
    """Commit whatever is uncommitted before leaving *current* for *target*.

    A checkout carries uncommitted changes across branches, which would drag
    another owner's half-finished edits onto ours. Committing them first keeps
    them on the branch they were made on. ``git_commit`` is idempotent (a clean
    tree returns no sha) and refuses ``main`` on its own.
    """
    if not current or current == target:
        return
    from app.core import workspaced_client as wsd

    try:
        await wsd.git_commit_workspace(
            str(room_id), f"chore: park local changes before switching to {target}"
        )
    except Exception as exc:
        logger.warning(
            "session_work: could not park the working tree of room %s on %s: %s",
            room_id, current, exc,
        )


async def _revive_parked_task(task, db) -> None:
    """The agent is writing on a task parked BLOCKED (tool budget ran out).

    The user's reply may have reached the dialogue as an ordinary turn instead
    of going through ``waiting_reply`` — either way the work is running again,
    so the board must not keep showing it as blocked.
    """
    from app.models.task import DispatchStatus, TaskState

    if task.state != TaskState.BLOCKED:
        return
    task.state = TaskState.IN_PROGRESS
    task.dispatch_status = DispatchStatus.DISPATCHED
    db.add(task)
    await db.flush()
    try:
        from app.core.waiting_reply import clear_task_awaiting

        await clear_task_awaiting(task, db)
    except Exception as exc:
        logger.warning("session_work: could not clear awaiting flag on %s: %s", task.id, exc)


async def ensure_worktree_branch(session, db, *, user_request: str | None = None) -> str | None:
    """Pin the workspace HEAD to the branch this dialogue's work belongs on.

    Called from the workspace tool loop right before a tool that could touch
    the volume. Returns the branch HEAD was pinned to, or ``None`` when nothing
    was pinned — not a dialogue session, the room opted out
    (``git_config.direct_session_mode = "off"``), or git refused. Best-effort
    and idempotent: refusing the turn over a git detail would be worse than
    writing where HEAD already is, which is what happened before this plan.
    """
    if not _is_dialogue_session(session):
        return None

    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import (
        _current_branch,
        _load_room,
        ensure_task_branch,
        get_flow_config,
        session_branch_name,
    )

    room_id = session.room_id
    cfg = get_flow_config(await _load_room(room_id, db))
    if cfg.direct_session_mode == "off":
        return None

    # A session bound to a task (``user_agent_p1``) already has a branch, and
    # its work merges through that task's own Gate 1.
    task_id = getattr(session, "task_id", None)
    if task_id is not None:
        from sqlalchemy.future import select

        from app.models.task import Task

        task = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
        if task is not None:
            return await ensure_task_branch(room_id, task, db)

    # Work this dialogue already opened a task for and that has not been merged
    # yet: keep building on it rather than opening a second branch for the same
    # conversation.
    own_task = await open_dialogue_task(session, db)
    if own_task is not None:
        await _revive_parked_task(own_task, db)
        await _remember(session, db, **{SNAPSHOT_WORK_TOUCHED: True})
        return await ensure_task_branch(room_id, own_task, db)

    branch = session_branch_name(session.id)
    current = await _current_branch(room_id)

    if current != branch:
        await _park_before_switch(room_id, current, branch)
        try:
            result = await wsd.create_workspace_branch(
                str(room_id), branch, base=cfg.working_branch
            )
        except Exception as exc:
            logger.warning(
                "session_work: could not open work branch %s in room %s: %s",
                branch, room_id, exc,
            )
            return None
        if isinstance(result, Mapping) and result.get("ok") is False:
            logger.warning(
                "session_work: workspaced refused work branch %s in room %s",
                branch, room_id,
            )
            return None
        logger.info(
            "session_work: session %s writes on %s (base %s)",
            session.id, branch, cfg.working_branch,
        )

    remembered: dict[str, Any] = {SNAPSHOT_WORK_TOUCHED: True}
    if recorded_work_branch(session) != branch:
        remembered[SNAPSHOT_WORK_BRANCH] = branch
    if user_request and not recorded_work_request(session):
        remembered[SNAPSHOT_WORK_REQUEST] = user_request[:4000]
    await _remember(session, db, **remembered)

    return branch


# ── turning the branch into a task (Stage 2) ─────────────────────────────────

_MAX_TITLE = 80
_MAX_REQUEST_IN_DESCRIPTION = 4000
_MAX_REPLY_IN_DESCRIPTION = 4000
_FALLBACK_TITLE = "Follow-up from the dialogue"


def _strip_mention(text: str) -> str:
    """Drop the leading ``@nickname`` — it addresses the agent, it is not part
    of what was asked."""
    stripped = text.lstrip()
    if not stripped.startswith("@"):
        return text.strip()
    head, _, tail = stripped.partition(" ")
    return tail.strip() if tail.strip() else head.strip()


def derive_task_title(user_request: str | None) -> str:
    """A board title out of the request that started the burst — no LLM call.

    First sentence, cut on a word boundary. A dialogue is written for a human
    reader, so its first line is already a serviceable title; anything cleverer
    would cost a model round-trip for a label.
    """
    text = " ".join(_strip_mention(str(user_request or "")).split())
    if not text:
        return _FALLBACK_TITLE

    for end in (". ", "! ", "? ", "\n"):
        head = text.split(end)[0]
        if head != text and len(head) >= 12:
            text = head
            break
    text = text.rstrip(" .!?,:;—-")
    if len(text) <= _MAX_TITLE:
        return text or _FALLBACK_TITLE

    cut = text[:_MAX_TITLE]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return (cut.rstrip(" .,:;—-") + "…") or _FALLBACK_TITLE


def derive_task_description(user_request: str | None, agent_reply: str | None) -> str:
    """What was asked, plus what the agent reported back.

    This is the whole "spec": no ChatMaster pass, no approval. It is also what
    a re-dispatch would feed the agent, so the previous result belongs in it.
    """
    request = str(user_request or "").strip()[:_MAX_REQUEST_IN_DESCRIPTION]
    reply = str(agent_reply or "").strip()[:_MAX_REPLY_IN_DESCRIPTION]
    parts = [request or _FALLBACK_TITLE]
    if reply:
        parts.append(f"---\n**Agent's result:**\n{reply}")
    return "\n\n".join(parts)


async def _branch_has_work(room_id, base: str, branch: str) -> bool | None:
    """True when *branch* carries commits of its own against *base*.

    The three-dot diff is the same one Gate 1 classifies risk from, so "there
    is something to record" and "there is something to review" cannot disagree.
    A failed diff answers ``None`` (falsy): inventing a task out of an
    unreadable workspace would be worse than leaving the branch for the next
    turn, and it must not be mistaken for "empty" by the branch cleanup.
    """
    from app.core import workspaced_client as wsd

    try:
        resp = await wsd.git_diff_workspace(
            str(room_id), from_ref=base, to_ref=branch, stat=True, merge_base=True
        )
    except Exception as exc:
        logger.warning(
            "session_work: could not diff %s against %s in room %s: %s",
            branch, base, room_id, exc,
        )
        return None
    stat = resp.get("diff", "") if isinstance(resp, Mapping) else str(resp or "")
    return bool(str(stat).strip())


async def _discard_empty_branch(session, db, room_id, branch: str) -> None:
    """Drop a ``session/<id>`` branch that ended up with no commits of its own.

    Such a branch is only ever created just-in-time before a tool that *could*
    write; when the turn wrote nothing (or its output was gitignored) it would
    otherwise sit in the Git panel forever. ``git_discard_branch`` tags it under
    ``refs/legion/trash/`` first, so the delete is recoverable. The recorded
    branch is forgotten too — ``ensure_session_branch`` would otherwise recreate
    it at the start of the next turn.
    """
    from app.core import workspaced_client as wsd

    try:
        res = await wsd.git_discard_branch_workspace(str(room_id), branch)
        discarded = not isinstance(res, Mapping) or res.get("ok") is not False
    except Exception as exc:
        logger.warning(
            "session_work: could not discard empty branch %s in room %s: %s",
            branch, room_id, exc,
        )
        return
    if not discarded:
        logger.warning(
            "session_work: workspaced refused to discard empty branch %s in room %s",
            branch, room_id,
        )
        return
    await _remember(
        session, db, **{SNAPSHOT_WORK_BRANCH: None, SNAPSHOT_WORK_REQUEST: None}
    )
    logger.info("session_work: discarded empty branch %s (room %s)", branch, room_id)


async def _agent_nickname(agent_id, db) -> str | None:
    if agent_id is None:
        return None
    try:
        from app.models.agent import Agent

        agent = await db.get(Agent, agent_id)
    except Exception:
        return None
    return getattr(agent, "nickname", None) if agent is not None else None


async def adopt_session_branch_as_task(
    session, db, *, user_request: str | None = None, agent_reply: str | None = None
):
    """Write the dialogue's work down as a task, once there is work to write.

    Returns the ``Task`` (the one just created, or the one this dialogue
    already has open), or ``None`` when the dialogue produced nothing — no
    branch, or a branch without commits of its own. That "nothing" case is the
    common one: most turns read.

    The task is deliberately cheap: title and description come from the
    conversation, there is no spec, no approval, and no dispatch of its own —
    the agent has already done the work. ``dispatch_status`` is claimed the way
    ``proxy_session`` claims a task it is running, so the board cannot start a
    second executor on the same branch; Stage 3 releases it.
    """
    from app.core.room_git_flow import (
        _load_room,
        commit_branch_worktree,
        get_flow_config,
        task_branch_name,
    )

    existing = await open_dialogue_task(session, db)
    if existing is not None:
        return existing

    branch = recorded_work_branch(session)
    if not branch:
        return None

    room_id = session.room_id
    cfg = get_flow_config(await _load_room(room_id, db))

    # bash-only work is still uncommitted at this point — fold it in before
    # asking whether the branch has anything on it.
    await commit_branch_worktree(
        room_id, branch, db, reason=f"finalize dialogue work on {branch}"
    )

    has_work = await _branch_has_work(room_id, cfg.working_branch, branch)
    if not has_work:
        logger.info(
            "session_work: session %s left no commits on %s — no task recorded",
            session.id, branch,
        )
        # ``None`` = the diff itself failed: the branch may well hold work, so
        # it is only dropped when git positively said it is empty.
        if has_work is False:
            await _discard_empty_branch(session, db, room_id, branch)
        return None

    from app.core import workspaced_client as wsd
    from app.models.task import (
        TASK_ORIGIN_DIRECT_SESSION,
        DispatchStatus,
        Task,
        TaskState,
    )

    request = user_request or recorded_work_request(session)
    agent_id = getattr(session, "agent_id", None)
    task = Task(
        room_id=room_id,
        goal_id=getattr(session, "goal_id", None),
        title=derive_task_title(request),
        description=derive_task_description(request, agent_reply),
        state=TaskState.IN_PROGRESS,
        dispatch_status=DispatchStatus.DISPATCHED,
        assigned_agent_id=agent_id,
        created_by_agent_id=agent_id,
        origin=TASK_ORIGIN_DIRECT_SESSION,
        origin_session_id=getattr(session, "id", None),
        branch_name=branch,
    )
    db.add(task)
    await db.flush()

    # Give the branch its canonical name. Cosmetic: every gate reads
    # ``task.branch_name``, so a refusal here costs nothing but a stranger
    # name in ``git log``.
    canonical = task_branch_name(task.id)
    try:
        renamed = await wsd.rename_workspace_branch(str(room_id), branch, canonical)
    except Exception as exc:
        logger.warning(
            "session_work: could not rename %s → %s in room %s: %s",
            branch, canonical, room_id, exc,
        )
        renamed = {"ok": False}
    if isinstance(renamed, Mapping) and renamed.get("ok"):
        task.branch_name = canonical
        db.add(task)
        await db.flush()

    await _remember(
        session, db,
        **{
            SNAPSHOT_TASK_ID: str(task.id),
            SNAPSHOT_WORK_BRANCH: None,
            SNAPSHOT_WORK_REQUEST: None,
        },
    )

    from app.core.events import emit_event

    nickname = await _agent_nickname(agent_id, db)
    await emit_event(
        room_id=room_id if isinstance(room_id, UUID) else UUID(str(room_id)),
        event_type="TASK_CREATED",
        content=f"Task from a dialogue: {task.title}",
        db=db,
        task_id=task.id,
        sender_agent_id=agent_id,
        metadata={
            # The board refreshes on this event; the chat does not draw a card
            # for it. The one card a dialogue burst produces is the Gate 1
            # result (Stage 3) — announcing the task as well would be noise.
            "silent": True,
            "origin": "direct_session",
            "session_id": str(getattr(session, "id", "")),
            "agent_nickname": nickname,
            "task_title": task.title,
            "branch": task.branch_name,
        },
    )
    logger.info(
        "session_work: session %s recorded task %s (%s) on %s",
        session.id, task.id, task.title, task.branch_name,
    )

    # internal design notes Stage 6: a direct @-mention dialogue turned
    # into a task — archive the user's own words (reference links, pasted
    # configs) to .legion/plans/task-{id}.md so a rework reads them verbatim.
    try:
        from app.core import plan_dialogue

        await plan_dialogue.archive_for_task(
            room_id, task.id, session, db,
            goal_id=getattr(session, "goal_id", None),
            task_title=task.title,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "session_work: plan-dialogue archive failed for task %s: %s",
            task.id, exc,
        )

    return task


# ── the end of a turn: Gate 1 (Stage 3) ──────────────────────────────────────

async def _merge_without_gate(room, task, db) -> dict[str, Any]:
    """``direct_session_mode = "branch_only"``: fold the branch in, no checks.

    For rooms whose suite is too slow to run after every writing turn. The work
    still goes through a branch and is still recorded as a task — only the
    verdict is skipped. A conflict leaves the task in review for a human.
    """
    from app.core.events import emit_event
    from app.core.room_git_flow import maybe_open_promotion, merge_task_branch
    from app.models.task import TaskState

    room_id = getattr(room, "id", room)
    result: dict[str, Any] = {
        "ci_passed": None, "risk": None, "merged": False,
        "merge_commit": None, "state": task.state,
    }
    try:
        merged = await merge_task_branch(room_id, task, db)
    except Exception as exc:
        logger.warning("session_work: branch_only merge failed for task %s: %s", task.id, exc)
        merged = {"ok": False}

    if not merged.get("ok"):
        task.state = TaskState.WAITING_REVIEW
        db.add(task)
        await db.flush()
        result["state"] = task.state
        return result

    task.merge_commit = merged.get("sha")
    task.state = TaskState.COMPLETED
    db.add(task)
    await db.commit()
    result.update(merged=True, merge_commit=task.merge_commit, state=task.state)

    await emit_event(
        room_id=room_id if isinstance(room_id, UUID) else UUID(str(room_id)),
        event_type="TASK_MERGED",
        content=f"Task '{task.title}' merged into {merged.get('target')} (without checks).",
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "branch": merged.get("branch"),
            "target": merged.get("target"),
            "merge_commit": task.merge_commit,
            "origin": getattr(task, "origin", None),
        },
    )
    try:
        await maybe_open_promotion(str(room_id), db)
    except Exception as exc:
        logger.warning("session_work: promotion hook failed for task %s: %s", task.id, exc)
    return result


async def finalize_session_work(
    session, db, *, reason: str, user_request: str | None = None,
    agent_reply: str | None = None, incomplete: bool = False,
) -> dict[str, Any] | None:
    """Close the current burst of dialogue work: record it, then run Gate 1.

    Called at the end of a turn, right after the agent's reply is already on
    its way to the user — the conversation keeps its speed and the gate follows
    with its own events. Also called on the session-closing paths (Stage 4) so
    a turn that never got its finalization is not lost.

    Idempotent and quiet: a dialogue that wrote nothing since the last
    finalization returns ``None`` without touching git or the board.

    ``incomplete`` (waiting_reply_safeguard S4): the turn ended on a forced
    text reply after the tool-call budget ran out. The work exists but the
    agent did not finish it, so the task is parked ``BLOCKED`` waiting for the
    user rather than sent through the review gate as a finished submission.
    """
    if not _is_dialogue_session(session):
        return None
    if not _snapshot(session).get(SNAPSHOT_WORK_TOUCHED):
        return None

    from app.core.room_git_flow import _load_room, finalize_task_gate1, get_flow_config

    room = await _load_room(session.room_id, db)
    cfg = get_flow_config(room)
    if cfg.direct_session_mode == "off":
        await _remember(session, db, **{SNAPSHOT_WORK_TOUCHED: None})
        return None

    task = await adopt_session_branch_as_task(
        session, db, user_request=user_request, agent_reply=agent_reply
    )
    if task is None:
        # Nothing was written after all (a turn that only ran `bash ls`): stop
        # asking on every later turn.
        await _remember(session, db, **{SNAPSHOT_WORK_TOUCHED: None})
        return None

    if incomplete:
        # waiting_reply_safeguard S4: the agent ran out of tool budget mid-work.
        # Park the task BLOCKED waiting for the user — not WAITING_REVIEW — and
        # flag it so a reply like "finish" re-opens it. The branch stays; the
        # next dialogue turn keeps building on it (``_OPEN_TASK_STATES``).
        from app.core.events import emit_event
        from app.core.waiting_reply import mark_task_awaiting
        from app.models.task import DispatchStatus, TaskState

        if task.state not in (TaskState.COMPLETED, TaskState.CANCELLED):
            task.state = TaskState.BLOCKED
        task.dispatch_status = DispatchStatus.READY
        db.add(task)
        await db.flush()
        await mark_task_awaiting(
            task, "continuation_offer", db,
            prompt=(agent_reply or "")[-2000:],
        )
        await _remember(session, db, **{SNAPSHOT_WORK_TOUCHED: None})
        await emit_event(
            room_id=room.id if isinstance(room.id, UUID) else UUID(str(room.id)),
            event_type="TASK_BLOCKED",
            content=(
                f"“{task.title}”: the agent stopped before finishing (tool budget "
                "exhausted). Reply to continue from the same branch."
            ),
            db=db,
            task_id=task.id,
            metadata={"task_id": str(task.id), "reason": "tool_budget_exhausted"},
        )
        logger.info(
            "session_work: session %s (%s) → task %s parked BLOCKED (incomplete)",
            session.id, reason, task.id,
        )
        return {"state": task.state, "incomplete": True, "merged": False}

    try:
        if cfg.direct_session_mode == "branch_only":
            result = await _merge_without_gate(room, task, db)
        else:
            result = await finalize_task_gate1(
                room, task, db,
                emit_skipped=False,
                auto_merge_normal=cfg.direct_session_auto_merge == "normal_too",
            )
    finally:
        # Release the claim taken when the task was recorded, whatever the
        # outcome — a task nobody is running must not read as "already
        # dispatched" on the board.
        try:
            from app.models.task import DispatchStatus

            task.dispatch_status = DispatchStatus.READY
            db.add(task)
            await db.flush()
        except Exception as exc:
            logger.warning(
                "session_work: could not release dispatch claim on task %s: %s", task.id, exc
            )
        await _remember(session, db, **{SNAPSHOT_WORK_TOUCHED: None})

    # waiting_reply_safeguard S1: a task left in review out of a dialogue is
    # waiting for the user's word. Flag it (and keep the agent's closing note)
    # so a bare "finish" / "wrap it up" routes back to this task.
    try:
        _final_state = str(result.get("state") or "")
        _final_state = getattr(result.get("state"), "value", _final_state)
        if _final_state == "waiting_review":
            from app.core.waiting_reply import mark_task_awaiting

            _reply = (agent_reply or "").strip()
            _offer = bool(
                _reply
                and (_reply.rstrip().endswith("?") or "remaining" in _reply.lower()
                     or "would you like" in _reply.lower())
            )
            await mark_task_awaiting(
                task,
                "continuation_offer" if _offer else "review",
                db,
                prompt=_reply[-2000:] if _reply else None,
            )
    except Exception as exc:
        logger.warning(
            "session_work: could not flag task %s as awaiting user: %s", task.id, exc
        )

    logger.info(
        "session_work: finalized session %s (%s) → task %s state=%s merged=%s",
        session.id, reason, task.id, result.get("state"), result.get("merged"),
    )
    return result


# ── sweeping what a crash left behind (Stage 4) ──────────────────────────────

_SWEEP_MAX_AGE_DAYS = 7
_SWEEP_KINDS = ("user_agent_p1", "user_agent_p2")


async def sweep_orphan_session_work(room_id, db, *, max_age_days: int = _SWEEP_MAX_AGE_DAYS) -> int:
    """Recover dialogue work whose finalization never ran to completion.

    Every close path — turn end, ``close_session``, rotation, idle,
    ``orphaned_restart`` — ends in :func:`finalize_session_work`, which always
    clears ``SNAPSHOT_WORK_TOUCHED`` once it runs to completion: either it did
    something, or it cleanly found nothing to record. If that flag is still
    set on a session that is already *closed*, the one attempt that should
    have cleared it never finished — the backend restarted mid-call, or
    workspaced was unreachable when the close path tried. What is left behind
    is exactly what §8.4 of the plan worries about: a branch nobody adopted,
    or a task still holding the ``DISPATCHED`` claim `finalize_session_work`
    took when it created it (the same shape as the proxy_session dispatch
    incident this plan explicitly avoids repeating).

    Called lazily whenever the next direct session opens in the room — one
    query over recently closed dialogue sessions, filtered against the
    snapshot already sitting on each row. No branch listing: the snapshot,
    not git, is the source of truth for "is there unfinished work here".
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy.future import select

    from app.models.session.base import Session

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    candidates = (
        await db.execute(
            select(Session).where(
                Session.room_id == room_id,
                Session.kind.in_(_SWEEP_KINDS),
                Session.closed_at.isnot(None),
                Session.closed_at >= cutoff,
            )
        )
    ).scalars().all()

    swept = 0
    for orphan in candidates:
        if not _snapshot(orphan).get(SNAPSHOT_WORK_TOUCHED):
            continue
        try:
            result = await finalize_session_work(orphan, db, reason="sweep")
        except Exception as exc:
            logger.warning(
                "session_work: sweep could not finalize session %s: %s", orphan.id, exc,
            )
            continue
        if result is not None:
            swept += 1
            logger.info("session_work: sweep recovered orphaned work from session %s", orphan.id)
    return swept
