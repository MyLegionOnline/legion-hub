"""Room-level git branching model for the agentic CI/CD workflow.

internal design notes **Stage 0 — foundation only.** This module owns:

* the flow-config accessors (`working_branch` / `ci_provider` /
  `promote_policy`), read from ``Room.git_config`` with defaults;
* the base-branch guarantee (`main` + `dev`) for a room repo, including a
  best-effort startup backfill for repos that predate this stage.

**Stage 1 — task branches now target ``working_branch``.** ``ensure_task_branch``
/ ``ensure_goal_branch`` / ``merge_task_branch`` are wired into
``proxy_session`` (New Chat Flow) and ``chat_master`` (simple agent loop):
a task is developed on ``task/<id>`` branched off ``dev`` (or a goal
branch) and merged back there on completion. ``main`` no longer moves
automatically — Gate 2 (``dev → main``) is a later stage.

**Stage 2 — Gate 1 pre-merge checks.** ``run_task_checks`` runs the room's
``quick`` checks (declared in ``.legion/run.json``) on the task branch
*before* a human review. Red checks park the task in ``BLOCKED`` and skip
the merge; green checks record ``task.ci_status="passed"`` and the flow
continues. A room with no checks is unaffected (``CI_SKIPPED``).

**Stage 3 — risk classification + trivial auto-merge.** ``classify_task_risk``
scores the task diff vs its base deterministically (``trivial`` / ``normal`` /
``risky``); a router ``risk_hint`` may only raise it. Once Gate 1 is green,
``chat_master`` auto-merges a ``trivial`` task straight into ``dev``
(``TASK_AUTO_MERGED``) and leaves everything else in ``WAITING_REVIEW`` for a
human. ``revert_task_merge`` is the one-click undo for an auto-merge.

**Stage 4 — Gate 2 (``dev → main``).** ``open_promotion`` bundles every task
merged into ``dev`` since the last promotion, diffs the bundle against ``main``,
runs the full check suite on ``dev`` and parks a ``Promotion`` in
``waiting_review`` (published as ``CHECKPOINT_REQUEST``). ``resolve_promotion``
is the human decision: ``approve`` does a clean ``merge --no-ff dev`` into
``main`` (``PROMOTED_TO_MAIN``; a real conflict escalates as ``GIT_CONFLICT``
and ``main`` is untouched), ``reject`` closes the gate and leaves ``dev`` alone.
``main`` never moves any other way (§0.1). ``maybe_open_promotion`` is the
``promote_policy`` hook fired after each task completes.

**Stage 6 — push the working branch for external CI.** When a room runs
``ci_provider == "github_actions"``, ``open_promotion`` pushes ``dev`` to the
remote (plain ``push``, never ``main``) so GitHub Actions can run on it, and
records the outcome in ``Promotion.dev_pushed``. Best-effort — a failed push
never blocks the gate.

**Stage 7 — PR + external CI status (variant A).** When ``ci_provider ==
"github_actions"``, ``open_promotion`` skips the in-container full suite: it
pushes ``dev`` (Stage 6) and opens a real ``dev → main`` pull request
(``Promotion.pr_url``) so the user's GitHub Actions run. ``ci_status`` stays
pending until ``apply_external_ci_status`` (fed by the ``workflow_run`` /
``check_suite`` webhook) stamps it. On approve, ``resolve_promotion`` merges the
PR through the forge API (``git_forge``) and fetches ``main`` back onto the
volume, instead of a local merge.

**Stage 9 — deploy hook after ``main``.** Once ``resolve_promotion`` approves a
promotion (``PROMOTED_TO_MAIN``), ``dispatch_deploy_hook`` fires the room's
post-promotion hook: when ``git_config.deploy_hook_url`` is set it POSTs a
JSON payload listing the promoted commits, signed with an HMAC-SHA256 of the
raw body keyed by ``git_config.deploy_hook_secret`` (``X-Legion-Signature-256``);
when ``git_config.deploy_push_main`` is true it also pushes ``main`` to the
remote so a push-triggered pipeline runs. Best-effort — a failed hook emits
``DEPLOY_HOOK_FAILED`` and never rolls back the promotion.

**Stage 5 — whole goal → ``dev``.** A goal developed on its own branch does not
complete the moment its last task lands. ``open_goal_review`` runs the full
check suite on ``goal/<id>`` and parks the goal in ``review_state="waiting_review"``
(published as ``CHECKPOINT_REQUEST`` with the ``dev..goal/<id>`` bundle diff).
``resolve_goal_review`` is the human decision: ``approve`` does a clean
``merge --no-ff goal/<id>`` into ``dev`` (no ``-X theirs``; a conflict escalates
as ``GIT_CONFLICT`` and ``dev`` is untouched) and then completes the goal;
``reject`` sets ``review_state="rejected"`` and the goal stays ``ACTIVE`` so
agents can iterate. ``maybe_open_goal_review`` is the hook ``maybe_complete_goal``
calls before it would otherwise finalize a goal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

RELEASE_BRANCH = "main"
DEFAULT_WORKING_BRANCH = "dev"
DEFAULT_CI_PROVIDER = "workspace"      # "none" | "workspace" | "github_actions"
DEFAULT_PROMOTE_POLICY = "manual"     # "manual" | "on_each_task" | "scheduled"

_CI_PROVIDERS = {"none", "workspace", "github_actions"}
_PROMOTE_POLICIES = {"manual", "on_each_task", "scheduled"}

# internal design notes Stage 0 — how work done inside a direct
# user ↔ agent dialogue reaches ``dev``:
#   "gate1"       — branch + task + the full Gate 1 (checks, risk, auto-merge
#                   or review card); same rules a dispatched task gets;
#   "branch_only" — branch + task, merged straight into the working branch
#                   without checks or risk (rooms whose suite is too slow to
#                   run after every writing turn);
#   "off"         — the pre-plan behaviour: the dialogue writes on the working
#                   branch and no task is recorded.
DEFAULT_DIRECT_SESSION_MODE = "gate1"
# Which risk classes an auto-merge may cover for those tasks. "risk" is the
# same rule dispatched tasks follow (``trivial`` only). "normal_too" also folds
# ``normal`` in — the user was in the conversation and already saw the work.
# ``risky`` is never auto-merged under either value.
DEFAULT_DIRECT_SESSION_AUTO_MERGE = "risk"

_DIRECT_SESSION_MODES = {"gate1", "branch_only", "off"}
_DIRECT_SESSION_AUTO_MERGE = {"risk", "normal_too"}


@dataclass(frozen=True)
class FlowConfig:
    """Resolved CI/CD flow settings for a room."""

    working_branch: str
    release_branch: str
    ci_provider: str
    promote_policy: str
    direct_session_mode: str = DEFAULT_DIRECT_SESSION_MODE
    direct_session_auto_merge: str = DEFAULT_DIRECT_SESSION_AUTO_MERGE


def get_flow_config(room: Any) -> FlowConfig:
    """Read the room's CI/CD flow settings from ``git_config``, applying
    defaults and clamping unknown values back to the default.

    Accepts a ``Room`` instance or anything with a ``git_config`` mapping.
    """
    raw = getattr(room, "git_config", None)
    cfg: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    working_branch = str(cfg.get("working_branch") or "").strip() or DEFAULT_WORKING_BRANCH

    ci_provider = str(cfg.get("ci_provider") or "").strip() or DEFAULT_CI_PROVIDER
    if ci_provider not in _CI_PROVIDERS:
        logger.warning("room_git_flow: unknown ci_provider %r, using default", ci_provider)
        ci_provider = DEFAULT_CI_PROVIDER

    promote_policy = str(cfg.get("promote_policy") or "").strip() or DEFAULT_PROMOTE_POLICY
    if promote_policy not in _PROMOTE_POLICIES:
        logger.warning("room_git_flow: unknown promote_policy %r, using default", promote_policy)
        promote_policy = DEFAULT_PROMOTE_POLICY

    direct_mode = (
        str(cfg.get("direct_session_mode") or "").strip() or DEFAULT_DIRECT_SESSION_MODE
    )
    if direct_mode not in _DIRECT_SESSION_MODES:
        logger.warning(
            "room_git_flow: unknown direct_session_mode %r, using default", direct_mode
        )
        direct_mode = DEFAULT_DIRECT_SESSION_MODE

    direct_auto_merge = (
        str(cfg.get("direct_session_auto_merge") or "").strip()
        or DEFAULT_DIRECT_SESSION_AUTO_MERGE
    )
    if direct_auto_merge not in _DIRECT_SESSION_AUTO_MERGE:
        logger.warning(
            "room_git_flow: unknown direct_session_auto_merge %r, using default",
            direct_auto_merge,
        )
        direct_auto_merge = DEFAULT_DIRECT_SESSION_AUTO_MERGE

    return FlowConfig(
        working_branch=working_branch,
        release_branch=RELEASE_BRANCH,
        ci_provider=ci_provider,
        promote_policy=promote_policy,
        direct_session_mode=direct_mode,
        direct_session_auto_merge=direct_auto_merge,
    )


# ── Base branches ────────────────────────────────────────────────────────────

async def ensure_base_branches(room_id: "UUID | str") -> dict[str, Any]:
    """Guarantee the room repo has both ``main`` and ``dev`` (idempotent).

    Thin wrapper over ``workspaced``; safe to call whether or not the repo
    already has the branches. Never changes HEAD.
    """
    from app.core import workspaced_client as wsd

    return await wsd.ensure_base_branches(str(room_id))


def is_non_fast_forward(errors: "list[str]") -> bool:
    """Whether a push failed because the remote moved (Pull, then Push helps).

    Only git's own wording for a diverged branch counts: ``! [rejected] …
    (non-fast-forward)`` / ``(fetch first)``. A ``! [remote rejected] …`` is the
    *server* refusing the update (GitHub ``Internal Server Error``, a protected
    branch, a failing hook, secret scanning) and Pulling cannot fix it — the old
    bare ``"rejected"`` match mislabelled all of those as "remote moved".
    """
    blob = " ".join(errors or []).lower()
    return any(
        marker in blob
        for marker in (
            "non-fast-forward",
            "fetch first",
            "tip of your current branch is behind",
            "! [rejected]",
        )
    )


async def backfill_local_excludes(room_id: "UUID | str") -> dict[str, Any]:
    """Backfill ``.git/info/exclude`` for a room repo that predates
    internal design notes Stage 1 (idempotent, no history change).

    Thin wrapper over ``workspaced``. Returns ``{ok, updated}``.
    """
    from app.core import workspaced_client as wsd

    return await wsd.backfill_local_excludes(str(room_id))


async def backfill_userfiles(engine) -> None:
    """Ensure ``/.userfiles`` exists for every existing room.

    Unlike the Git branch/exclude backfill below, this intentionally includes
    rooms without a configured remote.  The workspaced endpoint is idempotent
    and creates the directory even when no repository exists yet.
    """
    from sqlalchemy.future import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.models.room import Room

    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as db:
            rooms = (await db.execute(select(Room))).scalars().all()
            created_or_ready = 0
            for room in rooms:
                try:
                    result = await backfill_local_excludes(room.id)
                    if result.get("ok") or result.get("error") == "no git repo":
                        created_or_ready += 1
                    if result.get("userfiles_protected") is not True:
                        # A 200 from a workspaced that predates the Git
                        # exclusion looks like success but protects nothing.
                        logger.error(
                            "backfill_userfiles: workspaced did not confirm Git "
                            "exclusion of .userfiles for room %s — it is probably "
                            "running older code; restart it. Private uploads are "
                            "refused until it reports userfiles_protected.",
                            room.id,
                        )

                    from app.core.userfiles_policy import (
                        apply_userfiles_policy,
                        emit_userfiles_ready,
                    )

                    policy = await apply_userfiles_policy(
                        room,
                        result,
                        db,
                        source="startup_backfill",
                    )
                    if (
                        not policy["conflict"]
                        and (result.get("userfiles_created") or result.get("updated"))
                    ):
                        await emit_userfiles_ready(
                            room.id,
                            db,
                            source="startup_backfill",
                            size=0,
                            error=None,
                        )
                except Exception as exc:
                    logger.warning("backfill_userfiles: room %s skipped: %s", room.id, exc)
    except Exception:
        logger.exception("backfill_userfiles: could not list rooms")
        return

    if created_or_ready:
        logger.info("backfill_userfiles: ensured private directory for %d room(s)", created_or_ready)


async def backfill_dev_branches(engine) -> None:
    """Startup backfill for every room that already has a repo bound:

    * create ``dev`` for repos whose ``git_init`` predates internal design notes
      Stage 0;
    * write the local excludes (project ignores + ``.legion/git.lock``) into
      ``.git/info/exclude`` for repos that predate internal design notes
      Stage 1 — done in the same pass (internal design notes §3.2).

    Best-effort and non-fatal — a room whose ``workspaced`` is unreachable
    is simply retried on the next boot. Mirrors ``skill_backfill``.
    """
    from sqlalchemy.future import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.models.room import Room

    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as db:
            rows = (await db.execute(select(Room.id, Room.git_config))).all()
    except Exception:
        logger.exception("backfill_dev_branches: could not list rooms")
        return

    room_ids = [
        rid
        for (rid, git_config) in rows
        if isinstance(git_config, Mapping) and git_config.get("repo_url")
    ]

    created = 0
    excludes_updated = 0
    for rid in room_ids:
        try:
            result = await ensure_base_branches(rid)
            if result.get("created_dev"):
                created += 1
        except Exception as exc:
            logger.warning("backfill_dev_branches: room %s skipped: %s", rid, exc)
        try:
            ex = await backfill_local_excludes(rid)
            if ex.get("updated"):
                excludes_updated += 1
        except Exception as exc:
            logger.warning(
                "backfill_dev_branches: room %s local-excludes skipped: %s", rid, exc
            )

    if created:
        logger.info("backfill_dev_branches: created `dev` for %d room(s)", created)
    if excludes_updated:
        logger.info(
            "backfill_dev_branches: backfilled local excludes for %d room(s)",
            excludes_updated,
        )


# ── Task / goal branches (Stage 1) ───────────────────────────────────────────

TASK_BRANCH_PREFIX = "task/"
GOAL_BRANCH_PREFIX = "goal/"
# internal design notes Stage 1 — a direct user ↔ agent dialogue holds
# its work here until there is enough of it to be worth a task. The branch is
# renamed to ``task/<id>`` when that task is written (Stage 2), so this prefix
# only ever names work that has not been adopted yet.
SESSION_BRANCH_PREFIX = "session/"


def _slug(text: str) -> str:
    """A short, git-ref-safe slug. Empty when ``text`` has no ASCII word
    characters (e.g. a Cyrillic title) — callers fall back to the bare id."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower().strip())
    return s.strip("-")[:32]


def task_branch_name(task_id: "UUID | str") -> str:
    return f"{TASK_BRANCH_PREFIX}{task_id}"


def goal_branch_name(goal_id: "UUID | str", title: str = "") -> str:
    slug = _slug(title)
    return f"{GOAL_BRANCH_PREFIX}{goal_id}-{slug}" if slug else f"{GOAL_BRANCH_PREFIX}{goal_id}"


def session_branch_name(session_id: "UUID | str") -> str:
    return f"{SESSION_BRANCH_PREFIX}{session_id}"


async def _working_branch(room_id: "UUID | str", db) -> str:
    """Resolve the room's agent target branch (``git_config.working_branch``)."""
    from sqlalchemy.future import select

    from app.models.room import Room

    room = (
        await db.execute(select(Room).where(Room.id == room_id))
    ).scalar_one_or_none()
    return get_flow_config(room).working_branch if room is not None else DEFAULT_WORKING_BRANCH


async def _load_room(room_id: "UUID | str", db):
    from sqlalchemy.future import select

    from app.models.room import Room

    return (
        await db.execute(select(Room).where(Room.id == room_id))
    ).scalar_one_or_none()


async def _load_goal(goal_id: "UUID | str", db):
    from sqlalchemy.future import select

    from app.models.goal import RoomGoal

    return (
        await db.execute(select(RoomGoal).where(RoomGoal.id == goal_id))
    ).scalar_one_or_none()


async def _task_base_branch(room_id: "UUID | str", task, db) -> str:
    """The branch a task branches off / merges back into — its goal branch
    when it has one, otherwise the room's ``working_branch``."""
    target = await _working_branch(room_id, db)
    if getattr(task, "goal_id", None):
        goal = await _load_goal(task.goal_id, db)
        if goal is not None and goal.branch_name:
            target = goal.branch_name
    return target


async def ensure_goal_branch(room_id: "UUID | str", goal, db) -> str:
    """Guarantee ``goal/<id>-<slug>`` exists, branched off ``working_branch``.

    Idempotent: returns the recorded name if the goal already has one; the
    underlying ``workspaced`` branch call also no-ops when the ref exists.
    Persists ``goal.branch_name`` (flush only — the caller commits).
    """
    from app.core import workspaced_client as wsd

    if goal.branch_name:
        return goal.branch_name

    base = await _working_branch(room_id, db)
    name = goal_branch_name(goal.id, getattr(goal, "title", "") or "")
    await wsd.create_workspace_branch(str(room_id), name, base=base)

    goal.branch_name = name
    db.add(goal)
    await db.flush()
    logger.info("ensure_goal_branch: room %s goal %s → %s (base %s)", room_id, goal.id, name, base)
    return name


async def ensure_task_branch(room_id: "UUID | str", task, db) -> str:
    """Create ``task/<id>`` for ``task`` and check it out.

    Base is the goal branch when ``task.goal_id`` is set (created on demand),
    otherwise the room's ``working_branch`` (``dev`` by default). Persists
    ``task.branch_name`` (flush only). Returns the branch name.
    """
    from app.core import workspaced_client as wsd

    branch = task_branch_name(task.id)
    base = await _working_branch(room_id, db)

    if getattr(task, "goal_id", None):
        goal = await _load_goal(task.goal_id, db)
        if goal is not None:
            base = await ensure_goal_branch(room_id, goal, db)

    await wsd.create_workspace_branch(str(room_id), branch, base=base)

    if getattr(task, "branch_name", None) != branch:
        task.branch_name = branch
        db.add(task)
        await db.flush()

    logger.info("ensure_task_branch: room %s task %s → %s (base %s)", room_id, task.id, branch, base)
    return branch


# A task in one of these states is still being worked on, so its branch is
# "live": something is holding it checked out and may be mid-edit.
_LIVE_TASK_STATES = frozenset({"assigned", "in_progress", "pausing", "resuming", "continued"})


async def _live_task_owns_head(room_id: "UUID | str", branch: str | None, db) -> bool:
    """True when ``branch`` is the branch of a task that is still running.

    Switching away from it would pull the working tree out from under a live
    agent mid-edit — worse than leaving HEAD where it is.
    """
    if not branch or not branch.startswith(TASK_BRANCH_PREFIX):
        return False

    raw = branch[len(TASK_BRANCH_PREFIX):]
    try:
        task_id = UUID(raw)
    except ValueError:
        return False

    from sqlalchemy.future import select

    from app.models.task import Task

    task = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
    if task is None:
        return False
    state = getattr(task.state, "value", task.state)
    return str(state) in _LIVE_TASK_STATES


async def _live_session_owns_head(
    room_id: "UUID | str", branch: str | None, db, *, exclude_session_id=None
) -> bool:
    """True when ``branch`` is the work branch of a still-open dialogue.

    internal design notes Stage 1: a dialogue that has started writing
    owns the working tree exactly the way a live task does, so a *different*
    direct session must not pull HEAD away from it. ``exclude_session_id`` is
    the caller's own session — its own branch is never "somebody else's".
    """
    if not branch or not branch.startswith(SESSION_BRANCH_PREFIX):
        return False

    raw = branch[len(SESSION_BRANCH_PREFIX):]
    try:
        session_id = UUID(raw)
    except ValueError:
        return False
    if exclude_session_id is not None and str(exclude_session_id) == str(session_id):
        return False

    from sqlalchemy.future import select

    from app.models.session.base import Session as SessionRow

    row = (
        await db.execute(select(SessionRow).where(SessionRow.id == session_id))
    ).scalar_one_or_none()
    return row is not None and row.closed_at is None


async def _current_branch(room_id: "UUID | str") -> str | None:
    """The workspace's checked-out branch (``git log`` reports it for free)."""
    from app.core import workspaced_client as wsd

    try:
        return (await wsd.git_log_workspace(str(room_id), limit=1)).get("branch")
    except Exception as exc:
        logger.warning("_current_branch: room %s unavailable: %s", room_id, exc)
        return None


async def ensure_session_branch(room_id: "UUID | str", session, db) -> str | None:
    """Pin the workspace HEAD for a direct user ↔ agent session.

    A direct session (an ``@nickname`` mention, or a chat with ChatMaster) never
    went through task dispatch, so nothing ever put HEAD on a known branch: the
    agent's tool writes auto-commit onto whatever branch was checked out last —
    typically ``dev``, but a stale ``task/<id>`` whenever the previous dispatch
    left one checked out. This pins the branch explicitly before the agent gets
    its workspace tools:

    * a session attached to a task (``user_agent_p1``) → that task's branch, so
      the work merges back through the task's normal Gate 1 path;
    * a session that already opened a task of its own out of the dialogue
      (internal design notes) and has not seen it merged yet → that
      task's branch, so the next turn keeps building on the same work;
    * a session that already has a work branch of its own (``session/<id>``,
      created just-in-time before its first writing tool) → that branch. The
      branch is never *created* here: an all-reading dialogue must not leave
      branches behind;
    * every other direct session → the room's ``working_branch``.

    Two cases are left alone. A *live* task branch: a running task holds the
    single working tree, and yanking it mid-edit would strand that task's work.
    And another open dialogue's work branch, for the same reason. There the
    direct session keeps writing where HEAD already is (the pre-existing
    behaviour) until the room grows a per-room git lock.

    Best-effort and idempotent: returns the branch HEAD was pinned to, or
    ``None`` when nothing was pinned (no repo, live task, failed checkout) — the
    caller carries on, since refusing the turn over a git detail would be worse.
    """
    from app.core import workspaced_client as wsd

    task_id = getattr(session, "task_id", None)
    if task_id is not None:
        from sqlalchemy.future import select

        from app.models.task import Task

        task = (
            await db.execute(select(Task).where(Task.id == task_id))
        ).scalar_one_or_none()
        if task is not None:
            return await ensure_task_branch(room_id, task, db)

    # The dialogue's own unfinished work outranks the working branch: its task
    # is still open (or its branch not yet adopted), so this turn continues
    # where the previous one stopped instead of starting over on `dev`.
    from app.core import session_work

    own_task = await session_work.open_dialogue_task(session, db)
    if own_task is not None:
        return await ensure_task_branch(room_id, own_task, db)

    own_branch = session_work.recorded_work_branch(session)
    if own_branch:
        current = await _current_branch(room_id)
        if current == own_branch:
            return own_branch
        result = await wsd.create_workspace_branch(str(room_id), own_branch)
        if not (isinstance(result, Mapping) and result.get("ok") is False):
            return own_branch
        logger.warning(
            "ensure_session_branch: could not return room %s session %s to %s",
            room_id, getattr(session, "id", None), own_branch,
        )

    branch = await _working_branch(room_id, db)
    current = await _current_branch(room_id)
    if current == branch:
        return branch

    if await _live_task_owns_head(room_id, current, db):
        logger.info(
            "ensure_session_branch: room %s keeps HEAD on live task branch %s "
            "(session %s not pinned to %s)",
            room_id, current, getattr(session, "id", None), branch,
        )
        return None

    if await _live_session_owns_head(
        room_id, current, db, exclude_session_id=getattr(session, "id", None)
    ):
        logger.info(
            "ensure_session_branch: room %s keeps HEAD on live dialogue branch %s "
            "(session %s not pinned to %s)",
            room_id, current, getattr(session, "id", None), branch,
        )
        return None

    result = await wsd.create_workspace_branch(str(room_id), branch)
    if isinstance(result, Mapping) and result.get("ok") is False:
        logger.warning(
            "ensure_session_branch: could not pin room %s session %s to %s",
            room_id, getattr(session, "id", None), branch,
        )
        return None
    return branch


async def commit_branch_worktree(
    room_id: "UUID | str", branch: str, db, *, reason: str, task_id=None
) -> str | None:
    """Commit any uncommitted working-tree changes onto *branch*.

    ``write_file`` auto-commits, but ``bash`` (and anything else that writes
    straight to the volume) does not. Without this, work done only through
    ``bash`` never enters the branch and is invisible to Gate 1 checks, risk
    classification, and the promotion flow.

    Only commits when the workspace HEAD is already on *branch*, so it can
    never strand changes on the wrong one. Best-effort: no repo, a detached /
    mismatched HEAD, or a transport error just returns ``None``. Emits
    ``FILE_CHANGED`` when a commit was actually created.
    """
    from app.core import workspaced_client as wsd

    if not branch:
        return None
    current = await _current_branch(room_id)
    if current != branch:
        return None

    try:
        res = await wsd.git_commit_workspace(str(room_id), f"agent (bash): {reason}")
    except Exception as exc:
        logger.warning("commit_branch_worktree: commit failed on %s: %s", branch, exc)
        return None

    sha = res.get("sha") if isinstance(res, Mapping) else None
    if not sha:
        return None

    logger.info("commit_branch_worktree: committed pending tree on %s (%s)", branch, sha[:8])
    try:
        from app.core.events import emit_event

        rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
        await emit_event(
            room_id=rid,
            event_type="FILE_CHANGED",
            content=f"Uncommitted working-tree changes saved to {branch}",
            db=db,
            task_id=task_id,
            metadata={"source": "auto_commit", "branch": branch, "sha": sha},
        )
    except Exception as exc:
        logger.warning("commit_branch_worktree: FILE_CHANGED emit failed on %s: %s", branch, exc)
    return sha


async def commit_working_tree(room_id: "UUID | str", task, db, *, reason: str) -> str | None:
    """``commit_branch_worktree`` for the branch of *task* (its usual caller)."""
    branch = getattr(task, "branch_name", None) or task_branch_name(task.id)
    return await commit_branch_worktree(
        room_id, branch, db, reason=reason, task_id=task.id
    )


async def merge_task_branch(room_id: "UUID | str", task, db) -> dict[str, Any]:
    """Merge the task branch into its base (goal branch or ``working_branch``).

    First refreshes the task branch against its base with a plain 3-way merge —
    the base has usually moved since the branch was cut, and the ``-X theirs``
    merge below would otherwise silently drop whatever landed in between. Then
    folds the branch back with ``-X theirs`` (last writer wins — acceptable for
    increments of one theme, §2.4). A genuine conflict at either step emits
    ``GIT_CONFLICT`` and is returned as ``conflict=True`` without moving the
    base branch.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    branch = getattr(task, "branch_name", None) or task_branch_name(task.id)
    target = await _task_base_branch(room_id, task, db)

    async def _emit_conflict(content: str, *, stage: str | None = None) -> None:
        rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
        meta = {"branch": branch, "target": target}
        if stage:
            meta["stage"] = stage
        await emit_event(
            room_id=rid,
            event_type="GIT_CONFLICT",
            content=content,
            db=db,
            task_id=task.id,
            metadata=meta,
        )

    # Fold in any bash-only work still sitting uncommitted on the task branch
    # before the merge reads it.
    await commit_working_tree(room_id, task, db, reason=f"finalize {branch} before merge")

    # Refresh the task branch against its base *before* merging it back. The base
    # (`dev` or a goal branch) has almost always moved since this branch was cut
    # — a sibling `trivial` task auto-merged in the meantime — and folding the
    # branch back with `-X theirs` (below) would silently take the branch's stale
    # copy of every file it happens to touch, discarding whatever landed on the
    # base in between. Merge the base *into* the branch first with a plain 3-way
    # merge: a genuine overlap now surfaces as a conflict for a human instead of
    # being dropped without a trace.
    refresh = await wsd.merge_workspace_branch(
        str(room_id),
        target,
        message=f"merge {target} into {branch} before Gate 1 merge",
        target=branch,
        theirs=False,
    )
    if refresh.get("conflict"):
        # internal design notes Stage 0 — mark the task so the "Divergence
        # review" button / card is offered. Flush only; the caller commits
        # (emit_event above already committed the GIT_CONFLICT row).
        if getattr(task, "divergence_state", None) not in ("reviewing", "reviewed"):
            task.divergence_state = "detected"
            db.add(task)
            await db.flush()
        await _emit_conflict(
            f"Task branch {branch} has diverged from {target} — "
            f"resolve the conflict manually before merging",
            stage="base_refresh",
        )
        return {**refresh, "branch": branch, "target": target}
    if refresh.get("ok") is False:
        # A non-conflict failure (e.g. the base branch does not exist yet in a
        # greenfield room). Not the case this guard is for — log it and fall
        # through to the merge below, which is the pre-refresh behaviour.
        logger.warning(
            "merge_task_branch: could not refresh %s against %s for task %s: %s",
            branch, target, task.id, refresh.get("stderr") or refresh,
        )

    title = (getattr(task, "title", "") or "").strip()
    result = await wsd.merge_workspace_branch(
        str(room_id),
        branch,
        message=f"merge: {branch} — {title}" if title else f"merge: {branch}",
        target=target,
        theirs=True,
    )

    if result.get("conflict"):
        await _emit_conflict(
            f"Conflict while merging branch {branch} into {target} — resolve manually"
        )

    return {**result, "branch": branch, "target": target}


# ── Gate 1 — pre-merge checks (Stage 2) ──────────────────────────────────────

async def _archive_ci_log(room_id: "UUID | str", task, run: dict[str, Any]) -> str | None:
    """Persist the check run as JSONL to S3 via the shared log archive.

    Returns the S3 key, or None when no archive backend is configured (the
    ``NoopLogArchive`` writes 0 bytes) or the upload failed.
    """
    from app.core.log_storage import get_log_archive

    key = f"ci/{room_id}/{task.id}/{int(time.time())}.jsonl"
    rows = [
        {"kind": "ci_check", **r}
        for r in run.get("results", [])
    ] or [{"kind": "ci_check", "note": run.get("error") or "no per-check output"}]
    try:
        written = await get_log_archive().flush(key, rows)
    except Exception as exc:
        logger.warning("run_task_checks: could not archive CI log for task %s: %s", task.id, exc)
        return None
    return key if written else None


def _failure_tail(run: dict[str, Any], limit: int = 1200) -> str:
    parts: list[str] = []
    if run.get("error"):
        parts.append(str(run["error"]))
    for r in run.get("results", []):
        if r.get("exit_code", 0) != 0 or r.get("timed_out"):
            head = f"[{r.get('name')}] exit={r.get('exit_code')}"
            if r.get("timed_out"):
                head += " (timed out)"
            parts.append(f"{head}\n{(r.get('tail') or '').strip()}")
    return "\n\n".join(parts)[:limit]


async def _resolve_checks_env(room_id: "UUID | str", db, scope: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Room variables (always) plus each declared check's ``secrets`` names.

    internal design notes Stage 6 — the CI gate has no human actor
    (it runs from ``chat_master`` / the promotion flow), so secrets resolve
    under the Room owner, same reasoning as bash's system path (§2.3). A
    name is an explicit ask from the check's own declaration, so — like
    bash(secrets=[...]) — it bypasses the scope allow-list.

    Returns ``(env_for_exec, secrets_for_masking)``; the latter excludes Room
    variables so masking never touches ordinary, non-secret log text.
    On *any* failure along the way (workspace/DB unreachable, a bad ``db``
    stand-in in tests, a secret gone missing, ...) this fails all the way
    open to ``({}, {})`` rather than blocking CI on a secrets-store hiccup —
    a check that genuinely needs a value will simply fail with its own clear
    error, same as an unset env var always would.
    """
    from sqlalchemy.future import select

    from app.core import workspaced_client as wsd
    from app.core.room_config import effective_variables
    from app.core.secrets import room_secrets
    from app.core.secrets.service import SecretService
    from app.models.room import Room

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    try:
        room_vars = await effective_variables(db, rid, None)
    except Exception as exc:
        logger.warning("_resolve_checks_env: could not read Room variables for room %s: %s", room_id, exc)
        return {}, {}

    try:
        runs_cfg = await wsd.get_workspace_runs(str(room_id))
    except Exception as exc:
        logger.warning("_resolve_checks_env: could not read .legion/run.json for room %s: %s", room_id, exc)
        return room_vars, {}

    names: list[str] = []
    for chk in runs_cfg.get("checks") or []:
        if scope == "quick" and not bool(chk.get("quick", True)):
            continue
        for name in chk.get("secrets") or []:
            if name not in names:
                names.append(name)
    if not names:
        return room_vars, {}

    try:
        room = (await db.execute(select(Room).where(Room.id == rid))).scalar_one_or_none()
        if room is None:
            return room_vars, {}

        secret_service = SecretService(db)
        resolved = await room_secrets.resolve_room_env(
            db,
            service=secret_service,
            room_id=rid,
            actor_user_id=room.creator_id,
            names=names,
            scope=None,
            environment=None,
        )
    except Exception as exc:
        logger.warning("_resolve_checks_env: secret resolution failed for room %s: %s", room_id, exc)
        return room_vars, {}
    return {**room_vars, **resolved}, resolved


def _mask_check_run(run: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    """Redact resolved Room-secret values out of a checks-run result in place."""
    from app.core.secrets.room_masking import mask_room_secrets

    if not secrets:
        return run
    if run.get("error"):
        run["error"] = mask_room_secrets(str(run["error"]), secrets)
    for r in run.get("results") or []:
        if r.get("tail"):
            r["tail"] = mask_room_secrets(str(r["tail"]), secrets)
    return run


async def run_task_checks(
    room_id: "UUID | str", task, db, scope: str = "quick", *, emit_skipped: bool = True
) -> dict[str, Any]:
    """Gate 1: run the room's ``quick`` checks on the task branch.

    Called by ``chat_master`` *before* moving a task to ``WAITING_REVIEW``.
    Persists ``task.ci_status`` / ``task.ci_log_key`` (flush only — the caller
    commits) and emits ``CI_FAILED`` / ``CI_PASSED`` / ``CI_SKIPPED``.

    Returns ``{"passed": bool, "ran": bool, "results": [...], "log_key": str|None}``.
    On any transport error the checks are treated as skipped (fail-open) so a
    transient workspace problem never wedges a task.

    ``emit_skipped=False`` keeps the "no checks configured" / "workspace was
    unavailable" cases out of the chat: a task per dialogue turn
    (internal design notes) would otherwise repeat that notice after
    every edit in a room that has no ``.legion/run.json``. The verdict itself
    is unchanged — only the announcement is dropped, and it is still logged.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    branch = getattr(task, "branch_name", None)

    # Commit any bash-only work still uncommitted on the task branch so the
    # checks (and the later risk diff) see the real state, and the checkout
    # inside run_checks isn't fighting a dirty tree.
    await commit_working_tree(room_id, task, db, reason="finalize before Gate 1 checks")

    check_env, check_secrets = await _resolve_checks_env(room_id, db, scope)
    try:
        run = await wsd.run_workspace_checks(str(room_id), scope=scope, ref=branch, env=check_env or None)
        run = _mask_check_run(run, check_secrets)
    except Exception as exc:
        logger.warning("run_task_checks: checks did not run for task %s: %s", task.id, exc)
        if emit_skipped:
            await emit_event(
                room_id=rid,
                event_type="CI_SKIPPED",
                content=f"Checks were not run for '{getattr(task, 'title', '')}' — {exc}",
                db=db,
                task_id=task.id,
                metadata={"reason": "workspace_unavailable"},
            )
        return {"passed": True, "ran": False, "results": [], "log_key": None}

    if not run.get("ran"):
        task.ci_status = None
        db.add(task)
        await db.flush()
        if emit_skipped:
            await emit_event(
                room_id=rid,
                event_type="CI_SKIPPED",
                content=f"No checks are defined for '{getattr(task, 'title', '')}' in .legion/run.json",
                db=db,
                task_id=task.id,
                metadata={"scope": scope},
            )
        else:
            logger.info(
                "run_task_checks: room %s has no checks configured — task %s skipped quietly",
                room_id, task.id,
            )
        return {"passed": True, "ran": False, "results": [], "log_key": None}

    log_key = await _archive_ci_log(room_id, task, run)
    passed = bool(run.get("passed"))
    task.ci_status = "passed" if passed else "failed"
    task.ci_log_key = log_key
    db.add(task)
    await db.flush()

    if passed:
        await emit_event(
            room_id=rid,
            event_type="CI_PASSED",
            content=f"Checks passed: '{getattr(task, 'title', '')}'",
            db=db,
            task_id=task.id,
            metadata={"scope": scope, "checks": [r.get("name") for r in run.get("results", [])]},
        )
    else:
        await emit_event(
            room_id=rid,
            event_type="CI_FAILED",
            content=(
                f"Checks failed for '{getattr(task, 'title', '')}':\n"
                f"{_failure_tail(run)}"
            ),
            db=db,
            task_id=task.id,
            metadata={
                "scope": scope,
                "log_key": log_key,
                "results": [
                    {k: r.get(k) for k in ("name", "exit_code", "timed_out")}
                    for r in run.get("results", [])
                ],
            },
        )

    return {
        "passed": passed,
        "ran": True,
        "results": run.get("results", []),
        "log_key": log_key,
    }


# ── Gate 1 — risk classification + trivial auto-merge (Stage 3) ───────────────

RISK_TRIVIAL = "trivial"
RISK_NORMAL = "normal"
RISK_RISKY = "risky"
_RISK_ORDER = {RISK_TRIVIAL: 0, RISK_NORMAL: 1, RISK_RISKY: 2}

DEFAULT_RISK_MAX_FILES = 3
DEFAULT_RISK_MAX_LINES = 60

# Touching any of these makes a diff ``risky`` regardless of its size — a
# human must look at the diff, and ``trivial`` auto-merge is off the table.
DEFAULT_SENSITIVE_PATHS = [
    r"(^|/)migrations/",
    r"(^|/)alembic/",
    r"(^|/)settings[^/]*\.py$",
    r"(^|/)requirements[^/]*\.txt$",
    r"(^|/)pyproject\.toml$",
    r"(^|/)package(-lock)?\.json$",
    r"(^|/)\.github/",
    r"(^|/)Dockerfile$",
    r"(^|/)docker-compose",
]

_STAT_FILE_RE = re.compile(r"^\s*(.+?)\s+\|\s+(?:Bin\b|\d+)")
_STAT_SUMMARY_RE = re.compile(
    r"(\d+)\s+files?\s+changed"
    r"(?:,\s+(\d+)\s+insertions?\(\+\))?"
    r"(?:,\s+(\d+)\s+deletions?\(-\))?"
)


def _parse_diff_stat(stat: str) -> tuple[int, int, list[str]]:
    """Return ``(files_changed, lines_changed, paths)`` from ``git diff --stat``.

    ``paths`` may be incomplete for a very large diff (git elides the middle of
    a long list) — that diff is over the file threshold anyway, so the missing
    names never change the verdict.
    """
    paths: list[str] = []
    files = lines = 0
    for raw in (stat or "").splitlines():
        line = raw.rstrip()
        summary = _STAT_SUMMARY_RE.search(line)
        if summary:
            files = int(summary.group(1))
            lines = int(summary.group(2) or 0) + int(summary.group(3) or 0)
            continue
        m = _STAT_FILE_RE.match(line)
        if m:
            name = m.group(1).strip()
            # Rename form: "old/{a => b}/name" or "old => new" — keep both sides
            # so a rename into a sensitive path is still caught.
            paths.append(name)
    if not files and paths:
        files = len(paths)
    return files, lines, paths


def _risk_config(room) -> tuple[int, int, list[str]]:
    raw = getattr(room, "git_config", None)
    cfg: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    thresholds = cfg.get("risk_thresholds")
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    try:
        max_files = int(thresholds.get("files", DEFAULT_RISK_MAX_FILES))
    except (TypeError, ValueError):
        max_files = DEFAULT_RISK_MAX_FILES
    try:
        max_lines = int(thresholds.get("lines", DEFAULT_RISK_MAX_LINES))
    except (TypeError, ValueError):
        max_lines = DEFAULT_RISK_MAX_LINES
    sensitive = cfg.get("risk_sensitive_paths")
    if not isinstance(sensitive, list) or not sensitive:
        sensitive = DEFAULT_SENSITIVE_PATHS
    return max_files, max_lines, [str(p) for p in sensitive]


def _match_sensitive(paths: list[str], patterns: list[str]) -> str | None:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            logger.warning("classify_task_risk: bad sensitive-path regex %r", p)
    for path in paths:
        for rx in compiled:
            if rx.search(path):
                return path
    return None


async def classify_task_risk(
    room_id: "UUID | str", task, db, hint: str | None = None
) -> str:
    """Gate 1: deterministically classify the task diff vs its base branch.

    * a sensitive path (``migrations/``, ``settings*.py``, ``.github/`` …) →
      ``risky``;
    * else over the file / line thresholds → ``normal``;
    * else → ``trivial`` (eligible for auto-merge into ``dev``).

    ``hint`` (from the router LLM, §0.4) may only *raise* the class. On any
    transport error the task is treated as ``normal`` — never auto-merged, but
    not forced into a diff-mandatory review either. Persists ``task.risk``
    (flush only — the caller commits).
    """
    from app.core import workspaced_client as wsd

    room = await _load_room(room_id, db)
    max_files, max_lines, sensitive = _risk_config(room)
    base = await _task_base_branch(room_id, task, db)
    branch = getattr(task, "branch_name", None) or task_branch_name(task.id)

    try:
        resp = await wsd.git_diff_workspace(
            str(room_id), from_ref=base, to_ref=branch, stat=True, merge_base=True
        )
        stat = resp.get("diff", "") if isinstance(resp, Mapping) else str(resp)
    except Exception as exc:
        logger.warning("classify_task_risk: diff failed for task %s: %s", task.id, exc)
        task.risk = RISK_NORMAL
        db.add(task)
        await db.flush()
        return RISK_NORMAL

    files, lines, paths = _parse_diff_stat(stat)
    hit = _match_sensitive(paths, sensitive)
    if hit is not None:
        risk = RISK_RISKY
    elif files > max_files or lines > max_lines:
        risk = RISK_NORMAL
    else:
        risk = RISK_TRIVIAL

    if hint in _RISK_ORDER and _RISK_ORDER[hint] > _RISK_ORDER[risk]:
        logger.info("classify_task_risk: router hint %r raised %s → %s", hint, risk, hint)
        risk = hint

    task.risk = risk
    db.add(task)
    await db.flush()
    logger.info(
        "classify_task_risk: room %s task %s → %s (files=%d lines=%d sensitive=%s)",
        room_id, task.id, risk, files, lines, hit,
    )
    return risk


def auto_merge_trivial_enabled(room) -> bool:
    """``git_config.auto_merge_trivial`` — default on; only an explicit
    ``false`` disables auto-merge of ``trivial`` tasks (§0.4)."""
    raw = getattr(room, "git_config", None)
    cfg: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return cfg.get("auto_merge_trivial", True) is not False


async def _task_requirements_for_card(task, db) -> list[dict[str, Any]]:
    """internal design notes Stage 7 — best-effort requirements
    payload for a ``TASK_WAITING_REVIEW`` card. Never raises: a requirements
    lookup failure must not block the review card itself."""
    try:
        from app.core.requirements import task_requirements_payload

        return await task_requirements_payload(task, db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("finalize_task_gate1: requirements payload failed for task %s: %s", task.id, exc)
        return []


async def finalize_task_gate1(
    room,
    task,
    db,
    *,
    scope: str = "quick",
    emit_skipped: bool = True,
    auto_merge_normal: bool = False,
    risk_hint: str | None = None,
    sender_type: str | None = "master",
) -> dict[str, Any]:
    """Run Gate 1 on a finished task and route it: merge, review, or blocked.

    The whole of Gate 1 in one place — checks, risk, and the three outcomes:

    * checks red → ``BLOCKED``, nothing merged (``CI_FAILED`` comes from
      ``run_task_checks``, ``TASK_STATE_CHANGED`` from here);
    * green + ``trivial`` (or ``normal`` with *auto_merge_normal*) and the room
      allows auto-merge → the branch is folded into its base and the task is
      ``COMPLETED`` (``TASK_AUTO_MERGED``);
    * green otherwise → ``WAITING_REVIEW``, nothing merged
      (``TASK_WAITING_REVIEW``); the branch is merged when a human approves.

    This used to live inline in ``chat_master.approve_and_start``. It is shared
    now because a dialogue produces exactly the same kind of finished task
    (internal design notes) and two copies of a merge policy drift.
    ``auto_merge_normal`` is that path's only difference: the user was in the
    conversation and already saw the work. ``risky`` is never auto-merged.

    The caller keeps whatever else belongs to it (the agent's reply, session
    bookkeeping, memory). Returns
    ``{"ci_passed", "risk", "merged", "merge_commit", "state"}``.
    """
    from app.core.events import emit_event
    from app.models.task import TaskState

    room_id = getattr(room, "id", room)
    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    title = getattr(task, "title", "") or ""

    try:
        run = await run_task_checks(room_id, task, db, scope, emit_skipped=emit_skipped)
    except Exception as exc:
        logger.warning("finalize_task_gate1: checks errored for task %s: %s", task.id, exc)
        run = {"passed": True}
    ci_passed = bool(run.get("passed", True))

    risk = None
    if ci_passed:
        try:
            risk = await classify_task_risk(room_id, task, db, hint=risk_hint)
        except Exception as exc:
            logger.warning("finalize_task_gate1: risk failed for task %s: %s", task.id, exc)
            risk = RISK_NORMAL

    task.state = TaskState.WAITING_REVIEW if ci_passed else TaskState.BLOCKED
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.flush()

    result: dict[str, Any] = {
        "ci_passed": ci_passed,
        "risk": risk,
        "merged": False,
        "merge_commit": None,
        "state": task.state,
    }

    if not ci_passed:
        await emit_event(
            room_id=rid,
            event_type="TASK_STATE_CHANGED",
            content=f"Task blocked — checks failed: {title}",
            db=db,
            task_id=task.id,
            sender_type=sender_type,
            metadata={
                "task_title": title,
                "task_state": "blocked",
                "sender_type": sender_type,
                "origin": getattr(task, "origin", None),
            },
        )
        return result

    mergeable = {RISK_TRIVIAL, RISK_NORMAL} if auto_merge_normal else {RISK_TRIVIAL}
    if risk not in mergeable or not auto_merge_trivial_enabled(room):
        requirements = await _task_requirements_for_card(task, db)
        await emit_event(
            room_id=rid,
            event_type="TASK_WAITING_REVIEW",
            content=f"Task awaiting review ({risk}): {title}",
            db=db,
            task_id=task.id,
            sender_type=sender_type,
            metadata={
                "task_title": title,
                "task_state": "waiting_review",
                "risk": risk,
                "ci_status": getattr(task, "ci_status", None),
                "sender_type": sender_type,
                "origin": getattr(task, "origin", None),
                **({"requirements": requirements} if requirements else {}),
            },
        )
        return result

    # Fold the branch into its base (`dev` or the goal branch) — never `main`.
    # GIT_CONFLICT is emitted by merge_task_branch itself.
    merge_result: dict[str, Any] = {}
    try:
        merge_result = await merge_task_branch(room_id, task, db)
        if merge_result.get("ok"):
            task.merge_commit = merge_result.get("sha")
            result["merged"] = True
            result["merge_commit"] = task.merge_commit
            logger.info(
                "finalize_task_gate1: merged %s into %s for task %s",
                merge_result.get("branch"), merge_result.get("target"), task.id,
            )
    except Exception as exc:
        logger.warning("finalize_task_gate1: merge failed for task %s: %s", task.id, exc)

    if not result["merged"]:
        # The merge did not land — a divergence against the base (the pre-merge
        # refresh hit a real conflict) or another git failure. Do NOT mark the
        # task auto-merged: leave it in WAITING_REVIEW so a human resolves it.
        task.state = TaskState.WAITING_REVIEW
        task.updated_at = datetime.utcnow()
        db.add(task)
        await db.commit()
        result["state"] = task.state
        requirements = await _task_requirements_for_card(task, db)
        await emit_event(
            room_id=rid,
            event_type="TASK_WAITING_REVIEW",
            content=f"Could not auto-merge the task ({risk}) — manual review needed: {title}",
            db=db,
            task_id=task.id,
            sender_type=sender_type,
            metadata={
                "task_title": title,
                "task_state": "waiting_review",
                "risk": risk,
                "merge_failed": True,
                "conflict": bool(merge_result.get("conflict")),
                "sender_type": sender_type,
                "origin": getattr(task, "origin", None),
                **({"requirements": requirements} if requirements else {}),
            },
        )
        return result

    task.state = TaskState.COMPLETED
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.commit()
    result["state"] = task.state

    await emit_event(
        room_id=rid,
        event_type="TASK_AUTO_MERGED",
        content=f"Task auto-merged into dev ({risk}): {title}",
        db=db,
        task_id=task.id,
        sender_type=sender_type,
        metadata={
            "task_title": title,
            "task_state": "completed",
            "risk": risk,
            "auto_merged": True,
            "merge_commit": task.merge_commit,
            "revertable": bool(task.merge_commit),
            "sender_type": sender_type,
            "origin": getattr(task, "origin", None),
        },
    )

    try:
        await maybe_open_promotion(str(room_id), db)
    except Exception as exc:
        logger.warning("finalize_task_gate1: promotion hook failed for task %s: %s", task.id, exc)

    return result


async def revert_task_merge(room_id: "UUID | str", task, db) -> dict[str, Any]:
    """Undo a task's merge into its base branch with a fresh revert commit.

    The one-click escape hatch for a ``trivial`` auto-merge (§ Stage 3).
    Needs ``task.merge_commit``. Emits ``TASK_MERGE_REVERTED`` (or
    ``GIT_CONFLICT`` if the revert does not apply cleanly).
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    sha = getattr(task, "merge_commit", None)
    if not sha:
        return {"ok": False, "error": "no recorded merge commit for this task"}

    target = await _task_base_branch(room_id, task, db)
    result = await wsd.git_revert_commit_workspace(str(room_id), sha, target=target)

    if result.get("conflict"):
        await emit_event(
            room_id=rid,
            event_type="GIT_CONFLICT",
            content=f"The rollback of the merge of task '{getattr(task, 'title', '')}' did not apply cleanly — resolve manually",
            db=db,
            task_id=task.id,
            metadata={"merge_commit": sha, "target": target},
        )
        return {**result, "target": target}

    if result.get("ok"):
        task.merge_commit = None
        db.add(task)
        await db.flush()
        await emit_event(
            room_id=rid,
            event_type="TASK_MERGE_REVERTED",
            content=f"The merge of task '{getattr(task, 'title', '')}' into {target} was rolled back",
            db=db,
            task_id=task.id,
            metadata={"reverted_merge": sha, "revert_commit": result.get("sha"), "target": target},
        )

    return {**result, "target": target}


# ── Gate 2 — dev → main promotion (Stage 4) ──────────────────────────────────

PROMOTION_WAITING = "waiting_review"
PROMOTION_APPROVED = "approved"
PROMOTION_REJECTED = "rejected"
PROMOTION_CONFLICT = "conflict"

_DIFF_STAT_CAP = 8000


class PromotionError(RuntimeError):
    """Base for Gate 2 failures the API maps to a 4xx."""


class PromotionEmpty(PromotionError):
    """``dev`` has nothing ``main`` does not already have (409)."""


class PromotionAlreadyOpen(PromotionError):
    """A ``waiting_review`` promotion already exists for the room (409)."""


async def _open_promotion_row(room_id: "UUID | str", db):
    from sqlalchemy.future import select

    from app.models.promotion import Promotion

    return (
        await db.execute(
            select(Promotion)
            .where(Promotion.room_id == room_id, Promotion.state == PROMOTION_WAITING)
            .order_by(Promotion.created_at.desc())
        )
    ).scalars().first()


async def _promotable_task_ids(room_id: "UUID | str", db) -> list[str]:
    """``str(task_id)`` for every COMPLETED task merged into ``dev`` that a
    prior *approved* promotion has not already carried to ``main``."""
    from sqlalchemy.future import select

    from app.models.promotion import Promotion
    from app.models.task import Task, TaskState

    approved = (
        await db.execute(
            select(Promotion.task_ids).where(
                Promotion.room_id == room_id, Promotion.state == PROMOTION_APPROVED
            )
        )
    ).all()
    already: set[str] = set()
    for (ids,) in approved:
        for tid in ids or []:
            already.add(str(tid))

    rows = (
        await db.execute(
            select(Task.id).where(
                Task.room_id == room_id,
                Task.state == TaskState.COMPLETED,
                Task.merge_commit.isnot(None),
            )
        )
    ).all()
    return [str(tid) for (tid,) in rows if str(tid) not in already]


async def _task_titles(task_ids: list[str], db) -> list[str]:
    if not task_ids:
        return []
    from sqlalchemy.future import select

    from app.models.task import Task

    try:
        ids = [UUID(t) for t in task_ids]
    except (ValueError, TypeError):
        return []
    rows = (await db.execute(select(Task.title).where(Task.id.in_(ids)))).all()
    return [r[0] for r in rows]


async def _push_branch_best_effort(room_id: "UUID | str", room, branch: str) -> bool:
    """Push ``branch`` to the room's remote, best-effort.

    Used by Gate 2 to publish ``dev`` for an external CI (GitHub Actions,
    variant A — Stage 6) and by the Stage 9 deploy hook to publish ``main``.
    Any failure is logged and returns ``False`` — every caller proceeds
    regardless.
    """
    git_config = getattr(room, "git_config", None)
    if not isinstance(git_config, Mapping) or not git_config.get("repo_url"):
        return False
    try:
        from app.core.git_sync import push_workspace_to_git
        from app.core.secrets.runtime import legacy_secret_value

        # Best-effort token: the legacy inline ``access_token`` covers most
        # rooms; secret-store credentials need the request context and are
        # wired in Stage 7 (variant A). A failed auth just returns False —
        # the caller still proceeds.
        legacy = legacy_secret_value(git_config, "access_token")
        token = legacy.reveal() if legacy else None

        rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
        result = await push_workspace_to_git(
            room_id=rid,
            git_config=dict(git_config),
            access_token=token,
            mode="push",
            branch=branch,
        )
    except Exception as exc:
        logger.warning(
            "_push_branch_best_effort: push of %s failed for room %s: %s",
            branch, room_id, exc,
        )
        return False
    if result.get("errors"):
        logger.warning(
            "_push_branch_best_effort: push of %s for room %s reported: %s",
            branch, room_id, result["errors"],
        )
        return False
    logger.info("_push_branch_best_effort: pushed %s to remote for room %s", branch, room_id)
    return True


async def _stamp_first_pushed(room, db) -> None:
    """Record ``git_config.first_pushed_at`` so the shared-``.gitignore`` seed
    (internal design notes Stage 2) runs at most once per room."""
    cfg = dict(getattr(room, "git_config", None) or {})
    if cfg.get("first_pushed_at"):
        return
    cfg["first_pushed_at"] = datetime.now(timezone.utc).isoformat()
    room.git_config = cfg
    try:
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(room, "git_config")
    except Exception:
        pass  # a test double / non-ORM room — the plain assignment is enough
    db.add(room)
    await db.flush()


async def _maybe_seed_shared_gitignore(
    room_id: "UUID | str", room, db, cfg: FlowConfig
) -> bool:
    """internal design notes Stage 2 — commit a shared ``.gitignore`` on the
    working branch the first time Legion publishes a repo it owns.

    Runs once per room, and only when the room has a ``repo_url`` bound, was
    **not** adopted from an external remote (``adopted_at`` unset — an adopted
    repo keeps the upstream project's ``.gitignore``), and has never published
    before (``first_pushed_at`` unset).

    If the remote already carries the release branch (Legion is not the
    publisher of record) the seed is skipped but ``first_pushed_at`` is still
    stamped so the check does not repeat every promotion. On a real seed it
    stamps ``first_pushed_at`` and emits ``GITIGNORE_SEEDED``. Best-effort —
    never raises.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    git_config = getattr(room, "git_config", None)
    if not isinstance(git_config, Mapping) or not git_config.get("repo_url"):
        return False
    if git_config.get("adopted_at") or git_config.get("first_pushed_at"):
        return False

    try:
        from app.core.git_forge import remote_branch_exists, resolve_forge_token

        token = await resolve_forge_token(db, room, request=None)
        if await remote_branch_exists(git_config, token, cfg.release_branch):
            await _stamp_first_pushed(room, db)
            return False
    except Exception as exc:
        logger.warning(
            "_maybe_seed_shared_gitignore: remote check failed for room %s: %s",
            room_id, exc,
        )

    try:
        result = await wsd.git_ensure_shared_gitignore(str(room_id), cfg.working_branch)
    except Exception as exc:
        logger.warning(
            "_maybe_seed_shared_gitignore: seed failed for room %s: %s", room_id, exc
        )
        return False

    await _stamp_first_pushed(room, db)

    if result.get("added"):
        rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
        await emit_event(
            room_id=rid,
            event_type="GITIGNORE_SEEDED",
            content="A shared `.gitignore` with baseline ignore rules was added to the repository",
            db=db,
            metadata={"sha": result.get("sha"), "branch": result.get("branch")},
        )
        logger.info("_maybe_seed_shared_gitignore: room %s seeded %s", room_id, result.get("sha"))
        return True
    return False


async def _open_promotion_pr(room, db, cfg: FlowConfig, n_tasks: int) -> str | None:
    """Stage 7 — open a ``dev → main`` pull request so the user's GitHub
    Actions run on it. Best-effort: a missing token or a forge error just
    leaves ``pr_url`` empty and the gate falls back to the workspace flow.

    internal design notes §4.2 — when ``main`` was never published to
    the remote the forge would answer 422; instead of that, either seed
    ``main`` once (``git_config.seed_main_on_first_promotion``) or skip the PR
    and emit a hint telling the user to push ``main`` once.
    """
    try:
        from app.core.git_forge import (
            create_pull_request,
            remote_branch_exists,
            resolve_forge_token,
        )

        git_config = getattr(room, "git_config", None)
        if not isinstance(git_config, Mapping) or not git_config.get("repo_url"):
            return None
        token = await resolve_forge_token(db, room, request=None)

        if not await remote_branch_exists(git_config, token, cfg.release_branch):
            seed = bool(git_config.get("seed_main_on_first_promotion") or False)
            if not seed:
                await _emit_pr_needs_main(room, db, cfg)
                return None
            # The one push of ``main`` allowed outside a promotion approve:
            # it publishes a ``main`` that is already the approved base.
            await _push_branch_best_effort(
                getattr(room, "id", None), room, cfg.release_branch
            )

        result = await create_pull_request(
            git_config,
            token,
            head=cfg.working_branch,
            base=cfg.release_branch,
            title=f"Promote {cfg.working_branch} → {cfg.release_branch} ({n_tasks} tasks)",
            body="Opened by Legion Gate 2. Approving in Legion merges this PR.",
        )
    except Exception as exc:
        logger.warning("_open_promotion_pr: room %s skipped: %s", getattr(room, "id", "?"), exc)
        return None
    if result.get("ok"):
        logger.info("_open_promotion_pr: room %s PR %s", getattr(room, "id", "?"), result.get("url"))
        return result.get("url")
    logger.warning("_open_promotion_pr: room %s PR failed: %s", getattr(room, "id", "?"), result.get("error"))
    return None


async def _emit_pr_needs_main(room, db, cfg: FlowConfig) -> None:
    """Tell the room that the ``dev → main`` PR was not opened because the
    remote has no ``main`` yet (internal design notes §4.2)."""
    from app.core.events import emit_event

    rid = getattr(room, "id", None)
    if rid is None:
        return
    try:
        await emit_event(
            room_id=rid,
            event_type="PROMOTION_PR_SKIPPED",
            content=(
                f"PR {cfg.working_branch} → {cfg.release_branch} was not opened: the remote "
                f"has no `{cfg.release_branch}` branch yet. Push `{cfg.release_branch}` "
                f"once (Push to Git) or enable seed_main_on_first_promotion."
            ),
            db=db,
            metadata={
                "reason": "remote_release_branch_missing",
                "release_branch": cfg.release_branch,
                "hint": "push_main_once",
            },
        )
    except Exception:
        logger.warning(
            "_emit_pr_needs_main: failed to emit for room %s", rid, exc_info=True
        )


async def apply_external_ci_status(
    room_id: "UUID | str", head_branch: str, conclusion: str | None, db, *, source: str = "github"
) -> dict[str, Any]:
    """Stage 7 — stamp ``Promotion.ci_status`` from a forge CI webhook.

    Finds the ``waiting_review`` promotion whose ``from_branch`` matches the
    workflow's ``head_branch`` and records ``passed`` / ``failed`` from
    ``conclusion`` (``success`` → ``passed``, anything else → ``failed``).
    Emits ``CI_STATUS`` into the room chat. A no-op (``matched=False``) when no
    open promotion targets that branch.
    """
    from sqlalchemy.future import select

    from app.core.events import emit_event
    from app.models.promotion import Promotion

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    status = "passed" if str(conclusion or "").lower() == "success" else "failed"

    promo = (
        await db.execute(
            select(Promotion)
            .where(
                Promotion.room_id == rid,
                Promotion.state == PROMOTION_WAITING,
                Promotion.from_branch == head_branch,
            )
            .order_by(Promotion.created_at.desc())
        )
    ).scalars().first()
    if promo is None:
        # internal design notes Stage 4 — no open promotion, but the
        # branch may be an open Goal gate (`goal/<id>` pushed via "Publish
        # branch"). Stamp the Goal's ``review_ci_status`` so the review card
        # reflects the forge CI without a manual round-trip.
        return await _apply_external_ci_status_to_goal(
            rid, head_branch, status, conclusion, db, source=source
        )

    promo.ci_status = status
    db.add(promo)
    await db.flush()
    await emit_event(
        room_id=rid,
        event_type="CI_STATUS",
        content=(
            f"External CI ({source}) for {head_branch}: "
            + ("success" if status == "passed" else f"failure ({conclusion})")
        ),
        db=db,
        metadata={
            "promotion_id": str(promo.id),
            "ci_status": status,
            "conclusion": conclusion,
            "source": source,
        },
    )
    logger.info(
        "apply_external_ci_status: room %s promotion %s → %s (%s)",
        room_id, promo.id, status, conclusion,
    )
    return {"matched": True, "promotion_id": str(promo.id), "ci_status": status}


async def _apply_external_ci_status_to_goal(
    rid: UUID, head_branch: str, status: str, conclusion: str | None, db, *, source: str
) -> dict[str, Any]:
    """internal design notes Stage 4 — stamp ``RoomGoal.review_ci_status``
    from a forge CI webhook whose ``head_branch`` is an open Goal gate.

    A no-op (``matched=False``) unless a Goal in this room is parked in
    ``review_state="waiting_review"`` on exactly that branch.
    """
    from sqlalchemy.future import select

    from app.core.events import emit_event
    from app.models.goal import RoomGoal

    goal = (
        await db.execute(
            select(RoomGoal).where(
                RoomGoal.room_id == rid,
                RoomGoal.branch_name == head_branch,
                RoomGoal.review_state == GOAL_REVIEW_WAITING,
            )
        )
    ).scalars().first()
    if goal is None:
        return {"matched": False}

    goal.review_ci_status = status
    db.add(goal)
    await db.flush()
    await emit_event(
        room_id=rid,
        event_type="CI_STATUS",
        content=(
            f"External CI ({source}) for goal '{goal.title}' ({head_branch}): "
            + ("success" if status == "passed" else f"failure ({conclusion})")
        ),
        db=db,
        metadata={
            "goal_id": str(goal.id),
            "ci_status": status,
            "conclusion": conclusion,
            "source": source,
        },
    )
    logger.info(
        "apply_external_ci_status: room %s goal %s → %s (%s)",
        rid, goal.id, status, conclusion,
    )
    return {"matched": True, "goal_id": str(goal.id), "ci_status": status}


async def open_promotion(
    room_id: "UUID | str", db, initiated_by: "UUID | str | None" = None, *, run_checks: bool = True
):
    """Open a Gate 2 promotion for ``dev → main``.

    Bundles every COMPLETED task merged into ``dev`` since the last approved
    promotion, records the ``git diff --stat main..dev`` bundle diff, runs the
    full check suite on ``dev`` and parks a ``Promotion`` in ``waiting_review``
    (published as ``CHECKPOINT_REQUEST`` + ``PROMOTION_OPENED``).

    Raises ``PromotionAlreadyOpen`` if a gate is already open, ``PromotionEmpty``
    if there is nothing to publish, ``PromotionError`` on a transport failure.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event
    from app.models.message import CHECKPOINT_REQUEST
    from app.models.promotion import Promotion

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))

    if await _open_promotion_row(room_id, db) is not None:
        raise PromotionAlreadyOpen("a promotion is already awaiting review for this room")

    room = await _load_room(room_id, db)
    cfg = get_flow_config(room)

    try:
        resp = await wsd.git_diff_workspace(
            str(room_id), from_ref=cfg.release_branch, to_ref=cfg.working_branch, stat=True
        )
        diff_stat = resp.get("diff", "") if isinstance(resp, Mapping) else str(resp)
    except Exception as exc:
        raise PromotionError(
            f"could not diff {cfg.working_branch}..{cfg.release_branch}: {exc}"
        ) from exc

    if not (diff_stat or "").strip():
        raise PromotionEmpty(
            f"{cfg.working_branch} has nothing {cfg.release_branch} does not already have"
        )

    task_ids = await _promotable_task_ids(room_id, db)

    # internal design notes Stage 2 — a greenfield repo Legion publishes gets
    # a committed shared `.gitignore` on `dev` before it ever leaves the volume,
    # so a human cloning the published repo sees the ordinary project ignores.
    # No-op for an adopted repo or one that has already been published.
    await _maybe_seed_shared_gitignore(room_id, room, db, cfg)

    ci_status: str | None
    ci_results: list = []
    if cfg.ci_provider == "github_actions":
        # Stage 7 (variant A): the user's GitHub Actions run the suite on the
        # PR. ci_status stays pending (None) until the workflow_run webhook
        # reports back; approve is blocked until then unless forced.
        ci_status = None
    elif run_checks and cfg.ci_provider != "none":
        try:
            check_env, check_secrets = await _resolve_checks_env(room_id, db, "full")
            run = await wsd.run_workspace_checks(
                str(room_id), scope="full", ref=cfg.working_branch, env=check_env or None
            )
            run = _mask_check_run(run, check_secrets)
            if run.get("ran"):
                ci_status = "passed" if run.get("passed") else "failed"
                ci_results = run.get("results", []) or []
            else:
                # No checks declared → green, same contract as Gate 1.
                ci_status = "passed"
        except Exception as exc:
            logger.warning("open_promotion: full checks did not run for room %s: %s", room_id, exc)
            ci_status = None
    else:
        ci_status = "passed"

    promo = Promotion(
        room_id=rid,
        from_branch=cfg.working_branch,
        to_branch=cfg.release_branch,
        task_ids=task_ids,
        diff_stat=(diff_stat or "")[:_DIFF_STAT_CAP],
        ci_status=ci_status,
        ci_results=ci_results,
        state=PROMOTION_WAITING,
        initiated_by=UUID(str(initiated_by)) if initiated_by else None,
    )
    db.add(promo)
    await db.flush()

    if cfg.ci_provider == "github_actions":
        promo.dev_pushed = await _push_branch_best_effort(room_id, room, cfg.working_branch)
        if promo.dev_pushed:
            promo.pr_url = await _open_promotion_pr(room, db, cfg, len(task_ids))
        db.add(promo)
        await db.flush()

    titles = await _task_titles(task_ids, db)
    await emit_event(
        room_id=rid,
        event_type="PROMOTION_OPENED",
        content=(
            f"Gate 2: {len(task_ids)} task(s) ready for publication "
            f"{cfg.working_branch} → {cfg.release_branch}"
        ),
        db=db,
        metadata={
            "promotion_id": str(promo.id),
            "task_ids": task_ids,
            "ci_status": ci_status,
            "dev_pushed": promo.dev_pushed,
            "pr_url": promo.pr_url,
        },
    )
    await emit_event(
        room_id=rid,
        event_type=CHECKPOINT_REQUEST,
        content=f"Publish {len(task_ids)} task(s) to {cfg.release_branch}?",
        db=db,
        metadata={
            "checkpoint_kind": "confirm",
            "promotion_id": str(promo.id),
            "diff_stat": (diff_stat or "")[:_DIFF_STAT_CAP],
            "task_titles": titles,
            "ci_status": ci_status,
            "ci_results": ci_results,
            "pr_url": promo.pr_url,
        },
    )
    logger.info(
        "open_promotion: room %s promotion %s — %d task(s), ci=%s",
        room_id, promo.id, len(task_ids), ci_status,
    )
    return promo


def _pr_number(pr_url: str | None) -> int | None:
    for marker in ("/pull/", "/pulls/", "/merge_requests/"):
        if pr_url and marker in pr_url:
            tail = pr_url.split(marker, 1)[1].strip("/").split("/")[0]
            try:
                return int(tail)
            except ValueError:
                return None
    return None


async def _merge_promotion_pr(room_id, promotion, db, git_token: str | None) -> dict[str, Any]:
    """Stage 7 — approve path for a PR-backed promotion: merge the PR through
    the forge API, then fetch ``main`` back so the workspace volume catches up.
    """
    from app.core import workspaced_client as wsd
    from app.core.git_forge import merge_pull_request, resolve_forge_token
    from app.core.git_sync import fetch_workspace_from_git

    number = _pr_number(promotion.pr_url)
    if number is None:
        return {"ok": False, "error": f"cannot parse PR number from {promotion.pr_url!r}"}

    room = await _load_room(room_id, db)
    git_config = getattr(room, "git_config", None) or {}
    token = git_token or await resolve_forge_token(db, room, request=None)

    merged = await merge_pull_request(git_config, token, number)
    if not merged.get("ok"):
        return {"ok": False, "error": f"PR #{number} merge failed: {merged.get('error')}"}

    # Pull the merge commit back onto the workspace volume's ``main`` so the
    # local repo tracks the remote. Best-effort: a failed fetch just means the
    # volume is briefly behind — a manual ``sync`` recovers it.
    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    try:
        cfg = get_flow_config(room)
        fetch_config = {**git_config, "branch": cfg.release_branch}
        # Stage 1.2.3: a network fetch through workspaced.
        from app.db.release import release_db

        await release_db(db, reason="git/fetch_promotion")
        await fetch_workspace_from_git(
            room_id=rid,
            git_config=fetch_config,
            access_token=token,
            target_branch=cfg.release_branch,
        )
        await wsd.create_workspace_branch(str(room_id), cfg.working_branch)  # HEAD back on dev
    except Exception as exc:
        logger.warning("_merge_promotion_pr: room %s post-merge fetch skipped: %s", room_id, exc)

    return {"ok": True, "sha": merged.get("sha"), "pr_number": number}


# ── Stage 9 — deploy hook after main ─────────────────────────────────────────

DEPLOY_HOOK_TIMEOUT = 10.0
_DEPLOY_COMMIT_LIMIT = 100


def _deploy_hook_config(room) -> tuple[str | None, str | None, bool]:
    """``(deploy_hook_url, deploy_hook_secret, deploy_push_main)`` from
    ``git_config`` — all optional, all opt-in (the hook is a no-op unless a
    URL or ``deploy_push_main`` is set)."""
    raw = getattr(room, "git_config", None)
    cfg: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    url = str(cfg.get("deploy_hook_url") or "").strip() or None
    secret = str(cfg.get("deploy_hook_secret") or "").strip() or None
    push_main = cfg.get("deploy_push_main", False) is True
    return url, secret, push_main


async def _collect_promoted_commits(
    room_id: "UUID | str", from_branch: str, to_branch: str, db
) -> list[dict[str, Any]]:
    """The commits ``from_branch`` carries that ``to_branch`` does not yet have
    (``git log to_branch..from_branch``). Best-effort — an empty list on any
    transport error. Must be called *before* the promotion merge, while the
    range still resolves."""
    from app.core import workspaced_client as wsd

    try:
        resp = await wsd.git_log_workspace(
            str(room_id),
            rev_range=f"{to_branch}..{from_branch}",
            limit=_DEPLOY_COMMIT_LIMIT,
        )
    except Exception as exc:
        logger.warning(
            "_collect_promoted_commits: log %s..%s failed for room %s: %s",
            to_branch, from_branch, room_id, exc,
        )
        return []
    entries = resp.get("entries", []) if isinstance(resp, Mapping) else []
    return [
        {k: e.get(k) for k in ("sha", "short_sha", "message", "author", "timestamp")}
        for e in entries
        if isinstance(e, Mapping)
    ]


async def dispatch_deploy_hook(
    room_id: "UUID | str", room, promotion, commits: list[dict[str, Any]], db
) -> dict[str, Any]:
    """Stage 9 — fire the room's post-promotion deploy hook.

    Called right after ``PROMOTED_TO_MAIN``. When ``git_config.deploy_hook_url``
    is set, POSTs a signed JSON payload (HMAC-SHA256 over the raw body keyed by
    ``git_config.deploy_hook_secret``, in the ``X-Legion-Signature-256`` header)
    listing the promoted commits. When ``git_config.deploy_push_main`` is true,
    also pushes ``main`` to the remote so a push-triggered pipeline runs.

    Best-effort: every failure is logged and reported as ``DEPLOY_HOOK_FAILED``;
    it never raises and never rolls back the promotion. Returns a summary dict.
    """
    from app.core.events import emit_event

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    url, secret, push_main = _deploy_hook_config(room)
    result: dict[str, Any] = {"posted": False, "pushed_main": False}

    if push_main:
        result["pushed_main"] = await _push_branch_best_effort(room_id, room, RELEASE_BRANCH)

    if not url:
        return result

    payload = {
        "event": "promoted_to_main",
        "room_id": str(rid),
        "promotion_id": str(promotion.id),
        "from_branch": promotion.from_branch,
        "to_branch": promotion.to_branch,
        "merge_commit": getattr(promotion, "merge_commit", None),
        "task_ids": list(promotion.task_ids or []),
        "commits": commits,
        "diff_stat": getattr(promotion, "diff_stat", None),
        "pushed_main": result["pushed_main"],
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Legion-Event": "promoted_to_main"}
    if secret:
        headers["X-Legion-Signature-256"] = "sha256=" + hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

    status_code: int | None = None
    try:
        async with httpx.AsyncClient(timeout=DEPLOY_HOOK_TIMEOUT) as client:
            resp = await client.post(url, content=body, headers=headers)
        status_code = resp.status_code
        ok = 200 <= status_code < 300
    except Exception as exc:
        logger.warning("dispatch_deploy_hook: POST %s failed for room %s: %s", url, room_id, exc)
        await emit_event(
            room_id=rid,
            event_type="DEPLOY_HOOK_FAILED",
            content=f"Deploy hook failed: {exc}",
            db=db,
            metadata={"promotion_id": str(promotion.id), "url": url},
        )
        return result

    result["posted"] = ok
    result["status_code"] = status_code

    if ok:
        await emit_event(
            room_id=rid,
            event_type="DEPLOY_TRIGGERED",
            content=f"Deploy hook called ({len(commits)} commit(s))",
            db=db,
            metadata={
                "promotion_id": str(promotion.id),
                "url": url,
                "commit_count": len(commits),
                "pushed_main": result["pushed_main"],
            },
        )
    else:
        await emit_event(
            room_id=rid,
            event_type="DEPLOY_HOOK_FAILED",
            content=f"Deploy hook returned HTTP {status_code}",
            db=db,
            metadata={"promotion_id": str(promotion.id), "url": url, "status_code": status_code},
        )
    logger.info(
        "dispatch_deploy_hook: room %s promotion %s → posted=%s (%s), pushed_main=%s",
        room_id, promotion.id, result["posted"], status_code, result["pushed_main"],
    )
    return result


async def resolve_promotion(
    room_id: "UUID | str",
    promotion,
    db,
    *,
    decision: str,
    note: str | None = None,
    force: bool = False,
    resolved_by: "UUID | str | None" = None,
    git_token: str | None = None,
) -> dict[str, Any]:
    """Resolve a ``waiting_review`` promotion.

    ``decision="approve"`` → clean ``merge --no-ff dev`` into ``main`` (no
    ``-X theirs``); a conflict sets ``state="conflict"`` + ``GIT_CONFLICT`` and
    ``main`` is untouched, success sets ``state="approved"`` + ``PROMOTED_TO_MAIN``.
    ``decision="reject"`` → ``state="rejected"``, ``dev`` lives on.

    Stage 7: when the promotion is backed by a GitHub PR (``promotion.pr_url``),
    ``approve`` merges the PR through the forge API (using ``git_token``) and
    then fetches ``main`` back so the workspace volume catches up, instead of a
    local ``merge``.

    ``approve`` is refused (``blocked=True``) when ``ci_status != "passed"``
    unless ``force=True``.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))

    if promotion.state != PROMOTION_WAITING:
        return {"ok": False, "error": f"promotion already {promotion.state}", "state": promotion.state}

    resolver = UUID(str(resolved_by)) if resolved_by else None

    if decision == "reject":
        promotion.state = PROMOTION_REJECTED
        promotion.note = note
        promotion.resolved_at = datetime.utcnow()
        promotion.resolved_by = resolver
        db.add(promotion)
        await db.flush()
        await emit_event(
            room_id=rid,
            event_type="PROMOTION_REJECTED",
            content=(
                f"Publication of {promotion.from_branch} → {promotion.to_branch} rejected"
                + (f": {note}" if note else "")
            ),
            db=db,
            metadata={"promotion_id": str(promotion.id), "note": note},
        )
        return {"ok": True, "state": PROMOTION_REJECTED}

    if decision != "approve":
        return {"ok": False, "error": f"unknown decision {decision!r}"}

    if promotion.ci_status != "passed" and not force:
        return {
            "ok": False,
            "blocked": True,
            "reason": "ci_not_passed",
            "ci_status": promotion.ci_status,
        }

    n = len(promotion.task_ids or [])

    # Snapshot the promoted commits before the merge — afterwards ``main..dev``
    # resolves to nothing (Stage 9 deploy-hook payload).
    promoted_commits = await _collect_promoted_commits(
        room_id, promotion.from_branch, promotion.to_branch, db
    )

    if getattr(promotion, "pr_url", None):
        pr_out = await _merge_promotion_pr(room_id, promotion, db, git_token)
        if not pr_out.get("ok"):
            return pr_out
        merge_sha = pr_out.get("sha")
    else:
        merge = await wsd.merge_workspace_branch(
            str(room_id),
            promotion.from_branch,
            target=promotion.to_branch,
            theirs=False,
            message=f"promote: {promotion.from_branch} → {promotion.to_branch} ({n} tasks)",
        )

        if merge.get("conflict"):
            promotion.state = PROMOTION_CONFLICT
            db.add(promotion)
            await db.flush()
            await emit_event(
                room_id=rid,
                event_type="GIT_CONFLICT",
                content=(
                    f"Conflict while merging {promotion.from_branch} → {promotion.to_branch} "
                    "— resolve manually, main untouched"
                ),
                db=db,
                metadata={"promotion_id": str(promotion.id), "target": promotion.to_branch},
            )
            return {"ok": False, "conflict": True, "state": PROMOTION_CONFLICT}

        if not merge.get("ok"):
            return {"ok": False, "error": merge.get("stderr") or "merge failed"}

        # The merge left the working tree checked out on `main`, where agent /
        # auto commits are refused (§0.1). Put HEAD back on the working branch —
        # create_workspace_branch just checks it out when it already exists.
        try:
            await wsd.create_workspace_branch(str(room_id), promotion.from_branch)
        except Exception as exc:
            logger.warning(
                "resolve_promotion: could not restore HEAD to %s for room %s: %s",
                promotion.from_branch, room_id, exc,
            )
        merge_sha = merge.get("sha")

    promotion.state = PROMOTION_APPROVED
    promotion.merge_commit = merge_sha
    promotion.note = note
    promotion.resolved_at = datetime.utcnow()
    promotion.resolved_by = resolver
    db.add(promotion)
    await db.flush()
    await emit_event(
        room_id=rid,
        event_type="PROMOTED_TO_MAIN",
        content=(
            f"{promotion.from_branch} published to {promotion.to_branch} "
            f"({n} task(s))"
        ),
        db=db,
        metadata={
            "promotion_id": str(promotion.id),
            "merge_commit": promotion.merge_commit,
            "task_ids": promotion.task_ids,
        },
    )
    logger.info(
        "resolve_promotion: room %s promotion %s approved → %s (%s)",
        room_id, promotion.id, promotion.to_branch, promotion.merge_commit,
    )

    # Stage 9 — post-promotion deploy hook. Best-effort; a failure here must
    # not undo an approved promotion.
    deploy: dict[str, Any] = {}
    try:
        room = await _load_room(room_id, db)
        deploy = await dispatch_deploy_hook(room_id, room, promotion, promoted_commits, db)
    except Exception as exc:
        logger.warning("resolve_promotion: deploy hook skipped for room %s: %s", room_id, exc)

    return {
        "ok": True,
        "state": PROMOTION_APPROVED,
        "merge_commit": promotion.merge_commit,
        "deploy": deploy,
    }


async def maybe_open_promotion(
    room_id: "UUID | str", db, initiated_by: "UUID | str | None" = None
):
    """``promote_policy`` hook — called after a task reaches COMPLETED.

    Opens a Gate 2 promotion automatically only under ``on_each_task``. A no-op
    (returns ``None``) for ``manual`` / ``scheduled``, when there is nothing to
    publish, or when a gate is already open. Never raises — a Gate 2 hiccup must
    not wedge task completion.
    """
    try:
        room = await _load_room(room_id, db)
        if get_flow_config(room).promote_policy != "on_each_task":
            return None
        return await open_promotion(room_id, db, initiated_by=initiated_by)
    except PromotionError:
        return None
    except Exception as exc:
        logger.warning("maybe_open_promotion: room %s skipped: %s", room_id, exc)
        return None


# ── Stage 5 — whole goal → dev review gate ───────────────────────────────────

GOAL_REVIEW_TESTING = "testing"
GOAL_REVIEW_WAITING = "waiting_review"
GOAL_REVIEW_APPROVED = "approved"
GOAL_REVIEW_REJECTED = "rejected"

# review_state values from which the gate may be (re)opened.
_GOAL_REVIEW_OPENABLE = {None, GOAL_REVIEW_REJECTED}
# review_state values that mean "in the gate — do not finalize the goal yet".
_GOAL_REVIEW_PENDING = {GOAL_REVIEW_TESTING, GOAL_REVIEW_WAITING}


class GoalReviewError(RuntimeError):
    """Base for Stage 5 failures the API maps to a 4xx."""


async def _goal_task_titles(goal_id: "UUID | str", db) -> list[str]:
    from sqlalchemy.future import select

    from app.models.task import Task

    rows = (
        await db.execute(
            select(Task.title)
            .where(Task.goal_id == goal_id)
            .order_by(Task.created_at.asc())
        )
    ).all()
    return [r[0] for r in rows]


async def _stamp_goal_task_merges(goal_id: "UUID | str", sha: str | None, db) -> None:
    """Record the ``goal → dev`` merge sha on the goal's COMPLETED tasks that
    have no task-level merge commit yet, so Gate 2 (``dev → main``) bundles
    them like a standalone task (§ Stage 5, "on a par with micro-tasks")."""
    if not sha:
        return
    from sqlalchemy.future import select

    from app.models.task import Task, TaskState

    rows = (
        await db.execute(
            select(Task).where(
                Task.goal_id == goal_id,
                Task.state == TaskState.COMPLETED,
                Task.merge_commit.is_(None),
            )
        )
    ).scalars().all()
    for t in rows:
        t.merge_commit = sha
        db.add(t)
    if rows:
        await db.flush()


async def open_goal_review(room_id: "UUID | str", goal, db, *, run_checks: bool = True):
    """Open the Stage 5 human gate for ``goal/<id> → dev``.

    Runs the full check suite on the goal branch, records the ``dev..goal/<id>``
    bundle diff and parks the goal in ``review_state="waiting_review"``
    (published as ``GOAL_REVIEW_OPENED`` + ``CHECKPOINT_REQUEST``).

    Raises ``GoalReviewError`` when the goal has no branch or the bundle diff
    cannot be computed. Persists ``goal.review_state`` / ``goal.review_ci_status``
    (flush only — the caller commits, and ``emit_event`` commits anyway).
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event
    from app.models.message import CHECKPOINT_REQUEST

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))

    if not getattr(goal, "branch_name", None):
        raise GoalReviewError("goal has no branch to review")

    room = await _load_room(room_id, db)
    cfg = get_flow_config(room)

    goal.review_state = GOAL_REVIEW_TESTING
    db.add(goal)
    await db.flush()

    try:
        resp = await wsd.git_diff_workspace(
            str(room_id), from_ref=cfg.working_branch, to_ref=goal.branch_name, stat=True
        )
        diff_stat = resp.get("diff", "") if isinstance(resp, Mapping) else str(resp)
    except Exception as exc:
        raise GoalReviewError(
            f"could not diff {cfg.working_branch}..{goal.branch_name}: {exc}"
        ) from exc

    ci_status: str | None
    ci_results: list = []
    if run_checks and cfg.ci_provider != "none":
        try:
            check_env, check_secrets = await _resolve_checks_env(room_id, db, "full")
            run = await wsd.run_workspace_checks(
                str(room_id), scope="full", ref=goal.branch_name, env=check_env or None
            )
            run = _mask_check_run(run, check_secrets)
            if run.get("ran"):
                ci_status = "passed" if run.get("passed") else "failed"
                ci_results = run.get("results", []) or []
            else:
                ci_status = "passed"  # no checks declared → green, Gate 1 contract
        except Exception as exc:
            logger.warning(
                "open_goal_review: full checks did not run for goal %s: %s", goal.id, exc
            )
            ci_status = None
    else:
        ci_status = "passed"

    goal.review_state = GOAL_REVIEW_WAITING
    goal.review_ci_status = ci_status
    db.add(goal)
    await db.flush()

    titles = await _goal_task_titles(goal.id, db)
    _gc = getattr(room, "git_config", None)
    can_publish = bool(isinstance(_gc, Mapping) and _gc.get("repo_url"))
    try:
        from app.core.requirements import goal_coverage_matrix

        coverage_matrix = await goal_coverage_matrix(goal.id, db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("open_goal_review: coverage matrix failed for goal %s: %s", goal.id, exc)
        coverage_matrix = []
    await emit_event(
        room_id=rid,
        event_type="GOAL_REVIEW_OPENED",
        content=(
            f"🎯 Goal '{goal.title}' is implemented — all tasks are done. "
            f"Next: "
            + (
                f"publish branch {goal.branch_name} to the repository for external CI, "
                "or "
                if can_publish
                else ""
            )
            + f"merge {goal.branch_name} → {cfg.working_branch}."
        ),
        db=db,
        metadata={
            "goal_id": str(goal.id),
            "branch": goal.branch_name,
            "ci_status": ci_status,
            "can_publish": can_publish,
        },
    )
    await emit_event(
        room_id=rid,
        event_type=CHECKPOINT_REQUEST,
        content=f"Goal '{goal.title}' is implemented — merge into {cfg.working_branch}?",
        db=db,
        metadata={
            "checkpoint_kind": "confirm",
            "goal_review": True,
            "goal_id": str(goal.id),
            "branch": goal.branch_name,
            "diff_stat": (diff_stat or "")[:_DIFF_STAT_CAP],
            "task_titles": titles,
            "ci_status": ci_status,
            "ci_results": ci_results,
            "can_publish": can_publish,
            **({"requirements_coverage": coverage_matrix} if coverage_matrix else {}),
        },
    )
    logger.info(
        "open_goal_review: room %s goal %s — branch %s, ci=%s",
        room_id, goal.id, goal.branch_name, ci_status,
    )
    return goal


async def resolve_goal_review(
    room_id: "UUID | str",
    goal,
    db,
    *,
    decision: str,
    note: str | None = None,
    force: bool = False,
    resolved_by: "UUID | str | None" = None,
) -> dict[str, Any]:
    """Resolve a goal parked in ``review_state="waiting_review"``.

    ``decision="approve"`` → clean ``merge --no-ff goal/<id>`` into ``dev`` (no
    ``-X theirs``); a conflict emits ``GIT_CONFLICT`` and ``dev`` is untouched,
    success sets ``review_state="approved"``, stamps the goal's tasks with the
    merge sha and finalizes the goal (``GOAL_COMPLETED``).
    ``decision="reject"`` → ``review_state="rejected"``, the goal stays
    ``ACTIVE`` so agents iterate.

    ``approve`` is refused (``blocked=True``) when ``review_ci_status != "passed"``
    unless ``force=True``.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))

    if goal.review_state != GOAL_REVIEW_WAITING:
        return {"ok": False, "error": f"goal review is {goal.review_state}", "state": goal.review_state}

    if decision == "reject":
        goal.review_state = GOAL_REVIEW_REJECTED
        db.add(goal)
        await db.flush()
        await emit_event(
            room_id=rid,
            event_type="GOAL_REVIEW_REJECTED",
            content=(
                f"The merge of goal '{goal.title}' was rejected"
                + (f": {note}" if note else "")
            ),
            db=db,
            metadata={"goal_id": str(goal.id), "note": note},
        )
        return {"ok": True, "state": GOAL_REVIEW_REJECTED}

    if decision != "approve":
        return {"ok": False, "error": f"unknown decision {decision!r}"}

    if goal.review_ci_status != "passed" and not force:
        return {
            "ok": False,
            "blocked": True,
            "reason": "ci_not_passed",
            "ci_status": goal.review_ci_status,
        }

    if not force:
        try:
            from app.core.requirements import check_coverage

            gap = await check_coverage(goal.id, db)
            uncovered_must = gap.get("uncovered_must") or []
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("resolve_goal_review: coverage check failed for goal %s: %s", goal.id, exc)
            uncovered_must = []
        if uncovered_must:
            return {
                "ok": False,
                "blocked": True,
                "reason": "uncovered_must",
                "uncovered_must": uncovered_must,
            }

    return await _promote_goal_branch_to_dev(room_id, goal, db, rid=rid, log_prefix="resolve_goal_review")


async def _promote_goal_branch_to_dev(
    room_id: "UUID | str", goal, db, *, rid: UUID, log_prefix: str
) -> dict[str, Any]:
    """Clean ``merge --no-ff goal/<id> → working_branch`` + stamp task merges +
    ``GOAL_PROMOTED_TO_DEV`` + finalize the goal. Shared by
    ``resolve_goal_review`` (open gate) and ``merge_goal_branch_direct``
    (terminal goal whose gate never opened). A conflict emits ``GIT_CONFLICT``
    and leaves ``working_branch`` untouched.
    """
    from app.core import workspaced_client as wsd
    from app.core.events import emit_event

    room = await _load_room(room_id, db)
    cfg = get_flow_config(room)

    merge = await wsd.merge_workspace_branch(
        str(room_id),
        goal.branch_name,
        target=cfg.working_branch,
        theirs=False,
        message=f"merge goal: {goal.branch_name} → {cfg.working_branch} — {goal.title}",
    )

    if merge.get("conflict"):
        await emit_event(
            room_id=rid,
            event_type="GIT_CONFLICT",
            content=(
                f"Conflict while merging goal '{goal.title}' ({goal.branch_name}) "
                f"into {cfg.working_branch} — resolve manually"
            ),
            db=db,
            metadata={"goal_id": str(goal.id), "branch": goal.branch_name, "target": cfg.working_branch},
        )
        return {"ok": False, "conflict": True}

    if not merge.get("ok"):
        return {"ok": False, "error": merge.get("stderr") or "merge failed"}

    sha = merge.get("sha")
    goal.review_state = GOAL_REVIEW_APPROVED
    db.add(goal)
    await db.flush()
    await _stamp_goal_task_merges(goal.id, sha, db)

    await emit_event(
        room_id=rid,
        event_type="GOAL_PROMOTED_TO_DEV",
        content=f"Goal '{goal.title}' merged into {cfg.working_branch} ({goal.branch_name})",
        db=db,
        metadata={"goal_id": str(goal.id), "branch": goal.branch_name, "merge_commit": sha},
    )

    from app.core.goal_progress import finalize_goal_completion

    await finalize_goal_completion(goal, rid, db)

    logger.info(
        "%s: room %s goal %s approved → %s (%s)",
        log_prefix, room_id, goal.id, cfg.working_branch, sha,
    )
    return {"ok": True, "state": GOAL_REVIEW_APPROVED, "merge_commit": sha}


async def merge_goal_branch_direct(room_id: "UUID | str", goal, db) -> dict[str, Any]:
    """internal design notes follow-up — fold a finished goal's still-unmerged
    branch into ``working_branch`` directly, for a goal whose Stage 5 gate never
    opened (finished before the gate existed, or a fail-open completion).

    Caller must have already checked: goal has ``branch_name``, the branch is
    unmerged, and ``review_state`` is not pending (``testing`` / ``waiting_review``).
    Same clean merge as an approved gate; a conflict → ``conflict=True``.
    """
    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))
    return await _promote_goal_branch_to_dev(
        room_id, goal, db, rid=rid, log_prefix="merge_goal_branch_direct"
    )


async def maybe_open_goal_review(goal, room_id: "UUID | str", db):
    """Hook ``maybe_complete_goal`` calls before finalizing a goal.

    Opens the Stage 5 gate and returns a truthy value (so the caller does *not*
    complete the goal) when the goal is developed on its own branch and not
    already in / past the gate. Returns ``None`` — caller finalizes normally —
    for a goal with no isolated branch, or on any transport hiccup (fail-open:
    a broken gate must not wedge goal completion).
    """
    try:
        if not getattr(goal, "branch_name", None):
            return None
        if goal.review_state in _GOAL_REVIEW_PENDING:
            return goal  # already awaiting review — hold
        if goal.review_state not in _GOAL_REVIEW_OPENABLE:
            return None  # "approved" (or unknown) — let the caller finalize
        return await open_goal_review(room_id, goal, db)
    except GoalReviewError:
        # Fail-open: a broken gate must not wedge goal completion. But
        # ``open_goal_review`` flips ``review_state`` to "testing" *before* the
        # diff/checks that can raise — leave it there and the goal finalizes
        # carrying a misleading "testing". Reset so a fail-open completion has a
        # clean (null) review_state.
        if getattr(goal, "review_state", None) == GOAL_REVIEW_TESTING:
            goal.review_state = None
            db.add(goal)
            try:
                await db.flush()
            except Exception:
                pass
        return None
    except Exception as exc:
        logger.warning("maybe_open_goal_review: goal %s skipped: %s", getattr(goal, "id", "?"), exc)
        return None


async def publish_goal_branch(
    room_id: "UUID | str",
    goal,
    db,
    *,
    access_token: str | None = None,
) -> dict[str, Any]:
    """internal design notes Stage 4 — push ``goal/<id>`` to the bound
    remote so the forge's own CI can run on it.

    Only valid while the goal gate is open (``review_state="waiting_review"``).
    Publishing is a pure ``git push`` — it never touches ``review_state`` or
    completes the goal; the user still approves → merges into ``dev`` afterwards
    (``resolve_goal_review``), optionally once the external CI is green.

    Returns ``{"ok": True, "branch": ...}`` on success (``GOAL_BRANCH_PUBLISHED``
    emitted, ``review_branch_pushed_at`` stamped) or
    ``{"ok": False, "error": ..., "hint": ...}`` — the caller maps that to a
    4xx. State is never changed on failure.
    """
    from app.core.events import emit_event
    from app.core.git_sync import push_workspace_to_git

    rid = room_id if isinstance(room_id, UUID) else UUID(str(room_id))

    if not getattr(goal, "branch_name", None):
        return {"ok": False, "error": "goal has no branch to publish"}
    if goal.review_state != GOAL_REVIEW_WAITING:
        return {
            "ok": False,
            "error": f"goal review is {goal.review_state!r}, not 'waiting_review'",
        }

    room = await _load_room(room_id, db)
    git_config = getattr(room, "git_config", None)
    if not isinstance(git_config, Mapping) or not git_config.get("repo_url"):
        return {"ok": False, "error": "room has no bound repository"}

    # Stage 1.2.3: a network push through workspaced.
    from app.db.release import release_db

    await release_db(db, reason="git/publish_goal_branch")
    result = await push_workspace_to_git(
        room_id=rid,
        git_config=dict(git_config),
        access_token=access_token,
        mode="push",
        branch=goal.branch_name,
    )
    errors = result.get("errors") or []
    if errors:
        return {
            "ok": False,
            "error": errors[0],
            "hint": "sync_then_promote" if is_non_fast_forward(errors) else None,
        }

    goal.review_branch_pushed_at = datetime.utcnow()
    db.add(goal)
    await db.flush()

    await emit_event(
        room_id=rid,
        event_type="GOAL_BRANCH_PUBLISHED",
        content=(
            f"Goal branch '{goal.title}' ({goal.branch_name}) was pushed to "
            f"the repository — you can wait for external CI, then confirm the merge."
        ),
        db=db,
        metadata={"goal_id": str(goal.id), "branch": goal.branch_name, "pushed": True},
    )
    logger.info(
        "publish_goal_branch: room %s goal %s pushed %s to remote",
        room_id, goal.id, goal.branch_name,
    )
    return {"ok": True, "branch": goal.branch_name}
