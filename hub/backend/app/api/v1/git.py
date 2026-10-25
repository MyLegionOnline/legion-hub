"""
Git integration API endpoints.

POST /api/v1/rooms/{room_id}/git/webhook  — receive GitHub/GitLab push webhook
POST /api/v1/rooms/{room_id}/git/sync     — manual full sync (fetch from remote)
POST /api/v1/rooms/{room_id}/git/push     — push workspace repo to remote

Stage 6 of internal design notes: the workspace volume is the
single source of truth. These endpoints drive the real git repo on the
workspace volume through ``workspaced`` (``git fetch`` / ``git push``). The
token is resolved Hub-side from the secret store and handed to
``workspaced`` for one operation only — it never reaches a user container,
because ``workspaced`` runs git on the host (``git_ops.py``), not inside
``ws-*``.

Room.git_config must be configured before calling these endpoints:
{
  "provider": "github" | "gitlab",
  "repo_url": "https://github.com/user/project",
  "branch": "main",
  "access_token": "ghp_..."
}
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.api import deps
from app.core import workspaced_client as wsd
from app.core.events import emit_event
from app.core.git_sync import (
    adopt_remote_repository,
    fetch_workspace_from_git,
    push_workspace_to_git,
)
from app.core.userfiles_policy import apply_userfiles_policy
from app.core.git_webhook import verify_git_webhook_signature
from app.core.request_id import request_id_from_request
from app.core.room_git_flow import (
    get_flow_config,
    is_non_fast_forward as _is_non_fast_forward,
    goal_branch_name,
    session_branch_name,
    task_branch_name,
    SESSION_BRANCH_PREFIX,
    TASK_BRANCH_PREFIX,
    GOAL_BRANCH_PREFIX,
)
from app.core.secrets.runtime import binding_id_from_config, legacy_secret_value, resolve_for_use
from app.core.secrets.service import SecretResourceAccessDenied
from app.core.secrets.store import SecretAccessDenied, SecretUnavailable
from app.models.secret import SecretPurpose, SecretResourceType
from app.db.session import get_session
from app.models.goal import RoomGoal
from app.models.promotion import Promotion
from app.models.room import Room, RoomUserLink
from app.models.session.base import Session as SessionRow
from app.models.task import Task, TaskState
from app.models.user import User

router = APIRouter()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# internal design notes Stage 2 — GET /rooms/{id}/git/state
# ---------------------------------------------------------------------------

_ACTIVE_TASK_STATES = {
    TaskState.ASSIGNED, TaskState.IN_PROGRESS,
    TaskState.PAUSING, TaskState.PAUSED, TaskState.RESUMING,
    TaskState.WAITING_REVIEW, TaskState.BLOCKED,
}
_TERMINAL_TASK_STATES = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.CONTINUED}
# An agent is (or is about to be) writing commits onto the task branch — it must
# not be pulled out from under it by a hand delete.
_RUNNING_TASK_STATES = {TaskState.IN_PROGRESS, TaskState.PAUSING, TaskState.RESUMING}


def _task_state_enum(task: Task) -> Optional[TaskState]:
    state = task.state
    if isinstance(state, TaskState):
        return state
    try:
        return TaskState(state)
    except ValueError:
        return None


class GitEntityOut(BaseModel):
    type: str        # task | goal | session
    id: str
    title: str
    state: Optional[str] = None


class BranchNodeOut(BaseModel):
    name: str
    kind: str        # main | dev | task | session | goal | other
    base: Optional[str] = None
    merged: Optional[bool] = None
    ahead: Optional[int] = None
    behind: Optional[int] = None
    entity: Optional[GitEntityOut] = None
    ci_status: Optional[str] = None
    risk: Optional[str] = None
    action: str = "none"   # none | merge | promote | return_to_review
    # internal design notes Stage 0 — task branch could not be refreshed
    # against its moved base; the "Divergence review" button is offered.
    diverged: bool = False
    divergence_state: Optional[str] = None  # detected | reviewing | reviewed | resolved
    # A resolved divergence retired this branch (folded in / superseded / discarded).
    # The tree hides it the same way it hides a merged branch.
    retired: bool = False
    # The user may discard this branch from the Git panel (tag-backup + delete).
    # False for the release/working branch, for a branch a live session is
    # committing onto, and for a task branch whose agent is running right now.
    can_delete: bool = False
    children: list["BranchNodeOut"] = []


BranchNodeOut.model_rebuild()


class GitConnectionOut(BaseModel):
    repo_url: Optional[str] = None
    branch: Optional[str] = None
    provider: Optional[str] = None
    adopted_at: Optional[str] = None
    first_pushed_at: Optional[str] = None
    ahead: Optional[int] = None
    behind: Optional[int] = None
    ci_provider: str = "none"
    # internal design notes Stage 4 — a remote/index collision pauses private
    # uploads until the user fixes the repository and rechecks it.
    userfiles_policy: str = "ready"
    userfiles_policy_conflict: bool = False
    userfiles_policy_paths: list[str] = []


class GitFlowOut(BaseModel):
    working_branch: str
    release_branch: str
    promote_policy: str


class GitGateOut(BaseModel):
    kind: str        # goal_review | promotion
    id: str
    title: str
    branch: str
    ci_status: Optional[str] = None
    diff_stat: Optional[str] = None
    can_publish: bool = False
    pr_url: Optional[str] = None


class GitStateOut(BaseModel):
    connection: GitConnectionOut
    flow: GitFlowOut
    branches: list[BranchNodeOut]
    gates: list[GitGateOut]


def _branch_kind(name: str, working_branch: str, release_branch: str) -> str:
    if name == release_branch:
        return "main"
    if name == working_branch:
        return "dev"
    if name.startswith(TASK_BRANCH_PREFIX):
        return "task"
    if name.startswith(SESSION_BRANCH_PREFIX):
        return "session"
    if name.startswith(GOAL_BRANCH_PREFIX):
        return "goal"
    return "other"


def _task_action(task: Task, merged: Optional[bool]) -> str:
    state = task.state
    if isinstance(state, str):
        try:
            state = TaskState(state)
        except ValueError:
            return "none"
    # A resolved divergence retired this branch on purpose (folded in, superseded
    # by a rework task, or discarded) — never offer to reopen it.
    if getattr(task, "divergence_state", None) == "resolved":
        return "none"
    if state == TaskState.WAITING_REVIEW:
        return "merge"
    if merged is False:
        # A COMPLETED task whose branch never landed (old completion, a merge
        # that silently failed, a manual finish): offer a direct fold into the
        # base — review already happened, no need to reopen it.
        if state == TaskState.COMPLETED:
            return "merge"
        # CANCELLED / CONTINUED with commits still outstanding: back to Gate 1.
        if state in _TERMINAL_TASK_STATES:
            return "return_to_review"
    return "none"


async def build_git_state(room_id: UUID, db: AsyncSession) -> GitStateOut:
    """Assemble the full git state for a room in one pass.

    Graceful degradation: workspaced unavailable → branches=[], connection from DB.
    """
    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    cfg = get_flow_config(room)
    gc: dict[str, Any] = room.git_config or {}

    # ── Load entities ──────────────────────────────────────────────────────
    tasks: list[Task] = (await db.execute(
        select(Task).where(Task.room_id == room_id, Task.branch_name.isnot(None))
    )).scalars().all()

    goals: list[RoomGoal] = (await db.execute(
        select(RoomGoal).where(RoomGoal.room_id == room_id)
    )).scalars().all()
    goal_map: dict[str, RoomGoal] = {str(g.id): g for g in goals}

    live_sessions: list[SessionRow] = (await db.execute(
        select(SessionRow).where(
            SessionRow.room_id == room_id,
            SessionRow.closed_at.is_(None),
        )
    )).scalars().all()

    # ── Build bases map ────────────────────────────────────────────────────
    bases: dict[str, str] = {cfg.working_branch: cfg.release_branch}
    for goal in goals:
        if goal.branch_name:
            bases[goal.branch_name] = cfg.working_branch
    for task in tasks:
        if task.branch_name:
            goal_branch = None
            if task.goal_id and str(task.goal_id) in goal_map:
                g = goal_map[str(task.goal_id)]
                goal_branch = g.branch_name
            bases[task.branch_name] = goal_branch or cfg.working_branch
    for sess in live_sessions:
        sb = session_branch_name(sess.id)
        bases[sb] = cfg.working_branch

    # ── Workspaced branch list (graceful) ──────────────────────────────────
    wsd_branches = await wsd.list_workspace_branches(str(room_id), bases)
    # A session branch outlives its session (closed dialogue, orphaned work):
    # it is always cut from working_branch, so measure it against that too —
    # otherwise it has no ahead/merged and no way back into dev.
    orphan_session_branches = [
        b["name"] for b in wsd_branches
        if b["name"].startswith(SESSION_BRANCH_PREFIX) and b["name"] not in bases
    ]
    if orphan_session_branches:
        for name in orphan_session_branches:
            bases[name] = cfg.working_branch
        wsd_branches = await wsd.list_workspace_branches(str(room_id), bases)
    branch_info: dict[str, dict] = {b["name"]: b for b in wsd_branches}

    # ── Remote ahead/behind (for connection header) ────────────────────────
    remote_ab: Optional[dict] = None
    if gc.get("repo_url"):
        try:
            remote_ab = await wsd.git_ahead_behind_remote_workspace(
                str(room_id), cfg.working_branch
            )
        except Exception:
            pass

    # ── Open gates ────────────────────────────────────────────────────────
    open_promo: Optional[Promotion] = (await db.execute(
        select(Promotion)
        .where(Promotion.room_id == room_id, Promotion.state == "waiting_review")
        .order_by(Promotion.created_at.desc())
        .limit(1)
    )).scalars().first()

    goal_reviews: list[RoomGoal] = (await db.execute(
        select(RoomGoal).where(
            RoomGoal.room_id == room_id,
            RoomGoal.review_state == "waiting_review",
        )
    )).scalars().all()

    # ── Entity lookup dicts ───────────────────────────────────────────────
    task_by_branch: dict[str, Task] = {t.branch_name: t for t in tasks if t.branch_name}
    goal_by_branch: dict[str, RoomGoal] = {g.branch_name: g for g in goals if g.branch_name}
    session_id_set: set[str] = {str(s.id) for s in live_sessions}

    # ── Build branch nodes ────────────────────────────────────────────────
    def _make_node(name: str) -> BranchNodeOut:
        info = branch_info.get(name, {})
        kind = _branch_kind(name, cfg.working_branch, cfg.release_branch)
        merged = info.get("merged")
        ahead = info.get("ahead")
        behind_val = info.get("behind")
        base = info.get("base") or bases.get(name)

        entity: Optional[GitEntityOut] = None
        ci_st: Optional[str] = None
        risk_val: Optional[str] = None
        action = "none"
        diverged = False
        retired = False
        divergence_state: Optional[str] = None
        # Anything but the two protected branches may be discarded by hand; the
        # per-kind blocks below take that back while work on it is live.
        can_delete = kind not in ("main", "dev")

        if kind == "task" and name in task_by_branch:
            t = task_by_branch[name]
            state_str = t.state.value if isinstance(t.state, TaskState) else str(t.state)
            entity = GitEntityOut(type="task", id=str(t.id), title=t.title or "", state=state_str)
            ci_st = t.ci_status
            risk_val = t.risk
            action = _task_action(t, merged)
            ds = getattr(t, "divergence_state", None)
            divergence_state = ds if isinstance(ds, str) else None
            if divergence_state in ("detected", "reviewing", "reviewed"):
                diverged = True
            if divergence_state == "resolved":
                retired = True
            if _task_state_enum(t) in _RUNNING_TASK_STATES:
                can_delete = False

        elif kind == "goal" and name in goal_by_branch:
            g = goal_by_branch[name]
            state_str = g.status.value if hasattr(g.status, "value") else str(g.status)
            entity = GitEntityOut(type="goal", id=str(g.id), title=g.title or "", state=state_str)
            if g.review_state == "waiting_review":
                action = "merge"  # goal → dev via review/resolve endpoint (also in gates[])
            elif (
                merged is False
                and g.review_state not in ("testing", "waiting_review")
                and str(getattr(g.status, "value", g.status)) == "completed"
            ):
                # Finished goal whose Stage 5 gate never opened (old goal, or a
                # fail-open completion) → direct fold, mirrors the COMPLETED
                # task shortcut. Frontend routes to .../goals/{gid}/merge-branch.
                action = "merge"

        elif kind == "session":
            raw_id = name[len(SESSION_BRANCH_PREFIX):]
            if raw_id in session_id_set:
                entity = GitEntityOut(type="session", id=raw_id, title="", state="live")
                # no action — badge only
                can_delete = False  # a live session is committing onto it
            elif merged is False:
                # Orphaned work of a closed dialogue that never became a task:
                # fold it straight into its base like any other loose branch.
                action = "merge"

        elif kind == "dev":
            # promote action when dev is ahead of main (open gate or pending work)
            if open_promo or (ahead is not None and ahead > 0):
                action = "promote"

        elif kind == "other" and merged is False:
            action = "merge"

        return BranchNodeOut(
            name=name,
            kind=kind,
            base=base,
            merged=merged,
            ahead=ahead,
            behind=behind_val,
            entity=entity,
            ci_status=ci_st,
            risk=risk_val,
            action=action,
            diverged=diverged,
            divergence_state=divergence_state,
            retired=retired,
            can_delete=can_delete,
        )

    # Collect all branch names: wsd result + virtual main/dev even if no wsd
    all_names: set[str] = {b["name"] for b in wsd_branches}
    all_names.add(cfg.release_branch)
    all_names.add(cfg.working_branch)

    nodes: dict[str, BranchNodeOut] = {n: _make_node(n) for n in all_names}

    # ── Tree assembly: parent → children ──────────────────────────────────
    # child_base maps branch name → parent name
    def _parent(name: str) -> Optional[str]:
        if name == cfg.release_branch:
            return None
        if name == cfg.working_branch:
            return cfg.release_branch
        b = bases.get(name) or branch_info.get(name, {}).get("base")
        return b

    roots: list[BranchNodeOut] = []
    for name, node in nodes.items():
        parent_name = _parent(name)
        if parent_name and parent_name in nodes:
            nodes[parent_name].children.append(node)
        else:
            roots.append(node)

    # put release_branch first in roots
    roots.sort(key=lambda n: (n.name != cfg.release_branch, n.name))

    # ── Gates ─────────────────────────────────────────────────────────────
    gates: list[GitGateOut] = []

    for g in goal_reviews:
        if not g.branch_name:
            continue
        gates.append(GitGateOut(
            kind="goal_review",
            id=str(g.id),
            title=g.title or "",
            branch=g.branch_name,
            ci_status=g.review_ci_status,
            can_publish=True,
        ))

    if open_promo:
        gates.append(GitGateOut(
            kind="promotion",
            id=str(open_promo.id),
            title=f"{open_promo.from_branch} → {open_promo.to_branch}",
            branch=open_promo.from_branch,
            ci_status=open_promo.ci_status,
            diff_stat=open_promo.diff_stat or None,
            can_publish=bool(gc.get("repo_url")),
            pr_url=open_promo.pr_url,
        ))

    # ── Connection ────────────────────────────────────────────────────────
    connection = GitConnectionOut(
        repo_url=gc.get("repo_url"),
        branch=gc.get("branch"),
        provider=gc.get("provider"),
        adopted_at=gc.get("adopted_at"),
        first_pushed_at=gc.get("first_pushed_at"),
        ahead=remote_ab.get("ahead") if remote_ab else None,
        behind=remote_ab.get("behind") if remote_ab else None,
        ci_provider=cfg.ci_provider,
        userfiles_policy=str(gc.get("userfiles_policy") or "ready"),
        userfiles_policy_conflict=gc.get("userfiles_policy") == "conflict",
        userfiles_policy_paths=list(gc.get("userfiles_policy_paths") or []),
    )

    return GitStateOut(
        connection=connection,
        flow=GitFlowOut(
            working_branch=cfg.working_branch,
            release_branch=cfg.release_branch,
            promote_policy=cfg.promote_policy,
        ),
        branches=roots,
        gates=gates,
    )


async def _get_room_with_git(room_id: UUID, db: AsyncSession) -> Room:
    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if not room.git_config or not room.git_config.get("repo_url"):
        raise HTTPException(
            status_code=422,
            detail="Room has no git_config. Set repo_url via PUT /rooms/{id}.",
        )
    return room


async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a room member")


async def _resolve_git_access_token(
    room: Room,
    user: User,
    request: Request,
    db: AsyncSession,
):
    """Resolve a room Git token for one Hub-side provider operation."""

    config = room.git_config or {}
    binding_id = binding_id_from_config(config, "access_token_secret_binding_id")
    if binding_id is None:
        return legacy_secret_value(config, "access_token")
    return await resolve_for_use(
        db=db,
        # Membership was checked by the caller, while the room credential is
        # owned by the binding creator.  Using the member here would make a
        # migrated shared-room binding inaccessible to every non-creator.
        owner_id=room.creator_id,
        resource_type=SecretResourceType.ROOM,
        resource_id=room.id,
        purpose=SecretPurpose.GIT_HTTPS,
        binding_id=binding_id,
        request_id=request_id_from_request(request),
        source_ip=request.client.host if request.client else None,
    )


def _map_workspaced_error(exc: Exception) -> HTTPException:
    """Translate workspaced HTTP/network errors into FastAPI exceptions.

    Mirrors the mapping in ``workspace_files.py``: a 4xx from workspaced is
    surfaced as-is (it carries a meaningful status — 404 for a missing
    repo, 409 for a sync precondition), other HTTP failures become 502,
    and a network failure becomes 503.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if 400 <= code < 500:
            return HTTPException(
                status_code=code,
                detail=f"workspaced error: {exc.response.text}",
            )
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"workspaced error: {exc.response.text}",
        )
    if isinstance(exc, httpx.RequestError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"workspaced unreachable: {exc}",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"workspaced error: {exc}",
    )


# A branch whose whole tree is one of these (or empty) is the synthetic
# ``git_init`` start — no real content to publish.
_EMPTY_TREE_PATHS = {".gitignore"}


async def _branch_has_content(room_id: UUID, branch: str) -> bool:
    """True if ``branch`` on the workspace volume carries real tracked files.

    Used by ``git_push`` to refuse publishing an empty branch over a live
    remote (internal design notes §4.1): a user who has just bound an
    existing repository and hits "push" on ``main`` before the first
    ``dev → main`` promotion would otherwise clobber the remote with the
    synthetic empty tree. The check is tied to the branch that is *actually*
    being pushed, not HEAD, so pushing ``dev`` (which the agents have worked
    on) from a room whose ``main`` is still empty is not blocked.
    """
    try:
        entries = await wsd.git_ls_tree_workspace(str(room_id), ref=branch)
    except Exception:
        # If we can't tell, don't block the push — the push itself will
        # surface whatever workspaced-side failure is happening.
        return True
    paths = {str(e.get("path") or "").strip() for e in entries if e.get("path")}
    return bool(paths - _EMPTY_TREE_PATHS)


async def _handle_ci_status_webhook(
    room: Room, event: str, payload: dict, db: AsyncSession
) -> dict[str, Any]:
    """internal design notes Stage 7 — apply a GitHub Actions CI verdict to the
    open ``dev → main`` promotion.

    ``event`` is ``workflow_run`` or ``check_suite``; both nest a node under
    their own key with ``status`` / ``conclusion`` / ``head_branch``. Only a
    completed run carries a verdict — anything else is acknowledged and ignored.
    """
    node = payload.get(event)
    node = node if isinstance(node, dict) else {}
    run_status = str(node.get("status") or "").lower()
    conclusion = node.get("conclusion")
    head_branch = str(node.get("head_branch") or "")

    if run_status and run_status != "completed":
        return {"ok": True, "ignored": "run not completed"}
    if not head_branch:
        return {"ok": True, "ignored": "no head_branch"}

    from app.core.room_git_flow import apply_external_ci_status

    try:
        result = await apply_external_ci_status(
            room.id, head_branch, conclusion, db, source="github"
        )
    except Exception as exc:
        logger.warning("git webhook CI status failed for room %s: %s", room.id, exc)
        return {"ok": False, "errors": [str(exc)]}
    return {"ok": True, **result}


# ── POST /rooms/{room_id}/git/webhook ────────────────────────────────────────

@router.post("/rooms/{room_id}/git/webhook")
async def git_webhook(
    room_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> Any:
    """
    Receive a push webhook from GitHub or GitLab.

    The endpoint is intentionally unauthenticated at the HTTP level, but the
    provider signature is mandatory and is checked before JSON parsing.

    A native ``git fetch`` pulls everything that changed — the per-file
    ``paths`` from the payload are no longer needed.
    """
    room = await _get_room_with_git(room_id, db)
    config = room.git_config or {}
    provider = config.get("provider", "github")
    secret_value = None
    access_token_value = None
    try:
        raw_body = await request.body()
        binding_id = binding_id_from_config(config, "webhook_secret_binding_id")
        if binding_id is not None:
            secret_value = await resolve_for_use(
                db=db,
                owner_id=room.creator_id,
                resource_type=SecretResourceType.ROOM,
                resource_id=room.id,
                purpose=SecretPurpose.GIT_WEBHOOK,
                binding_id=binding_id,
                request_id=request_id_from_request(request),
                source_ip=request.client.host if request.client else None,
            )
        else:
            # Migration-only fallback. New public configurations cannot write
            # this field and strict mode deliberately rejects it.
            secret_value = legacy_secret_value(config, "webhook_secret")
        if secret_value is None or not verify_git_webhook_signature(
            provider,
            raw_body,
            request.headers,
            secret_value.reveal(),
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

        try:
            payload = json.loads(raw_body)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid webhook payload") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid webhook payload")

        # internal design notes Stage 7 — a GitHub Actions status webhook
        # (``workflow_run`` / ``check_suite``) stamps ``Promotion.ci_status``
        # for the open ``dev → main`` gate. No git fetch, no token needed.
        gh_event = request.headers.get("X-GitHub-Event", "").strip().lower()
        if gh_event in ("workflow_run", "check_suite"):
            return await _handle_ci_status_webhook(room, gh_event, payload, db)

        access_binding_id = binding_id_from_config(config, "access_token_secret_binding_id")
        if access_binding_id is not None:
            access_token_value = await resolve_for_use(
                db=db,
                owner_id=room.creator_id,
                resource_type=SecretResourceType.ROOM,
                resource_id=room.id,
                purpose=SecretPurpose.GIT_HTTPS,
                binding_id=access_binding_id,
                request_id=request_id_from_request(request),
                source_ip=request.client.host if request.client else None,
            )
        else:
            access_token_value = legacy_secret_value(config, "access_token")

        token_str = access_token_value.reveal() if access_token_value else None

        # Keep user credentials in Hub memory only; do not send them to a
        # workspace container. ``workspaced`` runs git on the host
        # (git_ops.py:347), so the token never reaches a user container.
        try:
            result = await fetch_workspace_from_git(
                room_id=room_id,
                git_config=config,
                access_token=token_str,
            )
        except Exception as exc:
            logger.warning("git webhook fetch failed for room %s: %s", room_id, exc)
            return {"ok": False, "synced": 0, "errors": [str(exc)]}
        if result.get("userfiles_policy_conflict") or (
            not result.get("errors") and result.get("synced")
        ):
            await apply_userfiles_policy(
                room,
                result,
                db,
                source="remote_webhook",
            )
        if not result.get("errors"):
            return {"ok": True, "synced": 1, "errors": []}
        return {
            "ok": False,
            "synced": 0,
            "errors": result["errors"],
            "userfiles_policy_conflict": bool(result.get("userfiles_policy_conflict")),
            "userfiles_tracked": result.get("userfiles_tracked") or [],
        }
    except (SecretAccessDenied, SecretResourceAccessDenied, SecretUnavailable):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        ) from None
    finally:
        secret_value = None
        access_token_value = None


# ── POST /rooms/{room_id}/git/sync ───────────────────────────────────────────

class GitSyncResponse(BaseModel):
    ok: bool
    synced: int
    errors: list[str]
    skipped: int = 0
    empty_blocked: int = 0
    truncated: bool = False
    aborted: bool = False
    conflict: bool = False
    # internal design notes Stage 4 — a seed `.gitignore` add/add conflict was
    # reconciled to the remote's file instead of surfacing as a conflict.
    gitignore_autoresolved: bool = False
    # The local branch the remote was merged into.
    branch: Optional[str] = None
    userfiles_policy_conflict: bool = False
    userfiles_tracked: list[str] = []
    userfiles_tracked_details: list[dict[str, Any]] = []


class GitSyncBody(BaseModel):
    # Per-branch Pull from the Git panel: fetch remote ``<branch>`` into the
    # local ``<branch>``. Omitted → the configured branch into working_branch.
    branch: Optional[str] = None


@router.post("/rooms/{room_id}/git/sync", response_model=GitSyncResponse)
async def git_sync(
    room_id: UUID,
    request: Request,
    body: Optional[GitSyncBody] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Manually sync from the configured branch into the room.

    A native ``git fetch`` + fast-forward merge on the workspace volume (one
    operation, full history). On a non-fast-forward conflict a
    ``GIT_CONFLICT`` event is emitted to the room chat so the user can
    resolve manually.

    An optional body ``{"branch": "dev"}`` pulls that same-named branch from
    the remote into the local one instead — for when a branch was edited on
    the remote side (the Git panel's per-branch Pull button).
    """
    await _verify_membership(room_id, current_user, db)
    room = await _get_room_with_git(room_id, db)
    pull_branch = (body.branch or "").strip() if body else ""
    branch_kwargs = (
        {"remote_branch": pull_branch, "target_branch": pull_branch} if pull_branch else {}
    )

    try:
        token_value = await _resolve_git_access_token(room, current_user, request, db)
    except (SecretAccessDenied, SecretResourceAccessDenied, SecretUnavailable):
        raise HTTPException(status_code=403, detail="Git credential is unavailable") from None
    token_str = token_value.reveal() if token_value else None
    try:
        try:
            result = await fetch_workspace_from_git(
                room_id=room_id,
                git_config=room.git_config,
                access_token=token_str,
                **branch_kwargs,
            )
        except Exception as exc:
            raise _map_workspaced_error(exc)
        conflict = bool(result.get("conflict"))
        if result.get("userfiles_policy_conflict") or (
            not result.get("errors") and result.get("synced")
        ):
            await apply_userfiles_policy(
                room,
                result,
                db,
                source="remote_pull",
            )
        if conflict:
            try:
                await emit_event(
                    room_id=room_id,
                    event_type="GIT_CONFLICT",
                    content=(
                        "Conflict during git fetch — the history diverges from remote, "
                        "resolve it manually"
                    ),
                    db=db,
                    metadata={
                        "remote_url": room.git_config.get("repo_url"),
                        "branch": pull_branch or room.git_config.get("branch", "main"),
                    },
                )
            except Exception:
                logger.warning(
                    "git_sync: failed to emit GIT_CONFLICT for room %s",
                    room_id, exc_info=True,
                )
        gitignore_autoresolved = bool(result.get("gitignore_autoresolved"))
        if gitignore_autoresolved:
            try:
                await emit_event(
                    room_id=room_id,
                    event_type="GITIGNORE_RECONCILED",
                    content=(
                        "seed-`.gitignore` aligned to the remote version when merging histories"
                    ),
                    db=db,
                    metadata={
                        "remote_url": room.git_config.get("repo_url"),
                        "branch": result.get("branch"),
                    },
                )
            except Exception:
                logger.warning(
                    "git_sync: failed to emit GITIGNORE_RECONCILED for room %s",
                    room_id, exc_info=True,
                )
        return GitSyncResponse(
            ok=not result.get("errors"),
            synced=result.get("synced", 0),
            errors=result.get("errors", []),
            conflict=conflict,
            gitignore_autoresolved=gitignore_autoresolved,
            branch=result.get("branch"),
            userfiles_policy_conflict=bool(result.get("userfiles_policy_conflict")),
            userfiles_tracked=result.get("userfiles_tracked") or [],
            userfiles_tracked_details=result.get("userfiles_tracked_details") or [],
        )
    finally:
        token_value = None


# ── POST /rooms/{room_id}/git/adopt ──────────────────────────────────────────

class GitAdoptBody(BaseModel):
    # An explicit human choice to let the transplant rewrite a local ``main``
    # that already has work on it (§3.3 divergence dialog, "accept remote as base").
    force: bool = False


class GitAdoptResponse(BaseModel):
    ok: bool
    adopted: bool = False
    diverged: bool = False
    dropped_init: bool = False
    errors: list[str] = []
    userfiles_policy_conflict: bool = False
    userfiles_tracked: list[str] = []
    userfiles_tracked_details: list[dict[str, Any]] = []


@router.post("/rooms/{room_id}/git/adopt", response_model=GitAdoptResponse)
async def git_adopt(
    room_id: UUID,
    request: Request,
    body: Optional[GitAdoptBody] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Adopt the configured external repository as this room's base
    (internal design notes §3).

    Run once, when a repo is bound for the first time (``git_config`` has no
    ``adopted_at``). It pins the room's local ``main`` to the remote's ``main``
    so the first ``dev → main`` promotion diffs only the room's own changes and
    the promotion push fast-forwards without a synthetic root commit. A
    subsequent bind of the same URL goes through the plain ``sync`` path
    instead.

    On ``diverged`` (local ``main`` already has work that is not an ancestor of
    the remote) nothing is rewritten and a ``GIT_CONFLICT`` event is emitted;
    the caller may retry with ``{"force": true}`` to transplant anyway.
    """
    await _verify_membership(room_id, current_user, db)
    room = await _get_room_with_git(room_id, db)
    force = bool(body and body.force)

    try:
        token_value = await _resolve_git_access_token(room, current_user, request, db)
    except (SecretAccessDenied, SecretResourceAccessDenied, SecretUnavailable):
        raise HTTPException(status_code=403, detail="Git credential is unavailable") from None
    token_str = token_value.reveal() if token_value else None
    try:
        try:
            result = await adopt_remote_repository(
                room_id=room_id,
                git_config=room.git_config,
                access_token=token_str,
                force=force,
            )
        except Exception as exc:
            raise _map_workspaced_error(exc)
    finally:
        token_value = None

    if result.get("userfiles_policy_conflict") or result.get("adopted"):
        await apply_userfiles_policy(
            room,
            result,
            db,
            source="remote_adopt",
        )

    if result.get("adopted"):
        room.git_config = {
            **(room.git_config or {}),
            "adopted_at": datetime.now(timezone.utc).isoformat(),
        }
        flag_modified(room, "git_config")
        await db.commit()
        try:
            await emit_event(
                room_id=room_id,
                event_type="GIT_ADOPTED",
                content=(
                    "External repository accepted as the room's base — "
                    "local `main` linked to remote"
                ),
                db=db,
                metadata={
                    "remote_url": room.git_config.get("repo_url"),
                    "branch": room.git_config.get("branch", "main"),
                    "dropped_init": bool(result.get("dropped_init")),
                    "forced": force,
                },
            )
        except Exception:
            logger.warning(
                "git_adopt: failed to emit GIT_ADOPTED for room %s", room_id, exc_info=True
            )
    elif result.get("diverged"):
        try:
            await emit_event(
                room_id=room_id,
                event_type="GIT_CONFLICT",
                content=(
                    "Local `main` has diverged from remote — choose: merge the histories "
                    "(Pull from Git) or accept the external repository as the base"
                ),
                db=db,
                metadata={
                    "remote_url": room.git_config.get("repo_url"),
                    "branch": room.git_config.get("branch", "main"),
                    "adopt_diverged": True,
                },
            )
        except Exception:
            logger.warning(
                "git_adopt: failed to emit GIT_CONFLICT for room %s", room_id, exc_info=True
            )

    return GitAdoptResponse(
        ok=bool(result.get("ok")),
        adopted=bool(result.get("adopted")),
        diverged=bool(result.get("diverged")),
        dropped_init=bool(result.get("dropped_init")),
        errors=result.get("errors", []),
        userfiles_policy_conflict=bool(result.get("userfiles_policy_conflict")),
        userfiles_tracked=result.get("userfiles_tracked") or [],
        userfiles_tracked_details=result.get("userfiles_tracked_details") or [],
    )


# ── POST /rooms/{room_id}/git/push ───────────────────────────────────────────

class GitPushResponse(BaseModel):
    ok: bool
    pushed: int
    errors: list[str]
    # internal design notes §4.3 — a machine-readable hint the frontend
    # turns into "sync → promote → push" guidance instead of a force button when
    # the remote rejected the push as non-fast-forward.
    hint: Optional[str] = None


class GitPushBody(BaseModel):
    # internal design notes Stage 6: push an arbitrary local branch (e.g.
    # ``dev``) instead of the configured ``git_config["branch"]``. Omitted →
    # the configured branch, unchanged behaviour.
    branch: Optional[str] = None


@router.post("/rooms/{room_id}/git/push", response_model=GitPushResponse)
async def git_push(
    room_id: UUID,
    request: Request,
    mode: Literal["push", "force"] = "push",
    body: Optional[GitPushBody] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Push the room repository to a git branch.

    A single native ``git push`` on the workspace volume (one commit
    history, not 180 Contents-API calls). By default the configured
    ``git_config["branch"]`` is pushed; an optional body ``{"branch": "dev"}``
    pushes that branch instead (internal design notes Stage 6) — the local
    branch of that name is sent regardless of which branch is checked out.

    ``mode`` selects the push strategy: the default ``"push"`` is rejected
    by the remote if the branch has diverged (the response then carries
    ``hint="sync_then_promote"``); ``"force"`` applies ``--force-with-lease``
    so the user can override divergence only with an explicit, acknowledged
    choice. A branch with no real content (empty tree, or only ``.gitignore``)
    is refused with 409 — pushing it would clobber a live remote with an empty
    tree (internal design notes §4.1). The check is tied to the branch
    actually being pushed, so a ``dev`` push is not blocked by an empty ``main``.
    """
    await _verify_membership(room_id, current_user, db)
    room = await _get_room_with_git(room_id, db)
    branch = body.branch if body and body.branch else None
    push_branch = branch or (room.git_config or {}).get("branch") or "main"

    if not await _branch_has_content(room_id, push_branch):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Branch `{push_branch}` has no changes yet — pushing it would "
                f"overwrite the remote with an empty tree. Promote "
                f"dev → main first, or push `dev` if you only need a backup."
            ),
        )

    try:
        token_value = await _resolve_git_access_token(room, current_user, request, db)
    except (SecretAccessDenied, SecretResourceAccessDenied, SecretUnavailable):
        raise HTTPException(status_code=403, detail="Git credential is unavailable") from None
    token_str = token_value.reveal() if token_value else None
    try:
        result = await push_workspace_to_git(
            room_id=room_id,
            git_config=room.git_config,
            access_token=token_str,
            mode=mode,
            branch=branch,
        )
    finally:
        token_value = None

    errors = result.get("errors") or []
    if result.get("userfiles_push_blocked"):
        paths = result.get("userfiles_tracked") or []
        await apply_userfiles_policy(
            room,
            result,
            db,
            source="push_guard",
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "userfiles_tracked",
                "message": (
                    "Push blocked: private room files are tracked by Git. "
                    "Remove them from the index or move deliberately publishable "
                    "files into the project before retrying."
                ),
                "paths": paths,
                "action": "git rm -r --cached .userfiles",
            },
        )
    resp: dict[str, Any] = {"ok": not errors, **result}
    if errors and mode != "force" and _is_non_fast_forward(errors):
        resp["hint"] = "sync_then_promote"
    return resp


@router.get("/rooms/{room_id}/git/state", response_model=GitStateOut)
async def get_git_state(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """internal design notes Stage 2 — one-stop git state for the sidebar panel.

    Returns connection, flow config, branch tree, and open gates in a single
    request. workspaced unavailable → 200 with ``branches: []``, never 502.
    """
    await _verify_membership(room_id, current_user, db)
    return await build_git_state(room_id, db)


# ── DELETE /rooms/{room_id}/git/branch ───────────────────────────────────────

class GitDeleteBranchResponse(BaseModel):
    ok: bool
    deleted: bool = False
    backup_ref: Optional[str] = None
    detail: Optional[str] = None


@router.delete("/rooms/{room_id}/git/branch", response_model=GitDeleteBranchResponse)
async def delete_git_branch(
    room_id: UUID,
    name: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Discard a branch from the Git panel — the user's own "this branch is not needed".

    The counterpart of the ``delete_branch`` divergence resolution, but
    available on any branch without waiting for a divergence: a goal branch
    whose goal was never really finished, a task branch closed by hand, a
    stray ``other`` branch. Same recoverable mechanics — ``workspaced`` tags
    the tip under ``refs/legion/trash/`` before ``git branch -D``, so the
    commits stay reachable.

    Refused for the release/working branch, for a branch a live session is
    committing onto, and for a task branch whose agent is running right now
    (``can_delete`` on the branch node mirrors these rules). The linked task
    is cancelled and an open goal gate is closed, so the tree and the gate
    queue do not keep pointing at a branch that no longer exists. Deleting a
    local branch never touches the remote.
    """
    await _verify_membership(room_id, current_user, db)
    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    cfg = get_flow_config(room)
    branch = (name or "").strip()
    if not branch:
        raise HTTPException(status_code=422, detail="branch name is required")
    if branch in (cfg.release_branch, cfg.working_branch):
        raise HTTPException(
            status_code=409,
            detail=f"Branch '{branch}' cannot be deleted — it is the thread's main branch.",
        )

    # Live session working on this branch → refuse (its commits would vanish
    # mid-flight and the container would keep writing to a dead ref).
    live_sessions: list[SessionRow] = (await db.execute(
        select(SessionRow).where(
            SessionRow.room_id == room_id,
            SessionRow.closed_at.is_(None),
        )
    )).scalars().all()
    if any(session_branch_name(s.id) == branch for s in live_sessions):
        raise HTTPException(
            status_code=409,
            detail="A live session is running on the branch — close it, then delete.",
        )

    task: Optional[Task] = (await db.execute(
        select(Task).where(Task.room_id == room_id, Task.branch_name == branch)
    )).scalars().first()
    if task is not None and _task_state_enum(task) in _RUNNING_TASK_STATES:
        raise HTTPException(
            status_code=409,
            detail="A task is running — stop it, then delete the branch.",
        )

    try:
        res = await wsd.git_discard_branch_workspace(str(room_id), branch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("delete_git_branch: workspaced failed for %s: %s", branch, exc)
        raise HTTPException(status_code=502, detail=f"workspaced unavailable: {exc}") from None

    if res.get("busy"):
        raise HTTPException(status_code=409, detail="The working tree is busy — try again.")
    if not res.get("ok"):
        raise HTTPException(
            status_code=409,
            detail=res.get("stderr") or "Could not delete the branch.",
        )

    backup_ref = res.get("backup_ref")
    deleted = bool(res.get("deleted"))

    # ── Detach the entities that pointed at the branch ────────────────────
    goal: Optional[RoomGoal] = (await db.execute(
        select(RoomGoal).where(RoomGoal.room_id == room_id, RoomGoal.branch_name == branch)
    )).scalars().first()

    if task is not None:
        if _task_state_enum(task) not in _TERMINAL_TASK_STATES:
            task.state = TaskState.CANCELLED
        task.divergence_state = "resolved"
        db.add(task)

    if goal is not None and goal.review_state in ("testing", "waiting_review"):
        # The Stage 5 gate card would otherwise offer "Merge into dev" on a branch
        # that is gone. Rejecting it leaves the goal ACTIVE, as a reject does.
        goal.review_state = "rejected"
        db.add(goal)

    await db.flush()

    await emit_event(
        room_id=room_id,
        event_type="BRANCH_DELETED",
        content=(
            f"Branch '{branch}' deleted by the user"
            + (f" (backup: {backup_ref})" if backup_ref else "")
            + ("; task cancelled" if task is not None else "")
            + ("; goal gate closed" if goal is not None and goal.review_state == "rejected" else "")
            + "."
        ),
        db=db,
        task_id=task.id if task is not None else None,
        metadata={
            "branch": branch,
            "backup_ref": backup_ref,
            "deleted_by": str(current_user.id),
            "task_id": str(task.id) if task is not None else None,
            "goal_id": str(goal.id) if goal is not None else None,
        },
    )
    await db.commit()

    return GitDeleteBranchResponse(
        ok=True,
        deleted=deleted,
        backup_ref=backup_ref,
        detail=None if deleted else res.get("stderr"),
    )
