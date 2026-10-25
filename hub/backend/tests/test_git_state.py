"""internal design notes Stage 2 — build_git_state.

Covers:
  * room without repo → connection.repo_url=None, branches contain main/dev,
    gates empty;
  * task with goal appears nested under goal branch, task without goal under dev;
  * open Promotion visible in gates AND as action='promote' on dev;
  * workspaced unavailable → 200-like result with empty branches, not 502.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.models.task import TaskState


# ── helpers ──────────────────────────────────────────────────────────────────

def _room(gc: Optional[dict] = None):
    r = MagicMock()
    r.id = uuid4()
    r.git_config = gc or {}
    r.creator_id = uuid4()
    return r


def _task(room_id, branch, state=TaskState.WAITING_REVIEW, goal_id=None, title="Task A"):
    t = MagicMock()
    t.id = uuid4()
    t.room_id = room_id
    t.branch_name = branch
    t.state = state
    t.goal_id = goal_id
    t.title = title
    t.ci_status = None
    t.risk = None
    t.divergence_state = None
    return t


def _goal(room_id, branch, title="Goal X", review_state=None):
    g = MagicMock()
    g.id = uuid4()
    g.room_id = room_id
    g.branch_name = branch
    g.title = title
    g.review_state = review_state
    g.review_ci_status = None
    g.status = MagicMock(value="active")
    return g


def _promo(room_id, from_branch="dev", to_branch="main"):
    p = MagicMock()
    p.id = uuid4()
    p.room_id = room_id
    p.from_branch = from_branch
    p.to_branch = to_branch
    p.state = "waiting_review"
    p.ci_status = None
    p.diff_stat = ""
    p.pr_url = None
    p.created_at = MagicMock()
    return p


def _db_returning(*rows_by_call):
    """Returns an AsyncMock db.execute() that yields successive scalars().all()
    or scalars().first() results per call."""
    call_idx = [0]
    results = list(rows_by_call)

    async def execute(_stmt):
        idx = call_idx[0]
        call_idx[0] += 1
        rows = results[idx] if idx < len(results) else []
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = rows if isinstance(rows, list) else [rows]
        scalars_mock.first.return_value = rows[0] if isinstance(rows, list) and rows else (rows if not isinstance(rows, list) else None)
        mock = MagicMock()
        mock.scalars.return_value = scalars_mock
        return mock

    return execute


def _flat_branches(state: git_api.GitStateOut) -> dict[str, git_api.BranchNodeOut]:
    """Flatten the branch tree into name → node."""
    result = {}

    def _walk(nodes):
        for n in nodes:
            result[n.name] = n
            _walk(n.children)

    _walk(state.branches)
    return result


# ── test: room without repo ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_repo_connection_is_empty():
    room = _room()  # no git_config
    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning(
        [],   # tasks
        [],   # goals
        [],   # live_sessions
        [],   # open_promo (first())
        [],   # goal_reviews
    )

    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=[]), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    assert state.connection.repo_url is None
    assert state.gates == []
    # main and dev are present even with no workspaced data
    names = _flat_branches(state).keys()
    assert "main" in names
    assert "dev" in names


# ── test: task without goal nested under dev ──────────────────────────────────

@pytest.mark.asyncio
async def test_task_without_goal_under_dev():
    room = _room()
    t = _task(room.id, branch="task/48", state=TaskState.WAITING_REVIEW)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning(
        [t],  # tasks
        [],   # goals
        [],   # live_sessions
        None, # open_promo (first())
        [],   # goal_reviews
    )

    wsd_branches = [
        {"name": "main",    "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",     "base": "main","ahead": 1,    "behind": 0,   "merged": False},
        {"name": "task/48", "base": "dev", "ahead": 2,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    flat = _flat_branches(state)
    assert "task/48" in flat
    assert flat["task/48"].action == "merge"
    assert flat["task/48"].kind == "task"

    # task/48 is a child of dev
    dev_node = flat["dev"]
    child_names = [c.name for c in dev_node.children]
    assert "task/48" in child_names


# ── test: orphaned session branch is measured against dev and mergeable ─────

@pytest.mark.asyncio
async def test_orphaned_session_branch_offers_merge_into_dev():
    room = _room()
    sid = uuid4()
    name = f"session/{sid}"

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([], [], [], None, [])  # no live session for sid

    def _listing(_room_id, bases):
        return [
            {"name": "main", "base": None, "ahead": None, "behind": None, "merged": None},
            {"name": "dev", "base": "main", "ahead": 0, "behind": 0, "merged": True},
            {"name": name, "base": bases.get(name),
             "ahead": 3 if name in bases else None, "behind": 0 if name in bases else None,
             "merged": False if name in bases else None},
        ]

    lister = AsyncMock(side_effect=_listing)
    with patch.object(git_api.wsd, "list_workspace_branches", lister), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    flat = _flat_branches(state)
    node = flat[name]
    assert node.kind == "session"
    assert node.base == "dev"
    assert node.ahead == 3
    assert node.action == "merge"
    assert node.can_delete is True
    assert name in [c.name for c in flat["dev"].children]


# ── test: terminal task actions (merge shortcut vs return-to-review) ─────────

@pytest.mark.asyncio
async def test_completed_unmerged_offers_direct_merge():
    room = _room()
    t = _task(room.id, branch="task/70", state=TaskState.COMPLETED)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([t], [], [], None, [])

    wsd_branches = [
        {"name": "main",    "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",     "base": "main","ahead": 0,    "behind": 0,   "merged": False},
        {"name": "task/70", "base": "dev", "ahead": 2,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    assert _flat_branches(state)["task/70"].action == "merge"


@pytest.mark.asyncio
async def test_completed_goal_unmerged_offers_direct_merge():
    room = _room()
    goal_id = uuid4()
    goal_branch = f"goal/{goal_id}"
    g = _goal(room.id, branch=goal_branch)
    g.id = goal_id
    g.status = MagicMock(value="completed")
    g.review_state = None

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([], [g], [], None, [])

    wsd_branches = [
        {"name": "main",       "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",        "base": "main","ahead": 0,    "behind": 0,   "merged": False},
        {"name": goal_branch,  "base": "dev", "ahead": 4,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    assert _flat_branches(state)[goal_branch].action == "merge"


@pytest.mark.asyncio
async def test_cancelled_unmerged_still_returns_to_review():
    room = _room()
    t = _task(room.id, branch="task/71", state=TaskState.CANCELLED)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([t], [], [], None, [])

    wsd_branches = [
        {"name": "main",    "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",     "base": "main","ahead": 0,    "behind": 0,   "merged": False},
        {"name": "task/71", "base": "dev", "ahead": 1,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    assert _flat_branches(state)["task/71"].action == "return_to_review"


# ── test: diverged task branch (internal design notes Stage 0) ──────────

@pytest.mark.asyncio
async def test_diverged_task_branch_flag():
    room = _room()
    t = _task(room.id, branch="task/88", state=TaskState.WAITING_REVIEW)
    t.divergence_state = "detected"

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([t], [], [], None, [])

    wsd_branches = [
        {"name": "main",    "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",     "base": "main","ahead": 0,    "behind": 0,   "merged": False},
        {"name": "task/88", "base": "dev", "ahead": 2,    "behind": 3,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    node = _flat_branches(state)["task/88"]
    assert node.diverged is True
    assert node.divergence_state == "detected"


@pytest.mark.asyncio
async def test_resolved_divergence_not_flagged():
    room = _room()
    t = _task(room.id, branch="task/89", state=TaskState.COMPLETED)
    t.divergence_state = "resolved"

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([t], [], [], None, [])

    wsd_branches = [
        {"name": "main",    "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",     "base": "main","ahead": 0,    "behind": 0,   "merged": False},
        {"name": "task/89", "base": "dev", "ahead": 0,    "behind": 0,   "merged": True},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    assert _flat_branches(state)["task/89"].diverged is False


# ── test: task with goal nested under goal branch ────────────────────────────

@pytest.mark.asyncio
async def test_task_with_goal_under_goal_branch():
    room = _room()
    goal_id = uuid4()
    goal_branch = f"goal/{goal_id}"
    g = _goal(room.id, branch=goal_branch)
    g.id = goal_id
    t = _task(room.id, branch="task/55", state=TaskState.WAITING_REVIEW, goal_id=goal_id)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning(
        [t],   # tasks
        [g],   # goals
        [],    # live_sessions
        None,  # open_promo (first())
        [],    # goal_reviews
    )

    wsd_branches = [
        {"name": "main",       "base": None,       "ahead": None, "behind": None, "merged": None},
        {"name": "dev",        "base": "main",      "ahead": 1,    "behind": 0,   "merged": False},
        {"name": goal_branch,  "base": "dev",       "ahead": 3,    "behind": 0,   "merged": False},
        {"name": "task/55",    "base": goal_branch, "ahead": 1,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    flat = _flat_branches(state)
    # task/55 should be a child of goal branch
    goal_node = flat.get(goal_branch)
    assert goal_node is not None
    child_names = [c.name for c in goal_node.children]
    assert "task/55" in child_names


# ── test: open Promotion → gates + dev action='promote' ──────────────────────

@pytest.mark.asyncio
async def test_open_promotion_in_gates_and_dev():
    room = _room()
    promo = _promo(room.id)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)

    def execute_fn(stmt):
        pass

    # Need to handle first() vs all() differently for open_promo vs others
    call_idx = [0]
    returns = [
        [],     # tasks
        [],     # goals
        [],     # live_sessions
        promo,  # open_promo (first())
        [],     # goal_reviews
    ]
    async def execute(_stmt):
        idx = call_idx[0]
        call_idx[0] += 1
        val = returns[idx]
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [val] if val and not isinstance(val, list) else (val or [])
        scalars_mock.first.return_value = val if not isinstance(val, list) else (val[0] if val else None)
        mock = MagicMock()
        mock.scalars.return_value = scalars_mock
        return mock

    db.execute = execute

    wsd_branches = [
        {"name": "main", "base": None,  "ahead": None, "behind": None, "merged": None},
        {"name": "dev",  "base": "main","ahead": 3,    "behind": 0,   "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    # promotion in gates
    assert any(g.kind == "promotion" for g in state.gates)
    promo_gate = next(g for g in state.gates if g.kind == "promotion")
    assert promo_gate.branch == "dev"

    # dev node has action=promote
    flat = _flat_branches(state)
    assert flat["dev"].action == "promote"


# ── test: workspaced unavailable → empty branches, not 502 ──────────────────

@pytest.mark.asyncio
async def test_workspaced_unavailable_returns_empty_branches():
    room = _room()
    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([], [], [], None, [])

    # list_workspace_branches already degrades to [] on error (per Stage 1),
    # so simulate that behaviour directly
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=[]), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    # No 502 raised, and virtual main/dev still appear
    flat = _flat_branches(state)
    assert "main" in flat
    assert "dev" in flat


# ── test: can_delete flag (hand-delete a branch from the Git panel) ─────────

@pytest.mark.asyncio
async def test_can_delete_flags_per_branch_kind():
    """main/dev are protected, a goal branch is deletable, and a task branch
    whose agent is running right now is not."""
    room = _room()
    goal_id = uuid4()
    goal_branch = f"goal/{goal_id}"
    g = _goal(room.id, branch=goal_branch, review_state="waiting_review")
    g.id = goal_id
    running = _task(room.id, branch="task/91", state=TaskState.IN_PROGRESS, goal_id=goal_id)
    idle = _task(room.id, branch="task/92", state=TaskState.WAITING_REVIEW, goal_id=goal_id)

    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([running, idle], [g], [], None, [g])

    wsd_branches = [
        {"name": "main",      "base": None,   "ahead": None, "behind": None, "merged": None},
        {"name": "dev",       "base": "main", "ahead": 0,    "behind": 0,    "merged": False},
        {"name": goal_branch, "base": "dev",  "ahead": 4,    "behind": 0,    "merged": False},
        {"name": "task/91",   "base": goal_branch, "ahead": 1, "behind": 0,  "merged": False},
        {"name": "task/92",   "base": goal_branch, "ahead": 1, "behind": 0,  "merged": False},
        {"name": "spike/foo", "base": "dev",  "ahead": 1,    "behind": 0,    "merged": False},
    ]
    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=wsd_branches), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None):
        state = await git_api.build_git_state(room.id, db)

    flat = _flat_branches(state)
    assert flat["main"].can_delete is False
    assert flat["dev"].can_delete is False
    assert flat[goal_branch].can_delete is True
    assert flat["spike/foo"].can_delete is True
    assert flat["task/91"].can_delete is False   # agent is running
    assert flat["task/92"].can_delete is True
