"""internal design notes — "Divergence review".

Covers:
  * resolve_divergence take_branch → merge_workspace_branch, merge_commit
    stamped, task COMPLETED + divergence_state="resolved", TASK_MERGED emitted;
  * resolve_divergence delete_branch → git_discard_branch, task CANCELLED;
  * resolve_divergence rework → fresh task created + dispatch requested,
    source task CONTINUED, branch NOT deleted;
  * resolve_divergence unknown decision / stale (report !ok) → ok=False;
  * take_branch when the merge conflicts → ok=False, task untouched;
  * run_divergence_review with no ChatMaster → deterministic analysis, still
    emits DIVERGENCE_REVIEW and moves the task to "reviewed";
  * endpoints: 422 without a branch, 409 when already resolved,
    422 on an unknown decision.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.core import divergence_review as dr
from app.models.task import TaskState


def _task(state=TaskState.WAITING_REVIEW, branch="task/abc", divergence_state="reviewed",
          description="Lay out the hero from the mockup images/variant-c-hero.png"):
    t = MagicMock()
    t.id = uuid4()
    t.room_id = uuid4()
    t.branch_name = branch
    t.state = state
    t.goal_id = None
    t.title = "Hero section"
    t.description = description
    t.merge_commit = None
    t.origin = None
    t.rework_count = 0
    t.assigned_agent_id = uuid4()
    t.divergence_state = divergence_state
    return t


def _db():
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    return db


_OK_REPORT = {"ok": True, "branch": "task/abc", "base": "dev", "overlap_files": ["f.txt"]}


# ── take_branch ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_take_branch_folds_and_completes():
    task = _task()
    db = _db()
    with patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_divergence_workspace", new_callable=AsyncMock, return_value=_OK_REPORT), \
         patch("app.core.workspaced_client.merge_workspace_branch", new_callable=AsyncMock,
               return_value={"ok": True, "sha": "cafe1234", "conflict": False}), \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock) as emit:
        res = await dr.resolve_divergence(task.room_id, task, db, "take_branch")

    assert res["ok"] is True
    assert task.merge_commit == "cafe1234"
    assert task.state == TaskState.COMPLETED
    assert task.divergence_state == "resolved"
    assert {c.kwargs.get("event_type") for c in emit.call_args_list} == {"DIVERGENCE_RESOLVED", "TASK_MERGED"}


@pytest.mark.asyncio
async def test_take_branch_conflict_leaves_task_untouched():
    task = _task()
    db = _db()
    with patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_divergence_workspace", new_callable=AsyncMock, return_value=_OK_REPORT), \
         patch("app.core.workspaced_client.merge_workspace_branch", new_callable=AsyncMock,
               return_value={"ok": False, "conflict": True, "stderr": "modify/delete"}), \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock):
        res = await dr.resolve_divergence(task.room_id, task, db, "take_branch")

    assert res["ok"] is False
    assert task.divergence_state == "reviewed"
    assert task.state == TaskState.WAITING_REVIEW


# ── delete_branch ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_branch_discards_and_cancels():
    task = _task()
    db = _db()
    with patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_discard_branch_workspace", new_callable=AsyncMock,
               return_value={"ok": True, "deleted": True, "backup_ref": "refs/legion/trash/task-abc@x"}), \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock) as emit:
        res = await dr.resolve_divergence(task.room_id, task, db, "delete_branch")

    assert res["ok"] is True and res["backup_ref"].startswith("refs/legion/trash/")
    assert task.state == TaskState.CANCELLED
    assert task.divergence_state == "resolved"
    assert emit.call_args_list[0].kwargs["event_type"] == "DIVERGENCE_RESOLVED"


# ── rework ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rework_spawns_and_dispatches_task():
    task = _task()
    db = _db()
    ev = MagicMock()
    ev.event_metadata = {"analysis": {"rework_brief": "move the X fix onto dev"}}
    scalars = MagicMock(); scalars.first.return_value = ev
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=scalars)))

    with patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_divergence_workspace", new_callable=AsyncMock, return_value=_OK_REPORT), \
         patch("app.core.workspaced_client.git_discard_branch_workspace", new_callable=AsyncMock,
               return_value={"ok": True, "deleted": True, "backup_ref": "refs/legion/trash/task-abc@x"}) as discard, \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock), \
         patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock, return_value="started") as disp:
        res = await dr.resolve_divergence(task.room_id, task, db, "rework", note="header only")

    assert res["ok"] is True and res["decision"] == "rework"
    assert task.state == TaskState.CONTINUED
    assert task.divergence_state == "resolved"
    disp.assert_awaited_once()
    discard.assert_awaited_once()  # stale branch retired, not left dangling
    # a new Task was added (source task is add()ed too, but a Task instance with
    # the rework title must be among them)
    added_titles = [getattr(c.args[0], "title", "") for c in db.add.call_args_list]
    assert any("Divergence rework" in t for t in added_titles)

    # The rework task must carry the source context forward, not just the
    # CM paraphrase: original description verbatim + the mockup path it named.
    rework_desc = next(
        getattr(c.args[0], "description", "")
        for c in db.add.call_args_list
        if "Divergence rework" in getattr(c.args[0], "title", "")
    )
    assert "move the X fix onto dev" in rework_desc          # the CM brief
    assert "## Context of the original task" in rework_desc
    assert "images/variant-c-hero.png" in rework_desc          # mockup pointer preserved
    assert "header only" in rework_desc                       # user note preserved


@pytest.mark.asyncio
async def test_source_context_block_lists_assigned_skills():
    task = _task(description="see design/hero.spec.json")
    db = _db()
    rows = [("Layout from the mockup", "ui-from-design")]
    db.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=rows)))

    block = await dr._source_context_block(task, db)

    assert "design/hero.spec.json" in block
    assert "Layout from the mockup" in block
    assert "Follow their process" in block


# ── guards ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_decision_rejected():
    task = _task()
    res = await dr.resolve_divergence(task.room_id, task, _db(), "nuke_from_orbit")
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_stale_report_blocks_take_branch():
    task = _task()
    with patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_divergence_workspace", new_callable=AsyncMock,
               return_value={"ok": False, "error": "ref not found: task/abc"}):
        res = await dr.resolve_divergence(task.room_id, task, _db(), "take_branch")
    assert res["ok"] is False


# ── run_divergence_review, no ChatMaster ───────────────────────────────────

@pytest.mark.asyncio
async def test_run_review_without_chatmaster_uses_fallback():
    task = _task(divergence_state="detected")
    db = _db()
    scalars = MagicMock(); scalars.first.return_value = task
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=scalars)))

    ctx = {"branch": "task/abc", "base": "dev",
           "report": {"ok": True, "branch": "task/abc", "base": "dev",
                      "branch_commits": [], "base_commits": [], "overlap_files": ["f.txt"],
                      "merge_base": {}, "base_head": {}, "branch_head": {}},
           "digest": "..."}

    with patch("app.core.room_git_flow._load_room", new_callable=AsyncMock, return_value=MagicMock()), \
         patch("app.core.divergence_review.build_divergence_context", new_callable=AsyncMock, return_value=ctx), \
         patch("app.core.divergence_review._resolve_cm_llm", new_callable=AsyncMock,
               side_effect=dr.DivergenceReviewError("no ChatMaster")), \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock) as emit:
        await dr.run_divergence_review(task.room_id, task.id, db)

    assert task.divergence_state == "reviewed"
    call = emit.call_args_list[-1]
    assert call.kwargs["event_type"] == "DIVERGENCE_REVIEW"
    analysis = call.kwargs["metadata"]["analysis"]
    assert analysis["_fallback"] is True
    assert analysis["recommendation"] in dr.DECISIONS


@pytest.mark.asyncio
async def test_run_review_emits_degraded_card_when_report_fails():
    """workspaced /git/divergence unreachable (stale service) → still a card,
    task lands on 'reviewed', never stuck on 'reviewing'."""
    task = _task(divergence_state="detected")
    db = _db()
    scalars = MagicMock(); scalars.first.return_value = task
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=scalars)))

    with patch("app.core.room_git_flow._load_room", new_callable=AsyncMock, return_value=MagicMock()), \
         patch("app.core.room_git_flow._task_base_branch", new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.workspaced_client.git_divergence_workspace", new_callable=AsyncMock,
               return_value={"ok": False, "error": "workspaced unreachable: boom"}), \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock) as emit:
        await dr.run_divergence_review(task.room_id, task.id, db)

    assert task.divergence_state == "reviewed"
    call = emit.call_args_list[-1]
    assert call.kwargs["event_type"] == "DIVERGENCE_REVIEW"
    analysis = call.kwargs["metadata"]["analysis"]
    assert analysis["_degraded"] is True
    assert analysis["recommendation"] in dr.DECISIONS


def test_fallback_analysis_has_required_fields():
    a = dr._fallback_analysis({"base": "dev", "branch": "task/x", "branch_commits": [], "base_commits": [],
                               "overlap_files": ["a.py"], "merge_base": {}, "base_head": {}})
    for key in ("summary", "recommendation", "take_branch_consequence", "rework_brief"):
        assert a[key]


@pytest.mark.asyncio
async def test_redundant_branch_recommends_delete_without_llm():
    """A branch whose work is already in the base (rework landed) → the review
    skips the LLM and recommends delete_branch."""
    task = _task(divergence_state="detected")
    db = _db()
    scalars = MagicMock(); scalars.first.return_value = task
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=scalars)))

    ctx = {"branch": "task/abc", "base": "dev",
           "report": {"ok": True, "redundant": True, "branch": "task/abc", "base": "dev",
                      "branch_commits": [], "base_commits": [], "overlap_files": [],
                      "merge_base": {}, "base_head": {}, "branch_head": {}},
           "digest": "..."}

    with patch("app.core.room_git_flow._load_room", new_callable=AsyncMock, return_value=MagicMock()), \
         patch("app.core.divergence_review.build_divergence_context", new_callable=AsyncMock, return_value=ctx), \
         patch("app.core.divergence_review._resolve_cm_llm", new_callable=AsyncMock) as cm, \
         patch("app.core.divergence_review.emit_event", new_callable=AsyncMock) as emit:
        await dr.run_divergence_review(task.room_id, task.id, db)

    cm.assert_not_awaited()  # LLM skipped entirely
    analysis = emit.call_args_list[-1].kwargs["metadata"]["analysis"]
    assert analysis["recommendation"] == "delete_branch"
    assert analysis["_redundant"] is True


# ── endpoints ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_endpoint_request_review_no_branch_422():
    from app.api.v1 import tasks as tasks_api
    task = _task(branch=None)
    db = AsyncMock()
    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.request_divergence_review(task.room_id, task.id, db=db, current_user=MagicMock())
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_endpoint_resolve_already_resolved_409():
    from app.api.v1 import tasks as tasks_api
    from app.schemas.task import DivergenceResolveRequest
    task = _task(divergence_state="resolved")
    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.resolve_divergence_review(
                task.room_id, task.id,
                body=DivergenceResolveRequest(decision="take_branch"),
                db=AsyncMock(), current_user=MagicMock(),
            )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_endpoint_resolve_unknown_decision_422():
    from app.api.v1 import tasks as tasks_api
    from app.schemas.task import DivergenceResolveRequest
    task = _task()
    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks._get_task", new_callable=AsyncMock, return_value=task):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.resolve_divergence_review(
                task.room_id, task.id,
                body=DivergenceResolveRequest(decision="bogus"),
                db=AsyncMock(), current_user=MagicMock(),
            )
    assert exc.value.status_code == 422
