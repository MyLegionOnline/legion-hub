"""Fixes for the Pull conflict review acceptance findings (no database)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import divergence_review, remote_divergence, room_git_flow

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "conflict_explanation": {"type": "string"},
        "recommendation": {"type": "string", "enum": ["rework", "keep_local"]},
    },
    "required": ["summary", "conflict_explanation", "recommendation"],
}


async def _analyse(monkeypatch, answer: dict) -> dict:
    monkeypatch.setattr(divergence_review, "_resolve_cm_llm", AsyncMock(return_value=(None, {}, None)))
    monkeypatch.setattr(divergence_review, "release_db", AsyncMock())
    monkeypatch.setattr(
        "app.core.chat_master.get_llm_json_completion",
        AsyncMock(return_value=SimpleNamespace(text=json.dumps(answer))),
    )
    return await divergence_review.analyse_with_chatmaster(
        None, None, digest="digest", prompt="prompt", schema=_SCHEMA,
        default_decision="rework", fallback=lambda: {"_fallback": True}, validate_required=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("filler", ["...", "…", " - ", "TBD", "n/a", "short"])
async def test_placeholder_fields_fall_back_and_are_logged(monkeypatch, caplog, filler):
    answer = {"summary": filler, "conflict_explanation": "Both sides edited .hero margins.",
              "recommendation": "rework"}
    with caplog.at_level(logging.WARNING, logger=divergence_review.logger.name):
        analysis = await _analyse(monkeypatch, answer)
    assert analysis == {"_fallback": True}
    assert "summary" in caplog.text and "rejected" in caplog.text


@pytest.mark.asyncio
async def test_real_analysis_is_accepted(monkeypatch):
    answer = {"summary": "Both sides changed the hero block.",
              "conflict_explanation": "Margins differ in style.css.", "recommendation": "keep_local"}
    assert await _analyse(monkeypatch, answer) == answer


def _row(**changes):
    values = {"state": "reviewed", "analysis": {"_fallback": True}, "rework_task_id": None, "decision": None}
    return SimpleNamespace(**{**values, **changes})


def test_only_an_undecided_fallback_review_can_be_rerun():
    assert remote_divergence.can_rereview(_row())
    assert not remote_divergence.can_rereview(_row(analysis={"summary": "ChatMaster answer"}))
    assert not remote_divergence.can_rereview(_row(rework_task_id=uuid4(), decision="rework"))
    assert not remote_divergence.can_rereview(_row(state="resolved"))


def test_prompt_fixes_the_recommendation_order():
    prompt = remote_divergence._SYSTEM_PROMPT
    assert prompt.index("-> rework") < prompt.index("-> take_remote") < prompt.index("-> keep_local")


@pytest.mark.asyncio
async def test_merge_task_branch_records_the_merge_commit(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(room_git_flow, "commit_working_tree", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "app.core.workspaced_client.merge_workspace_branch",
        AsyncMock(return_value={"ok": True, "conflict": False, "sha": "f" * 40}),
    )
    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=f"task/{tid}", title="T", merge_commit=None)
    await room_git_flow.merge_task_branch("room-1", task, None)
    assert task.merge_commit == "f" * 40
