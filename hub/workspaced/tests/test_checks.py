"""Stage 2 of internal design notes — pre-merge checks (git-layer / workspaced).

Covers ``app.checks.run_checks``:
  * a room with no matching checks is a no-op (``ran=False``, ``passed=True``);
  * ``scope="quick"`` skips checks flagged ``quick: false``;
  * every command green → ``passed=True``; one red / timed-out → ``passed=False``;
  * a failing ``ref`` checkout aborts before any check runs.
"""
from __future__ import annotations

import pytest

from app import checks


def _exec_fn(script):
    calls: list[tuple[list[str], int, dict | None]] = []

    async def exec_fn(cmd, timeout_sec, env=None):
        calls.append((cmd, timeout_sec, env))
        return script(cmd)

    exec_fn.calls = calls
    return exec_fn


@pytest.mark.asyncio
async def test_no_checks_is_a_noop(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [])
    out = await checks.run_checks("r", "quick", None, 60, exec_fn=_exec_fn(lambda c: {}))
    assert out == {"passed": True, "ran": False, "scope": "quick", "results": [], "error": None}


@pytest.mark.asyncio
async def test_quick_scope_skips_non_quick(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [
        {"name": "lint", "command": "ruff .", "quick": True},
        {"name": "e2e", "command": "playwright test", "quick": False},
    ])
    fn = _exec_fn(lambda c: {"exit_code": 0})
    out = await checks.run_checks("r", "quick", None, 60, exec_fn=fn)
    assert out["ran"] is True and out["passed"] is True
    assert [r["name"] for r in out["results"]] == ["lint"]


@pytest.mark.asyncio
async def test_full_scope_runs_everything(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [
        {"name": "lint", "command": "ruff .", "quick": True},
        {"name": "e2e", "command": "playwright test", "quick": False},
    ])
    out = await checks.run_checks("r", "full", None, 60, exec_fn=_exec_fn(lambda c: {"exit_code": 0}))
    assert [r["name"] for r in out["results"]] == ["lint", "e2e"]


@pytest.mark.asyncio
async def test_one_red_check_fails_the_run(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [
        {"name": "a", "command": "true"},
        {"name": "b", "command": "false"},
    ])

    def script(cmd):
        return {"exit_code": 0} if "true" in cmd[-1] else {"exit_code": 1, "stderr": "boom"}

    out = await checks.run_checks("r", "quick", None, 60, exec_fn=_exec_fn(script))
    assert out["passed"] is False
    assert out["results"][1]["exit_code"] == 1
    assert "boom" in out["results"][1]["tail"]


@pytest.mark.asyncio
async def test_timed_out_check_fails_the_run(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [{"name": "slow", "command": "sleep 999"}])
    out = await checks.run_checks(
        "r", "quick", None, 5,
        exec_fn=_exec_fn(lambda c: {"exit_code": -1, "timed_out": True}),
    )
    assert out["passed"] is False
    assert out["results"][0]["timed_out"] is True


@pytest.mark.asyncio
async def test_ref_checkout_failure_aborts(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [{"name": "a", "command": "true"}])
    fn = _exec_fn(lambda c: {"exit_code": 1, "stderr": "unknown ref"})
    out = await checks.run_checks("r", "quick", "task/nope", 60, exec_fn=fn)
    assert out["passed"] is False and out["ran"] is True
    assert "unknown ref" in out["error"]
    # only the checkout ran — no check commands
    assert len(fn.calls) == 1


@pytest.mark.asyncio
async def test_ref_checkout_success_then_runs(monkeypatch):
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [{"name": "a", "command": "pytest -q"}])
    fn = _exec_fn(lambda c: {"exit_code": 0})
    out = await checks.run_checks("r", "quick", "task/abc", 60, exec_fn=fn)
    assert out["passed"] is True
    assert "checkout task/abc" in fn.calls[0][0][-1]
    assert fn.calls[1][0][-1] == "pytest -q"


@pytest.mark.asyncio
async def test_env_reaches_check_commands_but_not_the_checkout(monkeypatch):
    """internal design notes Stage 6 — Hub-resolved env goes to every
    check command, never to the ``git checkout`` ahead of them."""
    monkeypatch.setattr(checks, "get_checks", lambda room_id: [{"name": "a", "command": "pytest -q"}])
    fn = _exec_fn(lambda c: {"exit_code": 0})
    out = await checks.run_checks(
        "r", "quick", "task/abc", 60, exec_fn=fn, env={"DEPLOY_TOKEN": "supersecretvalue"}
    )
    assert out["passed"] is True
    assert fn.calls[0][2] is None  # checkout
    assert fn.calls[1][2] == {"DEPLOY_TOKEN": "supersecretvalue"}  # check command
