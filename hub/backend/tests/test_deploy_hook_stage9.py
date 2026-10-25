"""Stage 9 of internal design notes — post-promotion deploy hook.

Covers ``room_git_flow``:
  * ``_deploy_hook_config`` reads ``deploy_hook_url`` / ``deploy_hook_secret`` /
    ``deploy_push_main`` from ``git_config`` (all opt-in);
  * ``_collect_promoted_commits`` asks ``workspaced`` for ``main..dev`` and is
    fail-open on a transport error;
  * ``dispatch_deploy_hook``:
      - no URL configured → no POST, ``posted=False``;
      - URL + secret → one signed POST (HMAC-SHA256 of the raw body in
        ``X-Legion-Signature-256``) carrying the promoted commits, then
        ``DEPLOY_TRIGGERED``;
      - non-2xx response / network error → ``DEPLOY_HOOK_FAILED``, no raise;
      - ``deploy_push_main`` → ``main`` pushed to the remote;
  * ``resolve_promotion`` approve fires the hook with the pre-merge commit list.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def add(self, obj):  # noqa: D401
        pass

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


class _Resp:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _FakeClient:
    """Records the single POST ``dispatch_deploy_hook`` makes."""

    last: "_FakeClient | None" = None

    def __init__(self, *a, status_code=200, boom=False, **kw):
        self.status_code = status_code
        self.boom = boom
        self.calls: list[dict] = []
        _FakeClient.last = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, content=None, headers=None):
        self.calls.append({"url": url, "content": content, "headers": headers or {}})
        if self.boom:
            raise RuntimeError("connection reset")
        return _Resp(self.status_code)


def _promo(**kw):
    base = dict(
        id=uuid4(),
        from_branch="dev",
        to_branch="main",
        task_ids=["t1", "t2"],
        diff_stat=" a.py | 2 +-\n 1 file changed",
        merge_commit="mergesha",
        resolved_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture()
def emit(monkeypatch):
    m = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", m)
    return m


def _types(emit) -> list[str]:
    return [c.kwargs["event_type"] for c in emit.await_args_list]


def _patch_client(monkeypatch, **kw):
    monkeypatch.setattr(
        room_git_flow.httpx, "AsyncClient", lambda *a, **ckw: _FakeClient(**kw)
    )


# ── _deploy_hook_config ─────────────────────────────────────────────────────

def test_deploy_hook_config_defaults_off():
    assert room_git_flow._deploy_hook_config(SimpleNamespace(git_config={})) == (None, None, False)
    assert room_git_flow._deploy_hook_config(SimpleNamespace(git_config=None)) == (None, None, False)


def test_deploy_hook_config_reads_values():
    room = SimpleNamespace(git_config={
        "deploy_hook_url": " https://ci.example/hook ",
        "deploy_hook_secret": "s3cr3t",
        "deploy_push_main": True,
    })
    assert room_git_flow._deploy_hook_config(room) == ("https://ci.example/hook", "s3cr3t", True)


# ── _collect_promoted_commits ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_promoted_commits_uses_range(monkeypatch):
    log = AsyncMock(return_value={"entries": [
        {"sha": "a" * 40, "short_sha": "aaaaaaaa", "message": "feat", "author": "x", "timestamp": "t"},
        {"sha": "b" * 40, "short_sha": "bbbbbbbb", "message": "fix", "author": "y", "timestamp": "t2"},
    ]})
    monkeypatch.setattr("app.core.workspaced_client.git_log_workspace", log)

    out = await room_git_flow._collect_promoted_commits(str(uuid4()), "dev", "main", _FakeDB())

    assert [c["message"] for c in out] == ["feat", "fix"]
    assert log.await_args.kwargs["rev_range"] == "main..dev"


@pytest.mark.asyncio
async def test_collect_promoted_commits_fail_open(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_log_workspace",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    out = await room_git_flow._collect_promoted_commits(str(uuid4()), "dev", "main", _FakeDB())
    assert out == []


# ── dispatch_deploy_hook ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_no_url_is_noop(monkeypatch, emit):
    _patch_client(monkeypatch)
    room = SimpleNamespace(git_config={})

    out = await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, _promo(), [], _FakeDB())

    assert out == {"posted": False, "pushed_main": False}
    assert _types(emit) == []


@pytest.mark.asyncio
async def test_dispatch_posts_signed_payload(monkeypatch, emit):
    _patch_client(monkeypatch, status_code=202)
    room = SimpleNamespace(git_config={
        "deploy_hook_url": "https://ci.example/hook",
        "deploy_hook_secret": "topsecret",
    })
    commits = [{"sha": "a" * 40, "short_sha": "aaaaaaaa", "message": "feat", "author": "x", "timestamp": "t"}]
    promo = _promo()

    out = await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, promo, commits, _FakeDB())

    assert out["posted"] is True and out["status_code"] == 202
    call = _FakeClient.last.calls[0]
    assert call["url"] == "https://ci.example/hook"

    body = call["content"]
    expected = "sha256=" + hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    assert call["headers"]["X-Legion-Signature-256"] == expected
    assert call["headers"]["X-Legion-Event"] == "promoted_to_main"

    payload = json.loads(body)
    assert payload["event"] == "promoted_to_main"
    assert payload["merge_commit"] == "mergesha"
    assert payload["commits"] == commits
    assert payload["task_ids"] == ["t1", "t2"]

    assert "DEPLOY_TRIGGERED" in _types(emit)


@pytest.mark.asyncio
async def test_dispatch_unsigned_when_no_secret(monkeypatch, emit):
    _patch_client(monkeypatch)
    room = SimpleNamespace(git_config={"deploy_hook_url": "https://ci.example/hook"})

    await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, _promo(), [], _FakeDB())

    assert "X-Legion-Signature-256" not in _FakeClient.last.calls[0]["headers"]


@pytest.mark.asyncio
async def test_dispatch_non_2xx_reports_failure(monkeypatch, emit):
    _patch_client(monkeypatch, status_code=500)
    room = SimpleNamespace(git_config={"deploy_hook_url": "https://ci.example/hook"})

    out = await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, _promo(), [], _FakeDB())

    assert out["posted"] is False and out["status_code"] == 500
    assert "DEPLOY_HOOK_FAILED" in _types(emit)


@pytest.mark.asyncio
async def test_dispatch_network_error_reports_failure(monkeypatch, emit):
    _patch_client(monkeypatch, boom=True)
    room = SimpleNamespace(git_config={"deploy_hook_url": "https://ci.example/hook"})

    out = await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, _promo(), [], _FakeDB())

    assert out["posted"] is False
    assert "DEPLOY_HOOK_FAILED" in _types(emit)


@pytest.mark.asyncio
async def test_dispatch_push_main(monkeypatch, emit):
    _patch_client(monkeypatch)
    push = AsyncMock(return_value=True)
    monkeypatch.setattr(room_git_flow, "_push_branch_best_effort", push)
    room = SimpleNamespace(git_config={"deploy_push_main": True})

    out = await room_git_flow.dispatch_deploy_hook(str(uuid4()), room, _promo(), [], _FakeDB())

    assert out["pushed_main"] is True
    assert push.await_args.args[2] == "main"


# ── resolve_promotion wiring ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_approve_fires_deploy_hook(monkeypatch, emit):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={})),
    )
    monkeypatch.setattr(
        "app.core.workspaced_client.merge_workspace_branch",
        AsyncMock(return_value={"ok": True, "conflict": False, "sha": "mainsha"}),
    )
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", AsyncMock())
    commits = [{"sha": "c" * 40, "short_sha": "cccccccc", "message": "x", "author": "a", "timestamp": "t"}]
    monkeypatch.setattr(
        room_git_flow, "_collect_promoted_commits", AsyncMock(return_value=commits)
    )
    dispatch = AsyncMock(return_value={"posted": True, "pushed_main": False})
    monkeypatch.setattr(room_git_flow, "dispatch_deploy_hook", dispatch)

    promo = _promo(ci_status="passed", state="waiting_review", note=None,
                   resolved_by=None, pr_url=None)
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve"
    )

    assert out["ok"] is True
    dispatch.assert_awaited_once()
    assert dispatch.await_args.args[3] == commits
    assert out["deploy"] == {"posted": True, "pushed_main": False}


@pytest.mark.asyncio
async def test_resolve_reject_does_not_fire_deploy_hook(monkeypatch, emit):
    dispatch = AsyncMock()
    monkeypatch.setattr(room_git_flow, "dispatch_deploy_hook", dispatch)

    promo = _promo(ci_status="passed", state="waiting_review", note=None, resolved_by=None)
    await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="reject", note="later"
    )
    dispatch.assert_not_awaited()
