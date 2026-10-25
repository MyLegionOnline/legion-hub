"""Stage 7 of internal design notes — forge (GitHub) PR helpers.

``git_forge`` on mocked API responses:
  * ``parse_repo`` cleans ``.git`` / ``/blob/`` / ``/tree/`` suffixes;
  * ``create_pull_request`` → 201 returns the PR url/number; 422 falls back to
    the existing open PR; other status → ``ok=False``; a non-GitHub provider is
    refused;
  * ``merge_pull_request`` → 200 returns the merge sha, non-200 → ``ok=False``;
  * ``resolve_forge_token`` returns a legacy inline token and ``None`` for a
    binding without request context.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.core import git_forge


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    def __init__(self, *, post=None, get=None, put=None):
        self._post = post
        self._get = get
        self._put = put
        self.calls: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.calls.append(("POST", url, json))
        return self._post

    async def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params))
        return self._get

    async def put(self, url, json=None, headers=None):
        self.calls.append(("PUT", url, json))
        return self._put


def _patch(monkeypatch, client):
    monkeypatch.setattr(git_forge.httpx, "AsyncClient", lambda *a, **kw: client)
    return client


# ── parse_repo ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/acme/widget", ("github", "acme", "widget")),
        ("https://github.com/acme/widget.git", ("github", "acme", "widget")),
        ("https://github.com/acme/widget/blob/main/x.py", ("github", "acme", "widget")),
        ("https://github.com/acme/widget/tree/dev", ("github", "acme", "widget")),
    ],
)
def test_parse_repo(url, expected):
    assert git_forge.parse_repo({"repo_url": url}) == expected


def test_parse_repo_bad_url():
    with pytest.raises(ValueError):
        git_forge.parse_repo({"repo_url": "not-a-url"})


# ── create_pull_request ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_pr_created(monkeypatch):
    _patch(monkeypatch, _FakeClient(
        post=_Resp(201, {"html_url": "https://github.com/acme/widget/pull/7", "number": 7})
    ))
    out = await git_forge.create_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok",
        head="dev", base="main", title="promote",
    )
    assert out == {"ok": True, "url": "https://github.com/acme/widget/pull/7", "number": 7, "error": None}


@pytest.mark.asyncio
async def test_create_pr_already_exists_falls_back(monkeypatch):
    _patch(monkeypatch, _FakeClient(
        post=_Resp(422, {"message": "A pull request already exists"}, text="422"),
        get=_Resp(200, [{"html_url": "https://github.com/acme/widget/pull/3", "number": 3}]),
    ))
    out = await git_forge.create_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok",
        head="dev", base="main", title="promote",
    )
    assert out["ok"] is True and out["number"] == 3


@pytest.mark.asyncio
async def test_create_pr_error_status(monkeypatch):
    _patch(monkeypatch, _FakeClient(post=_Resp(403, {"message": "nope"}, text="forbidden")))
    out = await git_forge.create_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok",
        head="dev", base="main", title="promote",
    )
    assert out["ok"] is False and "403" in out["error"]


@pytest.mark.asyncio
async def test_create_pr_non_github_refused(monkeypatch):
    out = await git_forge.create_pull_request(
        {"repo_url": "https://gitlab.com/acme/widget", "provider": "gitlab"}, "tok",
        head="dev", base="main", title="promote",
    )
    assert out["ok"] is False and "gitlab" in out["error"]


@pytest.mark.asyncio
async def test_create_pr_network_error(monkeypatch):
    class _Boom(_FakeClient):
        async def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    _patch(monkeypatch, _Boom())
    out = await git_forge.create_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok",
        head="dev", base="main", title="promote",
    )
    assert out["ok"] is False


# ── merge_pull_request ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_pr_ok(monkeypatch):
    _patch(monkeypatch, _FakeClient(put=_Resp(200, {"sha": "deadbeef", "merged": True})))
    out = await git_forge.merge_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok", 7
    )
    assert out == {"ok": True, "sha": "deadbeef", "error": None}


@pytest.mark.asyncio
async def test_merge_pr_conflict(monkeypatch):
    _patch(monkeypatch, _FakeClient(put=_Resp(405, {"message": "not mergeable"}, text="405")))
    out = await git_forge.merge_pull_request(
        {"repo_url": "https://github.com/acme/widget"}, "tok", 7
    )
    assert out["ok"] is False and "405" in out["error"]


# ── resolve_forge_token ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_forge_token_legacy(monkeypatch):
    monkeypatch.setattr(
        "app.core.secrets.runtime.allow_legacy_secret_read", lambda: True, raising=False
    )
    room = SimpleNamespace(git_config={"access_token": "ghp_legacy"}, id="r", creator_id="c")
    assert await git_forge.resolve_forge_token(None, room, request=None) == "ghp_legacy"


@pytest.mark.asyncio
async def test_resolve_forge_token_binding_needs_request(monkeypatch):
    room = SimpleNamespace(
        git_config={"access_token_secret_binding_id": "00000000-0000-0000-0000-000000000001"},
        id="r", creator_id="c",
    )
    assert await git_forge.resolve_forge_token(None, room, request=None) is None
