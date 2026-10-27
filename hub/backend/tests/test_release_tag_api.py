from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app.api import deps
from app.api.v1 import git
from app.core import deploy_publish
from app.db.session import get_session

ROOM = uuid4()
SHA = "a" * 40


@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(git.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_current_user] = lambda: SimpleNamespace(id=uuid4())
    app.dependency_overrides[get_session] = lambda: None
    async def member(*args):
        pass
    monkeypatch.setattr(git, "_verify_membership", member)
    return app


@pytest.mark.asyncio
async def test_create_release_tag_checks_then_publishes_without_card(api, monkeypatch):
    calls = []
    plan = deploy_publish.ReleaseTagPlan("v1.2.3", SHA, False)
    async def check(db, room, tag, *, ref):
        calls.append(("check", room, tag, ref))
        return plan
    async def publish(db, room, passed, *, message):
        calls.append(("publish", room, passed, message))
    monkeypatch.setattr(deploy_publish, "check_release_tag", check)
    monkeypatch.setattr(deploy_publish, "publish_release_tag", publish)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        response = await client.post(f"/api/v1/rooms/{ROOM}/release-tags", json={"tag": "v1.2.3", "ref": SHA, "message": "notes"})
    assert response.status_code == 200
    assert response.json() == {"tag": "v1.2.3", "sha": SHA}
    assert calls == [("check", ROOM, "v1.2.3", SHA), ("publish", ROOM, plan, "notes")]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["tag_exists", "tag_commit_not_published", "deploy_publication_unverified", "repo_not_configured"])
async def test_create_returns_error_code_and_does_not_publish(api, monkeypatch, code):
    async def check(*args, **kwargs):
        raise deploy_publish.DeployPublishError(code, "explanation")
    async def publish(*args, **kwargs):
        pytest.fail("must not publish")
    monkeypatch.setattr(deploy_publish, "check_release_tag", check)
    monkeypatch.setattr(deploy_publish, "publish_release_tag", publish)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        response = await client.post(f"/api/v1/rooms/{ROOM}/release-tags", json={"tag": "v1.2.3"})
    assert response.status_code == 409
    assert response.json()["detail"] == {"code": code, "message": "explanation"}


@pytest.mark.asyncio
async def test_membership_blocks_read_and_write(api, monkeypatch):
    async def forbidden(*args):
        raise HTTPException(status_code=403, detail="Not a room member")
    monkeypatch.setattr(git, "_verify_membership", forbidden)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        assert (await client.get(f"/api/v1/rooms/{ROOM}/release-tags")).status_code == 403
        assert (await client.post(f"/api/v1/rooms/{ROOM}/release-tags", json={"tag": "v1.2.3"})).status_code == 403


@pytest.mark.asyncio
async def test_suggestions_proxy(api, monkeypatch):
    async def suggestions(room):
        assert room == str(ROOM)
        return {"ok": True, "latest": "v1.2.3", "patch": "v1.2.4", "minor": "v1.3.0"}
    monkeypatch.setattr(git.wsd, "git_release_tags_workspace", suggestions)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        response = await client.get(f"/api/v1/rooms/{ROOM}/release-tags")
    assert response.json() == {"latest": "v1.2.3", "patch": "v1.2.4", "minor": "v1.3.0"}
