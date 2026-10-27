"""``main`` that skipped the promotion is not published without a person's yes,
and release tags go out only through ``release_tag``, after one."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core import deploy_publish
from app.core.deploy_publish import DeployPublishError, ReleaseTagPlan
from tests.test_deploy_publish import OTHER, ROOM, SHA, _noop, _World

SNEAKY = [{"sha": OTHER, "message": "chore: commit on main"}]


# ── publishing main ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_main_with_commits_that_skipped_the_promotion_waits_for_a_person(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    world.heads = {"main": ("found", OTHER)}
    world.bypass = SNEAKY
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.ensure_published(None, ROOM)
    assert err.value.code == "main_needs_approval" and err.value.commits == SNEAKY
    assert world.pushed == []
    assert world.bypass_excluded == [OTHER]  # what the forge has is not reported

    done = await deploy_publish.ensure_published(None, ROOM, allow_main_bypass=True)
    assert done.sha == SHA and world.pushed == ["main"]


@pytest.mark.asyncio
async def test_a_main_never_pushed_is_checked_against_dev_only(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    await deploy_publish.ensure_published(None, ROOM)
    assert world.bypass_excluded == [None] and world.pushed == ["main"]


@pytest.mark.asyncio
async def test_other_branches_and_an_up_to_date_main_are_not_checked(monkeypatch):
    world = _World(monkeypatch)
    world.bypass = SNEAKY
    world.local = {"main": SHA, "dev": SHA}
    world.heads = {"main": ("found", SHA)}
    assert (await deploy_publish.ensure_published(None, ROOM)).pushed is False
    await deploy_publish.ensure_published(None, ROOM, branch="dev")
    assert world.bypass_excluded == [] and world.pushed == ["dev"]


def _ctx():
    return SimpleNamespace(db=None, room_id=ROOM, session_id=uuid4())


@pytest.mark.asyncio
async def test_publish_for_deploy_asks_before_publishing_such_a_main(monkeypatch):
    from app.core.tools import _publish_for_deploy

    calls = []

    async def ensure(db, room_id, allow_main_bypass=False, **kwargs):
        calls.append(allow_main_bypass)
        if not allow_main_bypass:
            raise DeployPublishError("main_needs_approval", "'main' carries 1 commit(s)", commits=SNEAKY)
        return deploy_publish.PublishedCommit(SHA, "main", pushed=True)

    answers = iter([False, True])
    asked = []

    async def confirm(db, **kwargs):
        asked.append(kwargs["commits"])
        return next(answers)

    monkeypatch.setattr(deploy_publish, "ensure_published", ensure)
    monkeypatch.setattr(deploy_publish, "confirm_main_bypass_publish", confirm)
    monkeypatch.setattr(_publish_for_deploy, "release_tool_db", _noop)
    tool = _publish_for_deploy.PublishForDeployTool()

    denied = await tool.execute(_ctx(), {})
    assert denied.startswith("Error: [main_needs_approval]") and "task branch" in denied
    assert calls == [False] and asked == [SNEAKY]

    allowed = await tool.execute(_ctx(), {})
    assert SHA in allowed and calls == [False, False, True]


# ── release tags ────────────────────────────────────────────────────────────


class _Tags:
    def __init__(self, monkeypatch):
        self.check = {"ok": True, "tag": "v1.0.0", "sha": SHA, "exists": False}
        self.create = {"ok": True, "tag": "v1.0.0", "sha": SHA, "exists": False, "created": True}
        self.push_errors = []
        self.calls = []

        async def release_tag(room_id, tag, ref="main", message="", create=False):
            self.calls.append(("create" if create else "check", tag, ref, message))
            return self.create if create else self.check

        async def push_tag(room_id, git_config, tag, access_token=None):
            self.calls.append(("push", tag))
            return {"pushed": 0 if self.push_errors else 1, "errors": self.push_errors}

        monkeypatch.setattr("app.core.workspaced_client.git_release_tag_workspace", release_tag)
        monkeypatch.setattr("app.core.git_sync.push_tag_to_git", push_tag)


@pytest.mark.asyncio
async def test_a_tag_on_a_published_main_commit_is_planned(monkeypatch):
    world = _World(monkeypatch)
    tags = _Tags(monkeypatch)
    plan = await deploy_publish.check_release_tag(None, ROOM, "v1.0.0")
    assert plan == ReleaseTagPlan("v1.0.0", SHA, False, "main")
    assert tags.calls == [("check", "v1.0.0", "main", "")]


@pytest.mark.asyncio
async def test_workspace_refusals_keep_their_code(monkeypatch):
    world = _World(monkeypatch)
    tags = _Tags(monkeypatch)
    tags.check = {"ok": False, "code": "not_on_main", "error": "abc is not on 'main'"}
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_release_tag(None, ROOM, "v1.0.0", ref="dev")
    assert err.value.code == "tag_not_on_main"
    tags.check = {"ok": False, "code": "tag_exists", "error": "tag 'v1.0.0' already exists"}
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_release_tag(None, ROOM, "v1.0.0")
    assert err.value.code == "tag_exists"


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["missing", "outside_branch"])
async def test_a_commit_the_repository_main_lacks_is_not_tagged(monkeypatch, relation):
    world = _World(monkeypatch)
    _Tags(monkeypatch)
    world.relation = (relation, None)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_release_tag(None, ROOM, "v1.0.0")
    assert err.value.code == "tag_commit_not_published" and "publish_for_deploy" in str(err.value)


@pytest.mark.asyncio
async def test_a_forge_that_cannot_be_asked_stops_the_tag(monkeypatch):
    world = _World(monkeypatch)
    _Tags(monkeypatch)
    world.relation = ("unknown", None)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_release_tag(None, ROOM, "v1.0.0")
    assert err.value.code == "deploy_publication_unverified"


@pytest.mark.asyncio
async def test_publishing_creates_then_pushes_exactly_the_tag(monkeypatch):
    world = _World(monkeypatch)
    tags = _Tags(monkeypatch)
    plan = ReleaseTagPlan("v1.0.0", SHA, False)
    await deploy_publish.publish_release_tag(None, ROOM, plan, message="notes")
    assert tags.calls == [("create", "v1.0.0", SHA, "notes"), ("push", "v1.0.0")]


@pytest.mark.asyncio
async def test_a_tag_the_repository_has_elsewhere_is_reported_as_existing(monkeypatch):
    world = _World(monkeypatch)
    tags = _Tags(monkeypatch)
    tags.push_errors = ["! [rejected] v1.0.0 -> v1.0.0 (already exists)"]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.publish_release_tag(None, ROOM, ReleaseTagPlan("v1.0.0", SHA, False))
    assert err.value.code == "tag_exists"
    tags.push_errors = ["fatal: unable to access"]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.publish_release_tag(None, ROOM, ReleaseTagPlan("v1.0.0", SHA, False))
    assert err.value.code == "tag_not_published"


# ── the tool ────────────────────────────────────────────────────────────────


def _tool_world(monkeypatch, *, approve: bool):
    from app.core.tools import _release_tag

    seen = {"published": [], "asked": []}

    async def check(db, room_id, tag, *, ref="main", request_id=None):
        return ReleaseTagPlan(tag, SHA, False)

    async def confirm(db, **kwargs):
        seen["asked"].append(kwargs)
        return approve

    async def publish(db, room_id, plan, *, message="", request_id=None):
        seen["published"].append((plan.tag, message))

    async def subject(ctx, sha):
        return "promote: dev -> main"

    monkeypatch.setattr(deploy_publish, "check_release_tag", check)
    monkeypatch.setattr(deploy_publish, "confirm_release_tag", confirm)
    monkeypatch.setattr(deploy_publish, "publish_release_tag", publish)
    monkeypatch.setattr(_release_tag, "_subject", subject)
    monkeypatch.setattr(_release_tag, "release_tool_db", _noop)
    return _release_tag.ReleaseTagTool(), seen


@pytest.mark.asyncio
async def test_the_tool_publishes_only_after_a_yes(monkeypatch):
    tool, seen = _tool_world(monkeypatch, approve=True)
    out = await tool.execute(_ctx(), {"tag": "v1.0.0", "message": "notes"})
    assert out.startswith("Released 'v1.0.0'") and SHA in out
    assert seen["published"] == [("v1.0.0", "notes")]
    assert seen["asked"][0]["subject"] == "promote: dev -> main"


@pytest.mark.asyncio
async def test_the_tool_does_nothing_on_a_no(monkeypatch):
    tool, seen = _tool_world(monkeypatch, approve=False)
    out = await tool.execute(_ctx(), {"tag": "v1.0.0"})
    assert out.startswith("Error: [tag_needs_approval]") and seen["published"] == []


@pytest.mark.asyncio
async def test_the_tool_reports_a_refusal_with_its_code(monkeypatch):
    from app.core.tools import _release_tag

    async def check(db, room_id, tag, **kwargs):
        raise DeployPublishError("tag_invalid", "'1.0' is not a release tag")

    monkeypatch.setattr(deploy_publish, "check_release_tag", check)
    monkeypatch.setattr(_release_tag, "release_tool_db", _noop)
    out = await _release_tag.ReleaseTagTool().execute(_ctx(), {"tag": "1.0"})
    assert out == "Error: [tag_invalid] '1.0' is not a release tag"
    assert (await _release_tag.ReleaseTagTool().execute(_ctx(), {})).startswith("Error: tag is required")


def test_the_tool_is_offered_to_agents_only():
    import app.core.tools  # noqa: F401  (importing registers every tool)
    from app.core.tools.registry import REGISTRY

    tool = REGISTRY.get("release_tag")
    assert tool is not None and tool.roles == frozenset({"agent"}) and tool.mutating
