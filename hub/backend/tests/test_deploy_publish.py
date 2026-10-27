"""internal design notes Stage 15 — deploy only what was published.

No database: the forge, the workspace and the MCP call are faked, so these
cover the decisions (what is refused, what asks a person, what is verified
afterwards), not the transports.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.core import deploy_publish, git_forge
from app.core.deploy_publish import (
    DeployIntent,
    DeployPublishError,
    parse_deploy_intent,
    sha_matches,
)

PROJECT = "/srv/legion/landing"
SHA = "a" * 40
OTHER = "b" * 40


# ── parsing ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command, cwd, expected",
    [
        (["git", "-C", PROJECT, "checkout", SHA], None, DeployIntent("ref", SHA)),
        (["git", "-C", PROJECT + "/", "checkout", "--detach", SHA], None, DeployIntent("ref", SHA)),
        (["git", "checkout", SHA], PROJECT, DeployIntent("ref", SHA)),
        (["git", "-C", PROJECT, "checkout", "origin/main"], None, DeployIntent("ref", "main")),
        (["git", "-C", PROJECT, "switch", "--detach", SHA[:12]], None, DeployIntent("ref", SHA[:12])),
        (["git", "-C", PROJECT, "reset", "--hard", "origin/main"], None, DeployIntent("ref", "main")),
        (["git", "-C", PROJECT, "merge", "--ff-only", "origin/main"], None, DeployIntent("ref", "main")),
        (["git", "-C", PROJECT, "pull", "--ff-only"], None, DeployIntent("pull")),
        (["/usr/bin/git", "-C", PROJECT, "pull"], None, DeployIntent("pull")),
        (["git", "-c", "advice.detachedHead=false", "-C", PROJECT, "checkout", SHA], None, DeployIntent("ref", SHA)),
        (["git", "-C", "sub", "checkout", SHA], "/srv/legion", None),  # not the checkout
    ],
)
def test_parse_recognises_commands_that_move_the_checkout(command, cwd, expected):
    assert parse_deploy_intent(command, cwd=cwd, project_dir=PROJECT) == expected


@pytest.mark.parametrize(
    "command",
    [
        ["git", "-C", PROJECT, "fetch", "origin"],  # fetching moves nothing
        ["git", "-C", PROJECT, "status"],
        ["git", "-C", PROJECT, "checkout", "-b", "hotfix"],
        ["git", "-C", PROJECT, "checkout", SHA, "--", "index.html"],
        ["git", "-C", PROJECT, "checkout", "--", "index.html"],
        ["git", "-C", PROJECT, "reset", SHA],  # no --hard: leaves the tree alone
        ["git", "-C", PROJECT, "checkout", "HEAD~1"],  # only the server knows it
        ["git", "-C", PROJECT, "checkout", "FETCH_HEAD"],
        ["git", "-C", "/srv/other", "checkout", SHA],
        ["git", "checkout", SHA],  # no -C, no cwd: cannot tell where
        ["docker", "compose", "up", "-d"],
        [],
    ],
)
def test_parse_leaves_everything_else_alone(command):
    assert parse_deploy_intent(command, cwd=None, project_dir=PROJECT) is None


def test_sha_matches_accepts_abbreviations_but_not_prefix_noise():
    assert sha_matches(SHA, SHA)
    assert sha_matches(SHA, SHA[:12] + "\n")
    assert not sha_matches(SHA, OTHER)
    assert not sha_matches(SHA, "aaa")  # too short to mean anything
    assert not sha_matches(None, SHA)
    assert not sha_matches(SHA, "")


# ── the gate ────────────────────────────────────────────────────────────────


@dataclass
class _Target:
    branch: str = "main"
    git_config: dict = None

    def __post_init__(self):
        self.git_config = {"repo_url": "https://github.com/o/r", "provider": "github", "branch": self.branch}


class _World:
    """The forge and workspace as the gate sees them."""

    def __init__(self, monkeypatch, *, publication="main"):
        self.publication = publication
        self.relation = ("in_branch", SHA)
        self.heads = {}  # forge: branch -> ("found", sha) | ("missing", None) ...
        self.local = {}  # workspace: branch -> sha
        self.ahead = []  # commits local has over the forge
        self.unmerged = []
        self.bypass = []  # main commits that skipped the promotion
        self.bypass_excluded = []
        self.pushed = []
        self.push_errors = []
        self.token = "tok"

        async def target(db, room_id, request_id):
            return None, _Target(publication), self.token

        async def commit_relation(git_config, token, ref, base):
            return self.relation

        async def remote_branch_head(git_config, token, branch):
            return self.heads.get(branch, ("missing", None))

        async def local_head(room_id, branch):
            return self.local.get(branch)

        async def unpublished(room_id, local, remote):
            return self.ahead

        async def unmerged(room_id, branch, base):
            return self.unmerged if branch != base else []

        async def main_bypass(room_id, remote_sha):
            self.bypass_excluded.append(remote_sha)
            return self.bypass

        async def push(room_id, git_config, access_token=None, mode="push", branch=None):
            self.pushed.append(branch)
            if self.push_errors:
                return {"pushed": 0, "errors": self.push_errors}
            # a successful push moves the forge head to the local one
            self.heads[branch] = ("found", self.local[branch])
            return {"pushed": 1, "errors": []}

        monkeypatch.setattr(deploy_publish, "_target", target)
        monkeypatch.setattr(git_forge, "commit_relation", commit_relation)
        monkeypatch.setattr(git_forge, "remote_branch_head", remote_branch_head)
        monkeypatch.setattr(deploy_publish, "local_head", local_head)
        monkeypatch.setattr(deploy_publish, "_unpublished", unpublished)
        monkeypatch.setattr(deploy_publish, "unmerged_commits", unmerged)
        monkeypatch.setattr(deploy_publish, "main_bypass_commits", main_bypass)
        monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)
        monkeypatch.setattr("app.db.release.release_db", _noop)


async def _noop(*args, **kwargs):
    return None


ROOM = uuid4()


@pytest.mark.asyncio
async def test_a_commit_the_forge_has_never_seen_is_not_published(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("missing", None)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", SHA))
    assert err.value.code == "deploy_not_published"
    assert "publish_for_deploy" in str(err.value)


@pytest.mark.asyncio
async def test_a_published_commit_of_the_publication_branch_passes_with_its_full_sha(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("in_branch", SHA)
    check = await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", SHA[:10]))
    assert check.sha == SHA and not check.unmerged


@pytest.mark.asyncio
async def test_a_published_commit_outside_the_publication_branch_owes_a_card(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("outside_branch", SHA)
    world.unmerged = [{"sha": SHA, "message": "wip"}]
    check = await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", SHA))
    assert check.unmerged and check.commits == world.unmerged
    assert check.publication_branch == "main"


@pytest.mark.asyncio
async def test_a_branch_ref_whose_forge_copy_is_behind_the_workspace_is_not_published(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("in_branch", OTHER)  # what the forge has for 'main'
    world.local = {"main": SHA}  # the workspace is ahead
    world.ahead = [{"sha": SHA, "message": "merged locally, never pushed"}]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", "main"))
    assert err.value.code == "deploy_not_published"
    assert "previous version" in str(err.value)


@pytest.mark.asyncio
async def test_a_tag_or_remote_only_branch_is_not_blocked_for_having_no_local_copy(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("in_branch", SHA)
    check = await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", "v1.2.0"))
    assert check.sha == SHA


@pytest.mark.asyncio
async def test_a_forge_that_cannot_be_asked_is_never_read_as_permission(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("unknown", None)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", SHA))
    assert err.value.code == "deploy_publication_unverified"


@pytest.mark.asyncio
async def test_a_forge_the_hub_cannot_ask_leaves_the_old_behaviour(monkeypatch):
    world = _World(monkeypatch)
    world.relation = ("unsupported", None)
    assert await deploy_publish.check_intent(None, ROOM, DeployIntent("ref", SHA)) is None


@pytest.mark.asyncio
async def test_a_room_without_a_repository_is_not_gated(monkeypatch):
    from app.core import vps_deploy_key

    async def target(db, room_id, request_id):
        raise vps_deploy_key.DeployKeyError("repo_not_configured", "no repository")

    monkeypatch.setattr(deploy_publish, "_target", target)
    assert await deploy_publish.check_intent(None, ROOM, DeployIntent("pull")) is None


@pytest.mark.asyncio
async def test_pull_of_a_publication_branch_with_unpushed_commits_is_refused(monkeypatch):
    world = _World(monkeypatch)
    world.heads = {"main": ("found", OTHER)}
    world.local = {"main": SHA}
    world.ahead = [{"sha": SHA, "message": "x"}]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_intent(None, ROOM, DeployIntent("pull"))
    assert err.value.code == "deploy_not_published"


@pytest.mark.asyncio
async def test_pull_is_fine_when_the_forge_and_the_workspace_agree(monkeypatch):
    world = _World(monkeypatch)
    world.heads = {"main": ("found", SHA)}
    world.local = {"main": SHA}
    check = await deploy_publish.check_intent(None, ROOM, DeployIntent("pull"))
    assert check is not None and check.sha is None  # nothing to verify afterwards


@pytest.mark.asyncio
async def test_pull_of_a_publication_branch_never_pushed_is_refused(monkeypatch):
    world = _World(monkeypatch)
    world.heads = {"main": ("missing", None)}
    world.local = {"main": SHA}
    world.ahead = [{"sha": SHA, "message": "x"}]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.check_intent(None, ROOM, DeployIntent("pull"))
    assert "never pushed" in str(err.value)


# ── publishing ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publishing_an_up_to_date_branch_pushes_nothing(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    world.heads = {"main": ("found", SHA)}
    done = await deploy_publish.ensure_published(None, ROOM)
    assert done.sha == SHA and done.pushed is False and world.pushed == []


@pytest.mark.asyncio
async def test_publishing_pushes_the_branch_and_confirms_the_forge_moved(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    world.heads = {"main": ("found", OTHER)}
    done = await deploy_publish.ensure_published(None, ROOM)
    assert done.sha == SHA and done.pushed is True and done.verified is True
    assert world.pushed == ["main"]


@pytest.mark.asyncio
async def test_publishing_a_branch_outside_the_publication_branch_waits_for_a_person(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"dev": SHA}
    world.unmerged = [{"sha": SHA, "message": "feature"}]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.ensure_published(None, ROOM, branch="dev")
    assert err.value.code == "deploy_needs_approval"
    assert err.value.commits == world.unmerged
    assert world.pushed == []  # nothing left the Hub without the yes

    done = await deploy_publish.ensure_published(None, ROOM, branch="dev", allow_unmerged=True)
    assert done.sha == SHA and world.pushed == ["dev"]


@pytest.mark.asyncio
async def test_a_failed_push_says_so_and_names_no_other_transport(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    world.heads = {"main": ("found", OTHER)}
    world.push_errors = ["! [rejected] main -> main (non-fast-forward)"]
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.ensure_published(None, ROOM)
    assert err.value.code == "deploy_not_published"
    assert "non-fast-forward" in str(err.value)
    assert "another transport" in str(err.value)


@pytest.mark.asyncio
async def test_a_push_the_forge_does_not_reflect_is_not_called_published(monkeypatch):
    world = _World(monkeypatch)
    world.local = {"main": SHA}
    world.heads = {"main": ("found", OTHER)}

    async def lying_push(room_id, git_config, access_token=None, mode="push", branch=None):
        return {"pushed": 1, "errors": []}  # reports success, moves nothing

    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", lying_push)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.ensure_published(None, ROOM)
    assert "reported success" in str(err.value)


@pytest.mark.asyncio
async def test_a_branch_with_no_commits_has_nothing_to_publish(monkeypatch):
    _World(monkeypatch)
    with pytest.raises(DeployPublishError) as err:
        await deploy_publish.ensure_published(None, ROOM, branch="ghost")
    assert err.value.code == "deploy_not_published"


# ── the forge client ────────────────────────────────────────────────────────


def _forge(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(git_forge.httpx, "AsyncClient", factory)


GITHUB = {"repo_url": "https://github.com/o/r", "provider": "github"}
GITLAB = {"repo_url": "https://gitlab.com/o/r", "provider": "gitlab"}


@pytest.mark.asyncio
async def test_github_head_and_relation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/branches/main"):
            return httpx.Response(200, json={"commit": {"sha": SHA}})
        if path.endswith("/commits/feature"):
            return httpx.Response(200, json={"sha": OTHER})
        if path.endswith("/commits/gone"):
            return httpx.Response(404, json={})
        if "/compare/main..." in path:
            return httpx.Response(200, json={"status": "ahead"})
        return httpx.Response(500)

    _forge(monkeypatch, handler)
    assert await git_forge.remote_branch_head(GITHUB, "t", "main") == ("found", SHA)
    assert await git_forge.remote_branch_head(GITHUB, "t", "nope") == ("unknown", None)
    assert await git_forge.commit_relation(GITHUB, "t", "feature", "main") == ("outside_branch", OTHER)
    assert await git_forge.commit_relation(GITHUB, "t", "gone", "main") == ("missing", None)
    # without a token a 404 may be a private repository, not an unpushed commit
    assert await git_forge.commit_relation(GITHUB, None, "gone", "main") == ("unknown", None)


@pytest.mark.asyncio
async def test_github_behind_or_identical_means_the_commit_is_in_the_branch(monkeypatch):
    status = {"value": "behind"}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/compare/" in request.url.path:
            return httpx.Response(200, json={"status": status["value"]})
        return httpx.Response(200, json={"sha": SHA})

    _forge(monkeypatch, handler)
    assert await git_forge.commit_relation(GITHUB, "t", SHA, "main") == ("in_branch", SHA)
    status["value"] = "identical"
    assert await git_forge.commit_relation(GITHUB, "t", SHA, "main") == ("in_branch", SHA)
    status["value"] = "diverged"
    assert await git_forge.commit_relation(GITHUB, "t", SHA, "main") == ("outside_branch", SHA)


@pytest.mark.asyncio
async def test_gitlab_relation_reads_the_branches_that_contain_the_commit(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/refs"):
            return httpx.Response(200, json=[{"name": "main"}, {"name": "dev"}])
        return httpx.Response(200, json={"id": SHA})

    _forge(monkeypatch, handler)
    assert await git_forge.commit_relation(GITLAB, "t", SHA, "main") == ("in_branch", SHA)
    assert await git_forge.commit_relation(GITLAB, "t", SHA, "release") == ("outside_branch", SHA)


@pytest.mark.asyncio
async def test_a_forge_that_is_not_github_or_gitlab_is_unsupported():
    other = {"repo_url": "https://git.example.com/o/r", "provider": "gitea"}
    assert await git_forge.remote_branch_head(other, "t", "main") == ("unsupported", None)
    assert await git_forge.commit_relation(other, "t", SHA, "main") == ("unsupported", None)


@pytest.mark.asyncio
async def test_a_forge_outage_is_unknown_not_missing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    _forge(monkeypatch, handler)
    assert await git_forge.remote_branch_head(GITHUB, "t", "main") == ("unknown", None)
    assert await git_forge.commit_relation(GITHUB, "t", SHA, "main") == ("unknown", None)


# ── the gateway ─────────────────────────────────────────────────────────────


@pytest.fixture
def gateway(monkeypatch):
    from app.core import mcp_gateway, vps_bootstrap

    async def project_dir(db, room_id):
        return PROJECT

    monkeypatch.setattr(vps_bootstrap, "project_dir", project_dir)
    return mcp_gateway


@pytest.mark.asyncio
async def test_a_non_git_command_never_reaches_the_forge(gateway, monkeypatch):
    async def boom(*args, **kwargs):
        raise AssertionError("must not be consulted")

    monkeypatch.setattr(deploy_publish, "check_intent", boom)
    out = await gateway._vps_deploy_gate(
        {"command": ["systemctl", "status", "nginx"]}, room_id=ROOM, db=None, request_id=None
    )
    assert out == (None, None, None)


@pytest.mark.asyncio
async def test_a_checkout_of_unpublished_work_is_refused_with_its_code(gateway, monkeypatch):
    async def refuse(db, room_id, intent, request_id=None):
        raise DeployPublishError("deploy_not_published", "not on the repository")

    monkeypatch.setattr(deploy_publish, "check_intent", refuse)
    check, project_dir, error = await gateway._vps_deploy_gate(
        {"command": ["git", "-C", PROJECT, "checkout", SHA]}, room_id=ROOM, db=None, request_id=None
    )
    assert check is None and error.startswith("Error: [deploy_not_published]")


@pytest.mark.asyncio
async def test_a_checkout_outside_the_publication_branch_waits_for_a_person(gateway, monkeypatch):
    check = deploy_publish.DeployCheck(
        sha=SHA, ref="dev", unmerged=True, publication_branch="main",
        commits=[{"sha": SHA, "message": "wip"}],
    )
    asked = []

    async def check_intent(db, room_id, intent, request_id=None):
        return check

    async def confirm(db, **kwargs):
        asked.append(kwargs)
        return False

    monkeypatch.setattr(deploy_publish, "check_intent", check_intent)
    monkeypatch.setattr(deploy_publish, "confirm_unmerged_deploy", confirm)
    got, _dir, error = await gateway._vps_deploy_gate(
        {"command": ["git", "-C", PROJECT, "checkout", "dev"]}, room_id=ROOM, db=None, request_id=None
    )
    assert got is None and error.startswith("Error: [deploy_needs_approval]")
    assert asked and asked[0]["publication_branch"] == "main" and asked[0]["ref"] == "dev"

    async def yes(db, **kwargs):
        return True

    monkeypatch.setattr(deploy_publish, "confirm_unmerged_deploy", yes)
    got, project_dir, error = await gateway._vps_deploy_gate(
        {"command": ["git", "-C", PROJECT, "checkout", "dev"]}, room_id=ROOM, db=None, request_id=None
    )
    assert error is None and got is check and project_dir == PROJECT


@pytest.mark.asyncio
async def test_a_gate_that_crashes_stops_the_deploy_instead_of_waving_it_through(gateway, monkeypatch):
    async def crash(db, room_id, intent, request_id=None):
        raise RuntimeError("db gone")

    monkeypatch.setattr(deploy_publish, "check_intent", crash)
    _check, _dir, error = await gateway._vps_deploy_gate(
        {"command": ["git", "-C", PROJECT, "pull"]}, room_id=ROOM, db=None, request_id=None
    )
    assert error.startswith("Error: [deploy_publication_unverified]")


def _verify(gateway, monkeypatch, reply):
    sent = []

    async def billed(binding, tool, args, **kwargs):
        sent.append((tool, args))
        return reply

    monkeypatch.setattr(gateway, "_billed_mcp_call", billed)
    check = deploy_publish.DeployCheck(sha=SHA, ref="main")

    async def run():
        return await gateway._verify_vps_checkout(
            SimpleNamespace(), {"server": "prod", "_legion_conn": {"host": "h"}}, check, PROJECT,
            user_id=uuid4(), room_id=ROOM, upstream_headers={}, custom_config=None, request_id=None,
        )

    return run, sent


@pytest.mark.asyncio
async def test_a_server_that_landed_on_the_approved_commit_is_confirmed(gateway, monkeypatch):
    run, sent = _verify(gateway, monkeypatch, {"exit_code": 0, "stdout": SHA + "\n"})
    assert await run() is None
    tool, args = sent[0]
    assert tool == "vps_exec" and args["command"] == ["git", "-C", PROJECT, "rev-parse", "HEAD"]
    assert args["_legion_conn"] == {"host": "h"}


@pytest.mark.asyncio
async def test_a_server_that_ended_up_elsewhere_is_a_mismatch_not_a_success(gateway, monkeypatch):
    run, _ = _verify(gateway, monkeypatch, {"exit_code": 0, "stdout": OTHER + "\n"})
    error = await run()
    assert error.startswith("Error: [deploy_sha_mismatch]")
    assert OTHER[:10] in error and SHA[:10] in error and "do not report success" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["Error: [unavailable] ssh down", {"exit_code": 128, "stdout": ""}, {"exit_code": 0, "stdout": "  "}])
async def test_a_checkout_that_cannot_be_read_back_is_unverified_never_assumed_good(gateway, monkeypatch, reply):
    run, _ = _verify(gateway, monkeypatch, reply)
    assert (await run()).startswith("Error: [deploy_sha_unverified]")


@pytest.mark.asyncio
async def test_a_read_back_that_raises_is_unverified(gateway, monkeypatch):
    async def billed(*args, **kwargs):
        raise RuntimeError("wallet empty")

    monkeypatch.setattr(gateway, "_billed_mcp_call", billed)
    check = deploy_publish.DeployCheck(sha=SHA, ref="main")
    error = await gateway._verify_vps_checkout(
        SimpleNamespace(), {"server": "prod"}, check, PROJECT,
        user_id=uuid4(), room_id=ROOM, upstream_headers={}, custom_config=None, request_id=None,
    )
    assert error.startswith("Error: [deploy_sha_unverified]")


# ── the tool ────────────────────────────────────────────────────────────────


def _ctx():
    return SimpleNamespace(db=None, room_id=ROOM, session_id=uuid4())


@pytest.mark.asyncio
async def test_the_tool_returns_the_commit_to_deploy_and_how(monkeypatch):
    from app.core.tools import _publish_for_deploy

    async def ensure(db, room_id, **kwargs):
        return deploy_publish.PublishedCommit(SHA, "main", pushed=True)

    monkeypatch.setattr(deploy_publish, "ensure_published", ensure)
    monkeypatch.setattr(_publish_for_deploy, "release_tool_db", _noop)
    out = await _publish_for_deploy.PublishForDeployTool().execute(_ctx(), {})
    assert SHA in out and f"git checkout {SHA}" in out and "pushed just now" in out


@pytest.mark.asyncio
async def test_the_tool_asks_before_publishing_unmerged_work_and_stops_on_no(monkeypatch):
    from app.core.tools import _publish_for_deploy

    calls = []

    async def ensure(db, room_id, allow_unmerged=False, **kwargs):
        calls.append(allow_unmerged)
        if not allow_unmerged:
            raise DeployPublishError("deploy_needs_approval", "dev is ahead", commits=[{"sha": SHA, "message": "x"}])
        return deploy_publish.PublishedCommit(SHA, "dev", pushed=True)

    answers = iter([False, True])

    async def confirm(db, **kwargs):
        return next(answers)

    async def publication(ctx):
        return "main"

    monkeypatch.setattr(deploy_publish, "ensure_published", ensure)
    monkeypatch.setattr(deploy_publish, "confirm_unmerged_deploy", confirm)
    monkeypatch.setattr(_publish_for_deploy, "_publication_branch", publication)
    monkeypatch.setattr(_publish_for_deploy, "release_tool_db", _noop)
    tool = _publish_for_deploy.PublishForDeployTool()

    denied = await tool.execute(_ctx(), {"branch": "dev"})
    assert denied.startswith("Error: [deploy_needs_approval]") and calls == [False]

    allowed = await tool.execute(_ctx(), {"branch": "dev"})
    assert SHA in allowed and calls == [False, False, True]


@pytest.mark.asyncio
async def test_the_tool_reports_a_refusal_with_its_code(monkeypatch):
    from app.core.tools import _publish_for_deploy

    async def ensure(db, room_id, **kwargs):
        raise DeployPublishError("deploy_not_published", "push rejected")

    monkeypatch.setattr(deploy_publish, "ensure_published", ensure)
    monkeypatch.setattr(_publish_for_deploy, "release_tool_db", _noop)
    out = await _publish_for_deploy.PublishForDeployTool().execute(_ctx(), {})
    assert out == "Error: [deploy_not_published] push rejected"


def test_the_tool_is_offered_to_agents_only():
    import app.core.tools  # noqa: F401  (importing registers every tool)
    from app.core.tools.registry import REGISTRY

    tool = REGISTRY.get("publish_for_deploy")
    assert tool is not None and tool.roles == frozenset({"agent"}) and tool.mutating
