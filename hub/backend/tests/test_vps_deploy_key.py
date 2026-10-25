"""internal design notes §16b, Stage 13 — the repository credential a server gets.

The stage exists because "we ship code with git" (§3.10) had no mechanism: a
prepared server could not read the Room's repository at all, and the one
agent-visible transport left was base64 through the model's own context.
These tests pin the four properties that make the git path real *and* keep it
from becoming the short, dangerous version of itself:

* the Room's own ``git_https`` token — a full-account PAT — never reaches the
  server, in any argument of any call and in any file;
* the key registered on the forge is read-only, and it is registered by id so
  revoking removes that key rather than guessing;
* a forge that will not take the key degrades to "add it yourself", not to a
  failed preparation and not to a credential recorded as working;
* the deploy recipe's own git commands classify as writes into the project
  directory, so the transport this stage enables does not cost an approval
  card per update — which was the failure mode Stage 10 measured.

The DB-backed half reuses ``test_vps_bootstrap``'s fake server, extended with
just enough git to answer ``ls-remote``.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

# The fake server is worth reusing rather than re-deriving: it is the only
# thing that makes "the key really is on the box and really is checked"
# something a test can assert. `tests/` is not a package, so make it
# importable by path regardless of how pytest was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_vps_bootstrap import FakeServer  # noqa: E402

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


# ── pure checks (no database, no server) ───────────────────────────────────


def test_managed_block_is_replaced_never_duplicated():
    """Re-issuing rewrites our fence and leaves everyone else's lines alone."""
    from app.core.vps_deploy_key import replace_block

    original = "Host bastion\n  User ops\n"
    once = replace_block(original, "shop-abcd1234", "Host legion-shop\n  User git\n")
    twice = replace_block(once, "shop-abcd1234", "Host legion-shop\n  User git\n")
    assert once == twice
    assert twice.count("legion deploy shop-abcd1234") == 2  # begin + end marker
    assert "Host bastion" in twice and "User ops" in twice

    # A second Room's block coexists rather than overwriting the first.
    both = replace_block(twice, "blog-99887766", "Host legion-blog\n  User git\n")
    assert "Host legion-shop" in both and "Host legion-blog" in both

    # Revoke drops only ours.
    after = replace_block(both, "shop-abcd1234", None)
    assert "Host legion-shop" not in after
    assert "Host legion-blog" in after and "Host bastion" in after


def test_ssh_config_pins_the_forge_and_uses_only_our_key():
    from app.core import vps_deploy_key as dk

    block = dk.ssh_config_block(
        alias="legion-shop-abcd1234", forge="github.com",
        identity_path=dk.identity_path("shop-abcd1234"),
    )
    # The alias is per Room: two Rooms on one box must not both claim
    # `Host github.com`, where the first block would silently win.
    assert "Host legion-shop-abcd1234" in block
    assert "Host github.com\n" not in block
    # ...but the host key is still checked against the forge's real name.
    assert "HostKeyAlias github.com" in block
    assert "StrictHostKeyChecking yes" in block
    # Only this key, and never an interactive prompt.
    assert "IdentitiesOnly yes" in block and "BatchMode yes" in block
    assert dk.identity_path("shop-abcd1234") in block
    assert dk.ssh_alias("a-1") != dk.ssh_alias("a-2")


def test_allowlist_rejects_anything_this_module_did_not_plan():
    """Same guarantee as the wizard's: an unlisted command is a bug, not a card."""
    from app.core import vps_deploy_key as dk

    dk._assert_allowed(["git", "ls-remote", "--exit-code", "git@x:o/r.git", "HEAD"])
    dk._assert_allowed(["sudo", "-n", "-u", "legion", "-H", "git", "ls-remote", "url"])
    dk._assert_allowed(["mkdir", "-p", dk.SSH_DIR])
    for forbidden in (
        ["git", "push", "origin", "main"],
        ["sudo", "-n", "-u", "root", "-H", "git", "ls-remote", "url"],
        ["curl", "https://example.com/x.sh"],
        ["mkdir", "-p", "/etc/ssh"],
    ):
        with pytest.raises(dk.DeployKeyError) as excinfo:
            dk._assert_allowed(forbidden)
        assert excinfo.value.code == "command_not_allowed"


def test_deploy_recipe_git_commands_need_no_approval_card():
    """Stage 13's policy half: a checkout in the project dir is `write_safe`.

    Without this, the first ``git clone`` and then *every* update stopped on a
    human's card — approval fatigue (§17 risk 7) on the very transport §3.10
    calls the main one.
    """
    from app.core import mcp_policy

    def risk(argv):
        return mcp_policy.classify_call("vps_exec", {"command": argv}).value

    assert risk(["git", "clone", "git@legion-shop:o/r.git", "/srv/legion/shop"]) == "write_safe"
    assert risk(["git", "-C", "/srv/legion/shop", "fetch", "--prune", "origin"]) == "write_safe"
    assert risk(["git", "-C", "/srv/legion/shop", "checkout", "--detach", "abc123"]) == "write_safe"
    assert risk(["git", "-C", "/srv/legion/shop", "reset", "--hard", "abc123"]) == "write_safe"
    assert risk(["git", "-C", "/srv/legion/shop", "rev-parse", "HEAD"]) == "read"

    # Outside the working directories it is still a person's decision.
    assert risk(["git", "-C", "/etc/nginx", "checkout", "x"]) == "unknown"
    # A destination the classifier cannot see (it depends on `cwd`) is not
    # assumed safe — the recipe always names one.
    assert risk(["git", "clone", "git@legion-shop:o/r.git"]) == "unknown"
    assert risk(["git", "fetch"]) == "unknown"
    # `git` options that mean "run this program instead" do not look like
    # paths, so an allowlist of flags — not a path scan — is what stops them.
    assert risk(["git", "--exec-path=/tmp/evil", "-C", "/srv/legion/shop", "fetch"]) == "unknown"
    assert risk(["git", "-c", "core.sshCommand=/tmp/x", "-C", "/srv/legion/shop", "fetch"]) == "unknown"
    # A deletion keeps earning a look even in the working directory.
    assert risk(["git", "-C", "/srv/legion/shop", "clean", "-fdx"]) == "unknown"


def test_git_deploy_purpose_is_unreachable_from_the_agent():
    """A `git_deploy` token must not be resolvable by name from `bash(secrets=…)`."""
    from app.core.secrets.room_secrets import ROOM_SECRET_PURPOSES
    from app.models.secret import SecretPurpose

    assert SecretPurpose.GIT_DEPLOY not in ROOM_SECRET_PURPOSES
    assert SecretPurpose.VPS_ACCESS not in ROOM_SECRET_PURPOSES


# ── the forge calls, against a stubbed HTTP client ─────────────────────────


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _FakeForgeClient:
    """Records what the forge API was actually asked to do."""

    posts: list[tuple[str, dict, dict]] = []
    deletes: list[tuple[str, dict]] = []
    get_payload: dict = {"ssh_keys": ["ssh-ed25519 AAAAFAKEHOSTKEY"]}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).posts.append((url, json or {}, headers or {}))
        return _FakeResponse(201, {"id": 4242})

    async def delete(self, url, headers=None):
        type(self).deletes.append((url, headers or {}))
        return _FakeResponse(204)

    async def get(self, url, headers=None, params=None):
        return _FakeResponse(200, type(self).get_payload)


@pytest.mark.asyncio
async def test_deploy_key_is_registered_read_only_and_deleted_by_id():
    """The forge — not our policy layer — is what has to refuse a push."""
    from app.core import git_forge

    _FakeForgeClient.posts, _FakeForgeClient.deletes = [], []
    config = {"provider": "github", "repo_url": "https://github.com/acme/shop"}
    with patch.object(git_forge.httpx, "AsyncClient", _FakeForgeClient):
        added = await git_forge.add_deploy_key(
            config, "ghp_token", title="Legion · PROD", public_key="ssh-ed25519 AAAA x"
        )
        removed = await git_forge.delete_deploy_key(config, "ghp_token", added["id"])

    assert added == {"ok": True, "id": "4242", "error": None}
    url, payload, _ = _FakeForgeClient.posts[0]
    assert url.endswith("/repos/acme/shop/keys")
    assert payload["read_only"] is True
    assert removed["ok"] is True
    assert _FakeForgeClient.deletes[0][0].endswith("/repos/acme/shop/keys/4242")


@pytest.mark.asyncio
async def test_gitlab_deploy_key_cannot_push_either():
    from app.core import git_forge

    _FakeForgeClient.posts = []
    config = {"provider": "gitlab", "repo_url": "https://gitlab.com/acme/shop"}
    with patch.object(git_forge.httpx, "AsyncClient", _FakeForgeClient):
        added = await git_forge.add_deploy_key(
            config, "glpat", title="Legion · PROD", public_key="ssh-ed25519 AAAA x"
        )

    assert added["ok"] is True
    url, payload, headers = _FakeForgeClient.posts[0]
    assert "/projects/acme%2Fshop/deploy_keys" in url
    assert payload["can_push"] is False
    assert headers["PRIVATE-TOKEN"] == "glpat"


@pytest.mark.asyncio
async def test_no_forge_token_is_a_reported_refusal_not_a_silent_skip():
    from app.core import git_forge

    config = {"provider": "github", "repo_url": "https://github.com/acme/shop"}
    result = await git_forge.add_deploy_key(config, None, title="t", public_key="k")
    assert result["ok"] is False and result["id"] is None


# ── DB-backed runs ─────────────────────────────────────────────────────────

db_test = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    pytest.importorskip("nacl")

ROOM_GIT_TOKEN = "ghp_room_owner_full_account_token"
GIT_CONFIG = {
    "provider": "github",
    "repo_url": "https://github.com/acme/shop",
    "branch": "main",
    "access_token": ROOM_GIT_TOKEN,
}


class RepoFakeServer(FakeServer):
    """The wizard's fake server, plus enough git to answer ``ls-remote``."""

    def __init__(self, *, ls_remote_ok=True, **kwargs):
        super().__init__(**kwargs)
        self.ls_remote_ok = ls_remote_ok
        self.ls_remote_calls: list[list[str]] = []

    def _exec(self, argv):
        if argv[:1] == ["sudo"]:
            # `sudo -n -u legion -H <cmd>` — the wizard runs as root and has
            # to prove the credential works for the login that will use it.
            assert argv[:5] == ["sudo", "-n", "-u", "legion", "-H"], argv
            return self._exec(argv[5:])
        if argv[:2] == ["git", "ls-remote"]:
            self.ls_remote_calls.append(argv)
            if self.ls_remote_ok:
                return {"exit_code": 0, "stdout": "abc123\tHEAD\n", "stderr": ""}
            return {"exit_code": 128, "stdout": "", "stderr": "Permission denied (publickey)."}
        return super()._exec(argv)


@asynccontextmanager
async def _db():
    """One session, deterministically closed.

    Not ``async for db in get_session(): … break``: breaking out of that
    generator leaves the session (and its pooled connection) open, and a
    later test's ``engine.dispose()`` then hands a connection bound to a
    finished event loop to the next test — "attached to a different loop",
    several tests away from the one that caused it.
    """
    import test_vps_bootstrap as tb

    async with tb._session_factory()() as session:
        yield session


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    import test_vps_bootstrap as tb

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=tb._SHARED_KEYRING))


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()


async def _make_room_with_repo(db, room_id, owner_id):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=owner_id, email=f"dk-{owner_id}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="Web Shop", creator_id=owner_id, git_config=dict(GIT_CONFIG)))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=owner_id))
    await db.commit()


async def _legion_remote(room_id, owner_id, fake):
    """A ``Remote`` logged in as ``legion`` — what the card's buttons build.

    The fake server authenticates for real (it checks the key against its own
    ``authorized_keys``), so this has to carry the everyday key the
    preparation actually installed rather than a placeholder.
    """
    from app.core import vps_deploy_key as dk
    from app.core.secrets.room_secrets import resolve_room_secret
    from app.models.secret import SecretKind, SecretPurpose

    async with _db() as db:
        private = await resolve_room_secret(
            db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_KEY", purpose=SecretPurpose.VPS_ACCESS,
            expected_kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
    conn = {"user": "legion", "auth": {"kind": "key", "private_key": private}}

    async def _call(tool, arguments):
        return await fake(
            None, room_id=room_id, actor_user_id=owner_id, tool=tool,
            arguments=arguments, legion_conn=conn,
        )

    return dk.Remote(call=_call, user="legion")


async def _prepare(room_id, owner_id, *, fake=None, forge_ok=True, host_keys=None):
    """Run the whole wizard, with the forge stubbed at the function boundary."""
    from app.core import git_forge, vps_bootstrap as vb, vps_deploy_key, vps_inventory
    import test_vps_bootstrap as tb

    fake = fake or RepoFakeServer()
    registered: list[dict] = []
    deleted: list[str] = []

    async def _add_deploy_key(config, token, *, title, public_key):
        registered.append({"config": config, "token": token, "title": title, "key": public_key})
        if not forge_ok:
            return {"ok": False, "id": None, "error": "403: repository is archived"}
        return {"ok": True, "id": "4242", "error": None}

    async def _delete_deploy_key(config, token, key_id):
        deleted.append(key_id)
        return {"ok": True, "error": None}

    async def _host_keys(config):
        return ["ssh-ed25519 AAAAFAKEHOSTKEY"] if host_keys is None else host_keys

    async with _db() as db:
        await vps_inventory.add_server(
            db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
            name="PROD", host="203.0.113.10", user="root", root_password="root-password-value",
        )
        await db.commit()

    with patch.object(vb, "execute_vps_bootstrap_tool", fake), \
            patch.object(git_forge, "add_deploy_key", _add_deploy_key), \
            patch.object(git_forge, "delete_deploy_key", _delete_deploy_key), \
            patch.object(git_forge, "forge_ssh_host_keys", _host_keys), \
            patch.object(vps_deploy_key, "_VERIFY_BACKOFF_SEC", 0.0):
        async with _db() as db:
            await vb.start(
                db, room_id=room_id, actor_user_id=owner_id, server="PROD",
                profile=vb.BootstrapProfile(git=True), session_factory=tb._session_factory(),
            )
        for _ in range(600):
            await asyncio.sleep(0.02)
            report = vb.get_run(room_id, "PROD")
            if report["status"] == "awaiting_fingerprint":
                await vb.confirm_fingerprint(room_id, "PROD", accept=True)
                continue
            if report["status"] in ("succeeded", "failed"):
                await _settle()
                return report, fake, registered, deleted
        raise AssertionError("preparation run did not finish")


async def _settle():
    """Let the wizard's background task finish releasing its session.

    ``BootstrapRun.status`` is set before the run emits its closing Room
    event, so a poller can see "succeeded" while the task still holds a
    database connection. Tearing the engine down underneath it leaves that
    connection in the pool bound to a dead event loop, and the failure then
    surfaces two tests later as "attached to a different loop".
    """
    for _ in range(100):
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if not pending:
            return
        await asyncio.sleep(0.02)


@db_test
@pytest.mark.asyncio
async def test_prepared_server_can_read_the_repository_and_the_room_token_stays_home():
    """Stage 13 acceptance, both halves, on one run."""
    from app.core import vps_deploy_key as dk, vps_inventory
    from app.core.vps_bootstrap import room_slug
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, fake, registered, _ = await _prepare(room_id, owner_id)

    assert report["status"] == "succeeded", report
    steps = {s["id"]: s for s in report["steps"]}
    assert steps["repo"]["status"] == "ok", steps["repo"]

    slug = room_slug("Web Shop", room_id)
    key_path = dk.identity_path(slug)
    assert fake.files[key_path].startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    config = fake.files[dk.SSH_CONFIG_PATH]
    assert f"Host {dk.ssh_alias(slug)}" in config and "IdentitiesOnly yes" in config
    assert "github.com ssh-ed25519 AAAAFAKEHOSTKEY" in fake.files[dk.KNOWN_HOSTS_PATH]

    # The credential was *proven* before it was recorded, and proven as the
    # login that will use it — not as root.
    assert fake.ls_remote_calls, "the wizard never checked the credential"
    assert fake.ls_remote_calls[0][:2] == ["git", "ls-remote"]

    # The Room's own Git token — a full-account PAT — never left the Hub.
    for tool, arguments in fake.calls:
        assert ROOM_GIT_TOKEN not in repr(arguments), (tool, "room git token in a call argument")
    for path, content in fake.files.items():
        assert ROOM_GIT_TOKEN not in content, f"room git token written to {path}"
    assert registered[0]["token"] == ROOM_GIT_TOKEN  # used *by the Hub*, against the forge API

    async with _db() as db:
        server = (await vps_inventory.list_servers(db, room_id))[0]
        assert server["deploy_credential"] == "ssh"
        assert server["deploy_key_id"] == "4242"
        assert server["deploy_remote"] == f"git@{dk.ssh_alias(slug)}:acme/shop.git"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_a_forge_that_refuses_the_key_leaves_a_skipped_step_not_a_failed_run():
    """Degradation, per §16b: show the key, ask the person, keep the server."""
    from app.core import vps_inventory
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, fake, _, _ = await _prepare(room_id, owner_id, forge_ok=False)

    assert report["status"] == "succeeded", report
    steps = {s["id"]: s for s in report["steps"]}
    assert steps["repo"]["status"] == "skipped"
    assert "deploy key" in steps["repo"]["detail"]
    assert report["inventory"]["deploy_public_key"].startswith("ssh-ed25519 ")
    assert any("Check access" in s for s in report["suggestions"])

    # Nothing claims the server can read the repository, because it cannot.
    async with _db() as db:
        assert (await vps_inventory.list_servers(db, room_id))[0]["deploy_credential"] == "none"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_a_room_without_a_repository_skips_the_step_and_says_so():
    from app.db.session import engine
    import test_vps_bootstrap as tb

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await tb._make_room(db, room_id, owner_id)  # no git_config

    report, _, registered, _ = await _prepare(room_id, owner_id)

    assert report["status"] == "succeeded", report
    steps = {s["id"]: s for s in report["steps"]}
    assert steps["repo"]["status"] == "skipped"
    assert steps["repo"]["error_code"] == "repo_not_configured"
    assert registered == []  # the forge was never called

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_reissue_replaces_the_key_and_revoke_removes_it_everywhere():
    """Idempotence + the revoke acceptance: no second pair, no orphan key."""
    from app.core import git_forge, vps_deploy_key as dk, vps_inventory
    from app.core.vps_bootstrap import room_slug
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, fake, registered, deleted = await _prepare(room_id, owner_id)
    assert report["status"] == "succeeded"
    slug = room_slug("Web Shop", room_id)
    first_key = fake.files[dk.identity_path(slug)]

    remote = await _legion_remote(room_id, owner_id, fake)
    forge_deleted: list[str] = []

    async def _delete_deploy_key(config, token, key_id):
        forge_deleted.append(key_id)
        return {"ok": True, "error": None}

    async def _add_deploy_key(config, token, *, title, public_key):
        registered.append({"key": public_key, "token": token})
        return {"ok": True, "id": "5353", "error": None}

    async def _host_keys(config):
        return ["ssh-ed25519 AAAAFAKEHOSTKEY"]

    with patch.object(git_forge, "add_deploy_key", _add_deploy_key), \
            patch.object(git_forge, "delete_deploy_key", _delete_deploy_key), \
            patch.object(git_forge, "forge_ssh_host_keys", _host_keys), \
            patch.object(dk, "_VERIFY_BACKOFF_SEC", 0.0):
        async with _db() as db:
            result = await dk.issue(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
                server="PROD", remote=remote,
            )
        assert result["status"] == "ok" and result["key_id"] == "5353"
        # The previous key was deleted by the id the inventory recorded, so
        # the repository does not accumulate live keys per re-run.
        assert forge_deleted == ["4242"]
        assert fake.files[dk.identity_path(slug)] != first_key
        assert fake.files[dk.SSH_CONFIG_PATH].count("HostKeyAlias") == 1

        async with _db() as db:
            revoked = await dk.revoke(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
                server="PROD", remote=remote,
            )

    assert revoked["forge_key_removed"] is True
    assert forge_deleted == ["4242", "5353"]
    assert dk.identity_path(slug) not in fake.files
    assert "HostKeyAlias" not in fake.files[dk.SSH_CONFIG_PATH]

    async with _db() as db:
        server = (await vps_inventory.list_servers(db, room_id))[0]
        assert server["deploy_credential"] == "none"
        assert server["deploy_key_id"] is None

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_https_fallback_keeps_the_token_out_of_every_agent_path():
    """The pasted read-only token is a `git_deploy` secret, not a Room secret."""
    from app.core import git_forge, vps_deploy_key as dk, vps_inventory
    from app.core.secrets.room_secrets import list_room_secrets
    from app.core.vps_bootstrap import room_slug
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, fake, _, _ = await _prepare(room_id, owner_id)
    assert report["status"] == "succeeded"
    slug = room_slug("Web Shop", room_id)

    remote = await _legion_remote(room_id, owner_id, fake)
    deploy_token = "ghp_repo_scoped_read_only_token"

    with patch.object(dk, "_VERIFY_BACKOFF_SEC", 0.0):
        async with _db() as db:
            result = await dk.issue(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
                server="PROD", remote=remote, https_token=deploy_token,
            )

    assert result["credential"] == "https"
    assert result["remote_url"] == "https://github.com/acme/shop.git"
    creds = fake.files[dk.credentials_path(slug)]
    assert creds == f"https://x-access-token:{deploy_token}@github.com\n"
    assert "helper = store" in fake.files[dk.GITCONFIG_PATH]

    async with _db() as db:
        # Not in the Room's "Secrets" list → unreachable from bash(secrets=…).
        listed = await list_room_secrets(db, room_id=room_id, actor_user_id=owner_id)
        names = [row["env_name"] for row in listed]
        assert "VPS_PROD_DEPLOY_TOKEN" not in names
        server = (await vps_inventory.list_servers(db, room_id))[0]
        assert server["deploy_credential"] == "https"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_a_credential_that_does_not_work_is_never_recorded_as_working():
    """`ls-remote` fails → `deploy_verify_failed`, and the inventory still says none."""
    from app.core import vps_inventory
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, _, _, _ = await _prepare(room_id, owner_id, fake=RepoFakeServer(ls_remote_ok=False))

    steps = {s["id"]: s for s in report["steps"]}
    assert steps["repo"]["status"] == "skipped"
    assert steps["repo"]["error_code"] == "deploy_verify_failed"
    async with _db() as db:
        assert (await vps_inventory.list_servers(db, room_id))[0]["deploy_credential"] == "none"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_deleting_a_server_revokes_its_repository_credential():
    """§16b: a removed server must not leave a live read key on the repository."""
    from app.core import git_forge, vps_deploy_key as dk, vps_inventory
    from app.db.session import engine

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async with _db() as db:
        await _make_room_with_repo(db, room_id, owner_id)

    report, _, _, _ = await _prepare(room_id, owner_id)
    assert report["status"] == "succeeded"

    forge_deleted: list[str] = []

    async def _delete_deploy_key(config, token, key_id):
        forge_deleted.append(key_id)
        return {"ok": True, "error": None}

    with patch.object(git_forge, "delete_deploy_key", _delete_deploy_key):
        async with _db() as db:
            await dk.revoke(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
                server="PROD", remote=None,
            )
            await vps_inventory.remove_server(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id, name="PROD",
            )
            await db.commit()

    assert forge_deleted == ["4242"]
    async with _db() as db:
        assert await vps_inventory.list_servers(db, room_id) == []

    await engine.dispose()
