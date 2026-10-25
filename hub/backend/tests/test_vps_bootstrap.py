"""internal design notes Stage 8 tests — the server preparation wizard.

Two halves. The pure ones (allowlist, sudoers text, key generation, slug)
need nothing but an import. The rest run the real
``core/vps_bootstrap.py`` state machine against a *fake server*: every
``legion-vps-mcp`` call is intercepted and served by :class:`FakeServer`,
which keeps a dict of files, a set of users and a list of authorized keys and
answers each tool the way the real service would — including refusing to
authenticate ``legion`` until a matching key is actually in
``authorized_keys``. That is what makes "a second run adds no second entry"
and "a broken sudoers is never installed" real assertions rather than mock
call-count bookkeeping.

DB-gated on INIT_DB_TEST_URL and PyNaCl, same as tests/test_vps_inventory.py.
"""

import asyncio
import base64
import os
from unittest.mock import patch
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


# ── pure checks (no database) ──────────────────────────────────────────────


def test_allowlist_accepts_planned_commands_and_rejects_everything_else():
    from app.core import vps_bootstrap as vb

    vb._assert_allowed(["id", "-un"])
    vb._assert_allowed(["visudo", "-c", "-f", vb.SUDOERS_TMP_PATH])
    vb._assert_allowed(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "git"])
    vb._assert_allowed(["systemctl", "is-active", "ssh.socket"])

    for forbidden in (
        ["curl", "https://example.com/install.sh"],
        ["rm", "-rf", "/"],
        ["systemctl", "poweroff"],
        # Right binary, wrong environment prefix: the only env assignment the
        # allowlist knows is DEBIAN_FRONTEND (apt's debconf prompt).
        ["env", "LD_PRELOAD=/tmp/x.so", "apt-get", "update"],
    ):
        with pytest.raises(vb.BootstrapError) as excinfo:
            vb._assert_allowed(forbidden)
        assert excinfo.value.code == "command_not_allowed"


def test_sudoers_is_an_allowlist_and_follows_the_profile():
    from app.core import vps_bootstrap as vb

    minimal = vb.sudoers_content(vb.BootstrapProfile(git=True))
    assert "LEGION_DOCKER" not in minimal and "LEGION_NGINX" not in minimal
    assert "nginx" not in minimal
    assert f"{vb.LEGION_USER} ALL=(root) NOPASSWD:" in minimal
    # No blanket grant anywhere: every entry is a fully-qualified path.
    assert "ALL=(ALL)" not in minimal and "NOPASSWD: ALL" not in minimal

    full = vb.sudoers_content(vb.BootstrapProfile(docker=True, nginx=True, certbot=True))
    assert "/usr/sbin/nginx -t" in full
    assert "/usr/bin/systemctl restart nginx" in full
    assert "/usr/bin/certbot" in full
    # A wildcard verb would hand out the machine — units are named one by one.
    assert "/usr/bin/systemctl restart *" not in full

    # Stage 10: the privileged write path (write-file.ts `systemWritePlan`) and
    # the deploy recipe's "enable the site" step both go through LEGION_WRITE.
    # A missing entry here does not fail loudly — it fails at the last step of
    # a deploy with `sudo_not_permitted`, which is how cp/mv/chmod/chown/rm
    # was found missing against a live VPS (commit d93cef8).
    for command in ("cp", "mv", "chmod", "chown", "rm", "ln", "mkdir"):
        assert f"/usr/bin/{command}" in minimal, command

    # Stage 10 step 3 validates a candidate config before it reaches /etc.
    # sudoers matches arguments literally, so `nginx -t` does not cover
    # `nginx -t -c <file>`, and the grant is one fixed filename — which
    # `skill-vps-deploy` must name verbatim, or every check is refused.
    assert f"/usr/sbin/nginx -t -c {vb.NGINX_CHECK_PATH}" in full
    assert vb.NGINX_CHECK_PATH.startswith("/tmp/")
    # Bare `/usr/sbin/nginx` would let an arbitrary `-c` set `user root` and
    # load modules, so it must never appear as a grant on its own.
    assert "/usr/sbin/nginx," not in full


def test_generated_key_is_ed25519_and_round_trips():
    from app.core import vps_bootstrap as vb

    private, public = vb.generate_ed25519_keypair()
    assert private.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert public.startswith("ssh-ed25519 ")
    assert vb._public_line_for(private) == public


def test_room_slug_never_collides_between_rooms():
    from app.core import vps_bootstrap as vb

    a, b = uuid4(), uuid4()
    assert vb.room_slug("Web Shop", a) != vb.room_slug("Web Shop", b)
    # A name with no Latin letters still yields a usable directory name.
    assert vb.room_slug("我的项目", a).strip("-")


# ── the fake server ────────────────────────────────────────────────────────


class FakeServer:
    """Enough of an Ubuntu box to run the whole wizard against."""

    def __init__(self, *, fingerprint="SHA256:fakefingerprintvalue", visudo_ok=True, restrict_supported=True):
        self.fingerprint = fingerprint
        self.visudo_ok = visudo_ok
        self.restrict_supported = restrict_supported
        self.files: dict[str, str] = {"/etc/os-release": 'PRETTY_NAME="Ubuntu 24.04 LTS"\n'}
        self.users: set[str] = {"root"}
        self.installed = {"sudo"}
        self.calls: list[tuple[str, dict]] = []
        self.apt_installs: list[list[str]] = []

    # -- helpers -----------------------------------------------------------

    def _error(self, code, message="", **details):
        from app.core.mcp_gateway import VpsToolError

        return VpsToolError(code, message or code, details)

    def _authorized_body(self):
        from app.core import vps_bootstrap as vb

        return self.files.get(vb.AUTHORIZED_KEYS_PATH, "")

    def _check_login(self, conn):
        from app.core import vps_bootstrap as vb

        if conn["user"] != vb.LEGION_USER:
            return
        public = vb._public_line_for(conn["auth"]["private_key"])
        content = self._authorized_body()
        if f"restrict {public}" in content:
            # An sshd too old for `restrict` rejects the whole line, which on
            # the client side is indistinguishable from "no key installed".
            if self.restrict_supported:
                return
            raise self._error("auth_failed")
        if public in content:
            return
        raise self._error("auth_failed")

    # -- the tool surface --------------------------------------------------

    async def __call__(self, db, *, room_id, actor_user_id, tool, arguments, legion_conn, request_id=None):
        self.calls.append((tool, arguments))
        if tool == "vps_host_key":
            return {"server": arguments["server"], "fingerprint": self.fingerprint}
        self._check_login(legion_conn)
        if tool == "vps_read_file":
            path = arguments["path"]
            if path not in self.files:
                raise self._error("path_not_found", path)
            return {"content_base64": base64.b64encode(self.files[path].encode()).decode()}
        if tool == "vps_write_file":
            self.files[arguments["path"]] = base64.b64decode(arguments["content_base64"]).decode()
            return {"path": arguments["path"], "bytes": 1}
        if tool == "vps_list_dir":
            raise self._error("path_not_found", arguments["path"])
        if tool == "vps_exec":
            return self._exec(list(arguments["command"]))
        raise AssertionError(f"unexpected tool {tool}")

    def _exec(self, argv):
        from app.core import vps_bootstrap as vb

        def ok(stdout=""):
            return {"exit_code": 0, "stdout": stdout, "stderr": ""}

        def fail(stderr="", code=1):
            return {"exit_code": code, "stdout": "", "stderr": stderr}

        if argv[:2] == ["env", "DEBIAN_FRONTEND=noninteractive"]:
            argv = argv[2:]
        head = argv[0]
        if argv == ["id", "-un"]:
            return ok("legion" if vb.LEGION_USER in self.users else "root")
        if argv[:2] == ["id", "-u"]:
            return ok("1001") if argv[2] in self.users else fail("no such user")
        if head == "uname":
            return ok("6.8.0-31-generic")
        if head == "which":
            return ok("\n".join(f"/usr/bin/{b}" for b in argv[1:] if b in self.installed))
        if head == "ss":
            return ok("tcp LISTEN 0 128 0.0.0.0:22 0.0.0.0:*")
        if head == "useradd":
            self.users.add(argv[-1])
            return ok()
        if head in ("passwd", "mkdir", "chown", "chmod"):
            return ok()
        if argv[:2] == ["rm", "-f"]:
            self.files.pop(argv[2], None)
            return ok()
        if head == "visudo":
            return ok() if self.visudo_ok else fail("syntax error near line 3")
        if head == "mv":
            if argv[1] in self.files:
                self.files[argv[2]] = self.files.pop(argv[1])
            return ok()
        if head == "apt-get":
            if argv[1] == "install":
                packages = [a for a in argv[3:]]
                self.apt_installs.append(packages)
                self.installed.update(p.split("-")[0] for p in packages)
                self.installed.update(packages)
            return ok()
        if head in ("systemctl", "sshd"):
            return ok("systemd 255")
        if head == "dpkg-query":
            pkg = argv[-1]
            return ok("install ok installed") if pkg in self.installed else fail("no packages found matching " + pkg)
        raise AssertionError(f"fake server got an unplanned command: {argv}")


# ── DB-backed runs ─────────────────────────────────────────────────────────

db_test = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    pytest.importorskip("nacl")


def _keyring():
    from app.core.secret_keyring import SecretKeyring

    return SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


_SHARED_KEYRING = _keyring() if TEST_DB_URL else None

if TEST_DB_URL:
    from app.core.secrets import runtime as _secrets_runtime

    _secrets_runtime.configure_runtime_keyring(_SHARED_KEYRING)


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=_SHARED_KEYRING))


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()


def _session_factory():
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _make_room(db, room_id, owner_id):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=owner_id, email=f"vb-{owner_id}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="Web Shop", creator_id=owner_id))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=owner_id))
    await db.commit()


async def _drive(room_id, owner_id, server, *, profile=None, accept_fingerprint=True, fake=None):
    """Start a run, answer the host-key card, wait for it to finish."""
    from app.core import vps_bootstrap as vb
    from app.db.session import get_session

    fake = fake or FakeServer()
    profile = profile or vb.BootstrapProfile(git=True)

    with patch.object(vb, "execute_vps_bootstrap_tool", fake):
        async for db in get_session():
            await vb.start(
                db, room_id=room_id, actor_user_id=owner_id, server=server,
                profile=profile, session_factory=_session_factory(),
            )
            break
        for _ in range(400):
            await asyncio.sleep(0.02)
            report = vb.get_run(room_id, server)
            if report["status"] == "awaiting_fingerprint":
                await vb.confirm_fingerprint(room_id, server, accept=accept_fingerprint)
                continue
            if report["status"] in ("succeeded", "failed"):
                return report, fake
        raise AssertionError("preparation run did not finish")


async def _add_server(room_id, owner_id, name="PROD"):
    from app.core import vps_inventory
    from app.db.session import get_session

    async for db in get_session():
        await vps_inventory.add_server(
            db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
            name=name, host="203.0.113.10", user="root", root_password="root-password-value",
        )
        await db.commit()
        break


@db_test
@pytest.mark.asyncio
async def test_prepare_from_bare_root_password_makes_the_server_usable():
    """§3.6 acceptance: one root password in, a working `legion` login out."""
    from app.core import vps_bootstrap as vb, vps_inventory
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose
    from app.core.secrets.room_secrets import resolve_room_secret

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    report, fake = await _drive(
        room_id, owner_id, "PROD", profile=vb.BootstrapProfile(git=True, nginx=True)
    )

    assert report["status"] == "succeeded", report
    # `repo` (Stage 13) is skipped here because this Room has no external
    # repository — see tests/test_vps_deploy_key.py for its own runs.
    assert [s["status"] for s in report["steps"] if s["id"] not in ("harden", "repo")] == ["ok"] * 9
    assert next(s for s in report["steps"] if s["id"] == "repo")["error_code"] == "repo_not_configured"
    assert report["inventory"]["os"] == "Ubuntu 24.04 LTS"

    # The everyday credential now exists and is the key we authorized.
    async for db in get_session():
        rows = await vps_inventory.list_servers(db, room_id)
        server = rows[0]
        assert server["user"] == vb.LEGION_USER
        assert server["prepared_at"]
        assert server["has_key"] is True
        assert server["fingerprint"] == fake.fingerprint

        private = await resolve_room_secret(
            db, service=_service(db), room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_KEY", purpose=SecretPurpose.VPS_ACCESS,
            expected_kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
        assert vb._public_line_for(private) in fake.files[vb.AUTHORIZED_KEYS_PATH]
        break

    # sudoers was validated before being put in place, and the wizard offers
    # (never performs) the two follow-ups from §3.6 step 10 / §3.7.
    assert vb.SUDOERS_PATH in fake.files and vb.SUDOERS_TMP_PATH not in fake.files
    assert any("root credentials" in s for s in report["suggestions"])
    assert any("Full" in s for s in report["suggestions"])
    # The opt-out default holds: sshd was not touched.
    assert vb.SSHD_DROPIN_PATH not in fake.files
    assert report["steps"][-1]["status"] == "skipped"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_second_run_is_idempotent():
    """§3.6 acceptance: no second authorized_keys entry, no second sudoers drop."""
    from app.core import vps_bootstrap as vb
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    first, fake = await _drive(room_id, owner_id, "PROD")
    assert first["status"] == "succeeded"
    keys_after_first = fake.files[vb.AUTHORIZED_KEYS_PATH]
    sudoers_after_first = fake.files[vb.SUDOERS_PATH]
    users_after_first = set(fake.users)

    second, fake = await _drive(room_id, owner_id, "PROD", fake=fake)
    assert second["status"] == "succeeded", second
    assert fake.files[vb.AUTHORIZED_KEYS_PATH] == keys_after_first
    assert fake.files[vb.SUDOERS_PATH] == sudoers_after_first
    assert fake.users == users_after_first
    assert keys_after_first.count("ssh-ed25519") == 1
    # The fingerprint is pinned now, so the second run never has to ask again.
    assert second["pending_fingerprint"] is None
    assert "already" in second["steps"][2]["detail"]

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_broken_sudoers_is_not_installed_and_the_report_says_what_was_done():
    """§3.6 acceptance: `visudo -c` fails → nothing installed, server still works."""
    from app.core import vps_bootstrap as vb
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    report, fake = await _drive(room_id, owner_id, "PROD", fake=FakeServer(visudo_ok=False))

    assert report["status"] == "failed"
    assert report["error_code"] == "sudoers_invalid"
    steps = {s["id"]: s for s in report["steps"]}
    assert steps["sudoers"]["status"] == "failed"
    # Neither the candidate nor a live drop-in survives a rejected file.
    assert vb.SUDOERS_PATH not in fake.files
    assert vb.SUDOERS_TMP_PATH not in fake.files
    # "what is done, what is not": earlier steps keep their own outcome, later ones
    # say plainly that they never ran.
    assert steps["user"]["status"] == "ok" and steps["key"]["status"] == "ok"
    assert steps["verify"]["status"] == "skipped" and steps["verify"]["detail"]
    # A failed run never claims the server is prepared.
    from app.core import vps_inventory

    async for db in get_session():
        assert (await vps_inventory.list_servers(db, room_id))[0]["prepared_at"] is None
        break

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_declined_fingerprint_stops_before_authenticating():
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    report, fake = await _drive(room_id, owner_id, "PROD", accept_fingerprint=False)

    assert report["status"] == "failed"
    assert report["error_code"] == "host_key_rejected"
    # Exactly one call was made — the fingerprint probe, which authenticates
    # nothing (§3.5).
    assert [tool for tool, _ in fake.calls] == ["vps_host_key"]

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_restrict_option_is_dropped_when_the_server_rejects_it():
    """An sshd too old for `restrict` must not cost the operator the run."""
    from app.core import vps_bootstrap as vb
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    report, fake = await _drive(room_id, owner_id, "PROD", fake=FakeServer(restrict_supported=False))

    assert report["status"] == "succeeded", report
    assert not fake.files[vb.AUTHORIZED_KEYS_PATH].startswith("restrict ")
    assert "restrict" in report["steps"][3]["detail"]

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_hardening_writes_a_validated_drop_in_only_when_asked():
    from app.core import vps_bootstrap as vb
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    report, fake = await _drive(
        room_id, owner_id, "PROD",
        profile=vb.BootstrapProfile(git=True, disable_root_password_login=True),
    )

    assert report["status"] == "succeeded", report
    assert "PasswordAuthentication no" in fake.files[vb.SSHD_DROPIN_PATH]
    assert report["steps"][-1]["status"] == "ok"

    await engine.dispose()


@db_test
@pytest.mark.asyncio
async def test_agent_has_no_route_into_the_bootstrap():
    """§3.6 acceptance: "a call with the actor vps_bootstrap from an agent session does not exist".

    Two independent guarantees, checked from the agent's side of the fence:
    no VPS tool the model is offered can reach this module, and the resolver
    the agent's calls *do* go through refuses to read a ``*_ROOT_*`` secret
    even when that is the only credential the server has.
    """
    from app.core import mcp_policy, vps_bootstrap, vps_inventory
    from app.db.session import engine, get_session

    tool_names = set(mcp_policy._STATIC_TOOL_CLASS) | mcp_policy._WRITE_TOOLS | mcp_policy._EXEC_TOOLS
    assert not any("prepare" in name or "bootstrap" in name for name in tool_names)
    assert not any(
        name.startswith("vps_") for name in dir(vps_bootstrap) if name.endswith("_tool")
    )

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        break
    await _add_server(room_id, owner_id)

    async for db in get_session():
        with pytest.raises(vps_inventory.ServerCredentialsMissing):
            await vps_inventory.resolve_connection(
                db, service=_service(db), room_id=room_id, actor_user_id=owner_id, server="PROD",
            )
        break

    await engine.dispose()


# ── failure modes an ordinary user hits on a stock Ubuntu box ──────────────


def _bare_run(**profile_kwargs):
    from app.core import vps_bootstrap as vb

    return vb.BootstrapRun(
        room_id=uuid4(), server="PROD", environment=None, actor_user_id=uuid4(),
        profile=vb.BootstrapProfile(**profile_kwargs),
    )


class _ScriptedSession:
    """A ``_Session`` stand-in: canned answers per command, plus a call log."""

    def __init__(self, answers, files=None):
        self._answers = answers
        self.files = dict(files or {})
        self.commands: list[list[str]] = []

    async def call(self, tool, arguments):
        raise AssertionError("the scripted session makes no raw tool calls")

    async def exec(self, argv, *, sudo=False, timeout_sec=60):
        from app.core import vps_bootstrap as vb

        vb._assert_allowed(argv)
        self.commands.append(list(argv))
        answer = self._answers.get(tuple(argv))
        if answer is None:
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        return answer

    async def run_ok(self, argv, *, sudo=False, code, timeout_sec=60):
        from app.core import vps_bootstrap as vb

        result = await self.exec(argv, sudo=sudo, timeout_sec=timeout_sec)
        if result.get("exit_code") != 0:
            raise vb.BootstrapError(code, str(result.get("stderr") or ""))
        return result

    async def read_text(self, path, *, max_bytes=262144):
        return self.files.get(path)

    async def write_text(self, path, content, *, mode=None, owner=None):
        self.files[path] = content


def test_hardening_accepts_a_socket_activated_sshd():
    """Ubuntu 22.10+ runs sshd from ``ssh.socket``; ``reload ssh`` then fails.

    The drop-in is already what the next connection is started with, so the
    step is done — failing the run there reported a locked-out server on a
    machine that was fine, which is the worst possible false alarm on the one
    step that can actually lock someone out.
    """
    from app.core import vps_bootstrap as vb

    session = _ScriptedSession({
        ("systemctl", "reload", "ssh"): {
            "exit_code": 1, "stdout": "",
            "stderr": "Failed to reload ssh.service: Unit ssh.service is not active, cannot reload.",
        },
        ("systemctl", "is-active", "ssh.socket"): {"exit_code": 0, "stdout": "active\n", "stderr": ""},
    })
    run = _bare_run(disable_root_password_login=True)

    asyncio.run(vb._step_harden(session, run))

    assert ["systemctl", "is-active", "ssh.socket"] in session.commands
    assert "ssh.socket" in run.step("harden").detail


def test_hardening_still_fails_when_sshd_is_simply_broken():
    from app.core import vps_bootstrap as vb

    session = _ScriptedSession({
        ("systemctl", "reload", "ssh"): {"exit_code": 1, "stdout": "", "stderr": "Job failed"},
        ("systemctl", "is-active", "ssh.socket"): {"exit_code": 3, "stdout": "inactive\n", "stderr": ""},
    })

    with pytest.raises(vb.BootstrapError) as excinfo:
        asyncio.run(vb._step_harden(session, _bare_run(disable_root_password_login=True)))
    assert excinfo.value.code == "sshd_failed"


def test_repo_step_survives_a_transport_failure_instead_of_aborting_the_run():
    """The deploy-credential step is optional; the connection dying mid-step
    must leave the server prepared, not roll the whole wizard back."""
    from app.core import vps_bootstrap as vb
    from app.core import vps_deploy_key
    from app.core.mcp_gateway import VpsToolError

    run = _bare_run(git=True)
    session = _ScriptedSession({})

    async def _boom(*args, **kwargs):
        raise VpsToolError("unavailable", "The VPS service is unavailable: exec channel failed", {})

    with patch.object(vps_deploy_key, "issue", _boom):
        asyncio.run(vb._step_repo(None, session, run, object(), uuid4()))

    assert run.step("repo").status == "skipped"
    assert run.step("repo").error_code == "unavailable"
    assert run.suggestions and "repository" in run.suggestions[0]
