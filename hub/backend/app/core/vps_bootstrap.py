"""Server preparation wizard — ``internal design notes`` §3.6 / Stage 8.

Turns "a rented VPS and a root password" into "a prepared server" without a
single command typed by a human. The procedure runs as Hub code under the
system actor ``vps_bootstrap`` and drives the very same ``legion-vps-mcp``
service an agent uses, over the same Gateway surface
(``mcp_gateway.execute_vps_bootstrap_tool``).

Three properties it must have, and where each one lives:

* **Only a human starts it.** There is no tool, no agent-facing surface and
  no session path into this module — the single entry point is
  :func:`start`, called from ``POST /rooms/{id}/vps/{name}/prepare`` with a
  logged-in user. The root credentials it consumes are
  ``purpose=vps_access`` secrets that ``resolve_connection`` (the agent path)
  never reads (§3.4), and they are resolved under the *person's* user id, so
  the ``SecretAudit(resolve)`` row names them.

* **Its own allowlist, not the risk classes.** ``core/mcp_policy.py``
  classifies a command an agent thought up; here every command is one this
  module wrote, so the check is exact-prefix membership in
  :data:`_ALLOWED_COMMANDS` (:func:`_assert_allowed`). A command this module
  did not plan for is a bug, and fails the step instead of prompting a human.

* **Idempotence per step.** Every step checks the server's actual state
  first: a second run adds no second ``authorized_keys`` entry, no second
  sudoers drop-in and no second user. A step that cannot be completed leaves
  a report saying what was done and what was not, and stops — the server is
  never left half-configured *silently*.

The one deliberately dangerous step (disabling root password login) is
opt-in, runs last, and is skipped outright unless the ``legion`` account has
just been verified to work — losing the key must not mean losing the server
on the same run that created the key.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import vps_inventory
from app.core.events import emit_event
from app.core.mcp_gateway import (
    VpsServiceUnavailable,
    VpsToolError,
    execute_vps_bootstrap_tool,
)
from app.core.secrets.room_secrets import (
    EnvironmentLocked,
    RoomSecretError,
    RoomSecretNotAvailable,
    bind_room_secret,
    resolve_room_secret,
)
from app.models.secret import SecretKind, SecretPurpose

logger = logging.getLogger(__name__)

#: The system actor this procedure runs as. Present in the report and in
#: every Room event it emits, so "who touched this server" always has an
#: answer that is neither an agent nor an anonymous background job.
BOOTSTRAP_ACTOR = "vps_bootstrap"

#: The everyday account created on the server. "legion", not "deployer":
#: shorter, and it does not collide with the account name every third VPS
#: tutorial tells people to make (§3.6 step 3).
LEGION_USER = "legion"
LEGION_HOME = f"/home/{LEGION_USER}"
AUTHORIZED_KEYS_PATH = f"{LEGION_HOME}/.ssh/authorized_keys"
SUDOERS_PATH = "/etc/sudoers.d/90-legion"
#: Written first, validated, then renamed into place. The leading dot matters:
#: ``sudo`` ignores files in ``sudoers.d`` whose name contains one, so even a
#: crash between "write" and "visudo -c" cannot break sudo on the server.
SUDOERS_TMP_PATH = "/etc/sudoers.d/.90-legion.new"
SSHD_DROPIN_PATH = "/etc/ssh/sshd_config.d/90-legion.conf"
PROJECT_ROOT = "/srv/legion"
#: The one filename `nginx -t -c` is allowed to be pointed at (Stage 10, step 3
#: of the deploy recipe). Fixed rather than a wildcard because sudoers matches
#: arguments literally and `skill-vps-deploy` must name the same path.
NGINX_CHECK_PATH = "/tmp/legion-nginx-check.conf"

#: Units ``legion`` may drive through sudo (§3.6 step 5, "units from the list").
#: A fixed list, not a wildcard: ``systemctl restart *`` in a sudoers drop-in
#: would hand out the whole machine.
_SUDO_UNITS = ("nginx", "docker", "docker.socket")
_SUDO_UNIT_VERBS = ("status", "is-active", "reload", "restart", "start", "stop")

_PACKAGES = {
    "docker": ("docker.io", "docker-compose-v2"),
    "nginx": ("nginx",),
    "certbot": ("certbot", "python3-certbot-nginx"),
    "git": ("git",),
}
#: What to look for on `PATH` to decide a profile item is already installed.
_PROFILE_BINARIES = {"docker": "docker", "nginx": "nginx", "certbot": "certbot", "git": "git"}

_APT_TIMEOUT_SEC = 600
_DEFAULT_TIMEOUT_SEC = 60


class BootstrapError(RuntimeError):
    """A step failed. ``code`` is what the report and the UI show."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BootstrapBusy(RuntimeError):
    code = "prepare_already_running"


class BootstrapNotFound(RuntimeError):
    code = "prepare_run_not_found"


# ── the command allowlist (§3.6: "own allowlist instead of risk classes") ────

#: Exact argv prefixes this procedure may run. Matching is prefix equality on
#: the argv list — never a regex over a joined string — because ``vps_exec``
#: takes argv and quotes it, so there is no shell for a metacharacter to
#: reach. Anything not listed here raises before a call is made.
_ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("id", "-un"),
    ("id", "-u", LEGION_USER),
    ("uname", "-r"),
    ("which",),
    ("ss", "-tulnH"),
    ("dpkg-query", "-W", "-f=${Status}", "python3-certbot-nginx"),
    ("useradd", "--create-home", "--shell", "/bin/bash", LEGION_USER),
    ("passwd", "--lock", LEGION_USER),
    ("mkdir", "-p"),
    ("chown",),
    ("chmod",),
    ("rm", "-f"),
    ("mv", SUDOERS_TMP_PATH, SUDOERS_PATH),
    ("visudo", "-c", "-f"),
    ("apt-get", "update"),
    ("apt-get", "install", "-y"),
    ("systemctl", "--version"),
    ("systemctl", "reload", "ssh"),
    ("systemctl", "is-active", "ssh.socket"),
    ("sshd", "-t"),
)

#: The only environment prefix an allowlisted command may carry. ``apt-get``
#: will block forever on a debconf prompt without it.
_ALLOWED_ENV_PREFIX = ("env", "DEBIAN_FRONTEND=noninteractive")


def _assert_allowed(argv: list[str]) -> None:
    candidate = list(argv)
    if candidate[: len(_ALLOWED_ENV_PREFIX)] == list(_ALLOWED_ENV_PREFIX):
        candidate = candidate[len(_ALLOWED_ENV_PREFIX):]
    for prefix in _ALLOWED_COMMANDS:
        if candidate[: len(prefix)] == list(prefix):
            return
    raise BootstrapError(
        "command_not_allowed",
        f"{' '.join(argv[:3])}… is not in the preparation allowlist",
    )


# ── report model ───────────────────────────────────────────────────────────

STEPS: tuple[tuple[str, str], ...] = (
    ("connect", "Connection and host fingerprint"),
    ("inventory", "Server inventory"),
    ("user", f"User {LEGION_USER}"),
    ("key", "Access key"),
    ("sudoers", "Sudo access from a list"),
    ("profile", "Environment profile"),
    ("dirs", "Project directories"),
    ("repo", "Access to the Room's repository"),
    ("verify", "Verification of the new access"),
    ("finalize", "Writing to the inventory"),
    ("harden", "Disabling root password login"),
)


@dataclass
class StepReport:
    id: str
    title: str
    status: str = "pending"  # pending | running | ok | skipped | failed
    detail: str = ""
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "detail": self.detail, "error_code": self.error_code,
        }


@dataclass
class BootstrapProfile:
    """The wizard's checkboxes (§3.6 step 6 + the hardening opt-in)."""

    docker: bool = False
    nginx: bool = False
    certbot: bool = False
    git: bool = True
    disable_root_password_login: bool = False

    def packages(self) -> list[str]:
        out: list[str] = []
        for key in ("docker", "nginx", "certbot", "git"):
            if getattr(self, key):
                out.extend(_PACKAGES[key])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "docker": self.docker, "nginx": self.nginx, "certbot": self.certbot,
            "git": self.git, "disable_root_password_login": self.disable_root_password_login,
        }


@dataclass
class BootstrapRun:
    room_id: UUID
    server: str
    environment: Optional[str]
    actor_user_id: UUID
    profile: BootstrapProfile
    status: str = "running"  # running | awaiting_fingerprint | succeeded | failed
    steps: list[StepReport] = field(default_factory=lambda: [StepReport(i, t) for i, t in STEPS])
    inventory: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    pending_fingerprint: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    _decision: Optional[bool] = None
    _decided: asyncio.Event = field(default_factory=asyncio.Event)

    def step(self, step_id: str) -> StepReport:
        return next(s for s in self.steps if s.id == step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "environment": self.environment,
            "actor": BOOTSTRAP_ACTOR,
            "status": self.status,
            "profile": self.profile.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "inventory": self.inventory,
            "suggestions": self.suggestions,
            "pending_fingerprint": self.pending_fingerprint,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


#: In-process registry, keyed by (room, server, environment). Deliberately the
#: same shape of state as ``ask_user_registry``: a run is a live conversation
#: with one person watching one panel, not a durable job. A backend restart
#: loses the progress view — never the result, which lands in
#: ``VPS_<NAME>_PREPARED_AT`` and in the Room's event stream as it happens.
_runs: dict[tuple[UUID, str, Optional[str]], BootstrapRun] = {}


def _key(room_id: UUID, server: str, environment: Optional[str]) -> tuple[UUID, str, Optional[str]]:
    from app.core import room_config

    return (room_id, vps_inventory.validate_server_name(server), room_config.normalize_environment(environment))


def get_run(room_id: UUID, server: str, environment: Optional[str] = None) -> dict[str, Any]:
    run = _runs.get(_key(room_id, server, environment))
    if run is None:
        raise BootstrapNotFound(f"no preparation run for {server}")
    return run.to_dict()


def forget_run(room_id: UUID, server: str, environment: Optional[str] = None) -> None:
    """Drop the in-memory report for one server, if any.

    Called when a server is removed from the Room's inventory: without this,
    re-adding the same name later would resurrect the old server's report on
    the very first ``GET .../prepare`` and hide the profile picker again,
    exactly the bug this exists to prevent for delete+recreate. A missing
    entry is not an error — most servers are never re-added.
    """
    _runs.pop(_key(room_id, server, environment), None)


async def confirm_fingerprint(
    room_id: UUID, server: str, *, environment: Optional[str] = None, accept: bool
) -> dict[str, Any]:
    """Answer the wizard's own TOFU card (§3.5, §3.6 step 1).

    The decision is the operator's and applies to this run only: on "accept"
    the run pins ``VPS_<NAME>_FINGERPRINT`` itself, on "decline" it stops
    without having authenticated anywhere.
    """
    run = _runs.get(_key(room_id, server, environment))
    if run is None:
        raise BootstrapNotFound(f"no preparation run for {server}")
    if run.status != "awaiting_fingerprint":
        raise BootstrapBusy("this run is not waiting for a host-key decision")
    run._decision = bool(accept)
    run.status = "running"
    run._decided.set()
    return run.to_dict()


# ── key material ───────────────────────────────────────────────────────────


def generate_ed25519_keypair() -> tuple[str, str]:
    """``(OpenSSH private key, "ssh-ed25519 AAAA… legion@legion")``.

    Generated in the Hub and nowhere else (§3.6 step 4): the private half
    goes straight into the Room's encrypted store and only ever leaves it
    inside a ``_legion_conn`` block.
    """
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    return private, f"{public} {LEGION_USER}@legion"


def _public_line_for(private_key: str) -> str:
    loaded = serialization.load_ssh_private_key(private_key.encode("utf-8"), password=None)
    public = loaded.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    return f"{public} {LEGION_USER}@legion"


def room_slug(name: str, room_id: UUID) -> str:
    """A filesystem-safe directory name for ``/srv/legion/<room-slug>``.

    The short room id is always part of it. Two Rooms deploying to the same
    server must not land in the same directory, and a name-only slug collides
    easily — non-Latin names reduce to nothing at all, and "Web" and "web"
    reduce to the same thing.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40].strip("-")
    suffix = room_id.hex[:8]
    return f"{slug}-{suffix}" if slug else f"room-{suffix}"


async def project_dir(db: AsyncSession, room_id: UUID) -> str:
    """``/srv/legion/<room-slug>`` — where this Room's checkout lives.

    Derived, never stored: ``_step_dirs`` builds the very same path from the
    very same two inputs, so there is no second source of truth to drift.
    Stage 14 needs it outside a bootstrap run — the agent is told this path
    through ``vps_list_servers``, and ``vps_repo_status`` looks into it.
    """
    from app.models.room import Room

    room_name = await db.scalar(select(Room.name).where(Room.id == room_id))
    return f"{PROJECT_ROOT}/{room_slug(room_name or '', room_id)}"


def sudoers_content(profile: BootstrapProfile) -> str:
    """The ``/etc/sudoers.d/90-legion`` allowlist for this profile (§3.6 step 5).

    Every entry is a fully-qualified path with fixed arguments. This file —
    not ``core/mcp_policy.py`` — is the one boundary an agent cannot talk its
    way around (§3.7), so it grants the smallest set the chosen profile
    actually needs and nothing "just in case".
    """
    lines = [
        "# Managed by Legion (internal design notes §3.6). Do not edit by hand:",
        "# re-running the preparation wizard overwrites this file.",
        "Cmnd_Alias LEGION_INSPECT = /usr/bin/systemctl --version, /usr/bin/systemctl list-units",
        # legion-vps-mcp's atomicWrite (write-file.ts) stages every sudo write
        # under a fixed /tmp/.legion_tmp_* name before moving it into place —
        # unconditional (not profile-gated) because vps_write_file's
        # privileged path is core Stage 2/3 functionality, not an optional
        # profile feature; the actual gate on *when* sudo=true is allowed is
        # core/mcp_policy.py's write_system class, not this file. A bare
        # command with no argument list means "any arguments" to real sudo
        # (same idiom already used by LEGION_DOCKER below) — a standalone `*`
        # argument token is a hard visudo syntax error ("wildcards are not
        # allowed in command arguments"), caught here only because this was
        # the first time any wildcard Cmnd_Alias in this file was actually
        # run through real visudo against a live VPS rather than just
        # string-matched in a unit test.
        # `ln` and `mkdir` are here for Stage 10's own recipe: enabling an
        # nginx site is a symlink into /etc/nginx/sites-enabled, and a
        # privileged write sometimes needs its parent directory first. They
        # widen nothing in practice — unrestricted `cp` above already means
        # "root can place any file anywhere" — but leaving them out fails the
        # deploy at the last step, which is exactly how the missing
        # cp/mv/chmod/chown/rm was found (commit d93cef8).
        "Cmnd_Alias LEGION_WRITE = /usr/bin/cp, /usr/bin/mv, /usr/bin/chmod, "
        "/usr/bin/chown, /usr/bin/rm, /usr/bin/ln, /usr/bin/mkdir",
    ]
    aliases = ["LEGION_INSPECT", "LEGION_WRITE"]

    units = [u for u in _SUDO_UNITS if (profile.nginx or "nginx" not in u) and (profile.docker or "docker" not in u)]
    if units:
        cmds = ", ".join(
            f"/usr/bin/systemctl {verb} {unit}" for unit in units for verb in _SUDO_UNIT_VERBS
        )
        lines.append(f"Cmnd_Alias LEGION_SERVICE = {cmds}")
        aliases.append("LEGION_SERVICE")
    if profile.nginx:
        # The third entry is Stage 10's pre-install check. sudoers matches
        # arguments exactly, so `nginx -t` does not cover `nginx -t -c <file>`
        # — and validating a candidate config *before* it reaches /etc is the
        # stage's own acceptance criterion. Granting bare `/usr/sbin/nginx`
        # instead would be an escalation (an arbitrary `-c` can set `user
        # root` and load modules), so the path is a single fixed filename the
        # skill is told to always use; a wildcard argument is not an option
        # either (real visudo rejected one, commit d93cef8).
        lines.append(
            "Cmnd_Alias LEGION_NGINX = /usr/sbin/nginx -t, /usr/sbin/nginx -T, "
            f"/usr/sbin/nginx -s reload, /usr/sbin/nginx -t -c {NGINX_CHECK_PATH}"
        )
        aliases.append("LEGION_NGINX")
    if profile.docker:
        lines.append("Cmnd_Alias LEGION_DOCKER = /usr/bin/docker, /usr/bin/docker *")
        aliases.append("LEGION_DOCKER")
    if profile.certbot:
        lines.append("Cmnd_Alias LEGION_CERTBOT = /usr/bin/certbot, /usr/bin/certbot *")
        aliases.append("LEGION_CERTBOT")
    lines.append(
        "Cmnd_Alias LEGION_PKG = /usr/bin/apt-get update, /usr/bin/apt-get install -y *"
    )
    aliases.append("LEGION_PKG")
    lines.append(f"{LEGION_USER} ALL=(root) NOPASSWD: {', '.join(aliases)}")
    return "\n".join(lines) + "\n"


def sshd_hardening_content() -> str:
    return (
        "# Managed by Legion (internal design notes §3.6). Root password login is off;\n"
        "# the Room's VPS_<NAME>_KEY is now the only way in as legion.\n"
        "PermitRootLogin prohibit-password\n"
        "PasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\n"
    )


# ── the procedure ──────────────────────────────────────────────────────────


class _Session:
    """One preparation run's connection to the server, as a call helper."""

    def __init__(
        self,
        db: AsyncSession,
        run: BootstrapRun,
        *,
        conn: dict[str, Any],
        request_id: UUID,
    ) -> None:
        self.db = db
        self.run = run
        self.conn = conn
        self.request_id = request_id

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        return await execute_vps_bootstrap_tool(
            self.db,
            room_id=self.run.room_id,
            actor_user_id=self.run.actor_user_id,
            tool=tool,
            arguments={"server": self.run.server, **arguments},
            legion_conn=self.conn,
            request_id=self.request_id,
        )

    async def exec(
        self, argv: list[str], *, sudo: bool = False, timeout_sec: int = _DEFAULT_TIMEOUT_SEC
    ) -> dict[str, Any]:
        _assert_allowed(argv)
        result = await self.call(
            "vps_exec",
            {"command": argv, "sudo": sudo, "timeout_sec": timeout_sec},
        )
        if not isinstance(result, dict):
            raise BootstrapError("unexpected_result", f"{argv[0]} returned no result object")
        return result

    async def run_ok(
        self, argv: list[str], *, sudo: bool = False, code: str, timeout_sec: int = _DEFAULT_TIMEOUT_SEC
    ) -> dict[str, Any]:
        """``exec`` that turns a non-zero exit into a typed step failure."""
        result = await self.exec(argv, sudo=sudo, timeout_sec=timeout_sec)
        if result.get("exit_code") != 0:
            stderr = str(result.get("stderr") or result.get("stdout") or "").strip()
            raise BootstrapError(code, f"`{argv[0]}` exited with code {result.get('exit_code')}: {stderr[:300]}")
        return result

    async def read_text(self, path: str, *, max_bytes: int = 262144) -> Optional[str]:
        """File contents, or ``None`` when there is no such file."""
        try:
            result = await self.call("vps_read_file", {"path": path, "max_bytes": max_bytes})
        except VpsToolError as exc:
            if exc.code == "path_not_found":
                return None
            raise
        raw = base64.b64decode(str(result.get("content_base64") or ""))
        return raw.decode("utf-8", errors="replace")

    async def write_text(
        self, path: str, content: str, *, mode: Optional[str] = None, owner: Optional[str] = None
    ) -> None:
        args: dict[str, Any] = {
            "path": path,
            "content_base64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "sudo": True,
        }
        if mode:
            args["mode"] = mode
        if owner:
            args["owner"] = owner
        await self.call("vps_write_file", args)


async def _emit_step_event(db: AsyncSession, run: BootstrapRun, step: StepReport) -> None:
    """One Room event per finished step (§3.6, "every step is a Room event").

    Carries the step, its outcome and the actor — never a command line and
    never a credential: the detail text is written by this module, but a
    stderr excerpt can still quote whatever the server printed.
    """
    try:
        await emit_event(
            room_id=run.room_id,
            event_type="VPS_PREPARE_STEP",
            content=f"Preparing '{run.server}': {step.title} — {step.status}",
            db=db,
            metadata={
                "server": run.server,
                "environment": run.environment,
                "step": step.id,
                "status": step.status,
                "error_code": step.error_code,
                "actor": BOOTSTRAP_ACTOR,
            },
            sender_type="system",
        )
    except Exception:
        logger.warning("vps_bootstrap: failed to emit step event (room=%s)", run.room_id, exc_info=True)


#: How long the run waits for the operator to answer the host-key card before
#: giving up. Generous: they may have to open the provider's panel to compare
#: the fingerprint, which is exactly the check the card exists for.
_FINGERPRINT_TIMEOUT_SEC = 900


async def _await_fingerprint_decision(
    db: AsyncSession, run: BootstrapRun, *, fingerprint: str, expected: Optional[str]
) -> None:
    """Pause the run on the wizard's own host-key card and apply the answer."""
    run.pending_fingerprint = {
        "fingerprint": fingerprint,
        "expected_fingerprint": expected,
        "kind": "mismatch" if expected else "unknown",
    }
    run.status = "awaiting_fingerprint"
    run._decided.clear()
    try:
        await asyncio.wait_for(run._decided.wait(), timeout=_FINGERPRINT_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        run.pending_fingerprint = None
        raise BootstrapError("host_key_rejected", "the host fingerprint was not confirmed in time")
    run.pending_fingerprint = None
    if not run._decision:
        raise BootstrapError("host_key_rejected", "the operator did not confirm the host fingerprint")
    await vps_inventory.set_fingerprint(
        db, room_id=run.room_id, actor_user_id=run.actor_user_id,
        name=run.server, fingerprint=fingerprint, environment=run.environment,
    )
    await db.commit()
    try:
        await emit_event(
            room_id=run.room_id,
            event_type="VPS_HOST_TRUSTED",
            content=f"Preparing '{run.server}': the operator confirmed the fingerprint {fingerprint}",
            db=db,
            metadata={
                "server": run.server, "fingerprint": fingerprint,
                "previous_fingerprint": expected, "actor": BOOTSTRAP_ACTOR,
            },
            sender_type="system",
        )
    except Exception:
        logger.warning("vps_bootstrap: failed to emit host-trust event", exc_info=True)


async def _step_connect(db: AsyncSession, run: BootstrapRun, service: Any, request_id: UUID) -> _Session:
    """Probe the host key, settle trust with the operator, then authenticate."""
    coords = await vps_inventory.server_coordinates(
        db, run.room_id, run.server, environment=run.environment
    )
    # A prepared server's stored USER is already `legion`; root credentials
    # still belong to the root account, so a re-run must not try to use them
    # as `legion` (§3.6: the procedure is repeatable).
    root_user = "root" if coords.user == LEGION_USER else coords.user

    async def _root_session() -> _Session:
        conn = await vps_inventory.resolve_root_connection(
            db, service=service, room_id=run.room_id, actor_user_id=run.actor_user_id,
            server=run.server, user=root_user, environment=run.environment, request_id=request_id,
        )
        return _Session(db, run, conn=conn.legion_conn, request_id=request_id)

    session = await _root_session()
    probe = await session.call("vps_host_key", {})
    fingerprint = str((probe or {}).get("fingerprint") or "").strip()
    if not fingerprint:
        raise BootstrapError("host_key_unknown", "the server did not report a host key fingerprint")
    if coords.fingerprint != fingerprint:
        await _await_fingerprint_decision(
            db, run, fingerprint=fingerprint, expected=coords.fingerprint
        )
    # `vps_host_key` always burns its egress ticket on a raw, unpooled
    # connect-then-abort (ssh-client.ts probeHostKey) — the pool never sees
    # that connection, so the ticket cannot be reused for the real login
    # below regardless of whether the fingerprint matched. Re-resolving only
    # on mismatch left the already-trusted-server path (the common case on
    # any re-run) reusing the spent ticket, which legion-egress always
    # rejects as "ticket already used" — a 100%-reproducible failure on
    # every repeat prepare, caught testing against a real VPS.
    session = await _root_session()

    whoami = await session.run_ok(["id", "-un"], code="connect_failed")
    run.inventory["login"] = str(whoami.get("stdout") or "").strip()
    run.step("connect").detail = (
        f"{coords.host}:{coords.port}, login as {run.inventory['login']}, fingerprint {fingerprint}"
    )
    return session


async def _step_inventory(session: _Session, run: BootstrapRun) -> None:
    """Report what is already on the machine — never a silent "done"."""
    os_release = await session.read_text("/etc/os-release", max_bytes=8192) or ""
    pretty = next(
        (line.split("=", 1)[1].strip().strip('"') for line in os_release.splitlines()
         if line.startswith("PRETTY_NAME=")),
        "unknown OS",
    )
    kernel = await session.exec(["uname", "-r"])
    binaries = ["sudo", "docker", "nginx", "git", "certbot"]
    which = await session.exec(["which", *binaries])
    found = {
        line.rsplit("/", 1)[-1] for line in str(which.get("stdout") or "").splitlines() if line.strip()
    }
    #: `python3-certbot-nginx` has no binary of its own — `which certbot` finding
    #: certbot does not mean the nginx plugin is there (found live on TEST:
    #: certbot installed manually without it).
    plugin_check = await session.exec(["dpkg-query", "-W", "-f=${Status}", "python3-certbot-nginx"])
    run.inventory["certbot_nginx_plugin"] = (
        plugin_check.get("exit_code") == 0 and "installed" in str(plugin_check.get("stdout") or "")
    )
    ports: list[str] = []
    listening = await session.exec(["ss", "-tulnH"])
    if listening.get("exit_code") == 0:
        for line in str(listening.get("stdout") or "").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                ports.append(parts[4])
    sites: list[str] = []
    try:
        listed = await session.call("vps_list_dir", {"path": "/etc/nginx/sites-enabled"})
        entries = (listed or {}).get("entries") or []
        sites = [str(e.get("name")) for e in entries if isinstance(e, dict) and e.get("name")]
    except VpsToolError as exc:
        if exc.code != "path_not_found":
            raise

    run.inventory.update({
        "os": pretty,
        "kernel": str(kernel.get("stdout") or "").strip(),
        "installed": sorted(found),
        "listening": sorted(set(ports))[:40],
        "nginx_sites": sites,
    })
    plugin_note = "" if "certbot" not in found else (
        "; certbot nginx plugin: present" if run.inventory["certbot_nginx_plugin"]
        else "; certbot nginx plugin: absent (certbot --nginx will not work)"
    )
    run.step("inventory").detail = (
        f"{pretty}; installed: {', '.join(sorted(found)) or '—'}; "
        f"listening: {len(set(ports))} sockets; nginx sites: {len(sites)}{plugin_note}"
    )


async def _step_user(session: _Session, run: BootstrapRun) -> None:
    existing = await session.exec(["id", "-u", LEGION_USER])
    if existing.get("exit_code") == 0:
        run.step("user").detail = f"user {LEGION_USER} already exists (uid {str(existing.get('stdout') or '').strip()})"
        return
    await session.run_ok(
        ["useradd", "--create-home", "--shell", "/bin/bash", LEGION_USER],
        sudo=True, code="user_create_failed",
    )
    # `useradd` leaves the password field locked already; doing it explicitly
    # is the plan's `--disabled-password` in `useradd` vocabulary, and it is
    # what makes a re-run over a hand-made `legion` account safe too.
    await session.exec(["passwd", "--lock", LEGION_USER], sudo=True)
    run.step("user").detail = f"user {LEGION_USER} created without a password"


async def _step_key(
    db: AsyncSession, session: _Session, run: BootstrapRun, service: Any, request_id: UUID
) -> str:
    """Ensure ``VPS_<NAME>_KEY`` exists and its public half is authorized once."""
    env_name = f"VPS_{run.server}_KEY"
    reused = True
    try:
        private_key = await resolve_room_secret(
            db, service=service, room_id=run.room_id, actor_user_id=run.actor_user_id,
            env_name=env_name, purpose=SecretPurpose.VPS_ACCESS, environment=run.environment,
            scope=None, request_id=request_id, expected_kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
        public_line = _public_line_for(private_key)
    except RoomSecretNotAvailable:
        reused = False
        private_key, public_line = generate_ed25519_keypair()
        await bind_room_secret(
            db, service=service, room_id=run.room_id, actor_user_id=run.actor_user_id,
            env_name=env_name, value=private_key, environment=run.environment,
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.SSH_PRIVATE_KEY,
            request_id=request_id,
        )
        await db.commit()

    await session.run_ok(["mkdir", "-p", f"{LEGION_HOME}/.ssh"], sudo=True, code="key_install_failed")
    current = await session.read_text(AUTHORIZED_KEYS_PATH) or ""
    key_body = public_line.split()[1]
    if key_body in current:
        run.step("key").detail = (
            f"key {'reused' if reused else 'created'}; the authorized_keys entry already exists"
        )
    else:
        lines = [line for line in current.splitlines() if line.strip()]
        # `restrict` turns off port/agent forwarding, X11, PTY and user-rc for
        # this key — nothing `vps_exec`/SFTP needs. If the server's sshd is old
        # enough not to know the option, step 8 catches it (`auth_failed`) and
        # `_relax_authorized_key` rewrites the line without it.
        lines.append(f"restrict {public_line}")
        await session.write_text(
            AUTHORIZED_KEYS_PATH, "\n".join(lines) + "\n",
            mode="600", owner=f"{LEGION_USER}:{LEGION_USER}",
        )
        run.step("key").detail = f"key {'reused' if reused else 'created'}; the public part was added to authorized_keys"
    await session.exec(["chown", "-R", f"{LEGION_USER}:{LEGION_USER}", f"{LEGION_HOME}/.ssh"], sudo=True)
    await session.exec(["chmod", "700", f"{LEGION_HOME}/.ssh"], sudo=True)
    return private_key


async def _relax_authorized_key(session: _Session, private_key: str) -> bool:
    """Drop the ``restrict`` prefix after an ``auth_failed`` in step 8.

    Returns whether anything changed — an sshd too old for ``restrict``
    rejects the whole key line, which looks exactly like "the key was never
    installed" from the client side.
    """
    public_line = _public_line_for(private_key)
    current = await session.read_text(AUTHORIZED_KEYS_PATH) or ""
    if f"restrict {public_line}" not in current:
        return False
    await session.write_text(
        AUTHORIZED_KEYS_PATH,
        current.replace(f"restrict {public_line}", public_line),
        mode="600", owner=f"{LEGION_USER}:{LEGION_USER}",
    )
    return True


async def _step_sudoers(session: _Session, run: BootstrapRun) -> None:
    """Write the allowlist through ``visudo -c``; never install a broken one."""
    content = sudoers_content(run.profile)
    existing = await session.read_text(SUDOERS_PATH)
    if existing == content:
        run.step("sudoers").detail = f"{SUDOERS_PATH} already matches the profile"
        return
    await session.write_text(SUDOERS_TMP_PATH, content, mode="440", owner="root:root")
    check = await session.exec(["visudo", "-c", "-f", SUDOERS_TMP_PATH], sudo=True)
    if check.get("exit_code") != 0:
        await session.exec(["rm", "-f", SUDOERS_TMP_PATH], sudo=True)
        raise BootstrapError(
            "sudoers_invalid",
            "visudo rejected the generated file — it was not installed, sudo on the server is untouched: "
            + str(check.get("stderr") or "")[:300],
        )
    await session.run_ok(["mv", SUDOERS_TMP_PATH, SUDOERS_PATH], sudo=True, code="sudoers_invalid")
    await session.exec(["chown", "root:root", SUDOERS_PATH], sudo=True)
    await session.exec(["chmod", "440", SUDOERS_PATH], sudo=True)
    run.step("sudoers").detail = f"{SUDOERS_PATH} installed and checked by visudo"


async def _step_profile(session: _Session, run: BootstrapRun) -> None:
    installed = set(run.inventory.get("installed") or [])
    wanted = [
        key for key in ("docker", "nginx", "certbot", "git")
        if getattr(run.profile, key) and (
            _PROFILE_BINARIES[key] not in installed
            or (key == "certbot" and not run.inventory.get("certbot_nginx_plugin"))
        )
    ]
    if not wanted:
        run.step("profile").status = "skipped"
        run.step("profile").detail = "everything selected is already installed"
        return
    packages: list[str] = []
    for key in wanted:
        packages.extend(_PACKAGES[key])
    await session.run_ok(
        [*_ALLOWED_ENV_PREFIX, "apt-get", "update"],
        sudo=True, code="profile_install_failed", timeout_sec=_APT_TIMEOUT_SEC,
    )
    await session.run_ok(
        [*_ALLOWED_ENV_PREFIX, "apt-get", "install", "-y", *packages],
        sudo=True, code="profile_install_failed", timeout_sec=_APT_TIMEOUT_SEC,
    )
    run.step("profile").detail = "installed: " + ", ".join(packages)
    if "docker" in wanted:
        run.suggestions.append(
            f"{LEGION_USER} was deliberately not added to the docker group: membership in it is equivalent to root. "
            "Docker is available through sudo per the list in /etc/sudoers.d/90-legion."
        )


async def _step_dirs(db: AsyncSession, session: _Session, run: BootstrapRun) -> str:
    path = await project_dir(db, run.room_id)
    await session.run_ok(["mkdir", "-p", path], sudo=True, code="dirs_failed")
    await session.exec(["chown", "-R", f"{LEGION_USER}:{LEGION_USER}", path], sudo=True)
    await session.exec(["chmod", "750", path], sudo=True)
    run.inventory["project_dir"] = path
    run.step("dirs").detail = f"{path} is owned by {LEGION_USER}"
    return path


async def _step_repo(
    db: AsyncSession, session: _Session, run: BootstrapRun, service: Any, request_id: UUID
) -> None:
    """Give the server a read-only credential for the Room's repository (Stage 13).

    Placed after ``dirs`` and before ``verify`` because it writes into
    ``legion``'s home, which the ``user``/``key`` steps have just created,
    and because the acceptance for the run as a whole is "a prepared server
    can clone the project" — a server that cannot is prepared only on paper.

    Nothing here is fatal to the preparation. A Room with no external
    repository, a forge that will not take the key, a forge whose host keys
    are not published: each leaves the step ``skipped`` with a sentence
    saying what a person has to do, and the rest of the run continues. The
    one thing this step must never do is silently report success on a
    credential that does not work — hence ``git ls-remote`` inside
    :func:`vps_deploy_key.issue`, run as ``legion``, before anything is
    recorded.
    """
    from app.core import vps_deploy_key

    step = run.step("repo")
    if not run.profile.git:
        step.status = "skipped"
        step.detail = "git is not selected in the profile — repository access is not granted"
        return

    remote = vps_deploy_key.Remote(call=session.call, user=run.inventory.get("login") or "root")
    try:
        result = await vps_deploy_key.issue(
            db, service=service, room_id=run.room_id, actor_user_id=run.actor_user_id,
            server=run.server, remote=remote, environment=run.environment,
            request_id=request_id, require_prepared=False,
        )
    except (vps_deploy_key.DeployKeyError, VpsToolError) as exc:
        # A transport failure here used to abort the whole run: the machine
        # was already fully prepared, and still ended up without
        # `VPS_<NAME>_USER`, without the verification step and marked "not
        # prepared", because the last optional step could not reach the
        # forge. "Nothing here is fatal" has to cover the connection too, not
        # only the forge's answer.
        step.status = "skipped"
        step.error_code = getattr(exc, "code", "unavailable")
        step.detail = str(exc)
        run.suggestions.append(
            "The server is prepared, but it cannot read the Room's repository yet: "
            f"{exc}. The 'Grant repository access' button on the server card will repeat this step."
        )
        return

    run.inventory["deploy_remote"] = result["remote_url"]
    run.inventory["deploy_credential"] = result["credential"]
    if result["status"] == "manual":
        step.status = "skipped"
        step.detail = result["detail"]
        run.inventory["deploy_public_key"] = result["public_key"]
        run.suggestions.append(result["detail"])
        return
    step.detail = f"{result['detail']}; remote: {result['remote_url']}"


async def _step_verify(
    db: AsyncSession, run: BootstrapRun, *, private_key: str, project_dir: str,
    root_session: _Session, request_id: UUID,
) -> None:
    """Connect as ``legion`` with the new key and use it for real (§3.6 step 8)."""
    coords = await vps_inventory.server_coordinates(
        db, run.room_id, run.server, environment=run.environment
    )

    def _legion_session() -> _Session:
        conn = vps_inventory.build_connection(
            room_id=run.room_id, server=run.server, host=coords.host, port=coords.port,
            user=LEGION_USER, auth={"kind": "key", "private_key": private_key, "passphrase": None},
            fingerprint=coords.fingerprint, call_id=uuid4(),
        )
        return _Session(db, run, conn=conn, request_id=request_id)

    session = _legion_session()
    try:
        whoami = await session.exec(["id", "-un"])
    except VpsToolError as exc:
        if exc.code != "auth_failed" or not await _relax_authorized_key(root_session, private_key):
            raise
        run.step("key").detail += "; the restrict option was removed — the server's sshd does not accept it"
        session = _legion_session()
        whoami = await session.exec(["id", "-un"])

    if str(whoami.get("stdout") or "").strip() != LEGION_USER:
        raise BootstrapError("verify_failed", "the new key connected as the wrong user")
    await session.run_ok(["systemctl", "--version"], sudo=True, code="verify_failed")
    probe_path = f"{project_dir}/.legion_prepare_check"
    await session.write_text(probe_path, f"prepared {datetime.now(timezone.utc).isoformat()}\n")
    await session.run_ok(["rm", "-f", probe_path], code="verify_failed")
    run.step("verify").detail = f"login as {LEGION_USER} by key, sudo -n and writing to {project_dir} work"


async def _step_finalize(db: AsyncSession, run: BootstrapRun) -> None:
    prepared_at = datetime.now(timezone.utc).isoformat()
    await vps_inventory.set_server_variable(
        db, room_id=run.room_id, actor_user_id=run.actor_user_id, name=run.server,
        suffix="USER", value=LEGION_USER, environment=run.environment,
    )
    await vps_inventory.set_server_variable(
        db, room_id=run.room_id, actor_user_id=run.actor_user_id, name=run.server,
        suffix="PREPARED_AT", value=prepared_at, environment=run.environment,
    )
    await db.commit()
    run.step("finalize").detail = f"VPS_{run.server}_USER={LEGION_USER}, PREPARED_AT={prepared_at}"


async def _step_harden(session: _Session, run: BootstrapRun) -> None:
    """The opt-in step that can lock everyone out — hence last, and validated."""
    content = sshd_hardening_content()
    existing = await session.read_text(SSHD_DROPIN_PATH)
    if existing != content:
        await session.run_ok(["mkdir", "-p", "/etc/ssh/sshd_config.d"], sudo=True, code="sshd_failed")
        await session.write_text(SSHD_DROPIN_PATH, content, mode="644", owner="root:root")
        check = await session.exec(["sshd", "-t"], sudo=True)
        if check.get("exit_code") != 0:
            await session.exec(["rm", "-f", SSHD_DROPIN_PATH], sudo=True)
            raise BootstrapError(
                "sshd_invalid",
                "sshd -t rejected the configuration — it was removed, server login is unchanged: "
                + str(check.get("stderr") or "")[:300],
            )
    reload_result = await session.exec(["systemctl", "reload", "ssh"], sudo=True)
    note = ""
    if reload_result.get("exit_code") != 0:
        # Ubuntu 22.10+ — including the 26.04 this was found on — ships sshd
        # socket-activated: `ssh.service` is not running between connections,
        # so `reload` legitimately fails with "Unit ssh.service is not
        # active" while the drop-in already governs every new connection.
        # Failing the run there told the person their server was broken when
        # nothing was, on the one step whose failure reads as "locked out".
        socket_active = await session.exec(["systemctl", "is-active", "ssh.socket"])
        if str(socket_active.get("stdout") or "").strip() != "active":
            stderr = str(reload_result.get("stderr") or reload_result.get("stdout") or "").strip()
            raise BootstrapError(
                "sshd_failed",
                f"`systemctl reload ssh` exited with code {reload_result.get('exit_code')}: {stderr[:300]}",
            )
        note = "; sshd is started via ssh.socket — new connections already read the new config"
    run.step("harden").detail = (
        f"{SSHD_DROPIN_PATH}: root password login and password authentication disabled{note}"
    )


async def _run(
    session_factory: Callable[[], Any],
    run: BootstrapRun,
    request_id: UUID,
) -> None:
    from app.core.secrets import runtime as secrets_runtime

    async with session_factory() as db:
        service = secrets_runtime.service_for(db)
        failed_at: Optional[str] = None
        try:
            root_session = await _finish_step(db, run, "connect", _step_connect(db, run, service, request_id))
            await _finish_step(db, run, "inventory", _step_inventory(root_session, run))
            await _finish_step(db, run, "user", _step_user(root_session, run))
            private_key = await _finish_step(
                db, run, "key", _step_key(db, root_session, run, service, request_id)
            )
            await _finish_step(db, run, "sudoers", _step_sudoers(root_session, run))
            await _finish_step(db, run, "profile", _step_profile(root_session, run))
            project_dir = await _finish_step(db, run, "dirs", _step_dirs(db, root_session, run))
            await _finish_step(db, run, "repo", _step_repo(db, root_session, run, service, request_id))
            await _finish_step(
                db, run, "verify",
                _step_verify(
                    db, run, private_key=private_key, project_dir=project_dir,
                    root_session=root_session, request_id=request_id,
                ),
            )
            await _finish_step(db, run, "finalize", _step_finalize(db, run))
            if run.profile.disable_root_password_login:
                await _finish_step(db, run, "harden", _step_harden(root_session, run))
            else:
                harden = run.step("harden")
                harden.status = "skipped"
                harden.detail = "the checkbox is not set — sshd_config untouched"
        except _StepFailed as exc:
            failed_at = exc.step_id
        except Exception as exc:  # noqa: BLE001 — a bug here must still produce a report
            logger.warning("vps_bootstrap: run crashed (room=%s server=%s)", run.room_id, run.server, exc_info=True)
            run.error_code = "internal_error"
            run.error_message = str(exc)[:300]
            failed_at = None

        stopped = failed_at is not None or run.error_code is not None
        if stopped:
            # "An error at any step leaves a clear report 'what is done, what is
            # not'" — the steps that never ran say so instead of staying
            # `pending`, which in the UI would read as "still going".
            for step in run.steps:
                if step.status in ("pending", "running"):
                    step.status = "skipped"
                    step.detail = step.detail or "did not run — the preparation stopped earlier"
        run.status = "failed" if stopped else "succeeded"
        run.finished_at = datetime.now(timezone.utc)
        if run.status == "succeeded":
            run.suggestions.insert(
                0,
                "Root credentials are no longer needed — they can be removed with the 'Forget root credentials' button.",
            )
            run.suggestions.insert(
                1,
                "For the agent to be able not only to look at the server but also to change it, switch the Room "
                "to the 'Full' access mode.",
            )
        try:
            await emit_event(
                room_id=run.room_id,
                event_type="VPS_PREPARE_FINISHED",
                content=(
                    f"Preparation of server '{run.server}' {'completed' if run.status == 'succeeded' else 'stopped'}"
                    + (f": {run.error_message}" if run.error_message else "")
                ),
                db=db,
                metadata={
                    "server": run.server, "status": run.status,
                    "error_code": run.error_code, "actor": BOOTSTRAP_ACTOR,
                },
                sender_type="system",
            )
        except Exception:
            logger.warning("vps_bootstrap: failed to emit finish event", exc_info=True)


class _StepFailed(Exception):
    def __init__(self, step_id: str) -> None:
        super().__init__(step_id)
        self.step_id = step_id


async def _finish_step(db: AsyncSession, run: BootstrapRun, step_id: str, coro: Awaitable[Any]) -> Any:
    """Run one step, record its outcome, and stop the run if it failed.

    Every failure mode ends up in the same report shape: our own typed
    :class:`BootstrapError`, a structured ``VpsToolError`` from the service
    (``auth_failed``, ``sudo_not_permitted``, ``disk_full``, …), a secrets-layer
    refusal, or an unexpected bug. None of them leaves the run without a
    "what is done, what is not" answer.
    """
    step = run.step(step_id)
    step.status = "running"
    try:
        result = await coro
    except BootstrapError as exc:
        step.status, step.error_code, step.detail = "failed", exc.code, str(exc)
    except VpsToolError as exc:
        step.status, step.error_code, step.detail = "failed", exc.code, str(exc)
    except VpsServiceUnavailable as exc:
        step.status, step.error_code, step.detail = "failed", exc.code, str(exc)
    except EnvironmentLocked as exc:
        step.status, step.error_code, step.detail = "failed", exc.code, str(exc)
    except vps_inventory.VpsInventoryError as exc:
        step.status, step.error_code, step.detail = "failed", exc.code, str(exc)
    except RoomSecretError as exc:
        step.status, step.error_code, step.detail = "failed", "secret_error", str(exc)
    else:
        if step.status == "running":
            step.status = "ok"
        await _emit_step_event(db, run, step)
        return result

    run.error_code = step.error_code
    run.error_message = step.detail
    await _emit_step_event(db, run, step)
    raise _StepFailed(step_id)


async def start(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    profile: BootstrapProfile,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    session_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """Kick off a preparation run and return its initial report.

    ``db`` is only used for the pre-flight checks (the server exists, the
    room is not already preparing it); the run itself takes its own session,
    because it outlives the HTTP request that started it.
    """
    name = vps_inventory.validate_server_name(server)
    await vps_inventory.server_coordinates(db, room_id, name, environment=environment)

    key = _key(room_id, name, environment)
    existing = _runs.get(key)
    if existing is not None and existing.status in ("running", "awaiting_fingerprint"):
        raise BootstrapBusy(f"preparation of {name} is already in progress")

    run = BootstrapRun(
        room_id=room_id, server=name, environment=key[2],
        actor_user_id=actor_user_id, profile=profile,
    )
    _runs[key] = run

    if session_factory is None:
        from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
        from sqlalchemy.orm import sessionmaker

        from app.db.session import engine

        session_factory = sessionmaker(engine, class_=_AsyncSession, expire_on_commit=False)

    asyncio.create_task(_run(session_factory, run, request_id or uuid4()))
    return run.to_dict()
