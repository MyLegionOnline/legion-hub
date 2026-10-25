"""The repository credential one prepared server gets — ``internal design notes`` §16b, Stage 13.

Until this module existed, §3.10's "we ship code with git" was a promise with no
mechanism behind it: the preparation wizard installed ``git`` and made a
project directory, and then the server had no way at all to read the Room's
repository. An agent asked to deploy found a checkout it could not create, a
`room_config` that never mentioned a repository, and one "official" transport
(``vps_upload``) that wanted the whole bundle base64-encoded through its own
context. It did what anyone would do with those three facts and went looking
for a way around them.

What this module refuses to do is the short path: take the Room's own
``git_https`` token — a full-account PAT that can write every repository the
owner has — and copy it onto a rented machine we do not control (§3.5).
Instead each server gets its own credential:

* **SSH (preferred).** The Hub generates an ed25519 pair, the private half
  goes to that one server and is never stored in the Room or returned to the
  Hub, and the public half is registered on the forge as a **read-only**
  deploy key. Read-only is enforced by the forge, not by our policy layer, so
  a push from the server is refused by the thing that owns the repository.
* **HTTPS (fallback).** A read-only repository token the person pastes
  themselves, stored as ``purpose=git_deploy`` — a purpose that, exactly like
  ``vps_access``, is outside ``ROOM_SECRET_PURPOSES`` and therefore
  unreachable from ``bash(secrets=…)``, the ``.env`` sync, or any agent path.

Two smaller decisions worth naming, because both were bugs waiting to happen:

* The SSH config entry is a **per-Room alias** (``Host legion-<room-slug>``
  with ``HostKeyAlias`` pinned to the forge), not ``Host github.com``. Two
  Rooms deploying to the same box would otherwise write two blocks for the
  same host and the first one would silently win — the same collision
  ``room_slug`` already exists to prevent for directories.
* The forge's host keys are fetched from its published list and written to
  ``known_hosts`` outright. TOFU belongs to §3.5, where the host is *your*
  server and nobody publishes its key; a forge's keys are public, so
  ``StrictHostKeyChecking yes`` stays on and an interposed host is a hard
  failure rather than a prompt nobody sees.

Like ``vps_bootstrap.py`` this module drives ``legion-vps-mcp`` over the Hub's
own Gateway surface with its own fixed command allowlist rather than
``core/mcp_policy.py``: every command here is one this file wrote, so an
unlisted one is a bug and must fail rather than raise a card.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import git_forge, vps_inventory
from app.core.mcp_gateway import VpsToolError
from app.core.vps_bootstrap import (
    LEGION_HOME,
    LEGION_USER,
    generate_ed25519_keypair,
    room_slug,
)
from app.models.secret import SecretKind, SecretPurpose

logger = logging.getLogger(__name__)

SSH_DIR = f"{LEGION_HOME}/.ssh"
SSH_CONFIG_PATH = f"{SSH_DIR}/config"
KNOWN_HOSTS_PATH = f"{SSH_DIR}/known_hosts"
GITCONFIG_PATH = f"{LEGION_HOME}/.gitconfig"

#: The secret name the HTTPS fallback token is bound under.
DEPLOY_TOKEN_SUFFIX = vps_inventory.DEPLOY_TOKEN_SUFFIX

#: How long to keep retrying the `ls-remote` check after registering a key.
#: A freshly created GitHub deploy key is not always usable on the very next
#: connection, and a spurious "does not pass" here would send a person looking
#: for a problem that fixes itself in two seconds.
_VERIFY_ATTEMPTS = 3
_VERIFY_BACKOFF_SEC = 2.0
_LS_REMOTE_TIMEOUT_SEC = 45


class DeployKeyError(RuntimeError):
    """A typed failure. ``code`` is what the report, the UI and the API show."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ── the command allowlist (same idiom as vps_bootstrap._assert_allowed) ────

_ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("mkdir", "-p", SSH_DIR),
    ("chmod",),
    ("chown",),
    ("rm", "-f"),
    ("git", "ls-remote"),
    # The wizard runs as root, so the check has to drop to `legion` to prove
    # the credential works *for the login that will use it* — running it as
    # root would test root's ssh config and pass with the key unreadable.
    ("sudo", "-n", "-u", LEGION_USER, "-H", "git", "ls-remote"),
)


def _assert_allowed(argv: list[str]) -> None:
    for prefix in _ALLOWED_COMMANDS:
        if argv[: len(prefix)] == list(prefix):
            return
    raise DeployKeyError(
        "command_not_allowed",
        f"{' '.join(argv[:3])}… is not in the deploy-credential allowlist",
    )


# ── managed blocks in files the person may also have edited ────────────────


def _begin(slug: str, comment: str = "#") -> str:
    return f"{comment} >>> legion deploy {slug} >>>"


def _end(slug: str, comment: str = "#") -> str:
    return f"{comment} <<< legion deploy {slug} <<<"


def replace_block(text: str, slug: str, body: Optional[str], comment: str = "#") -> str:
    """Insert, replace or (``body=None``) drop this Room's block in ``text``.

    The files below — ``~/.ssh/config``, ``known_hosts``, ``~/.gitconfig`` —
    belong to the server's ``legion`` user, and another Room (or a person)
    may legitimately have lines in them. Rewriting the whole file would be
    the same class of bug as `authorized_keys` growing a duplicate entry on
    every re-run, so only the fenced region is ours.
    """
    begin, end = _begin(slug, comment), _end(slug, comment)
    lines = (text or "").splitlines()
    out: list[str] = []
    inside = False
    replaced = False
    for line in lines:
        if line.strip() == begin:
            inside = True
            continue
        if inside:
            if line.strip() == end:
                inside = False
                if body is not None:
                    out.extend([begin, *body.rstrip("\n").splitlines(), end])
                    replaced = True
            continue
        out.append(line)
    if body is not None and not replaced:
        while out and not out[-1].strip():
            out.pop()
        out.extend([begin, *body.rstrip("\n").splitlines(), end])
    while out and not out[-1].strip():
        out.pop()
    return ("\n".join(out) + "\n") if out else ""


def ssh_config_block(*, alias: str, forge: str, identity_path: str) -> str:
    """The ``~/.ssh/config`` entry a ``git@<alias>:owner/repo.git`` URL uses.

    ``IdentitiesOnly`` keeps ssh from walking every other key on the box (a
    forge counts wrong-key attempts against the connection, so a second Room's
    key can make this one fail), ``BatchMode`` turns a prompt into an error
    instead of a hang, and ``HostKeyAlias`` makes the ``known_hosts`` lookup
    use the forge's real name even though we connect through the alias.
    """
    return (
        f"Host {alias}\n"
        f"  HostName {forge}\n"
        f"  HostKeyAlias {forge}\n"
        f"  User git\n"
        f"  IdentityFile {identity_path}\n"
        f"  IdentitiesOnly yes\n"
        f"  BatchMode yes\n"
        f"  StrictHostKeyChecking yes\n"
    )


def gitconfig_block(*, forge: str, credentials_path: str) -> str:
    return (
        f'[credential "https://{forge}"]\n'
        f"\thelper = store --file={credentials_path}\n"
    )


# ── the server side of one issue/revoke ────────────────────────────────────


@dataclass
class Remote:
    """One conversation with the server, as ``root`` or as ``legion``.

    Both callers reach the same procedure: the preparation wizard (already
    holding a root session) and the "Grant repository access" button on a
    prepared server (which logs in as ``legion`` with the everyday key, so a
    Room that has already forgotten its root credentials can still issue one).
    """

    call: Callable[[str, dict[str, Any]], Awaitable[Any]]
    user: str

    @property
    def privileged(self) -> bool:
        return self.user != LEGION_USER

    async def read_text(self, path: str, *, max_bytes: int = 262144) -> Optional[str]:
        try:
            result = await self.call("vps_read_file", {"path": path, "max_bytes": max_bytes})
        except VpsToolError as exc:
            if exc.code in ("path_not_found", "permission_denied"):
                return None
            raise
        raw = base64.b64decode(str((result or {}).get("content_base64") or ""))
        return raw.decode("utf-8", errors="replace")

    async def write_text(self, path: str, content: str, *, mode: str) -> None:
        args: dict[str, Any] = {
            "path": path,
            "content_base64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "mode": mode,
        }
        if self.privileged:
            # Writing into legion's home as root leaves root-owned files the
            # login cannot replace on the next issue — and ssh refuses a
            # config it does not own anyway.
            args["sudo"] = True
            args["owner"] = f"{LEGION_USER}:{LEGION_USER}"
        await self.call("vps_write_file", args)

    async def exec(self, argv: list[str], *, timeout_sec: int = 60) -> dict[str, Any]:
        _assert_allowed(argv)
        result = await self.call(
            "vps_exec", {"command": argv, "sudo": False, "timeout_sec": timeout_sec}
        )
        if not isinstance(result, dict):
            raise DeployKeyError("unexpected_result", f"{argv[0]} returned no result object")
        return result

    async def exec_as_legion(self, argv: list[str], *, timeout_sec: int = 60) -> dict[str, Any]:
        if self.privileged:
            argv = ["sudo", "-n", "-u", LEGION_USER, "-H", *argv]
        return await self.exec(argv, timeout_sec=timeout_sec)

    async def remove(self, path: str) -> None:
        await self.exec(["rm", "-f", path])

    @staticmethod
    async def for_prepared_server(
        db: AsyncSession,
        *,
        service: Any,
        room_id: UUID,
        actor_user_id: UUID,
        server: str,
        environment: Optional[str] = None,
        request_id: Optional[UUID] = None,
        source_ip: Optional[str] = None,
    ) -> "Remote":
        """Log in as the everyday account, the way the card's buttons do.

        Deliberately the *everyday* credential, not root: issuing or revoking
        a repository credential must keep working on a server whose root
        password the Room has already forgotten (§3.6 step 10), which is the
        state every properly finished preparation leaves behind.
        """
        from app.core.mcp_gateway import execute_vps_bootstrap_tool

        conn = await vps_inventory.resolve_connection(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            server=server, environment=environment, request_id=request_id, source_ip=source_ip,
        )

        async def _call(tool: str, arguments: dict[str, Any]) -> Any:
            return await execute_vps_bootstrap_tool(
                db,
                room_id=room_id,
                actor_user_id=actor_user_id,
                tool=tool,
                arguments={"server": vps_inventory.validate_server_name(server), **arguments},
                legion_conn=conn.legion_conn,
                request_id=request_id,
            )

        return Remote(call=_call, user=str(conn.legion_conn.get("user") or LEGION_USER))

    async def ensure_ssh_dir(self) -> None:
        await self.exec(["mkdir", "-p", SSH_DIR])
        await self.exec(["chmod", "700", SSH_DIR])
        if self.privileged:
            await self.exec(["chown", f"{LEGION_USER}:{LEGION_USER}", SSH_DIR])


# ── repository facts the Room already has ──────────────────────────────────


@dataclass(frozen=True)
class RepoTarget:
    provider: str
    owner: str
    repo: str
    forge: str
    branch: str
    https_url: str
    git_config: dict[str, Any]

    @property
    def ssh_path(self) -> str:
        return f"{self.owner}/{self.repo}.git"


async def repo_target(db: AsyncSession, room_id: UUID) -> tuple[Any, RepoTarget]:
    """``(room, RepoTarget)`` or :class:`DeployKeyError` ``repo_not_configured``.

    "The Room has no external repository" is a fact a person fixes in the
    Room's Git settings, so it is its own error code all the way out to the
    agent (Stage 14) — never a generic failure that reads like something to
    retry.
    """
    from app.models.room import Room

    room = await db.get(Room, room_id)
    if room is None:
        raise DeployKeyError("repo_not_configured", "the room was not found")
    git_config = dict(getattr(room, "git_config", None) or {})
    if not str(git_config.get("repo_url") or "").strip():
        raise DeployKeyError(
            "repo_not_configured",
            "the Room has no external repository connected — connect one on the Git tab",
        )
    try:
        provider, owner, repo = git_forge.parse_repo(git_config)
    except ValueError as exc:
        raise DeployKeyError("repo_not_configured", f"could not parse the repository URL: {exc}") from None
    forge = git_forge.forge_host(git_config)
    if not forge:
        raise DeployKeyError("repo_not_configured", "the repository URL has no host name")
    return room, RepoTarget(
        provider=provider,
        owner=owner,
        repo=repo,
        forge=forge,
        branch=str(git_config.get("branch") or "main"),
        https_url=git_forge.https_clone_url(git_config),
        git_config=git_config,
    )


# ── issue ──────────────────────────────────────────────────────────────────


async def _existing_token(
    db: AsyncSession, *, service: Any, room_id: UUID, actor_user_id: UUID,
    server: str, environment: Optional[str], request_id: Optional[UUID],
    source_ip: Optional[str],
) -> Optional[str]:
    from app.core.secrets.room_secrets import RoomSecretNotAvailable, resolve_room_secret

    try:
        return await resolve_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=f"VPS_{server}_{DEPLOY_TOKEN_SUFFIX}", purpose=SecretPurpose.GIT_DEPLOY,
            environment=environment, scope=None, request_id=request_id, source_ip=source_ip,
            expected_kind_override=SecretKind.ROOM_ENV,
        )
    except RoomSecretNotAvailable:
        return None


async def _verify_ls_remote(remote: Remote, url: str) -> tuple[bool, str]:
    """``git ls-remote`` as ``legion``, retried over a new key's propagation."""
    stderr = ""
    for attempt in range(_VERIFY_ATTEMPTS):
        result = await remote.exec_as_legion(
            ["git", "ls-remote", "--exit-code", url, "HEAD"],
            timeout_sec=_LS_REMOTE_TIMEOUT_SEC,
        )
        if result.get("exit_code") == 0:
            return True, ""
        stderr = str(result.get("stderr") or result.get("stdout") or "").strip()
        if attempt + 1 < _VERIFY_ATTEMPTS:
            await asyncio.sleep(_VERIFY_BACKOFF_SEC)
    return False, stderr[:300]


async def issue(
    db: AsyncSession,
    *,
    service: Any,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    remote: Remote,
    environment: Optional[str] = None,
    https_token: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
    require_prepared: bool = True,
) -> dict[str, Any]:
    """Give one server a read-only credential for the Room's repository.

    Idempotent per server: a second call replaces the credential rather than
    adding a second one — the old forge key is deleted by the id the
    inventory recorded, and the server's files are rewritten in place. The
    result is a report, not a raw success flag, because the honest outcomes
    are three, not two: issued and verified, issued but the forge would not
    take the key (``status="manual"`` — the person adds it and presses
    "Check"), and failed.
    """
    name = vps_inventory.validate_server_name(server)
    coords = await vps_inventory.server_coordinates(db, room_id, name, environment=environment)
    if require_prepared and not coords.prepared_at:
        # The button on a server card needs the `legion` account to exist
        # before it can put anything in its home. The wizard's own `repo`
        # step passes ``require_prepared=False``: it runs *inside* the
        # preparation, after the account and the key but before the run has
        # stamped ``PREPARED_AT``.
        raise DeployKeyError(
            "server_not_prepared",
            "the server is not prepared yet — first the preparation wizard, which creates the legion user",
        )
    room, target = await repo_target(db, room_id)
    slug = room_slug(getattr(room, "name", "") or "", room_id)

    token = https_token or await _existing_token(
        db, service=service, room_id=room_id, actor_user_id=actor_user_id, server=name,
        environment=environment, request_id=request_id, source_ip=source_ip,
    )
    if token:
        return await _issue_https(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id, name=name,
            remote=remote, target=target, slug=slug, token=token, is_new_token=bool(https_token),
            environment=environment, request_id=request_id, source_ip=source_ip,
        )
    return await _issue_ssh(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name,
        remote=remote, room=room, target=target, slug=slug,
        environment=environment, request_id=request_id, source_ip=source_ip,
    )


def identity_path(slug: str) -> str:
    return f"{SSH_DIR}/id_legion_{slug}"


def credentials_path(slug: str) -> str:
    return f"{LEGION_HOME}/.git-credentials-{slug}"


def ssh_alias(slug: str) -> str:
    return f"legion-{slug}"


async def _issue_ssh(
    db: AsyncSession, *, room_id: UUID, actor_user_id: UUID, name: str,
    remote: Remote, room: Any, target: RepoTarget, slug: str,
    environment: Optional[str], request_id: Optional[UUID], source_ip: Optional[str],
) -> dict[str, Any]:
    host_keys = await git_forge.forge_ssh_host_keys(target.git_config)
    if not host_keys:
        # Without published host keys the only ways forward are TOFU (§3.5
        # says no) and turning host checking off (worse). Say so and point at
        # the fallback instead of installing something that cannot work.
        raise DeployKeyError(
            "forge_host_keys_unavailable",
            f"{target.forge} does not publish its SSH host keys in a machine-readable form — "
            "grant access over HTTPS (a read-only repository token)",
        )

    private_key, public_key = generate_ed25519_keypair()
    kind, body = public_key.split()[0], public_key.split()[1]
    public_line = f"{kind} {body} legion-deploy@{slug}"

    alias = ssh_alias(slug)
    key_path = identity_path(slug)
    remote_url = f"git@{alias}:{target.ssh_path}"

    await remote.ensure_ssh_dir()
    await remote.write_text(key_path, private_key, mode="600")
    config = await remote.read_text(SSH_CONFIG_PATH) or ""
    await remote.write_text(
        SSH_CONFIG_PATH,
        replace_block(config, slug, ssh_config_block(alias=alias, forge=target.forge, identity_path=key_path)),
        mode="600",
    )
    known = await remote.read_text(KNOWN_HOSTS_PATH) or ""
    await remote.write_text(
        KNOWN_HOSTS_PATH,
        replace_block(known, slug, "\n".join(f"{target.forge} {k}" for k in host_keys)),
        mode="644",
    )

    # Replace, never accumulate: the previous key for this server is deleted
    # before the new one is offered, so re-issuing does not leave a trail of
    # live keys on the repository nobody can attribute to a machine.
    previous_id = await _recorded(db, room_id, name, "DEPLOY_KEY_ID", environment)
    forge_token = await git_forge.resolve_forge_token(
        db, room, request_id=request_id, source_ip=source_ip
    )
    if previous_id and forge_token:
        await git_forge.delete_deploy_key(target.git_config, forge_token, previous_id)

    registered = await git_forge.add_deploy_key(
        target.git_config,
        forge_token,
        title=f"Legion · {name} · {slug}",
        public_key=public_line,
    )
    if not registered["ok"]:
        await _record(
            db, room_id=room_id, actor_user_id=actor_user_id, name=name,
            environment=environment, credential=None, key_id=None, remote_url=remote_url,
        )
        return {
            "status": "manual",
            "credential": "ssh",
            "public_key": public_line,
            "remote_url": remote_url,
            "key_id": None,
            "repo": _repo_summary(target),
            "detail": (
                f"The key was created and is on the server, but registering it in {target.forge} failed "
                f"({registered['error']}). Add the public key as a deploy key with read-only "
                "access and press 'Check access'."
            ),
        }

    ok, stderr = await _verify_ls_remote(remote, remote_url)
    if not ok:
        raise DeployKeyError(
            "deploy_verify_failed",
            f"the key is registered, but `git ls-remote` from the server fails: {stderr}",
        )

    await _record(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name, environment=environment,
        credential="ssh", key_id=registered["id"], remote_url=remote_url,
    )
    return {
        "status": "ok",
        "credential": "ssh",
        "public_key": public_line,
        "remote_url": remote_url,
        "key_id": registered["id"],
        "repo": _repo_summary(target),
        "detail": f"read-only deploy key granted to server '{name}', `git ls-remote` passes",
    }


async def _issue_https(
    db: AsyncSession, *, service: Any, room_id: UUID, actor_user_id: UUID, name: str,
    remote: Remote, target: RepoTarget, slug: str, token: str, is_new_token: bool,
    environment: Optional[str], request_id: Optional[UUID], source_ip: Optional[str],
) -> dict[str, Any]:
    from app.core.secrets.room_secrets import (
        RoomSecretNotAvailable,
        bind_room_secret,
        unbind_room_secret,
    )

    env_name = f"VPS_{name}_{DEPLOY_TOKEN_SUFFIX}"
    if is_new_token:
        try:
            await unbind_room_secret(
                db, service=service, room_id=room_id, actor_user_id=actor_user_id,
                env_name=env_name, environment=environment, purpose=SecretPurpose.GIT_DEPLOY,
                request_id=request_id, source_ip=source_ip,
            )
        except RoomSecretNotAvailable:
            pass
        await bind_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=env_name, value=token, environment=environment,
            purpose=SecretPurpose.GIT_DEPLOY, kind_override=SecretKind.ROOM_ENV,
            request_id=request_id, source_ip=source_ip,
        )
        await db.commit()

    username = "oauth2" if target.provider == "gitlab" else "x-access-token"
    creds_path = credentials_path(slug)
    await remote.write_text(creds_path, f"https://{username}:{token}@{target.forge}\n", mode="600")
    gitconfig = await remote.read_text(GITCONFIG_PATH) or ""
    await remote.write_text(
        GITCONFIG_PATH,
        replace_block(gitconfig, slug, gitconfig_block(forge=target.forge, credentials_path=creds_path)),
        mode="600",
    )

    ok, stderr = await _verify_ls_remote(remote, target.https_url)
    if not ok:
        raise DeployKeyError(
            "deploy_verify_failed",
            f"the token is stored, but `git ls-remote` from the server fails: {stderr}",
        )
    await _record(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name, environment=environment,
        credential="https", key_id=None, remote_url=target.https_url,
    )
    return {
        "status": "ok",
        "credential": "https",
        "public_key": None,
        "remote_url": target.https_url,
        "key_id": None,
        "repo": _repo_summary(target),
        "detail": (
            f"HTTPS token granted to server '{name}', `git ls-remote` passes. "
            f"Only this server holds a copy — the Room's token never left for the server."
        ),
    }


async def verify(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    remote: Remote,
    environment: Optional[str] = None,
) -> dict[str, Any]:
    """Re-check an already-installed credential and stamp the inventory.

    The one path out of ``status="manual"``: the person added the public key
    in the forge's own UI, so there is nothing to generate — only to confirm
    that the key they added actually works, and to record that it does.
    """
    name = vps_inventory.validate_server_name(server)
    _, target = await repo_target(db, room_id)
    remote_url = await _recorded(db, room_id, name, "DEPLOY_REMOTE", environment)
    if not remote_url:
        raise DeployKeyError(
            "repo_credential_missing",
            "this server has not been granted repository access yet",
        )
    ok, stderr = await _verify_ls_remote(remote, remote_url)
    credential = "https" if remote_url.startswith("https://") else "ssh"
    if not ok:
        return {
            "status": "manual",
            "credential": credential,
            "remote_url": remote_url,
            "repo": _repo_summary(target),
            "detail": f"`git ls-remote` from the server still fails: {stderr}",
        }
    await _record(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name, environment=environment,
        credential=credential,
        key_id=await _recorded(db, room_id, name, "DEPLOY_KEY_ID", environment),
        remote_url=remote_url,
    )
    return {
        "status": "ok",
        "credential": credential,
        "remote_url": remote_url,
        "repo": _repo_summary(target),
        "detail": "`git ls-remote` from the server passes",
    }


# ── revoke ─────────────────────────────────────────────────────────────────


async def revoke(
    db: AsyncSession,
    *,
    service: Any,
    room_id: UUID,
    actor_user_id: UUID,
    server: str,
    remote: Optional[Remote] = None,
    environment: Optional[str] = None,
    request_id: Optional[UUID] = None,
    source_ip: Optional[str] = None,
) -> dict[str, Any]:
    """Take the credential away: the forge key, the files, the inventory rows.

    Best-effort per part and in that order on purpose. Deleting the key at the
    forge is the part that actually revokes access — it works even if the
    server is unreachable, powered off or already someone else's — so it is
    never skipped because a later step might fail. ``remote=None`` (the
    server-deleted path) does the forge and inventory halves only, and says so
    in the report rather than implying the files are gone.
    """
    name = vps_inventory.validate_server_name(server)
    notes: list[str] = []

    key_id = await _recorded(db, room_id, name, "DEPLOY_KEY_ID", environment)
    remote_url = await _recorded(db, room_id, name, "DEPLOY_REMOTE", environment)
    forge_removed = False
    if key_id:
        try:
            room, target = await repo_target(db, room_id)
            forge_token = await git_forge.resolve_forge_token(
                db, room, request_id=request_id, source_ip=source_ip
            )
            result = await git_forge.delete_deploy_key(target.git_config, forge_token, key_id)
            forge_removed = bool(result["ok"])
            if not forge_removed:
                notes.append(
                    f"the key {key_id} could not be deleted in the forge ({result['error']}) — "
                    "delete it manually in the repository settings"
                )
        except DeployKeyError as exc:
            notes.append(f"the Room's repository is no longer configured ({exc.code}) — the key remains in the forge")

    if remote is not None:
        from app.models.room import Room

        room = await db.get(Room, room_id)
        slug = room_slug(getattr(room, "name", "") or "", room_id)
        try:
            await remote.remove(identity_path(slug))
            await remote.remove(credentials_path(slug))
            for path, comment in ((SSH_CONFIG_PATH, "#"), (KNOWN_HOSTS_PATH, "#"), (GITCONFIG_PATH, "#")):
                current = await remote.read_text(path)
                if current is None:
                    continue
                await remote.write_text(path, replace_block(current, slug, None, comment), mode="600")
        except (VpsToolError, DeployKeyError) as exc:
            notes.append(f"the files on the server could not be removed ({exc}) — the key in the forge is already revoked")
    else:
        notes.append("the server was not queried — non-working key files remain on it")

    for suffix in ("DEPLOY_CREDENTIAL", "DEPLOY_KEY_ID", "DEPLOY_REMOTE"):
        await vps_inventory.clear_server_variable(
            db, room_id=room_id, name=name, suffix=suffix, environment=environment
        )
    from app.core.secrets.room_secrets import RoomSecretNotAvailable, unbind_room_secret

    try:
        await unbind_room_secret(
            db, service=service, room_id=room_id, actor_user_id=actor_user_id,
            env_name=f"VPS_{name}_{DEPLOY_TOKEN_SUFFIX}", environment=environment,
            purpose=SecretPurpose.GIT_DEPLOY, request_id=request_id, source_ip=source_ip,
        )
    except RoomSecretNotAvailable:
        pass
    await db.commit()
    return {
        "status": "revoked",
        "forge_key_removed": forge_removed,
        "remote_url": remote_url,
        "notes": notes,
    }


# ── inventory plumbing ─────────────────────────────────────────────────────


def _repo_summary(target: RepoTarget) -> dict[str, Any]:
    return {
        "provider": target.provider,
        "url": target.https_url,
        "branch": target.branch,
        "forge": target.forge,
    }


async def room_repo_summary(db: AsyncSession, room_id: UUID) -> Optional[dict[str, Any]]:
    """The Room's repository as facts an agent may see, or ``None``.

    Provider/URL/branch/forge only — the same four fields the issue and revoke
    endpoints already return to the UI, and by construction not a token and
    not a key. ``None`` means "this Room has no external repository", which
    Stage 14 reports to an agent as ``repo_not_configured``: something a person
    fixes on the Room's Git tab, never something to retry.
    """
    try:
        _room, target = await repo_target(db, room_id)
    except DeployKeyError:
        return None
    return _repo_summary(target)


async def _recorded(
    db: AsyncSession, room_id: UUID, name: str, suffix: str, environment: Optional[str]
) -> Optional[str]:
    for row in await vps_inventory.list_servers(db, room_id, environment=environment):
        if row["name"] == name:
            return row.get({"DEPLOY_KEY_ID": "deploy_key_id", "DEPLOY_REMOTE": "deploy_remote"}[suffix])
    return None


async def _record(
    db: AsyncSession,
    *,
    room_id: UUID,
    actor_user_id: UUID,
    name: str,
    environment: Optional[str],
    credential: Optional[str],
    key_id: Optional[str],
    remote_url: str,
) -> None:
    """Write what was issued. ``credential=None`` means "installed, not proven".

    The distinction matters to everything downstream: Stage 14 tells an agent
    ``repo_credential_missing`` off this exact value, and a server whose key
    the forge never accepted must not read as one that can clone.
    """
    await vps_inventory.set_server_variable(
        db, room_id=room_id, actor_user_id=actor_user_id, name=name,
        suffix="DEPLOY_REMOTE", value=remote_url, environment=environment,
    )
    if credential:
        await vps_inventory.set_server_variable(
            db, room_id=room_id, actor_user_id=actor_user_id, name=name,
            suffix="DEPLOY_CREDENTIAL", value=credential, environment=environment,
        )
    else:
        await vps_inventory.clear_server_variable(
            db, room_id=room_id, name=name, suffix="DEPLOY_CREDENTIAL", environment=environment,
        )
    if key_id:
        await vps_inventory.set_server_variable(
            db, room_id=room_id, actor_user_id=actor_user_id, name=name,
            suffix="DEPLOY_KEY_ID", value=key_id, environment=environment,
        )
    else:
        await vps_inventory.clear_server_variable(
            db, room_id=room_id, name=name, suffix="DEPLOY_KEY_ID", environment=environment,
        )
    await db.commit()


__all__ = [
    "DeployKeyError",
    "Remote",
    "RepoTarget",
    "issue",
    "verify",
    "revoke",
    "repo_target",
    "replace_block",
    "ssh_config_block",
    "gitconfig_block",
    "identity_path",
    "credentials_path",
    "ssh_alias",
    "SSH_CONFIG_PATH",
    "KNOWN_HOSTS_PATH",
    "GITCONFIG_PATH",
]
