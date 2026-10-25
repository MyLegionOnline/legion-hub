"""Bash tool — runs shell commands in the workspace container.

cwd: sets the working directory for THIS call only; it is NOT preserved between calls.
Use `cd dir && cmd` to chain directory changes within a single call.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shlex
import time
import uuid
from typing import Any

from app.core import run_clock
from app.core.context_observations import fingerprint, record_observation
from app.core.secrets.room_masking import (
    mask_room_secrets,
    remember_bg_secrets,
    track_room_secrets,
)
from app.core.tools._workspace_wake import ensure_workspace_running
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate, release_tool_db

logger = logging.getLogger(__name__)

# internal design notes Stage 5 — must match workspaced.docker_manager.SECRETS_DIR.
_SECRETS_DIR = "/run/legion/secrets"

# Patterns from BASH_DENY_PATTERNS env var (comma-separated) plus fixed defaults.
# These are a soft guard; real isolation is the container resource limits.
_DENY_DEFAULTS = [
    r"rm\s+-rf\s+/(?:\s|$)",   # rm -rf / or rm -rf / (with space)
    r"mkfs\b",                  # disk formatting
    r":\(\)\s*\{.*:\|:&\s*\}", # fork bomb :(){:|:&};:
    r"dd\s+.*of=/dev/[sh]d",   # raw disk write
]

def _load_deny_patterns() -> list[re.Pattern[str]]:
    raw = os.getenv("BASH_DENY_PATTERNS", "")
    extra = [p.strip() for p in raw.split(",") if p.strip()] if raw else []
    return [re.compile(p, re.IGNORECASE) for p in _DENY_DEFAULTS + extra]

_DENY_PATTERNS: list[re.Pattern[str]] = _load_deny_patterns()


def _check_deny(cmd: str) -> str | None:
    for pat in _DENY_PATTERNS:
        if pat.search(cmd):
            return f"Error: command matches deny pattern '{pat.pattern}' and was blocked."
    return None

_NO_NETWORK_HINT = (
    "\n[note: network is DISABLED in this workspace — commands that need internet "
    "access (npm install, pip install, git clone, curl, …) will hang or fail. "
    "Do not retry them; ask the user to restart the workspace with network enabled.]"
)

_BG_MAX = 5  # max concurrent background processes per workspace

# A synchronous call is capped at 120 s, but a package install over a slow
# mirror routinely outruns that and forced the agent into background + polling
# (a tool-call step per poll). Recognised install commands get a longer window.
_DEFAULT_TIMEOUT_SEC = 30
_MAX_TIMEOUT_SEC = 120
_INSTALL_MAX_TIMEOUT_SEC = int(os.getenv("BASH_INSTALL_TIMEOUT_SECONDS", "900"))
_INSTALL_RE = re.compile(
    r"(?:^|[;&|(]\s*|\bsudo\s+)"
    r"(?:"
    r"(?:python3?\s+-m\s+)?pip3?\s+install|uv\s+(?:pip\s+install|sync|add)|"
    r"poetry\s+(?:install|add)|pipx\s+install|"
    r"apt(?:-get)?\s+(?:-\S+\s+)*(?:install|update|upgrade)|"
    r"(?:npm|pnpm|bun)\s+(?:install|i|ci|add)|yarn(?:\s+(?:install|add))?|"
    r"npx\s+playwright\s+install|playwright\s+install|"
    r"cargo\s+(?:install|build|fetch)|go\s+(?:mod\s+download|install)"
    r")\b"
)


def _is_install_command(cmd: str) -> bool:
    return bool(_INSTALL_RE.search(cmd))


def _effective_timeout(cmd: str, requested: Any) -> int:
    """Clamp ``timeout_sec``: 1–120 s, or up to the install cap for installs.

    An install command with no explicit timeout gets the install cap outright —
    the agent should not have to guess how slow the mirror is.
    """
    install = _is_install_command(cmd)
    ceiling = _INSTALL_MAX_TIMEOUT_SEC if install else _MAX_TIMEOUT_SEC
    if not requested:
        return ceiling if install else _DEFAULT_TIMEOUT_SEC
    return max(1, min(int(requested), ceiling))

# One lock per room: only one bash exec at a time per container (MVP constraint).
_ws_locks: dict[str, asyncio.Lock] = {}


def _ws_lock(room_id: str) -> asyncio.Lock:
    if room_id not in _ws_locks:
        _ws_locks[room_id] = asyncio.Lock()
    return _ws_locks[room_id]


def _classify_single_file_cat(cmd: str) -> str | None:
    """Return the raw path argument if cmd is exactly 'cat [-n] <path>', else None.

    Deliberately narrow: any pipe, redirect, second command, wildcard, or
    extra flag falls through to a plain shlex token mismatch and this
    returns None, so ambiguous commands always run for real. Piping through
    ``shlex.split`` also means a genuine ``|``/``;``/``&&`` in the command
    shows up as its own token and fails the length/shape check below.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    if not tokens or tokens[0] != "cat":
        return None
    rest = tokens[1:]
    if rest and rest[0] == "-n":
        rest = rest[1:]
    if len(rest) != 1:
        return None
    path = rest[0]
    if not path or path.startswith("-") or "*" in path or "?" in path:
        return None
    return path


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command in the workspace container. Returns exit_code, stdout, stderr.\n"
        "The workspace root is the current directory; cwd accepts a relative path "
        "(legacy /workspace/... is normalized).\n"
        "• cwd: working directory for this call only — NOT preserved between calls; "
        "use 'cd dir && cmd' to chain within one call.\n"
        "• background=true: start a temporary non-user-facing watcher. User-visible web/API/worker apps "
        "must use workspace_service instead when it is enabled; if workspace_service reports that "
        "services are disabled on this deployment, background=true is the way to run them. "
        "Returns a bg_id; use bash_wait(bg_id) to wait for it to finish (installs, builds — a sync call is "
        "capped at 120 s), bash_output(bg_id) to read its log and bash_kill(bg_id) to stop it.\n"
        "• Large output (>16 KB) is saved to /workspace/.legion/logs/ and you get the tail + path.\n"
        "• Headless-browser renders (Remotion, Playwright, Puppeteer) take 100+ processes/threads "
        "each and the container has a fixed process limit: run them with background=true and "
        "one browser at a time (e.g. `remotion render/still --concurrency=1`), never several in parallel.\n"
        f"• Max {_BG_MAX} concurrent background processes per workspace."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Shell command to run (executed as: bash -c <cmd>)",
            },
            "timeout_sec": {
                "type": "integer",
                "default": 30,
                "description": (
                    "Execution timeout in seconds (1–120); ignored when background=true. "
                    "Package-install commands (pip/apt/npm/…) get a longer window "
                    f"(up to {_INSTALL_MAX_TIMEOUT_SEC} s) automatically."
                ),
            },
            "cwd": {
                "type": "string",
                "default": "/workspace",
                "description": "Workspace-relative working directory (default root). NOT preserved between calls.",
            },
            "background": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Run as a background process. Returns bg_id immediately; "
                    "use bash_output(bg_id) to read output and bash_kill(bg_id) to stop."
                ),
            },
            "secrets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Names of Room secrets (from the Room's Secrets & variables tab) to "
                    "resolve and inject into this command's env. You only ever see the "
                    "name, never the value — reference it as $NAME in cmd. Room variables "
                    "are always in env already; this is only for secrets. A name bound as "
                    "a deploy SSH key is not put in env — it's written to a tmpfs file for "
                    "this command only and GIT_SSH_COMMAND is set automatically, so a plain "
                    "'git push'/'git clone' over ssh just works. Not usable with background=true."
                ),
            },
            "wake_on_exit": {
                "type": "boolean",
                "default": False,
                "description": (
                    "background=true only, direct chat only. For a process that may run longer "
                    "than a few minutes (a long build, a render, a download): start it with "
                    "wake_on_exit=true, tell the user briefly what you are waiting for, and END "
                    "YOUR TURN. You are resumed automatically — once — when the process exits, "
                    "with its exit code and log tail. Do not poll while waiting."
                ),
            },
            "wake_note": {
                "type": "string",
                "description": (
                    "With wake_on_exit: a short note to your future self — what to do once the "
                    "process finishes (it is shown to you when you are resumed)."
                ),
            },
            "environment": {
                "type": "string",
                "description": (
                    "Room environment to resolve variables/secrets for (e.g. 'staging', "
                    "'production'). Omit for the Room's shared (non-environment-scoped) values."
                ),
            },
        },
        "required": ["cmd"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = True
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd

        cmd = str(tool_input.get("cmd") or "").strip()
        try:
            timeout_sec = _effective_timeout(cmd, tool_input.get("timeout_sec"))
        except (TypeError, ValueError):
            return "Error: timeout_sec must be an integer"
        raw_cwd = str(tool_input.get("cwd") or "").strip()
        try:
            from app.core.path_contract import workspace_absolute_path
            cwd = workspace_absolute_path(raw_cwd)
        except ValueError as exc:
            return f"Error: invalid cwd: {exc}"
        background = bool(tool_input.get("background", False))
        rid = str(ctx.room_id)
        sid = str(ctx.session_id)

        if not cmd:
            return "Error: cmd is required"

        secrets_requested = tool_input.get("secrets")
        if secrets_requested is not None:
            if not isinstance(secrets_requested, list) or not all(
                isinstance(s, str) for s in secrets_requested
            ):
                return "Error: secrets must be a list of strings"
            if len(secrets_requested) > 20:
                return "Error: at most 20 secrets per call"
        environment = tool_input.get("environment")
        if environment is not None and not isinstance(environment, str):
            return "Error: environment must be a string"

        denied = _check_deny(cmd)
        if denied:
            return denied

        # A workspace stopped under the task (idle stop, quota switch) is
        # started here — the agent has no tool of its own for that.
        ws, err = await ensure_workspace_running(ctx, reason="bash")
        if err:
            return err

        exec_env, resolved_secrets, secret_files, err = await self._build_exec_env(
            ctx, secrets_requested, environment
        )
        if err:
            return err
        if background and secret_files:
            # A deploy-key file's lifetime is tied to the one exec that wrote
            # it — the launcher exec below returns as soon as the background
            # process is spawned, and the key would be removed before the
            # detached command actually uses it (plan §3.3).
            return "Error: deploy-key secrets cannot be used with background=true"

        if background:
            return await self._run_background(
                ctx, ws, cmd, cwd, wsd, exec_env, resolved_secrets, secret_files,
                wake_on_exit=bool(tool_input.get("wake_on_exit", False)),
                wake_note=str(tool_input.get("wake_note") or "").strip()[:2000] or None,
            )

        dedup = await self._maybe_dedup_cat(ctx, cmd, cwd)
        if dedup is not None:
            return dedup
        return await self._run_sync(
            ctx, ws, cmd, cwd, timeout_sec, sid, wsd, exec_env, resolved_secrets, secret_files
        )

    async def _build_exec_env(
        self,
        ctx: ToolContext,
        secrets_requested: list[str] | None,
        environment: str | None,
    ) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]], str | None]:
        """Resolve Room variables (always) plus explicitly requested Room secrets.

        internal design notes Stage 4 (variables/plain secrets) +
        Stage 5 (deploy keys). Variables are non-secret and go into every
        exec's env unconditionally; secrets only go in when the agent names
        them by env_name. A name bound as a deploy SSH key never touches env —
        it becomes a ``secret_files`` entry on tmpfs plus a ``GIT_SSH_COMMAND``
        pointing at it (plan §3.3); everything else lands in env as before.

        Returns ``(env, secrets_for_masking, secret_files, error)`` — on error
        the first three are empty and ``error`` is the exact string to hand
        back to the model (plan §3.1: a deterministic ``secret_not_available``,
        never distinguishing "missing" from "not in scope").
        """
        from fastapi import HTTPException
        from sqlalchemy.future import select

        from app.core.room_config import effective_variables
        from app.core.secrets import room_secrets
        from app.core.secrets.service import SecretService
        from app.models.room import Room

        try:
            room_vars = await effective_variables(ctx.db, ctx.room_id, environment)
        except HTTPException as exc:
            return {}, {}, [], f"Error: {exc.detail}"

        if not secrets_requested:
            return room_vars, {}, [], None

        room = (
            await ctx.db.execute(select(Room).where(Room.id == ctx.room_id))
        ).scalar_one_or_none()
        if room is None:
            return {}, {}, [], "Error: room not found"

        service = SecretService(ctx.db)
        try:
            env_secrets, ssh_keys = await room_secrets.resolve_room_secrets_mixed(
                ctx.db,
                service=service,
                room_id=ctx.room_id,
                actor_user_id=room.creator_id,
                names=secrets_requested,
                environment=environment,
            )
        except room_secrets.EnvironmentLocked as exc:
            approved = await self._confirm_environment_unlock(ctx, str(exc))
            if not approved:
                return {}, {}, [], f"Error: [{exc.code}] {exc}"
            try:
                env_secrets, ssh_keys = await room_secrets.resolve_room_secrets_mixed(
                    ctx.db,
                    service=service,
                    room_id=ctx.room_id,
                    actor_user_id=room.creator_id,
                    names=secrets_requested,
                    environment=environment,
                )
            except room_secrets.RoomSecretNotAvailable as exc2:
                return {}, {}, [], f"Error: [{exc2.code}] {exc2}"
            except room_secrets.RoomSecretError as exc2:
                return {}, {}, [], f"Error: {exc2}"
        except room_secrets.RoomSecretNotAvailable as exc:
            return {}, {}, [], f"Error: [{exc.code}] {exc}"
        except room_secrets.RoomSecretError as exc:
            return {}, {}, [], f"Error: {exc}"

        env = {**room_vars, **env_secrets}
        secret_files: list[dict[str, str]] = []
        for name, value in ssh_keys.items():
            secret_files.append(
                {"path": f"{_SECRETS_DIR}/{name}", "content": value, "mode": "0600"}
            )
        if len(ssh_keys) == 1:
            (only_path,) = (f["path"] for f in secret_files)
            env["GIT_SSH_COMMAND"] = (
                f'ssh -i {only_path} -o IdentitiesOnly=yes '
                "-o StrictHostKeyChecking=accept-new"
            )
        return env, {**env_secrets, **ssh_keys}, secret_files, None

    async def _confirm_environment_unlock(self, ctx: ToolContext, environment: str) -> bool:
        """Raise an ask-user confirmation card instead of failing outright.

        internal design notes §2.4 Stage 12: a protected environment
        without an active window does not just error out of a tool call — the
        agent's turn pauses on the same ask_user card mechanism
        (``ask_user_registry`` + an ``AGENT_QUESTION`` event) used for
        clarifying questions, and a human answers "Allow"/"Deny" from chat.
        "Allow" opens a Stage-12 confirmation window (default
        ``DEFAULT_UNLOCK_MINUTES``) for this ``(room, environment)`` pair and
        the caller retries the resolve once; anything else (deny, timeout, a
        failure to even post the card) keeps it locked.
        """
        from uuid import uuid4

        from app.core import ask_user_registry as _reg
        from app.core.events import emit_event
        from app.core.secrets import room_environment_gate

        minutes = room_environment_gate.DEFAULT_UNLOCK_MINUTES
        question_text = (
            f"The agent requested access to the secrets of the protected environment "
            f"'{environment}'. Allow for {minutes} minutes?"
        )
        questions = [
            {
                "question": question_text,
                "header": "Environment",
                "options": [
                    {
                        "label": "Allow",
                        "description": f"Open a window of {minutes} minutes for '{environment}'.",
                    },
                    {
                        "label": "Deny",
                        "description": "Keep the environment locked.",
                    },
                ],
            }
        ]
        question_id = str(uuid4())
        timeout_seconds = int(os.getenv("ENV_UNLOCK_CONFIRM_TIMEOUT_SECONDS", "300"))
        try:
            await emit_event(
                room_id=ctx.room_id,
                event_type="AGENT_QUESTION",
                content=question_text,
                db=ctx.db,
                task_id=ctx.task_id,
                metadata={
                    "question_id": question_id,
                    "session_id": str(ctx.session_id),
                    "questions": questions,
                    "timeout_seconds": timeout_seconds,
                },
            )
        except Exception:
            logger.exception(
                "bash: failed to emit environment-unlock question (room=%s env=%s)",
                ctx.room_id, environment,
            )
            return False

        pq = _reg.register(question_id, ctx.session_id, ctx.room_id, {"questions": questions})
        try:
            await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
        except asyncio.TimeoutError:
            _reg.expire(question_id)
            return False
        _reg.expire(question_id)

        selected = (pq.answer or {}).get("selected_options") or []
        if "Allow" not in selected:
            return False

        from sqlalchemy.future import select as _select

        from app.models.room import Room

        room = (
            await ctx.db.execute(_select(Room).where(Room.id == ctx.room_id))
        ).scalar_one_or_none()
        actor_user_id = room.creator_id if room else None
        await room_environment_gate.grant_unlock(
            ctx.db,
            room_id=ctx.room_id,
            environment=environment,
            actor_user_id=actor_user_id,
        )
        return True

    async def _maybe_dedup_cat(self, ctx: ToolContext, cmd: str, cwd: str) -> str | None:
        """Short-circuit a plain 'cat [-n] <file>' that already-read content covers.

        Cross-tool with read_file: both write into the same session read
        ledger (context_state.read_ledger), keyed by workspace-relative path
        + content fingerprint, independent of which tool did the reading.
        bash otherwise has none of read_file's/grep's duplicate protection,
        so an agent that switches tools mid-task bypasses it entirely — this
        closes that gap for the dominant case seen in practice (cat/cat -n
        re-dumping a whole file already seen via read_file or a prior cat).
        """
        raw_path = _classify_single_file_cat(cmd)
        if raw_path is None:
            return None

        from app.core import workspaced_client as wsd
        from app.core.context_observations import (
            load_context_state,
            record_observation,
            record_read_interval,
            subtract_interval,
        )
        from app.core.path_contract import WorkspacePathError, normalize_workspace_path

        combined = raw_path if raw_path.startswith("/") else f"{cwd.rstrip('/')}/{raw_path}"
        try:
            path = normalize_workspace_path(combined)
        except WorkspacePathError:
            return None

        content = await wsd.read_workspace_file(str(ctx.room_id), f"/workspace/{path}")
        if content is None or (content.startswith("data:") and ";base64," in content):
            return None  # missing/binary — let the real `cat` produce its own output/error

        total = len(content.splitlines())
        if total == 0:
            return None

        source_fp = fingerprint(content)
        state = await load_context_state(ctx.db, ctx.session_id)
        ledger = state.get("read_ledger") if isinstance(state, dict) else None
        entry = (ledger or {}).get(path) if isinstance(ledger, dict) else None
        if not isinstance(entry, dict) or entry.get("fingerprint") != source_fp:
            return None  # never read before, or the file changed since — run for real
        if subtract_interval(1, total, entry.get("ranges") or []):
            return None  # only partially covered — run for real

        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name="read_file",
                canonical_args={"path": path, "offset": 1, "limit": total},
                payload=content,
                source_fingerprint=source_fp,
                metadata={"kind": "file", "path": path, "total_lines": total},
            )
            await record_read_interval(
                ctx.db,
                session_id=ctx.session_id,
                path=path,
                source_fingerprint=source_fp,
                requested_start=1,
                requested_end=total,
                observation_ref=observation.ref,
            )
            receipt = observation.receipt()
        except Exception:
            receipt = f"[observation unavailable; path={path!r}]"

        from app.core.tools._session_state import duplicate_escalation_notice, note_duplicate_read

        # Same key format as read_file's duplicate counter — cat and read_file
        # on the same file share one escalation count, since both are "you
        # already saw this content" regardless of which tool asked.
        count = note_duplicate_read(str(ctx.session_id), f"read:{path}:{source_fp}")
        notice = duplicate_escalation_notice(repr(path), count)
        return (
            notice
            + f"[Skipped duplicate `cat`: {path!r} was already read in full this session "
            "via read_file or an earlier cat, content unchanged.] " + receipt
        )

    async def _run_background(
        self, ctx, ws, cmd, cwd, wsd, exec_env, resolved_secrets, secret_files,
        *, wake_on_exit: bool = False, wake_note: str | None = None,
    ) -> str:
        from sqlalchemy.future import select

        from app.models.workspace import WorkspaceBgProc

        rid = str(ctx.room_id)
        sid = str(ctx.session_id)

        # Enforce concurrency limit
        running_res = await ctx.db.execute(
            select(WorkspaceBgProc).where(
                WorkspaceBgProc.workspace_id == ws.id,
                WorkspaceBgProc.status == "running",
            )
        )
        running = running_res.scalars().all()
        if len(running) >= _BG_MAX:
            ids = ", ".join(p.bg_id for p in running)
            return (
                f"Error: maximum {_BG_MAX} background processes already running "
                f"for this workspace. Kill one first: {ids}"
            )

        bg_id = uuid.uuid4().hex[:12]
        log_rel = f".legion/bg/{bg_id}.log"  # relative to workspace root
        log_abs = f"/workspace/.legion/bg/{bg_id}.log"  # inside container

        # Launch: env vars carry cmd/log to avoid quoting issues.
        # setsid gives the child its own session so killing by PGID works.
        launch_script = (
            'mkdir -p "$(dirname "$BGLOG")" && '
            # The wrapper records the exit code next to the log: bash_wait reads it.
            'setsid bash -c \'bash -c "$BGCMD"; echo $? >"$BGLOG.exit"\' >"$BGLOG" 2>&1 </dev/null & '
            'BG_PID=$!; '
            'sleep 0.1; '
            'BG_PGID=$(ps -o pgid= -p $BG_PID 2>/dev/null | tr -d " "); '
            'echo "$BG_PID ${BG_PGID:-$BG_PID}"'
        )
        env = {**exec_env, "BGCMD": cmd, "BGLOG": log_abs}

        await release_tool_db(ctx, reason="bash/background")
        async with _ws_lock(rid):
            try:
                result = await asyncio.wait_for(
                    wsd.exec_in_workspace(
                        rid, ["bash", "-c", launch_script], cwd=cwd, timeout_sec=10,
                        env=env, secret_files=secret_files,
                    ),
                    timeout=20,
                )
            except asyncio.TimeoutError:
                return "Error: timed out launching background process"

        pid: int | None = None
        pgid: int | None = None
        out = (result.get("stdout") or "").strip()
        parts = out.split()
        try:
            pid = int(parts[0])
            pgid = int(parts[1]) if len(parts) > 1 else pid
        except (ValueError, IndexError):
            pass

        # Only a direct chat can be resumed by a wake: a dispatched task's agent
        # has no "end the turn and wait" — its turn ending is a dialogue with the
        # ChatMaster. There ``bash_wait`` is the tool.
        wake_ok = wake_on_exit and ctx.task_id is None
        proc = WorkspaceBgProc(
            workspace_id=ws.id,
            session_id=sid,
            bg_id=bg_id,
            pid=pid,
            pgid=pgid,
            log_path=log_rel,
            cmd=cmd,
            status="running",
            wake_on_exit=wake_ok,
            wake_note=wake_note if wake_ok else None,
        )
        ctx.db.add(proc)
        await ctx.db.flush()
        remember_bg_secrets(bg_id, resolved_secrets)

        lines = [
            f"bg_id: {bg_id}",
            f"status: running",
            f"pid: {pid}",
            f"log: /workspace/.legion/bg/{bg_id}.log",
            f"Use bash_output(bg_id={bg_id!r}) to read output.",
            f"Use bash_kill(bg_id={bg_id!r}) to stop.",
        ]
        if wake_ok:
            lines.append(
                "wake_on_exit: armed — end your turn now with a short message to the user; "
                "you will be resumed automatically when the process exits."
            )
        elif wake_on_exit:
            lines.append(
                "wake_on_exit ignored: only available in direct chat. "
                "Use bash_wait(bg_id) to wait for this process."
            )
        from app.core.workspace_services import services_enabled

        if services_enabled():
            lines.append(
                "If this is a user-visible app, declare it with workspace_service instead of keeping this background process."
            )
        return "\n".join(lines)

    async def _run_sync(self, ctx, ws, cmd, cwd, timeout_sec, sid, wsd, exec_env, resolved_secrets, secret_files) -> str:
        from datetime import datetime, timezone

        from app.models.workspace import WorkspaceExec

        rid = str(ctx.room_id)

        gen_before = await self._safe_generation(wsd, rid)

        # Up to 120 s in the container (installs: longer); the WorkspaceExec row
        # is written after.
        await release_tool_db(ctx, reason="bash/exec")
        install = _is_install_command(cmd)
        # An install can hold the exec for minutes. The room lock would then
        # freeze every other agent's bash in this workspace for that long, so
        # installs run outside it (background processes already do). Time spent
        # waiting on a mirror is not agent work: it is credited to the run's
        # deadline *while* it elapses.
        async with (
            contextlib.nullcontext() if install else _ws_lock(rid),
            run_clock.accruing(ctx.task_id) if install else contextlib.nullcontext(),
        ):
            t0 = time.monotonic()
            try:
                with track_room_secrets(resolved_secrets):
                    result = await asyncio.wait_for(
                        wsd.exec_in_workspace(
                            rid, ["bash", "-c", cmd], cwd=cwd, timeout_sec=timeout_sec,
                            env=exec_env or None, secret_files=secret_files,
                        ),
                        timeout=timeout_sec + 10,
                    )
            except asyncio.TimeoutError:
                msg = f"Error: execution timed out after {timeout_sec}s"
                if install:
                    msg += (
                        "\n[install still running in the container is not killed; run it with "
                        "background=true and use bash_wait(bg_id) for slow installs]"
                    )
                if not ws.network_enabled:
                    msg += _NO_NETWORK_HINT
                return msg

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        exit_code = result.get("exit_code", -1)
        stdout = mask_room_secrets(result.get("stdout", ""), resolved_secrets)
        stderr = mask_room_secrets(result.get("stderr", ""), resolved_secrets)

        exec_record = WorkspaceExec(
            workspace_id=ws.id,
            initiator=f"session:{sid}",
            command=cmd,
            tool_name="bash",
            exit_code=exit_code,
            stdout_tail=stdout[-8192:],
            stderr_tail=stderr[-8192:],
            duration_ms=elapsed_ms,
        )
        ctx.db.add(exec_record)
        ws.last_activity_at = datetime.now(timezone.utc)
        ctx.db.add(ws)
        await ctx.db.flush()

        # Stage 6: no repofile metadata sync needed — the volume is the
        # single source of truth and bash wrote to it directly.
        #
        # But bash, unlike write_file, emits no FILE_CHANGED of its own, so the
        # Repository panel and any live git-flow view never learn a command
        # touched the tree. Nudge them when the volume generation advanced.
        await self._notify_if_tree_changed(ctx, wsd, rid, gen_before)

        from app.core.room_storage import kernel_enospc_hint

        raw = "\n".join(filter(None, [
            f"exit_code: {exit_code}",
            f"stdout:\n{stdout}" if stdout else "",
            f"stderr:\n{stderr}" if stderr else "",
            _NO_NETWORK_HINT.lstrip("\n") if exit_code != 0 and not ws.network_enabled else "",
            kernel_enospc_hint(ctx.room_id, exit_code, stdout, stderr),
        ]))

        observation_ref = ""
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={
                    "cmd": cmd,
                    "cwd": cwd,
                    "timeout_sec": timeout_sec,
                },
                payload=raw,
                source_fingerprint=fingerprint(raw),
                metadata={"kind": "bash", "exit_code": exit_code},
            )
            observation_ref = f"\n[observation_ref={observation.ref}]"
        except Exception:
            pass

        # Large output: save full log to /workspace/.legion/logs/<id>.log
        if len(raw) > 16_384:
            exec_id = str(exec_record.id)
            log_rel = f".legion/logs/{exec_id}.log"
            try:
                await wsd.write_workspace_file(rid, log_rel, raw)
                tail = "\n".join(raw.splitlines()[-100:])
                return (
                    f"[Full output ({len(raw)} chars) saved to /workspace/.legion/logs/{exec_id}.log]\n"
                    f"[Showing last 100 lines:]\n{tail}\n"
                    f"[Use read_file(path='.legion/logs/{exec_id}.log') to read the full output.]"
                    f"{observation_ref}"
                )
            except Exception:
                pass  # fall through to normal truncation

        return _truncate(raw) + observation_ref

    @staticmethod
    async def _safe_generation(wsd, rid: str):
        """Best-effort read of the workspace volume generation ({epoch, revision})."""
        try:
            return await wsd.get_workspace_generation(rid)
        except Exception:
            return None

    async def _notify_if_tree_changed(self, ctx: ToolContext, wsd, rid: str, gen_before) -> None:
        """Emit FILE_CHANGED when a synchronous exec advanced the volume generation.

        The generation only moves when file *contents* on the volume changed
        (see workspaced/app/generation.py), so read-only commands (ls, cat,
        grep, git status, …) never produce an event. Background processes write
        after this returns and are intentionally not covered.
        """
        if gen_before is None:
            return
        gen_after = await self._safe_generation(wsd, rid)
        if gen_after is None or gen_after == gen_before:
            return
        try:
            from app.core.events import emit_event

            await emit_event(
                room_id=ctx.room_id,
                event_type="FILE_CHANGED",
                content="Working tree changed by a bash command",
                db=ctx.db,
                task_id=ctx.task_id,
                metadata={"source": "bash", "session_id": str(ctx.session_id)},
            )
        except Exception as exc:
            logger.warning("bash: FILE_CHANGED emit failed for room %s: %s", rid, exc)


REGISTRY.register(BashTool())
