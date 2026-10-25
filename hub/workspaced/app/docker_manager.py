from __future__ import annotations

import asyncio
import logging
import posixpath
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiodocker

from app.config import (
    WORKSPACE_DATA_DIR,
    WORKSPACE_IMAGE,
    WORKSPACE_NETWORK,
    WORKSPACE_VOLUME_NAME,
    WORKSPACE_UID,
    WORKSPACE_GID,
    WORKSPACE_TMP_MB,
    WORKSPACE_PIDS_LIMIT,
    WORKSPACE_PROJECT_QUOTA_MODE,
    EXEC_OUTPUT_LIMIT,
    PROXY_OUTPUT_LIMIT,
    PROXY_TIMEOUT_SEC,
)
from app.schemas import StartRequest, StatusResponse, ExecRequest, ExecResponse, SecretFile
from app.room_id import is_safe_room_id, require_safe_room_id

logger = logging.getLogger("workspaced.docker")

CONTAINER_PREFIX = "ws-"

# internal design notes Stage 5 — deploy-key material (and any other
# file-shaped secret) lives only here: a dedicated tmpfs mount outside
# /workspace, so it never reaches git, a snapshot, or repo_map, and dies with
# the container. Owned by the workspace user so unprivileged bash can read it.
SECRETS_DIR = "/run/legion/secrets"

# Runs inside the workspace container to bridge Docker exec stdin/stdout to a
# service's loopback port. stdin/stdout are copied as bytes.  This is
# intentionally not curl, shell interpolation, or base64: HTTP/WebSocket
# framing remains untouched between the gateway and the localhost service.
SERVICE_TCP_RELAY = """import os, socket, sys, threading
s = socket.create_connection(('127.0.0.1', int(os.environ['SERVICE_PORT'])), 10)
# The 10 s bounds only the connect.  Left on the socket it would also end any
# recv idle that long, closing a quiet HMR WebSocket and making Vite reload
# the page (a white flash) every ~10 s.
s.settimeout(None)
def upload():
    try:
        while True:
            # ``read`` waits until its full buffer is filled.  Browser GET
            # requests are normally far smaller than 64 KiB and their outer
            # HTTP stream can remain open while the response starts, causing
            # a deadlock before the local service sees its request line.
            data = sys.stdin.buffer.read1(65536)
            if not data:
                break
            s.sendall(data)
    except OSError:
        pass
    # The tunnel is gone.  Without a socket timeout the recv below would wait
    # on an idle WebSocket forever, so half-close: the service finishes any
    # response in flight, then closes its side and recv returns.
    try:
        s.shutdown(socket.SHUT_WR)
    except OSError:
        pass
threading.Thread(target=upload, daemon=True).start()
while True:
    data = s.recv(65536)
    if not data:
        break
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
"""

# internal design notes §3a (T2a): "In use now" on the plan screen
# asks for every running room at once — a stats call costs Docker ~1 s, so a
# room's figure is reused for this long.
STATS_CACHE_SEC = 15.0
_MIB = 1024 * 1024


def memory_from_stats(stats: Any) -> dict[str, Optional[float]]:
    """Memory figures of one Docker stats sample, in MiB.

    ``usage`` counts the page cache, which the kernel drops under pressure
    — it is not what the room holds. The resident figure subtracts the
    inactive file pages, as ``docker stats`` does: ``inactive_file`` on
    cgroup v2, ``total_inactive_file`` on cgroup v1.
    """
    s = stats[0] if isinstance(stats, list) and stats else stats
    mem = (s or {}).get("memory_stats") or {}
    usage = mem.get("usage")
    limit = mem.get("limit")
    detail = mem.get("stats") or {}
    inactive = detail.get("inactive_file")
    if inactive is None:
        inactive = detail.get("total_inactive_file", 0)
    rss = None
    if usage is not None:
        rss = max(0, int(usage) - int(inactive or 0))
    return {
        "memory_rss_mb": round(rss / _MIB, 1) if rss is not None else None,
        "memory_usage_mb": round(usage / _MIB, 1) if usage is not None else None,
        "memory_limit_mb": int(limit) // _MIB if limit is not None else None,
    }


def _validate_secret_path(path: str) -> str:
    norm = posixpath.normpath(path)
    if norm != path or not norm.startswith(SECRETS_DIR + "/") or norm == SECRETS_DIR:
        raise ValueError(f"secret file path must be a direct child of {SECRETS_DIR}")
    return norm


def _container_name(room_id: str) -> str:
    return f"{CONTAINER_PREFIX}{room_id}"


class ServiceTcpTunnel:
    """Raw bidirectional TCP stream backed by a Docker exec session.

    The exec process runs inside the workspace container, so its loopback is
    the service's loopback even when the container has NetworkMode=none.
    """

    def __init__(self, stream: Any, context: Any, docker_mgr: "DockerManager", room_id: str) -> None:
        self._stream = stream
        self._context = context
        self._docker_mgr = docker_mgr
        self._room_id = room_id
        self._closed = False

    async def write(self, data: bytes) -> None:
        await self._stream.write_in(data)

    async def read(self) -> bytes | None:
        while True:
            message = await self._stream.read_out()
            if message is None:
                return None
            if message.stream != 2:
                return message.data

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._context.__aexit__(None, None, None)
        finally:
            self._docker_mgr.touch_activity(self._room_id)


class DockerManager:
    def __init__(self) -> None:
        self._docker: Optional[aiodocker.Docker] = None
        self._last_activity: dict[str, float] = {}
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._destroyed_rooms: set[str] = set()
        # exec() leaves a timed-out process running in the container (see
        # exec() below) — tracked here so callers can wait for it to
        # actually finish before trusting a workspace snapshot as final.
        self._pending_execs: dict[str, list[Any]] = {}
        # room_id -> (monotonic time, stats payload); see STATS_CACHE_SEC.
        self._stats_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _room_lock(self, room_id: str) -> asyncio.Lock:
        return self._room_locks.setdefault(room_id, asyncio.Lock())

    async def connect(self) -> None:
        self._docker = aiodocker.Docker()

    async def close(self) -> None:
        if self._docker:
            await self._docker.close()

    def touch_activity(self, room_id: str) -> None:
        self._last_activity[room_id] = time.time()

    def last_activity(self, room_id: str) -> Optional[float]:
        return self._last_activity.get(room_id)

    async def start(self, room_id: str, req: StartRequest) -> StatusResponse:
        room_id = require_safe_room_id(room_id)
        async with self._room_lock(room_id):
            if room_id in self._destroyed_rooms:
                raise RuntimeError(f"Workspace for room {room_id} was permanently destroyed")
            return await self._start_unlocked(room_id, req)

    async def _start_unlocked(self, room_id: str, req: StartRequest) -> StatusResponse:
        name = _container_name(room_id)
        volume_path = f"{WORKSPACE_DATA_DIR}/{room_id}"

        # internal design notes Stage 1 — even an already-running container may
        # have been created by an older workspaced process, so repair the
        # private namespace before taking the fast path below.
        from app.file_ops import ensure_workspace_dir, chown_workspace_tree
        ensure_workspace_dir(room_id)

        if WORKSPACE_PROJECT_QUOTA_MODE == "required" and (
            req.storage_project_id is None
            or req.storage_quota_mb is None
            or req.container_layer_mb is None
        ):
            raise RuntimeError(
                "required project quotas need storage_project_id, storage_quota_mb, and container_layer_mb"
            )

        # T9: one XFS project covers every directory of one room owner. Do
        # this before the running-container fast path too: an open after a
        # missed billing callback must still repair a changed plan's limit.
        if req.storage_project_id is not None or req.storage_quota_mb is not None:
            if req.storage_project_id is None or req.storage_quota_mb is None:
                raise RuntimeError("storage project id and quota must be supplied together")
            from app.project_quota import project_quotas

            await project_quotas.configure(
                project_id=req.storage_project_id,
                quota_mb=req.storage_quota_mb,
                room_ids=list(dict.fromkeys([*req.storage_project_room_ids, room_id])),
            )

        existing = await self._find_container(name)
        if existing:
            info = await existing.show()
            state = info["State"]
            if state.get("Running"):
                return StatusResponse(
                    room_id=room_id,
                    status="running",
                    container_id=existing.id,
                )
            await existing.delete(force=True)

        # The room's subdirectory must exist before the container mounts it
        # (Subpath mounts fail on a missing path), and the whole tree must be
        # owned by the container user or npm install / builds / git will fail.
        chown_workspace_tree(room_id)

        host_config = {
            "Memory": req.mem_limit_mb * 1024 * 1024,
            "NanoCpus": req.cpu_limit_milli * 1_000_000,
            "PidsLimit": WORKSPACE_PIDS_LIMIT,
            # PID 1 is ``sleep infinity``, which never reaps re-parented
            # children: every render killed by a bash timeout left its node and
            # Chrome processes as zombies that kept their PIDs until the
            # container died (a video-render room, 2026-10-25: 186 zombies,
            # fork impossible).  Docker's init (tini) reaps them.
            "Init": True,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            # Docker's tmpfs default includes noexec, which silently breaks
            # anything an agent installs into a /tmp-rooted venv ("Permission
            # denied" running <venv>/bin/pip — the file itself is +x, the
            # mount just refuses to execute anything on it). This container
            # already grants unrestricted `bash`/code execution against
            # /workspace, so noexec on /tmp adds no real containment — it
            # only breaks a workflow (venv under /tmp) an agent will
            # reasonably try. "exec" here is the mount option, not a Python
            # package.
            "Tmpfs": {
                "/tmp": f"size={WORKSPACE_TMP_MB}m,exec",
                # Deploy keys and other file-shaped secrets (Stage 5). Small,
                # noexec (the tmpfs default — no "exec" suffix here, unlike
                # /tmp above), owned by the workspace user so bash can read a
                # key without root, and never mounted under /workspace.
                SECRETS_DIR: f"size=1m,uid={WORKSPACE_UID},gid={WORKSPACE_GID},mode=0700",
            },
        }
        if WORKSPACE_VOLUME_NAME:
            # Mount the same named volume workspaced uses, scoped to the room's
            # subdirectory — container and file/git API see identical files.
            # Requires Docker API >= 1.45 (Engine 26).
            host_config["Mounts"] = [{
                "Type": "volume",
                "Source": WORKSPACE_VOLUME_NAME,
                "Target": "/workspace",
                "VolumeOptions": {"Subpath": room_id},
            }]
        else:
            # Legacy: WORKSPACE_DATA_DIR is a real host path
            host_config["Binds"] = [f"{volume_path}:/workspace"]
        if not req.network_enabled:
            host_config["NetworkMode"] = "none"
        elif WORKSPACE_NETWORK:
            host_config["NetworkMode"] = WORKSPACE_NETWORK
        # Overlay2 accepts a writable-layer size only when Docker's own
        # data-root is XFS+pquota. Only the fail-closed cloud mode enables
        # this: ``auto`` exists for self-hosters who want to attempt XFS
        # project quotas but must still work on a Docker root without pquota.
        if WORKSPACE_PROJECT_QUOTA_MODE == "required" and req.container_layer_mb is not None:
            host_config["StorageOpt"] = {"size": f"{req.container_layer_mb}M"}

        config = {
            "Image": req.image or WORKSPACE_IMAGE,
            "Cmd": ["sleep", "infinity"],
            "WorkingDir": "/workspace",
            "User": f"{WORKSPACE_UID}:{WORKSPACE_GID}",
            "HostConfig": host_config,
        }

        container = await self._docker.containers.create_or_replace(name, config)
        await container.start()

        self.touch_activity(room_id)

        # Ensure git is initialised in the volume (idempotent; chowns after init)
        from app.git_ops import backfill_local_excludes, git_init, is_git_repo
        if not is_git_repo(room_id):
            git_init(room_id)
        else:
            # internal design notes Stage 3: a repo created before Stage 1
            # never got the local excludes (project ignores + .legion/git.lock)
            # in .git/info/exclude — backfill them once, in the same pass.
            backfill_local_excludes(room_id)

        return StatusResponse(
            room_id=room_id,
            status="running",
            container_id=container.id,
        )

    async def stop(self, room_id: str) -> str:
        room_id = require_safe_room_id(room_id)
        async with self._room_lock(room_id):
            return await self._stop_unlocked(room_id)

    async def _stop_unlocked(self, room_id: str) -> str:
        name = _container_name(room_id)
        container = await self._find_container(name)
        if not container:
            return "stopped"

        try:
            await container.stop(t=10)
        except aiodocker.exceptions.DockerError:
            pass
        try:
            await container.delete(force=True)
        except aiodocker.exceptions.DockerError:
            pass

        self._last_activity.pop(room_id, None)
        self._stats_cache.pop(room_id, None)
        return "stopped"

    async def destroy(self, room_id: str) -> dict[str, Any]:
        room_id = require_safe_room_id(room_id)
        async with self._room_lock(room_id):
            self._destroyed_rooms.add(room_id)
            return await self._destroy_unlocked(room_id)

    async def _destroy_unlocked(self, room_id: str) -> dict[str, Any]:
        """Permanently remove one room runtime and its private volume subtree.

        ``stop`` deliberately preserves the workspace volume for a later
        reopen.  Room deletion needs the stronger operation: remove the
        container first, then remove only ``WORKSPACE_DATA_DIR/<room_id>``.
        The operation is idempotent so a retry after a backend/workspaced
        disconnect is safe.
        """
        room_id = require_safe_room_id(room_id)
        name = _container_name(room_id)
        container = await self._find_container(name)
        container_removed = False

        if container:
            try:
                await container.stop(t=10)
            except aiodocker.exceptions.DockerError:
                # A container that exited between lookup and stop is already
                # safe to remove.  Delete below is the authoritative step.
                pass
            try:
                await container.delete(force=True)
                container_removed = True
            except aiodocker.exceptions.DockerError as exc:
                if getattr(exc, "status", None) != 404:
                    raise
                container_removed = True

        self._last_activity.pop(room_id, None)

        data_root = Path(WORKSPACE_DATA_DIR)
        room_root = data_root / room_id
        # The identifier validation above makes this equality a defensive
        # assertion against future changes to the path construction.
        if room_root.parent.resolve() != data_root.resolve():
            raise ValueError("Workspace path escaped the data root")

        data_removed = False
        if room_root.is_symlink() or room_root.is_file():
            await asyncio.to_thread(room_root.unlink)
            data_removed = True
        elif room_root.is_dir():
            await asyncio.to_thread(shutil.rmtree, room_root)
            data_removed = True

        return {
            "room_id": room_id,
            "status": "destroyed",
            "container_removed": container_removed,
            "data_removed": data_removed,
        }

    async def status(self, room_id: str) -> StatusResponse:
        name = _container_name(room_id)
        container = await self._find_container(name)

        if not container:
            return StatusResponse(room_id=room_id, status="stopped")

        info = await container.show()
        state = info["State"]

        if not state.get("Running"):
            return StatusResponse(
                room_id=room_id,
                status="stopped",
                container_id=container.id,
            )

        started_at = state.get("StartedAt", "")
        uptime = None
        if started_at:
            import datetime as dt
            try:
                start = dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                uptime = (dt.datetime.now(dt.timezone.utc) - start).total_seconds()
            except (ValueError, TypeError):
                pass

        memory: dict[str, Optional[float]] = {}
        try:
            memory = memory_from_stats(await container.stats(stream=False))
        except Exception:
            pass

        return StatusResponse(
            room_id=room_id,
            status="running",
            container_id=container.id,
            uptime_sec=round(uptime, 1) if uptime else None,
            # The resident memory, not the page cache (memory_from_stats).
            memory_usage_mb=memory.get("memory_rss_mb"),
            memory_limit_mb=memory.get("memory_limit_mb"),
        )

    async def stats(self, room_id: str) -> dict[str, Any]:
        """The room container's memory, cached for STATS_CACHE_SEC (T2a)."""
        room_id = require_safe_room_id(room_id)
        cached = self._stats_cache.get(room_id)
        now = time.monotonic()
        if cached is not None and now - cached[0] < STATS_CACHE_SEC:
            return {**cached[1], "cached": True}
        container = await self._find_container(_container_name(room_id))
        payload: dict[str, Any] = {
            "room_id": room_id,
            "status": "stopped",
            "memory_rss_mb": None,
            "memory_usage_mb": None,
            "memory_limit_mb": None,
        }
        if container is not None:
            info = await container.show()
            if (info.get("State") or {}).get("Running"):
                payload["status"] = "running"
                payload.update(memory_from_stats(await container.stats(stream=False)))
        if payload["status"] == "running":
            self._stats_cache[room_id] = (now, payload)
        else:
            self._stats_cache.pop(room_id, None)
        return {**payload, "cached": False}

    async def update_memory(self, room_id: str, mem_limit_mb: int) -> dict[str, Any]:
        """Change a running room's memory limit in place (``docker update``).

        Swap stays at Docker's default for a container created without one —
        twice the memory — and has to move with it: Docker refuses a memory
        limit above the swap limit already set.
        """
        room_id = require_safe_room_id(room_id)
        async with self._room_lock(room_id):
            container = await self._find_container(_container_name(room_id))
            if container is None:
                raise RuntimeError("workspace is not running")
            info = await container.show()
            if not (info.get("State") or {}).get("Running"):
                raise RuntimeError("workspace is not running")
            memory = int(mem_limit_mb) * _MIB
            await self._docker._query_json(
                f"containers/{container.id}/update",
                method="POST",
                data={"Memory": memory, "MemorySwap": memory * 2},
            )
            self._stats_cache.pop(room_id, None)
            return {"room_id": room_id, "memory_limit_mb": int(mem_limit_mb)}

    async def exec(self, room_id: str, req: ExecRequest) -> ExecResponse:
        name = _container_name(room_id)
        container = await self._find_container(name)

        if not container:
            raise RuntimeError(f"Container for room {room_id} is not running")

        info = await container.show()
        if not info["State"].get("Running"):
            raise RuntimeError(f"Container for room {room_id} is not running")

        exec_id = str(uuid.uuid4())
        env_list = [f"{k}={v}" for k, v in (req.env or {}).items()]

        if req.secret_files:
            await self._write_secret_files(container, req.secret_files)

        start_ts = time.monotonic()

        ex = await container.exec(
            req.cmd,
            stdout=True,
            stderr=True,
            workdir=req.cwd,
            environment=env_list or None,
        )

        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        total_size = 0
        truncated = False

        timed_out = False
        cleanup_scheduled = False
        try:
            try:
                async with ex.start(detach=False) as stream:
                    async with asyncio.timeout(req.timeout_sec):
                        while True:
                            msg = await stream.read_out()
                            if msg is None:
                                break
                            if total_size >= EXEC_OUTPUT_LIMIT:
                                truncated = True
                                break
                            data = msg.data
                            if msg.stream == 2:
                                stderr_parts.append(data)
                            else:
                                stdout_parts.append(data)
                            total_size += len(data)
            except TimeoutError:
                truncated = True
                timed_out = True
                # The process keeps running in the container after we stop
                # reading its output — track it so wait_for_idle() can tell
                # callers (e.g. review-verification) when it actually finishes.
                self._pending_execs.setdefault(room_id, []).append(ex)
                if req.secret_files:
                    # Deletion must wait for the process to actually exit —
                    # not for this request to return — or a still-running
                    # `git push` loses its key mid-use (plan §3.3).
                    cleanup_scheduled = True
                    asyncio.create_task(
                        self._cleanup_secret_files_when_done(container, ex, req.secret_files)
                    )
        finally:
            if req.secret_files and not cleanup_scheduled:
                await self._delete_secret_files(container, req.secret_files)

        duration_ms = int((time.monotonic() - start_ts) * 1000)

        inspect = await ex.inspect()
        # ExitCode is None while the process is still running (e.g. after a
        # timeout) — coerce to -1 so ExecResponse validation never 500s.
        exit_code = inspect.get("ExitCode") if inspect else None
        if exit_code is None:
            exit_code = -1
        if timed_out:
            stderr_parts.append(
                f"\n[exec timed out after {req.timeout_sec}s; "
                "the process may still be running in the container]".encode()
            )

        stdout_raw = b"".join(stdout_parts)
        stderr_raw = b"".join(stderr_parts)

        self.touch_activity(room_id)

        return ExecResponse(
            exec_id=exec_id,
            exit_code=exit_code,
            stdout=stdout_raw.decode("utf-8", errors="replace")[:EXEC_OUTPUT_LIMIT],
            stderr=stderr_raw.decode("utf-8", errors="replace")[:EXEC_OUTPUT_LIMIT],
            duration_ms=duration_ms,
            truncated=truncated,
        )

    async def _exec_simple(
        self, container: Any, cmd: list[str], *, stdin_data: Optional[bytes] = None
    ) -> None:
        """Run one short helper command inside the container and drain it.

        Used only for the secret-file write/chmod/rm side-commands (Stage
        5) — small, quiet, and never surfaced to the caller's stdout/stderr.
        """
        ex = await container.exec(
            cmd, stdout=True, stderr=True, stdin=stdin_data is not None
        )
        async with ex.start(detach=False) as stream:
            if stdin_data is not None:
                await stream.write_in(stdin_data)
                await stream.close()
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break

    async def _write_secret_files(self, container: Any, secret_files: list[SecretFile]) -> None:
        """Write each secret to tmpfs just before the command that needs it runs.

        Content goes over stdin (never interpolated into a shell string) so
        arbitrary key material — newlines, quotes, backslashes — is safe.
        ``$1`` is the script's first positional arg, not a shell-expanded
        piece of the command line.
        """
        for sf in secret_files:
            path = _validate_secret_path(sf.path)
            await self._exec_simple(
                container,
                ["sh", "-c", 'cat > "$1"', "sw", path],
                stdin_data=sf.content.encode("utf-8"),
            )
            await self._exec_simple(container, ["chmod", sf.mode or "0600", path])

    async def _delete_secret_files(self, container: Any, secret_files: list[SecretFile]) -> None:
        for sf in secret_files:
            try:
                path = _validate_secret_path(sf.path)
                await self._exec_simple(container, ["rm", "-f", path])
            except Exception:
                logger.warning("failed to delete secret file %s", sf.path, exc_info=True)

    async def _cleanup_secret_files_when_done(
        self, container: Any, ex: Any, secret_files: list[SecretFile]
    ) -> None:
        """Wait for a timed-out exec to actually finish, then remove its secret files.

        The tool-level timeout only stops us reading output (see the
        ``TimeoutError`` branch in :meth:`exec`); the process itself keeps
        running. Deleting the key file right away would yank it out from
        under a still-running ``git push`` (plan §3.3).
        """
        while True:
            try:
                info = await ex.inspect()
            except Exception:
                break
            if not info.get("Running"):
                break
            await asyncio.sleep(1.0)
        await self._delete_secret_files(container, secret_files)

    async def pending_exec_count(self, room_id: str) -> int:
        """Prune finished timed-out execs for room_id and return how many are still running."""
        pending = self._pending_execs.get(room_id)
        if not pending:
            return 0
        still_running = []
        for ex in pending:
            try:
                info = await ex.inspect()
                if info.get("Running"):
                    still_running.append(ex)
            except Exception:
                # Exec (or its container) is gone — treat as finished.
                pass
        if still_running:
            self._pending_execs[room_id] = still_running
        else:
            self._pending_execs.pop(room_id, None)
        return len(still_running)

    async def wait_for_idle(
        self, room_id: str, max_wait_sec: float = 20.0, poll_interval: float = 1.0
    ) -> dict[str, Any]:
        """Block until every timed-out exec for room_id has finished, or max_wait_sec elapses.

        Used before treating a workspace snapshot as final (e.g. task-close
        review verification) — a bash call that hit its tool timeout keeps
        writing to disk in the background, and a snapshot taken while that
        write is still in flight can look incomplete even though nothing
        actually failed.
        """
        start = time.monotonic()
        pending = await self.pending_exec_count(room_id)
        while pending and (time.monotonic() - start) < max_wait_sec:
            await asyncio.sleep(poll_interval)
            pending = await self.pending_exec_count(room_id)
        return {
            "pending": pending,
            "waited_sec": round(time.monotonic() - start, 1),
        }

    async def fetch_http(
        self,
        room_id: str,
        port: int,
        path: str = "",
        query: str = "",
    ) -> dict:
        """Fetch an HTTP resource from a server running inside the container.

        Works regardless of the container's network mode (including "none")
        because the request is made by curl executed inside the container
        against 127.0.0.1. The raw response (headers + body) is base64-encoded
        to survive the exec text stream, then parsed here.
        """
        import base64
        import shlex

        name = _container_name(room_id)
        container = await self._find_container(name)
        if not container:
            raise RuntimeError(f"Container for room {room_id} is not running")
        info = await container.show()
        if not info["State"].get("Running"):
            raise RuntimeError(f"Container for room {room_id} is not running")

        url = f"http://127.0.0.1:{port}/{path.lstrip('/')}"
        if query:
            url += f"?{query}"

        # Write to a temp file first so the curl exit code is observable
        # (a plain pipe would report base64's exit code instead).
        script = (
            f"out=/tmp/.preview.$$; "
            f"curl -s -i -L --max-redirs 5 --max-time {PROXY_TIMEOUT_SEC} "
            f"-o \"$out\" {shlex.quote(url)}; ec=$?; "
            f"if [ $ec -ne 0 ]; then echo \"CURLERR:$ec\"; "
            f"else base64 -w 0 \"$out\"; fi; rm -f \"$out\""
        )

        ex = await container.exec(["sh", "-c", script], stdout=True, stderr=True)

        parts: list[bytes] = []
        total = 0
        try:
            async with ex.start(detach=False) as stream:
                async with asyncio.timeout(PROXY_TIMEOUT_SEC + 10):
                    while True:
                        msg = await stream.read_out()
                        if msg is None:
                            break
                        if msg.stream == 2:
                            continue  # ignore stderr noise
                        parts.append(msg.data)
                        total += len(msg.data)
                        # base64 inflates by 4/3
                        if total > PROXY_OUTPUT_LIMIT * 4 // 3:
                            return {"error": "Response too large", "status_code": 502}
        except TimeoutError:
            return {"error": "Timed out fetching from workspace server", "status_code": 504}

        self.touch_activity(room_id)

        raw = b"".join(parts).strip()
        if not raw:
            return {"error": "Empty response from workspace", "status_code": 502}
        if raw.startswith(b"CURLERR:"):
            code = raw.decode("ascii", errors="replace").split(":", 1)[1].strip()
            hint = "connection refused — is the server running on this port?" if code == "7" else f"curl exit code {code}"
            return {"error": hint, "status_code": 502}

        try:
            data = base64.b64decode(raw, validate=False)
        except Exception:
            return {"error": "Failed to decode response", "status_code": 502}

        # curl -i -L prints one header block per hop; skip to the last one.
        status_code = 502
        content_type = ""
        while data.startswith(b"HTTP/"):
            sep = data.find(b"\r\n\r\n")
            if sep == -1:
                break
            header_block, data = data[:sep], data[sep + 4:]
            lines = header_block.split(b"\r\n")
            try:
                status_code = int(lines[0].split(b" ")[1])
            except (IndexError, ValueError):
                status_code = 502
            content_type = ""
            for line in lines[1:]:
                if line.lower().startswith(b"content-type:"):
                    content_type = line.split(b":", 1)[1].strip().decode("latin-1")

        return {
            "status_code": status_code,
            "content_type": content_type,
            "body_b64": base64.b64encode(data).decode("ascii"),
        }

    async def open_service_tcp_tunnel(self, room_id: str, port: int) -> ServiceTcpTunnel:
        """Open a binary tunnel to a service's *already assigned* loopback port.

        The caller has to obtain ``port`` from ``ServiceSupervisor``; this
        primitive deliberately does not accept hostnames or URLs.
        """
        if not 1 <= port <= 65535:
            raise RuntimeError("invalid service port")
        container = await self._find_container(_container_name(room_id))
        if not container:
            raise RuntimeError(f"Container for room {room_id} is not running")
        info = await container.show()
        if not info["State"].get("Running"):
            raise RuntimeError(f"Container for room {room_id} is not running")
        ex = await container.exec(
            ["python3", "-c", SERVICE_TCP_RELAY], stdout=True, stderr=True, stdin=True,
            environment={"SERVICE_PORT": str(port)},
        )
        context = ex.start(detach=False)
        stream = await context.__aenter__()
        return ServiceTcpTunnel(stream, context, self, room_id)

    async def exec_stream(
        self,
        room_id: str,
        req: "ExecRequest",
        disconnect_fn=None,
    ):
        """Async generator that yields NDJSON bytes from a running exec.

        Each line is one of:
          {"t":1,"d":"..."}  – stdout chunk
          {"t":2,"d":"..."}  – stderr chunk
          {"exit":N}          – process exited with code N
          {"timeout":true}    – timeout_sec elapsed, process may still run
          {"error":"..."}     – could not start or Docker error
        """
        import json as _json

        name = _container_name(room_id)
        container = await self._find_container(name)
        if not container:
            yield (_json.dumps({"error": f"Container for room {room_id} is not running"}) + "\n").encode()
            return
        info = await container.show()
        if not info["State"].get("Running"):
            yield (_json.dumps({"error": f"Container for room {room_id} is not running"}) + "\n").encode()
            return

        ex = await container.exec(
            req.cmd,
            stdout=True,
            stderr=True,
            workdir=req.cwd,
            environment=[f"{k}={v}" for k, v in (req.env or {}).items()] or None,
        )

        exit_code = -1
        try:
            async with ex.start(detach=False) as stream:
                deadline = asyncio.get_event_loop().time() + req.timeout_sec
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        yield b'{"timeout":true}\n'
                        break
                    try:
                        msg = await asyncio.wait_for(
                            stream.read_out(), timeout=min(1.0, remaining)
                        )
                    except asyncio.TimeoutError:
                        if disconnect_fn and await disconnect_fn():
                            break
                        continue
                    if msg is None:
                        break
                    chunk = _json.dumps({"t": msg.stream, "d": msg.data.decode("utf-8", errors="replace")}) + "\n"
                    yield chunk.encode()
        except Exception as exc:
            yield (_json.dumps({"error": str(exc)}) + "\n").encode()

        try:
            insp = await ex.inspect()
            exit_code = insp.get("ExitCode", -1) if insp else -1
        except Exception:
            exit_code = -1
        yield (_json.dumps({"exit": exit_code}) + "\n").encode()
        self.touch_activity(room_id)

    async def list_running(self) -> list[str]:
        return await self._list_container_ids(all=False)

    async def list_managed(self) -> list[str]:
        """Return all workspace containers, including stopped leftovers."""
        return await self._list_container_ids(all=True)

    async def remove_exited(self) -> list[str]:
        """Delete room containers that stopped without going through stop().

        ``stop`` deletes the container, but one the Docker daemon stopped
        itself (daemon or host restart, a crashed init) stays behind as
        ``exited`` with its writable layer — ``~/.npm``, ``~/.cache/pip`` —
        until the room is started again, which may be never
        (internal design notes §5.2, T5).
        """
        containers = await self._docker.containers.list(
            all=True,
            filters={"name": [CONTAINER_PREFIX], "status": ["exited", "dead"]},
        )
        removed = []
        for room_id in self._room_ids_of(containers):
            async with self._room_lock(room_id):
                container = await self._find_container(_container_name(room_id))
                if container is None:
                    continue
                info = await container.show()
                # Re-checked under the room lock: start() may have replaced it.
                if info["State"].get("Status") not in ("exited", "dead"):
                    continue
                try:
                    await container.delete(force=True)
                except aiodocker.exceptions.DockerError as exc:
                    if getattr(exc, "status", None) != 404:
                        logger.warning("failed to remove exited workspace %s: %s", room_id, exc)
                        continue
                self._last_activity.pop(room_id, None)
                removed.append(room_id)
        return removed

    async def _list_container_ids(self, *, all: bool) -> list[str]:
        containers = await self._docker.containers.list(
            all=all,
            filters={"name": [CONTAINER_PREFIX]},
        )
        return self._room_ids_of(containers)

    @staticmethod
    def _room_ids_of(containers: list[Any]) -> list[str]:
        room_ids = []
        for c in containers:
            names = c._container.get("Names", [])
            for n in names:
                n = n.lstrip("/")
                candidate = n[len(CONTAINER_PREFIX):] if n.startswith(CONTAINER_PREFIX) else ""
                if candidate and is_safe_room_id(candidate):
                    room_ids.append(candidate)
        return sorted(set(room_ids))

    async def _find_container(self, name: str):
        try:
            return await self._docker.containers.get(name)
        except aiodocker.exceptions.DockerError as e:
            if e.status == 404:
                return None
            raise
