"""Supervisor for declared workspace services.

Unlike legacy Runs this module never interprets a user command as a shell
string.  A service starts from an argv vector in its own process session, so
stopping the service reliably terminates its children as well.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from app.schemas import ExecRequest, ServiceRuntimeResponse

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
_LOG_LIMIT_BYTES = 5 * 1024 * 1024
_TAIL_LIMIT = 64 * 1024


@dataclass
class Runtime:
    service_id: str
    slug: str
    port: int
    pid: int | None
    pgid: int | None
    log_path: str


class ServiceSupervisor:
    def __init__(self, docker_mgr, port_range: tuple[int, int] = (3000, 3999)) -> None:
        self._docker_mgr = docker_mgr
        self._port_range = port_range
        self._locks: dict[str, asyncio.Lock] = {}
        self._runtime: dict[tuple[str, str], Runtime] = {}

    def _lock(self, room_id: str) -> asyncio.Lock:
        return self._locks.setdefault(room_id, asyncio.Lock())

    @staticmethod
    def log_path(service_id: str, slug: str) -> str:
        safe = _SAFE_NAME.sub("_", slug).strip("._") or "service"
        return f"/workspace/.legion/services/{safe}-{service_id[:12]}.log"

    async def _exec(self, room_id: str, cmd: list[str], *, cwd: str = "/workspace",
                    timeout: int = 15, env: dict[str, str] | None = None):
        return await self._docker_mgr.exec(
            room_id, ExecRequest(cmd=cmd, cwd=cwd, timeout_sec=timeout, env=env or {})
        )

    async def _alive(self, room_id: str, pid: int | None) -> bool:
        if not pid:
            return False
        # ``kill -0`` succeeds even for a zombie — a service process that has
        # already exited but was never reaped (the workspace container's PID 1
        # is ``sleep infinity``, which never waits on re-parented children).
        # A zombie no longer listens on its port, so counting it as alive
        # strands the service in a bogus "ready" state whose preview can never
        # connect. Read the process state from /proc and reject Z (zombie) and
        # X (dead). ``comm`` may contain spaces and parentheses, so the state
        # is taken as the first field after the final ") ".
        script = (
            'st=$(cat "/proc/$SERVICE_PID/stat" 2>/dev/null) || exit 1; '
            'st=${st##*) }; st=${st%% *}; '
            '[ "$st" = Z ] && exit 1; [ "$st" = X ] && exit 1; exit 0'
        )
        result = await self._exec(
            room_id, ["sh", "-c", script],
            timeout=5, env={"SERVICE_PID": str(pid)},
        )
        return result.exit_code == 0

    async def _port_available(self, room_id: str, port: int) -> bool:
        # Binding tests availability without exposing a port.  The per-room
        # lock covers concurrent starts; readiness below verifies that the
        # child actually listens on this exact reserved port.
        script = (
            "import socket,sys; p=int(__import__('os').environ['SERVICE_PORT']); "
            "s=socket.socket(); "
            "\ntry:\n s.bind(('127.0.0.1',p))\nexcept OSError:\n sys.exit(1)\n"
            "finally:\n s.close()"
        )
        result = await self._exec(
            room_id, ["python3", "-c", script], timeout=5,
            env={"SERVICE_PORT": str(port)},
        )
        return result.exit_code == 0

    async def reserve_port(self, room_id: str, requested: int | str = "auto") -> int:
        """Return an unused loopback port while holding the room start lock.

        Callers normally use this through ``start``.  It is public for the
        backend's explicit reserve_port internal API and tests.
        """
        async with self._lock(room_id):
            return await self._reserve_port_unlocked(room_id, requested)

    async def _reserve_port_unlocked(self, room_id: str, requested: int | str = "auto") -> int:
        if isinstance(requested, int):
            if not await self._port_available(room_id, requested):
                raise ServiceError("port_conflict", f"Port {requested} is already in use")
            return requested
        first, last = self._port_range
        for port in range(first, last + 1):
            if await self._port_available(room_id, port):
                return port
        raise ServiceError("port_conflict", "No free service port is available")

    async def start(self, room_id: str, spec) -> ServiceRuntimeResponse:
        async with self._lock(room_id):
            key = (room_id, spec.service_id)
            existing = self._runtime.get(key)
            if existing and await self._alive(room_id, existing.pid):
                return ServiceRuntimeResponse(
                    service_id=spec.service_id, slug=spec.slug, state="ready",
                    port=existing.port, pid=existing.pid, pgid=existing.pgid,
                    log_path=existing.log_path,
                )

            try:
                port = await self._reserve_port_unlocked(room_id, spec.port)
            except ServiceError as exc:
                return ServiceRuntimeResponse(
                    service_id=spec.service_id, slug=spec.slug, state="error",
                    port=spec.port if isinstance(spec.port, int) else None,
                    error_code=exc.code, error=str(exc),
                )

            log_path = self.log_path(spec.service_id, spec.slug)
            # argv never passes through a shell (see the launcher below), so a
            # literal ``$PORT``/``${PORT}`` token — the pattern declared
            # services are expected to use to forward the reserved port to
            # their stack (e.g. Vite's ``--port`` flag, which does not read
            # the PORT env var) — would otherwise reach the child process
            # unexpanded. Resolve it here, once, against the actual reserved
            # port.
            resolved_argv = [
                item.replace("${PORT}", str(port)).replace("$PORT", str(port))
                for item in spec.argv
            ]
            # Static Python launcher: argv/env cross the docker exec boundary
            # as JSON environment data, never as interpolated shell syntax.
            launcher = (
                "import json,os,subprocess; "
                "argv=json.loads(os.environ['SERVICE_ARGV']); "
                "env=os.environ.copy(); child_env=json.loads(os.environ['SERVICE_ENV']); "
                "prefix=child_env.pop('LEGION_ENV_PATH_PREPEND',''); "
                "\nif prefix: child_env['PATH']=prefix+':'+env.get('PATH','')\n"
                "env.update(child_env); "
                "log=os.environ['SERVICE_LOG']; os.makedirs(os.path.dirname(log),exist_ok=True); "
                "old=log+'.1'; "
                "\nif os.path.exists(log) and os.path.getsize(log)>5242880:\n"
                " os.replace(log,old)\n"
                "f=open(log,'ab',buffering=0); "
                "p=subprocess.Popen(argv,cwd=os.environ['SERVICE_CWD'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); "
                "print(str(p.pid)+' '+str(os.getpgid(p.pid)))"
            )
            env = {
                "SERVICE_ARGV": json.dumps(resolved_argv),
                "SERVICE_ENV": json.dumps({
                    key: value.replace("${PORT}", str(port)).replace("$PORT", str(port))
                    for key, value in spec.env.items()
                } | {"PORT": str(port)}),
                "SERVICE_LOG": log_path,
                "SERVICE_CWD": spec.cwd,
            }
            result = await self._exec(room_id, ["python3", "-c", launcher], cwd=spec.cwd, timeout=10, env=env)
            try:
                pid_s, pgid_s = result.stdout.strip().splitlines()[-1].split()
                runtime = Runtime(spec.service_id, spec.slug, port, int(pid_s), int(pgid_s), log_path)
            except (IndexError, ValueError):
                return ServiceRuntimeResponse(
                    service_id=spec.service_id, slug=spec.slug, state="error", port=port,
                    log_path=log_path, error_code="start_failed",
                    error=(result.stderr or result.stdout or "supervisor did not return PID").strip(),
                )
            self._runtime[key] = runtime

        ready, error = await self.wait_ready(room_id, spec.service_id, spec.readiness, spec.timeout_sec)
        if not ready:
            await self.stop(room_id, spec.service_id)
            return ServiceRuntimeResponse(
                service_id=spec.service_id, slug=spec.slug, state="error", port=port,
                pid=runtime.pid, pgid=runtime.pgid, log_path=log_path,
                error_code="readiness_failed", error=error,
            )
        return ServiceRuntimeResponse(
            service_id=spec.service_id, slug=spec.slug, state="ready", port=port,
            pid=runtime.pid, pgid=runtime.pgid, log_path=log_path,
        )

    async def wait_ready(self, room_id: str, service_id: str, readiness: dict[str, Any], timeout_sec: int):
        runtime = self._runtime.get((room_id, service_id))
        if runtime is None:
            return False, "service is not running"
        kind = str(readiness.get("type") or "tcp")
        path = str(readiness.get("path") or "/")
        statuses = {int(value) for value in (readiness.get("statuses") or [200])}
        deadline = asyncio.get_running_loop().time() + min(timeout_sec, 300)
        last_error = "service did not become ready"
        while asyncio.get_running_loop().time() < deadline:
            if not await self._alive(room_id, runtime.pid):
                return False, "service process exited before readiness"
            if kind == "tcp":
                tcp_probe = (
                    "import os,socket,sys; s=socket.socket(); s.settimeout(1); "
                    "\ntry:\n s.connect(('127.0.0.1',int(os.environ['SERVICE_PORT']))); sys.exit(0)\n"
                    "except OSError:\n sys.exit(1)\n"
                    "finally:\n s.close()"
                )
                probe = await self._exec(
                    room_id, ["python3", "-c", tcp_probe],
                    timeout=5, env={"SERVICE_PORT": str(runtime.port)},
                )
                if probe.exit_code == 0:
                    return True, ""
                last_error = "TCP listener is not available on reserved port"
            elif kind == "http":
                probe_script = (
                    "import os,sys,urllib.request,urllib.error; "
                    "u='http://127.0.0.1:'+os.environ['SERVICE_PORT']+os.environ['SERVICE_PATH']; "
                    "\ntry:\n r=urllib.request.urlopen(u,timeout=1); print(r.status); sys.exit(0 if r.status in {" + ",".join(map(str, statuses)) + "} else 1)\n"
                    "except Exception as e:\n print(e); sys.exit(1)"
                )
                probe = await self._exec(room_id, ["python3", "-c", probe_script], timeout=5,
                    env={"SERVICE_PORT": str(runtime.port), "SERVICE_PATH": path})
                if probe.exit_code == 0:
                    return True, ""
                last_error = (probe.stdout or probe.stderr or "HTTP readiness probe failed").strip()
            else:
                return False, f"unsupported readiness type: {kind}"
            await asyncio.sleep(0.25)
        return False, last_error

    async def _kill_group(self, room_id: str, pgid: int) -> None:
        # The workspace image's /bin/sh is dash, whose builtin ``kill`` does
        # not understand a ``--`` end-of-options marker (it errors with
        # "Illegal option --" before ever parsing the negative pgid), so the
        # signal is silently never delivered. Passing the negative pgid
        # directly, with no ``--``, is what dash's builtin actually accepts.
        script = 'kill -TERM -"$SERVICE_PGID" 2>/dev/null || true; sleep 1; kill -KILL -"$SERVICE_PGID" 2>/dev/null || true'
        await self._exec(room_id, ["sh", "-c", script], timeout=8, env={"SERVICE_PGID": str(pgid)})

    async def stop(self, room_id: str, service_id: str, fallback_pid: int | None = None,
                    fallback_pgid: int | None = None) -> ServiceRuntimeResponse:
        async with self._lock(room_id):
            runtime = self._runtime.get((room_id, service_id))
            if runtime is None:
                # A supervisor restart (e.g. uvicorn --reload during
                # development, or a redeploy) wipes this in-memory map while
                # the process keeps running inside the separate, long-lived
                # workspace container. Without a durable fallback identity
                # this silently no-ops, leaving a real process holding its
                # port forever and every future start failing with
                # ``port_conflict``. The caller's DB row is the source of
                # truth for that identity across a supervisor restart.
                if fallback_pid and await self._alive(room_id, fallback_pid):
                    await self._kill_group(room_id, fallback_pgid or fallback_pid)
                return ServiceRuntimeResponse(service_id=service_id, slug="", state="stopped")
            if await self._alive(room_id, runtime.pid):
                await self._kill_group(room_id, runtime.pgid or runtime.pid)
            self._runtime.pop((room_id, service_id), None)
            return ServiceRuntimeResponse(service_id=service_id, slug=runtime.slug, state="stopped", port=runtime.port,
                                          log_path=runtime.log_path)

    async def logs(self, room_id: str, service_id: str, cursor: int = 0, slug: str = "") -> dict[str, Any]:
        runtime = self._runtime.get((room_id, service_id))
        if runtime is not None:
            log_path = runtime.log_path
        elif slug:
            # The log path is a pure function of (service_id, slug) and lives on
            # the workspace's persistent volume, independent of whether this
            # process still remembers a live runtime for it.  A caller that
            # only knows the durable DB row (e.g. after a supervisor restart,
            # or when the last start attempt failed before registering a
            # runtime) must still be able to read diagnostics.
            log_path = self.log_path(service_id, slug)
        else:
            raise ServiceError("not_found", "service runtime is not known")
        script = (
            "import os; p=os.environ['SERVICE_LOG']; c=max(0,int(os.environ['SERVICE_CURSOR'])); "
            "\ntry:\n s=os.path.getsize(p); f=open(p,'rb'); f.seek(min(c,s)); d=f.read(65536); print(d.decode('utf-8','replace'),end=''); print('\\n__CURSOR__='+str(min(c,s)+len(d)))\n"
            "except FileNotFoundError:\n print('__CURSOR__=0')"
        )
        result = await self._exec(room_id, ["python3", "-c", script], timeout=10,
            env={"SERVICE_LOG": log_path, "SERVICE_CURSOR": str(cursor)})
        content, marker, next_cursor = result.stdout.rpartition("\n__CURSOR__=")
        return {"service_id": service_id, "log_path": log_path, "content": content,
                "cursor": int(next_cursor) if marker and next_cursor.isdigit() else cursor,
                "rotated": cursor > 0 and not marker}

    async def reconcile(self, room_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rehydrate runtime records after a backend restart and mark dead PIDs."""
        output = []
        async with self._lock(room_id):
            for record in records:
                pid = record.get("pid")
                alive = await self._alive(room_id, int(pid) if pid else None)
                if alive:
                    runtime = Runtime(str(record["service_id"]), str(record.get("slug") or ""), int(record["port"]),
                                      int(pid), record.get("pgid"), str(record.get("log_path") or self.log_path(str(record["service_id"]), str(record.get("slug") or ""))))
                    self._runtime[(room_id, runtime.service_id)] = runtime
                output.append({"service_id": record.get("service_id"), "alive": alive})
        return output

    async def service_port(self, room_id: str, service_id: str) -> int:
        """Return only a live supervisor-owned target, never a caller port."""
        runtime = self._runtime.get((room_id, service_id))
        if runtime is None or not await self._alive(room_id, runtime.pid):
            raise ServiceError("not_ready", "service runtime is not ready")
        return runtime.port

    def clear_room(self, room_id: str) -> None:
        """Forget PID records after Docker removes the workspace container."""
        for key in [key for key in self._runtime if key[0] == room_id]:
            self._runtime.pop(key, None)


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
