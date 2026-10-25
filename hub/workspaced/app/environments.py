"""Preparation of reproducible workspace dependency environments.

Install commands are selected from a small built-in profile registry.  The
backend supplies only a validated profile and manifest paths, never an
agent-provided shell script, so Browser start remains a bounded lifecycle
operation instead of another arbitrary command executor.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from app.schemas import ExecRequest

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
_RUNTIME_COMMANDS = {
    "python-pip": ("python3",),
    "node-npm": ("node", "npm"),
    "node-pnpm": ("node", "corepack"),
    "node-yarn": ("node", "corepack"),
    "none": (),
}


@dataclass(frozen=True)
class PreparedEnvironment:
    state: str
    fingerprint: str | None = None
    runtime_fingerprint: str | None = None
    runtime_env: dict[str, str] | None = None
    log_path: str | None = None
    error_code: str | None = None
    error: str | None = None
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "fingerprint": self.fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "runtime_env": self.runtime_env or {},
            "log_path": self.log_path,
            "error_code": self.error_code,
            "error": self.error,
            "cached": self.cached,
        }


class EnvironmentSupervisor:
    """Serialize profile preparation per room/environment and retain markers."""

    def __init__(self, docker_mgr) -> None:
        self._docker_mgr = docker_mgr
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _lock(self, room_id: str, environment_id: str) -> asyncio.Lock:
        return self._locks.setdefault((room_id, environment_id), asyncio.Lock())

    @staticmethod
    def _base_path(environment_id: str, slug: str) -> str:
        safe = _SAFE_NAME.sub("_", slug).strip("._") or "environment"
        return f"/workspace/.legion/environments/{safe}-{environment_id[:12]}"

    @classmethod
    def log_path(cls, environment_id: str, slug: str) -> str:
        return f"{cls._base_path(environment_id, slug)}/prepare.log"

    async def _exec(
        self, room_id: str, cmd: list[str], *, cwd: str = "/workspace", timeout: int = 120,
        env: dict[str, str] | None = None,
    ):
        return await self._docker_mgr.exec(
            room_id, ExecRequest(cmd=cmd, cwd=cwd, timeout_sec=timeout, env=env or {})
        )

    async def _fingerprint(self, room_id: str, spec) -> dict[str, Any]:
        # The script runs inside the workspace, where the volume and the exact
        # runtime binaries are visible.  ``commonpath`` prevents a malformed
        # internal caller from hashing files outside the declared root.
        script = r"""
import hashlib,json,os,shutil,subprocess,sys
root=os.path.realpath(os.environ['ENV_ROOT'])
manifests=json.loads(os.environ['ENV_MANIFESTS'])
commands=json.loads(os.environ['ENV_RUNTIME_COMMANDS'])
items=[]
for relative in manifests:
    path=os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([root,path]) != root:
        print(json.dumps({'error_code':'environment_manifest_invalid','error':'manifest escapes environment root'})); sys.exit(0)
    if not os.path.isfile(path):
        print(json.dumps({'error_code':'environment_manifest_missing','error':'manifest not found: '+relative})); sys.exit(0)
    with open(path,'rb') as handle: items.append((relative,hashlib.sha256(handle.read()).hexdigest()))
versions={}
for command in commands:
    executable=shutil.which(command)
    if not executable:
        print(json.dumps({'error_code':'runtime_unavailable','error':'required runtime is unavailable: '+command})); sys.exit(0)
    try:
        versions[command]=subprocess.check_output([executable,'--version'],stderr=subprocess.STDOUT,text=True,timeout=5).strip()
    except Exception as exc:
        print(json.dumps({'error_code':'runtime_unavailable','error':'could not inspect runtime '+command+': '+str(exc)})); sys.exit(0)
body={'profile':os.environ['ENV_PROFILE'],'manifests':items,'runtime':versions}
encoded=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
print(json.dumps({'fingerprint':hashlib.sha256(encoded).hexdigest(),'runtime_fingerprint':hashlib.sha256(json.dumps(versions,sort_keys=True).encode()).hexdigest()}))
"""
        result = await self._exec(
            room_id, ["python3", "-c", script], timeout=20,
            env={
                "ENV_ROOT": spec.root,
                "ENV_MANIFESTS": json.dumps(spec.manifests),
                "ENV_PROFILE": spec.profile,
                "ENV_RUNTIME_COMMANDS": json.dumps(_RUNTIME_COMMANDS.get(spec.profile, ())),
            },
        )
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return {"error_code": "environment_fingerprint_failed", "error": (result.stderr or result.stdout or "could not inspect environment")[:2000]}
        return payload

    async def _marker_matches(
        self, room_id: str, marker_path: str, fingerprint: str, profile: str, root: str,
    ) -> bool:
        script = r"""
import json,os,sys
try:
  with open(os.environ['ENV_MARKER'],encoding='utf-8') as h: marker=json.load(h)
  ok=marker.get('fingerprint') == os.environ['ENV_FINGERPRINT'] and marker.get('profile') == os.environ['ENV_PROFILE']
  if ok and os.environ['ENV_PROFILE'] == 'python-pip':
    ok=os.path.isfile(os.path.join(os.path.dirname(os.environ['ENV_MARKER']),'venv','bin','python'))
  if ok and os.environ['ENV_PROFILE'].startswith('node-'):
    ok=os.path.isdir(os.path.join(os.environ['ENV_ROOT'],'node_modules'))
except Exception: ok=False
sys.exit(0 if ok else 1)
"""
        result = await self._exec(
            room_id, ["python3", "-c", script], timeout=10,
            env={
                "ENV_MARKER": marker_path, "ENV_FINGERPRINT": fingerprint,
                "ENV_PROFILE": profile, "ENV_ROOT": root,
            },
        )
        return result.exit_code == 0

    async def _append_log(self, room_id: str, log_path: str, content: str) -> None:
        script = r"""
import os
path=os.environ['ENV_LOG']; os.makedirs(os.path.dirname(path),exist_ok=True)
with open(path,'a',encoding='utf-8',errors='replace') as handle: handle.write(os.environ['ENV_LOG_CONTENT'])
"""
        await self._exec(
            room_id, ["python3", "-c", script], timeout=15,
            env={"ENV_LOG": log_path, "ENV_LOG_CONTENT": content[-60_000:]},
        )

    async def _write_marker(self, room_id: str, marker_path: str, marker: dict[str, str]) -> None:
        script = r"""
import json,os
path=os.environ['ENV_MARKER']; os.makedirs(os.path.dirname(path),exist_ok=True)
tmp=path+'.tmp'
with open(tmp,'w',encoding='utf-8') as h: json.dump(json.loads(os.environ['ENV_MARKER_BODY']),h,sort_keys=True)
os.replace(tmp,path)
"""
        await self._exec(
            room_id, ["python3", "-c", script], timeout=15,
            env={"ENV_MARKER": marker_path, "ENV_MARKER_BODY": json.dumps(marker, sort_keys=True)},
        )

    @staticmethod
    def _runtime_env(profile: str, base_path: str) -> dict[str, str]:
        if profile == "python-pip":
            return {
                "VIRTUAL_ENV": f"{base_path}/venv",
                "LEGION_ENV_PATH_PREPEND": f"{base_path}/venv/bin",
            }
        return {}

    async def _install(self, room_id: str, spec, base_path: str) -> tuple[bool, str]:
        commands: list[list[str]]
        if spec.profile == "python-pip":
            venv_python = f"{base_path}/venv/bin/python"
            commands = [
                ["python3", "-m", "venv", f"{base_path}/venv"],
                [venv_python, "-m", "pip", "install", "--disable-pip-version-check", "--requirement", *spec.manifests],
            ]
        elif spec.profile == "node-npm":
            commands = [["npm", "ci", "--no-audit", "--no-fund"]]
        elif spec.profile == "node-pnpm":
            commands = [["corepack", "pnpm", "install", "--frozen-lockfile"]]
        elif spec.profile == "node-yarn":
            commands = [["corepack", "yarn", "install", "--immutable"]]
        else:
            return True, "profile none: no dependencies to install\n"
        output: list[str] = []
        for command in commands:
            result = await self._exec(room_id, command, cwd=spec.root, timeout=spec.timeout_sec)
            text = f"$ {' '.join(command)}\n{result.stdout}{result.stderr}"
            output.append(text if text.endswith("\n") else text + "\n")
            if result.exit_code != 0:
                return False, "".join(output)
        return True, "".join(output)

    async def prepare(self, room_id: str, spec) -> PreparedEnvironment:
        async with self._lock(room_id, spec.environment_id):
            fingerprint = await self._fingerprint(room_id, spec)
            log_path = self.log_path(spec.environment_id, spec.slug)
            if fingerprint.get("error_code"):
                message = str(fingerprint.get("error") or "environment preflight failed")
                await self._append_log(room_id, log_path, f"preflight failed: {message}\n")
                return PreparedEnvironment(
                    state="error", log_path=log_path, error_code=str(fingerprint["error_code"]), error=message,
                )
            value = str(fingerprint["fingerprint"])
            runtime = str(fingerprint["runtime_fingerprint"])
            base_path = self._base_path(spec.environment_id, spec.slug)
            marker_path = f"{base_path}/marker.json"
            runtime_env = self._runtime_env(spec.profile, base_path)
            if await self._marker_matches(room_id, marker_path, value, spec.profile, spec.root):
                await self._append_log(room_id, log_path, "environment cache hit\n")
                return PreparedEnvironment(
                    state="ready", fingerprint=value, runtime_fingerprint=runtime,
                    runtime_env=runtime_env, log_path=log_path, cached=True,
                )
            await self._append_log(room_id, log_path, f"preparing {spec.profile}\n")
            ok, output = await self._install(room_id, spec, base_path)
            await self._append_log(room_id, log_path, output)
            if not ok:
                return PreparedEnvironment(
                    state="error", fingerprint=value, runtime_fingerprint=runtime, log_path=log_path,
                    error_code="dependency_prepare_failed",
                    error=(output.strip().splitlines()[-1] if output.strip() else "dependency install failed")[:2000],
                )
            await self._write_marker(room_id, marker_path, {
                "fingerprint": value, "profile": spec.profile, "runtime_fingerprint": runtime,
            })
            return PreparedEnvironment(
                state="ready", fingerprint=value, runtime_fingerprint=runtime,
                runtime_env=runtime_env, log_path=log_path,
            )

    async def logs(self, room_id: str, log_path: str, cursor: int = 0) -> dict[str, Any]:
        script = (
            "import os; p=os.environ['ENV_LOG']; c=max(0,int(os.environ['ENV_CURSOR'])); "
            "\ntry:\n s=os.path.getsize(p); f=open(p,'rb'); f.seek(min(c,s)); d=f.read(65536); print(d.decode('utf-8','replace'),end=''); print('\\n__CURSOR__='+str(min(c,s)+len(d)))"
            "\nexcept FileNotFoundError:\n print('__CURSOR__=0')"
        )
        result = await self._exec(room_id, ["python3", "-c", script], timeout=10,
                                  env={"ENV_LOG": log_path, "ENV_CURSOR": str(max(0, cursor))})
        content, marker, next_cursor = result.stdout.rpartition("\n__CURSOR__=")
        return {
            "log_path": log_path, "content": content,
            "cursor": int(next_cursor) if marker and next_cursor.isdigit() else cursor,
            "rotated": cursor > 0 and not marker,
        }
