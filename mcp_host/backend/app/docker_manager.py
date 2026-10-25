from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

import aiodocker

from app.config import (
    MCP_HOST_CONTAINER_PREFIX,
    MCP_HOST_ALLOW_INSECURE_REGISTRY,
    MCP_HOST_HUB_URL,
    MCP_HOST_INSTALL_EGRESS_PROFILE,
    MCP_HOST_MAX_LOG_BYTES,
    MCP_HOST_MAX_MEMORY_MB,
    MCP_HOST_MAX_STORAGE_MB,
    MCP_HOST_NETWORK,
    MCP_HOST_PROBE_REQUEST_TIMEOUT_SEC,
    MCP_HOST_PROBE_TIMEOUT_SEC,
    MCP_HOST_PUBLIC_BASE_URL,
    MCP_HOST_RUNNER_DATA_VOLUME_PREFIX,
    MCP_HOST_RUNNER_IMAGE,
    MCP_HOST_RUNNER_NETWORK,
    MCP_HOST_RUNNER_NPM_REGISTRY,
    MCP_HOST_RUNNER_NPM_REGISTRY_HOSTS,
    MCP_HOST_RUNNER_PORT,
    MCP_HOST_RUNTIME_EGRESS_PROFILE,
    MCP_HOST_SECRETS_TIMEOUT_SEC,
    MCP_HOST_STORAGE_QUOTA_MODE,
    MCP_HOST_VOLUME_DRIVER,
    MCP_HOST_VOLUME_SIZE_OPTION,
    MCP_HOST_TOKEN,
)
from app.schemas import (
    ManagedStdioStartRequest,
    ResolvedArtifact,
    StartRequest,
    StatusResponse,
    ToolSnapshot,
)


_SAFE_FRAGMENT = re.compile(r"[^a-zA-Z0-9_.-]")


def _fragment(value: str) -> str:
    return _SAFE_FRAGMENT.sub("-", str(value))[:32].strip("-") or "item"


def _name(user_id: str, item_id: str) -> str:
    return f"{MCP_HOST_CONTAINER_PREFIX}{_fragment(user_id)[:12]}-{_fragment(item_id)[:12]}"


def _volume_name(user_id: str, item_id: str) -> str:
    return f"{MCP_HOST_RUNNER_DATA_VOLUME_PREFIX}{_fragment(user_id)[:12]}-{_fragment(item_id)[:12]}"


def _mb(value: Any) -> float | None:
    try:
        return round(float(value) / 1024 / 1024, 2)
    except (TypeError, ValueError):
        return None


def _safe_log_line(value: Any) -> str:
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    return value[:MCP_HOST_MAX_LOG_BYTES]


class PersonalDockerManager:
    def __init__(self) -> None:
        self.docker: aiodocker.Docker | None = None

    async def connect(self) -> None:
        self.docker = aiodocker.Docker()

    async def close(self) -> None:
        if self.docker:
            await self.docker.close()

    async def _find(self, name: str):
        assert self.docker is not None
        try:
            return await self.docker.containers.get(name)
        except aiodocker.exceptions.DockerError as exc:
            if exc.status == 404:
                return None
            raise

    async def _find_volume(self, name: str):
        assert self.docker is not None
        try:
            return await self.docker.volumes.get(name)
        except aiodocker.exceptions.DockerError as exc:
            if exc.status == 404:
                return None
            raise

    async def _ensure_volume(self, user_id: str, item_id: str, size_mb: int):
        assert self.docker is not None
        name = _volume_name(user_id, item_id)
        volume = await self._find_volume(name)
        if volume is not None:
            return volume
        if MCP_HOST_STORAGE_QUOTA_MODE not in {"enforced", "best_effort"}:
            raise RuntimeError("storage_quota_mode_invalid")
        labels = {
            "legion.mcp.managed": "true",
            "legion.mcp.user_id": _fragment(user_id),
            "legion.mcp.installed_item_id": _fragment(item_id),
            "legion.mcp.storage_limit_mb": str(size_mb),
            "legion.mcp.storage_quota_mode": MCP_HOST_STORAGE_QUOTA_MODE,
        }
        options = {MCP_HOST_VOLUME_SIZE_OPTION: f"{size_mb}m"}
        config: dict[str, Any] = {"Name": name, "Driver": MCP_HOST_VOLUME_DRIVER, "Labels": labels}
        if MCP_HOST_STORAGE_QUOTA_MODE == "enforced":
            config["DriverOpts"] = options
        else:
            # Local Docker development has no portable size enforcement. Keep
            # the requested allocation visible in labels and surface the mode.
            pass
        await self.docker.volumes.create(config)
        return await self._find_volume(name)

    async def _remove_container(self, container: Any) -> None:
        try:
            await container.stop(t=10)
        except Exception:
            pass
        await container.delete(force=True)

    async def _remove_volume(self, name: str) -> None:
        volume = await self._find_volume(name)
        if volume is None:
            return
        await volume.delete()

    def _legacy_config(self, req: StartRequest, name: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "Image": req.image,
            "Env": [f"{key}={value}" for key, value in req.env.items()],
            "HostConfig": {
                "Memory": req.mem_limit_mb * 1024 * 1024,
                "NanoCpus": req.cpu_limit_milli * 1_000_000,
                "PidsLimit": 256,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PortBindings": {f"{req.port}/tcp": [{"HostPort": str(req.port)}]},
                "NetworkMode": MCP_HOST_NETWORK,
            },
            "ExposedPorts": {f"{req.port}/tcp": {}},
            "Labels": {"legion.mcp.mode": "personal_image", "legion.mcp.item": name},
        }
        if req.args:
            config["Cmd"] = req.args
        return config

    def _managed_config(self, req: ManagedStdioStartRequest, profile: dict[str, Any], name: str, volume_name: str) -> dict[str, Any]:
        profile_json = json.dumps(profile, separators=(",", ":"), sort_keys=True)
        labels = {
            "legion.mcp.mode": "managed_stdio",
            "legion.mcp.item": name,
            "legion.mcp.user_id": _fragment(profile.get("user_id", "")),
            "legion.mcp.installed_item_id": _fragment(profile.get("installed_item_id", "")),
            "legion.mcp.memory_limit_mb": str(req.mem_limit_mb),
            "legion.mcp.storage_limit_mb": str(req.storage_limit_mb),
            "legion.mcp.correlation_id": profile["correlation_id"],
            "legion.mcp.install_egress_profile": MCP_HOST_INSTALL_EGRESS_PROFILE,
            "legion.mcp.runtime_egress_profile": MCP_HOST_RUNTIME_EGRESS_PROFILE,
        }
        return {
            "Image": MCP_HOST_RUNNER_IMAGE,
            "Cmd": ["node", "/app/src/index.js"],
            "User": "10001:10001",
            # This is a normalized profile and contains no secret values.
            "Env": [
                f"MCP_RUNNER_PROFILE_JSON={profile_json}",
                f"MCP_RUNNER_PORT={MCP_HOST_RUNNER_PORT}",
                "MCP_RUNNER_DATA_DIR=/data",
                f"MCP_RUNNER_NPM_REGISTRY={MCP_HOST_RUNNER_NPM_REGISTRY}",
                f"MCP_RUNNER_NPM_REGISTRY_HOSTS={MCP_HOST_RUNNER_NPM_REGISTRY_HOSTS}",
                f"MCP_RUNNER_ALLOW_INSECURE_REGISTRY={'1' if MCP_HOST_ALLOW_INSECURE_REGISTRY else '0'}",
                f"MCP_RUNNER_EGRESS_PROFILE={MCP_HOST_RUNTIME_EGRESS_PROFILE}",
            ],
            "ExposedPorts": {f"{MCP_HOST_RUNNER_PORT}/tcp": {}},
            "Labels": labels,
            "Healthcheck": {
                "Test": [
                    "CMD",
                    "node",
                    "-e",
                    "require('http').get('http://127.0.0.1:%d/healthz', r => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))" % MCP_HOST_RUNNER_PORT,
                ],
                "Interval": 5_000_000_000,
                "Timeout": 2_000_000_000,
                "Retries": 3,
            },
            "HostConfig": {
                "Memory": req.mem_limit_mb * 1024 * 1024,
                "NanoCpus": req.cpu_limit_milli * 1_000_000,
                "PidsLimit": 256,
                "Ulimits": [{"Name": "nofile", "Soft": 1024, "Hard": 4096}],
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "ReadonlyRootfs": True,
                "Privileged": False,
                "NetworkMode": MCP_HOST_RUNNER_NETWORK,
                # The only writable persistent path is the item-scoped volume.
                "Binds": [f"{volume_name}:/data:rw"],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
            },
        }

    async def start(self, user_id: str, item_id: str, req: StartRequest | ManagedStdioStartRequest) -> StatusResponse:
        if self.docker is None:
            raise RuntimeError("Docker is not connected")
        if isinstance(req, ManagedStdioStartRequest):
            return await self._start_managed(user_id, item_id, req)
        return await self._start_legacy(user_id, item_id, req)

    async def _start_legacy(self, user_id: str, item_id: str, req: StartRequest) -> StatusResponse:
        name = _name(user_id, item_id)
        old = await self._find(name)
        if old:
            info = await old.show()
            if info.get("State", {}).get("Running"):
                return await self.status(user_id, item_id, req.port)
            await old.delete(force=True)
        config = self._legacy_config(req, name)
        container = await self.docker.containers.create_or_replace(name, config)  # type: ignore[union-attr]
        await container.start()
        return StatusResponse(
            installed_item_id=item_id,
            user_id=user_id,
            status="running",
            mode="personal_image",
            container_id=container.id,
            endpoint=f"{MCP_HOST_PUBLIC_BASE_URL.rstrip('/') + '/' if MCP_HOST_PUBLIC_BASE_URL else 'http://'}{name}:{req.port}",
            port=req.port,
            memory_limit_mb=req.mem_limit_mb,
        )

    async def _start_managed(self, user_id: str, item_id: str, req: ManagedStdioStartRequest) -> StatusResponse:
        self._validate_managed_policy(req)
        secret_env = await self._consume_secret_lease(user_id, item_id, req)
        name = _name(user_id, item_id)
        volume_name = _volume_name(user_id, item_id)
        volume = await self._ensure_volume(user_id, item_id, req.storage_limit_mb)
        if volume is None:
            raise RuntimeError("storage_volume_create_failed")
        old = await self._find(name)
        if old:
            info = await old.show()
            if info.get("State", {}).get("Running"):
                return await self.status(user_id, item_id)
            await old.delete(force=True)
        correlation_id = req.correlation_id or _correlation_id(user_id, item_id)
        profile = {
            "mode": req.mode,
            "runtime_profile": req.runtime_profile,
            "transport": req.transport,
            "package": req.package,
            "command_args": req.command_args,
            "env_names": req.env_names,
            "shell": False,
            "resolved_artifact": req.resolved_artifact.model_dump() if req.resolved_artifact else None,
            "correlation_id": correlation_id,
            # Internal labels only; the runner ignores these fields.
            "user_id": user_id,
            "installed_item_id": item_id,
        }
        config = self._managed_config(req, profile, name, volume_name)
        container = await self.docker.containers.create_or_replace(name, config)
        await container.start()
        try:
            if req.env_names:
                await self._bootstrap_runner(
                    f"http://{name}:{MCP_HOST_RUNNER_PORT}",
                    secret_env,
                    correlation_id,
                )
        except Exception:
            await self._remove_container(container)
            raise
        finally:
            # Drop the host-side mapping immediately after the private
            # bootstrap request.  It is never part of Docker inspect data.
            secret_env.clear()
        endpoint = f"http://{name}:{MCP_HOST_RUNNER_PORT}/mcp"
        try:
            await self._wait_ready(endpoint, correlation_id)
            tools = await self._probe(endpoint, correlation_id)
            resolved_artifact = await self._runner_artifact(endpoint, correlation_id)
        except Exception as exc:
            await self._remove_container(container)
            return StatusResponse(
                installed_item_id=item_id,
                user_id=user_id,
                status="error",
                mode="managed_stdio",
                volume_name=volume_name,
                memory_limit_mb=req.mem_limit_mb,
                storage_limit_mb=req.storage_limit_mb,
                correlation_id=correlation_id,
                error_code="runner_not_ready",
                message="The managed MCP runner did not become ready.",
                log_tail=[_safe_log_line(str(exc))],
            )
        return await self.status(
            user_id,
            item_id,
            endpoint=endpoint,
            tools=tools,
            resolved_artifact=resolved_artifact,
        )

    @staticmethod
    def _validate_managed_policy(req: ManagedStdioStartRequest) -> None:
        try:
            parsed = urllib.parse.urlparse(MCP_HOST_RUNNER_NPM_REGISTRY)
            port = parsed.port
        except ValueError as exc:
            raise RuntimeError("registry_not_allowed") from exc
        allowed_hosts = {
            host.strip().lower()
            for host in MCP_HOST_RUNNER_NPM_REGISTRY_HOSTS.split(",")
            if host.strip()
        }
        valid_scheme = parsed.scheme == "https" or (MCP_HOST_ALLOW_INSECURE_REGISTRY and parsed.scheme == "http")
        if (
            not valid_scheme
            or parsed.username
            or parsed.password
            or parsed.hostname is None
            or parsed.hostname.lower() not in allowed_hosts
            or port not in (None, 443)
        ):
            raise RuntimeError("registry_not_allowed")
        if MCP_HOST_INSTALL_EGRESS_PROFILE != "npm_registry_only":
            raise RuntimeError("install_egress_policy_invalid")
        if MCP_HOST_RUNTIME_EGRESS_PROFILE not in {"mcp_runtime_v1", "none"}:
            raise RuntimeError("runtime_egress_policy_invalid")
        if req.mem_limit_mb > MCP_HOST_MAX_MEMORY_MB:
            raise RuntimeError("memory_quota_exceeded")
        if req.storage_limit_mb > MCP_HOST_MAX_STORAGE_MB:
            raise RuntimeError("storage_quota_exceeded")

    async def _consume_secret_lease(
        self, user_id: str, item_id: str, req: ManagedStdioStartRequest
    ) -> dict[str, str]:
        if not req.env_names:
            return {}
        if not req.secret_lease_token:
            raise RuntimeError("secret_lease_required")
        url = (
            f"{MCP_HOST_HUB_URL}/internal/secrets/personal-mcps/"
            f"{user_id}/{item_id}/secret-lease/consume"
        )
        payload = self._http_json(
            url,
            "POST",
            {"lease_token": req.secret_lease_token},
            req.correlation_id or _correlation_id(user_id, item_id),
            headers={"Authorization": f"Bearer {MCP_HOST_TOKEN}"},
            timeout=MCP_HOST_SECRETS_TIMEOUT_SEC,
        )
        raw = payload.get("env")
        if not isinstance(raw, dict) or set(raw) - set(req.env_names):
            raise RuntimeError("secret_lease_invalid")
        result: dict[str, str] = {}
        for name in req.env_names:
            if name in raw:
                value = raw[name]
                if not isinstance(value, str) or not value:
                    raise RuntimeError("secret_lease_invalid")
                result[name] = value
        return result

    async def _bootstrap_runner(
        self, base_url: str, secret_env: dict[str, str], correlation_id: str
    ) -> None:
        # The runner's HTTP server only starts listening after it resolves
        # the npm package version against the registry (a network call), so
        # a bootstrap attempt made the instant the container starts always
        # loses the race. Poll like `_wait_ready` does instead of a single
        # shot.
        url = f"{base_url}/bootstrap"
        deadline = time.monotonic() + MCP_HOST_PROBE_TIMEOUT_SEC
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                await asyncio.to_thread(
                    self._http_json, url, "POST", {"env": secret_env}, correlation_id
                )
                return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.25)
        raise RuntimeError("runner bootstrap timeout") from last_error

    async def _wait_ready(self, endpoint: str, correlation_id: str) -> None:
        health_url = endpoint.rsplit("/mcp", 1)[0] + "/readyz"
        deadline = time.monotonic() + MCP_HOST_PROBE_TIMEOUT_SEC
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                payload = await asyncio.to_thread(self._http_json, health_url, "GET", None, correlation_id)
                if payload.get("status") == "ready":
                    return
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.25)
        raise RuntimeError("runner readiness timeout") from last_error

    async def _probe(self, endpoint: str, correlation_id: str) -> ToolSnapshot:
        deadline = time.monotonic() + MCP_HOST_PROBE_TIMEOUT_SEC
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return await self._probe_once(endpoint, correlation_id)
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.25)
        raise RuntimeError("runner protocol probe timeout") from last_error

    async def _probe_once(self, endpoint: str, correlation_id: str) -> ToolSnapshot:
        initialize = {
            "jsonrpc": "2.0",
            "id": "legion-initialize",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "legion-mcp-host", "version": "1.0.0"},
            },
        }
        init = await asyncio.to_thread(self._http_json, endpoint, "POST", initialize, correlation_id)
        if not isinstance(init.get("result"), dict):
            raise RuntimeError("initialize probe failed")
        tools = await asyncio.to_thread(
            self._http_json,
            endpoint,
            "POST",
            {"jsonrpc": "2.0", "id": "legion-tools-list", "method": "tools/list", "params": {}},
            correlation_id,
        )
        result = tools.get("result")
        raw_tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw_tools, list):
            raise RuntimeError("tools/list probe failed")
        return _tool_snapshot(raw_tools)

    async def _runner_artifact(self, endpoint: str, correlation_id: str) -> ResolvedArtifact | None:
        try:
            payload = await asyncio.to_thread(
                self._http_json,
                endpoint.rsplit("/mcp", 1)[0] + "/metadata",
                "GET",
                None,
                correlation_id,
            )
        except Exception:
            return None
        artifact = payload.get("resolved_artifact")
        if not isinstance(artifact, dict):
            return None
        try:
            return ResolvedArtifact.model_validate(artifact)
        except Exception:
            return None

    @staticmethod
    def _http_json(
        url: str,
        method: str,
        body: dict[str, Any] | None,
        correlation_id: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
                **(headers or {}),
            },
        )
        with urllib.request.urlopen(
            request, timeout=timeout or MCP_HOST_PROBE_REQUEST_TIMEOUT_SEC
        ) as response:
            payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("runner returned invalid JSON")
            return payload

    async def stop(self, user_id: str, item_id: str) -> StatusResponse:
        container = await self._find(_name(user_id, item_id))
        if container:
            await self._remove_container(container)
        return await self.status(user_id, item_id)

    async def delete(self, user_id: str, item_id: str, confirmed: bool) -> StatusResponse:
        if not confirmed:
            raise ValueError("delete_confirmation_required")
        container = await self._find(_name(user_id, item_id))
        if container:
            await self._remove_container(container)
        volume_name = _volume_name(user_id, item_id)
        # Do not report deletion or release quota until Docker confirms that
        # the persistent volume was actually removed.
        await self._remove_volume(volume_name)
        return StatusResponse(installed_item_id=item_id, user_id=user_id, status="deleted", volume_name=volume_name)

    async def status(
        self,
        user_id: str,
        item_id: str,
        port: int | None = None,
        *,
        endpoint: str | None = None,
        tools: ToolSnapshot | None = None,
        resolved_artifact: ResolvedArtifact | None = None,
    ) -> StatusResponse:
        name = _name(user_id, item_id)
        container = await self._find(name)
        if not container:
            volume_name = _volume_name(user_id, item_id)
            volume = await self._find_volume(volume_name)
            if volume is not None:
                volume_info = await volume.show()
                volume_labels = volume_info.get("Labels", {}) or {}
                return StatusResponse(
                    installed_item_id=item_id,
                    user_id=user_id,
                    status="stopped",
                    mode="managed_stdio",
                    port=port,
                    volume_name=volume_name,
                    storage_limit_mb=_as_int(volume_labels.get("legion.mcp.storage_limit_mb")),
                )
            return StatusResponse(installed_item_id=item_id, user_id=user_id, status="stopped", port=port)
        info = await container.show()
        labels = info.get("Config", {}).get("Labels", {}) or info.get("Labels", {}) or {}
        mode = labels.get("legion.mcp.mode", "personal_image")
        state = info.get("State", {})
        memory_limit = _as_int(labels.get("legion.mcp.memory_limit_mb")) or _as_int(info.get("HostConfig", {}).get("Memory"), bytes_to_mb=True)
        storage_limit = _as_int(labels.get("legion.mcp.storage_limit_mb"))
        volume_name = _volume_name(user_id, item_id) if mode == "managed_stdio" else None
        if not state.get("Running"):
            exit_code = state.get("ExitCode")
            return StatusResponse(
                installed_item_id=item_id,
                user_id=user_id,
                status="error" if exit_code not in (None, 0) else "stopped",
                mode=mode if mode in {"managed_stdio", "personal_image"} else None,
                container_id=container.id,
                memory_limit_mb=memory_limit,
                storage_limit_mb=storage_limit,
                volume_name=volume_name,
                error_code="runner_exited" if exit_code not in (None, 0) else None,
            )
        started = state.get("StartedAt", "")
        uptime = None
        if started:
            try:
                uptime = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds()
            except (TypeError, ValueError):
                pass
        memory_usage, storage_used = await self._resource_usage(container, info)
        if mode == "managed_stdio":
            endpoint = endpoint or f"http://{name}:{MCP_HOST_RUNNER_PORT}/mcp"
            ready = await self._is_ready(endpoint)
            if ready and tools is None:
                try:
                    tools = await self._probe(endpoint, "status")
                except Exception:
                    ready = False
            runner_metrics = await self._runner_metrics(endpoint)
            if ready and resolved_artifact is None:
                resolved_artifact = await self._runner_artifact(endpoint, "status")
            if runner_metrics:
                metric_memory = _mb(runner_metrics.get("memory_usage_bytes"))
                metric_storage = _mb(runner_metrics.get("storage_used_bytes"))
                if metric_memory is not None:
                    memory_usage = metric_memory
                if metric_storage is not None:
                    storage_used = metric_storage
            return StatusResponse(
                installed_item_id=item_id,
                user_id=user_id,
                status="running" if not ready else "verified",
                mode="managed_stdio",
                container_id=container.id,
                endpoint=endpoint,
                internal_port=MCP_HOST_RUNNER_PORT,
                uptime_sec=round(uptime, 1) if uptime is not None else None,
                memory_usage_mb=memory_usage,
                memory_limit_mb=memory_limit,
                storage_used_mb=storage_used,
                storage_limit_mb=storage_limit,
                volume_name=volume_name,
                ready=ready,
                healthy=ready,
                tool_count=tools.count if tools else None,
                tools=tools,
                resolved_artifact=resolved_artifact,
                correlation_id=labels.get("legion.mcp.correlation_id"),
            )
        return StatusResponse(
            installed_item_id=item_id,
            user_id=user_id,
            status="running",
            mode="personal_image",
            container_id=container.id,
            endpoint=f"{MCP_HOST_PUBLIC_BASE_URL.rstrip('/') + '/' if MCP_HOST_PUBLIC_BASE_URL else 'http://'}{name}:{port}" if port else None,
            port=port,
            uptime_sec=round(uptime, 1) if uptime is not None else None,
            memory_usage_mb=memory_usage,
            memory_limit_mb=memory_limit,
        )

    async def _is_ready(self, endpoint: str) -> bool:
        try:
            payload = await asyncio.to_thread(self._http_json, endpoint.rsplit("/mcp", 1)[0] + "/readyz", "GET", None, "status")
            return payload.get("status") == "ready"
        except Exception:
            return False

    async def _runner_metrics(self, endpoint: str) -> dict[str, Any] | None:
        try:
            return await asyncio.to_thread(
                self._http_json,
                endpoint.rsplit("/mcp", 1)[0] + "/metrics",
                "GET",
                None,
                "metrics",
            )
        except Exception:
            return None

    async def _resource_usage(self, container: Any, info: dict[str, Any]) -> tuple[float | None, float | None]:
        memory_usage = _mb(info.get("MemoryStats", {}).get("Usage"))
        storage_used = _mb(info.get("SizeRw"))
        stats = getattr(container, "stats", None)
        if stats is not None:
            try:
                payload = await stats(stream=False)
                if isinstance(payload, dict):
                    memory_usage = _mb(payload.get("memory_stats", {}).get("usage")) or memory_usage
            except Exception:
                pass
        return memory_usage, storage_used


def _as_int(value: Any, *, bytes_to_mb: bool = False) -> int | None:
    try:
        number = int(value)
        return number // 1024 // 1024 if bytes_to_mb else number
    except (TypeError, ValueError):
        return None


def _correlation_id(user_id: str, item_id: str) -> str:
    return f"mcp-{_fragment(user_id)[:12]}-{_fragment(item_id)[:12]}-{int(time.time())}"


def _tool_snapshot(raw_tools: list[Any]) -> ToolSnapshot:
    """Return a bounded, deterministic tool catalog and fingerprint."""

    safe_tools: list[dict[str, Any]] = []
    names: list[str] = []
    for raw in raw_tools[:512]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name or len(name) > 128:
            continue
        safe_tools.append(
            {
                "name": name,
                "description": str(raw.get("description") or "")[:2000],
                "inputSchema": raw.get("inputSchema") or raw.get("input_schema") or {},
            }
        )
        names.append(name)
    canonical = json.dumps(safe_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ToolSnapshot(count=len(names), names=names, fingerprint=fingerprint)
