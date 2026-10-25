"""Durable registry and state machine for declared workspace services."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.workspace import (
    Workspace,
    WorkspaceEnvironment,
    WorkspaceManagedProcess,
    WorkspaceServiceEvent,
)
from app.websockets.manager import manager

_SLUG = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_KINDS = {"web", "api", "worker"}
_ENVIRONMENT_PROFILES = {"python-pip", "node-npm", "node-pnpm", "node-yarn", "none"}
_RUNTIME_TRANSITIONS = {
    "stopped": {"starting", "stopping", "error"},
    "starting": {"ready", "error", "stopping", "stopped"},
    "ready": {"stopping", "error", "degraded"},
    "degraded": {"starting", "stopping", "error", "ready"},
    "stopping": {"stopped", "error"},
    "error": {"starting", "stopping", "stopped"},
}
_LOCKS: dict[UUID, asyncio.Lock] = {}


class ServiceValidationError(ValueError):
    pass


def services_enabled() -> bool:
    return os.getenv("WORKSPACE_SERVICES_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def port_range() -> tuple[int, int]:
    raw = os.getenv("WORKSPACE_SERVICE_PORT_RANGE", "3000-3999")
    try:
        first, last = (int(value) for value in raw.split("-", 1))
    except (AttributeError, ValueError):
        raise ServiceValidationError("WORKSPACE_SERVICE_PORT_RANGE must be first-last")
    if not (1 <= first <= last <= 65535):
        raise ServiceValidationError("invalid workspace service port range")
    return first, last


def _workspace_relative_path(value: object, field: str) -> str:
    path = str(value or "")
    if not path or path.startswith("/") or "\\" in path or (
        path != "." and any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ServiceValidationError(f"{field} must be a workspace-relative path")
    return path


def normalize_environment(raw: object) -> dict[str, Any]:
    """Validate a reproducible, built-in dependency installation profile.

    The actual installer is selected inside workspaced, never supplied as an
    agent-controlled shell command.  This keeps the service declaration
    stack-agnostic without turning Browser start into an arbitrary executor.
    """
    if not isinstance(raw, dict):
        raise ServiceValidationError("environment is required for new service declarations")
    slug = str(raw.get("slug") or "")
    if not _SLUG.fullmatch(slug):
        raise ServiceValidationError("environment.slug must be lowercase kebab-case (1-63 chars)")
    profile = str(raw.get("profile") or "")
    if profile not in _ENVIRONMENT_PROFILES:
        raise ServiceValidationError(
            "environment.profile must be python-pip, node-npm, node-pnpm, node-yarn, or none"
        )
    root = _workspace_relative_path(raw.get("root") or ".", "environment.root")
    manifests_raw = raw.get("manifests") or []
    if not isinstance(manifests_raw, list) or any(not isinstance(item, str) for item in manifests_raw):
        raise ServiceValidationError("environment.manifests must be a list of workspace-relative paths")
    manifests = sorted({_workspace_relative_path(item, "environment.manifests") for item in manifests_raw})
    if len(manifests) != len(manifests_raw):
        raise ServiceValidationError("environment.manifests cannot contain duplicates")
    if profile == "python-pip":
        if not manifests or any(not path.endswith(".txt") for path in manifests):
            raise ServiceValidationError("python-pip requires one or more requirements *.txt manifests")
    elif profile == "node-npm" and manifests != ["package-lock.json"]:
        raise ServiceValidationError("node-npm requires manifests: [package-lock.json]")
    elif profile == "node-pnpm" and manifests != ["pnpm-lock.yaml"]:
        raise ServiceValidationError("node-pnpm requires manifests: [pnpm-lock.yaml]")
    elif profile == "node-yarn" and manifests != ["yarn.lock"]:
        raise ServiceValidationError("node-yarn requires manifests: [yarn.lock]")
    elif profile == "none" and manifests:
        raise ServiceValidationError("environment.profile none cannot declare manifests")
    return {"slug": slug, "profile": profile, "root": root, "manifests": manifests}


def normalize_declaration(raw: dict[str, Any]) -> dict[str, Any]:
    slug = str(raw.get("slug") or "")
    if not _SLUG.fullmatch(slug):
        raise ServiceValidationError("slug must be lowercase kebab-case (1-63 chars)")
    kind = str(raw.get("kind") or "")
    if kind not in _KINDS:
        raise ServiceValidationError("kind must be web, api, or worker")
    title = str(raw.get("title") or "").strip()
    if not title or len(title) > 160:
        raise ServiceValidationError("title is required and must be at most 160 characters")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ServiceValidationError("argv must be a non-empty list of strings")
    cwd = str(raw.get("cwd") or "")
    # ``.`` is the canonical workspace root.  It must be accepted because
    # projects such as a root-level Django app keep their entry point
    # (manage.py) directly under /workspace.  Keep rejecting dot segments in
    # nested paths so the stored declaration remains unambiguous.
    if not cwd or cwd.startswith("/") or "\\" in cwd or (
        cwd != "." and any(part in {"", ".", ".."} for part in cwd.split("/"))
    ):
        raise ServiceValidationError("cwd must be a workspace-relative path")
    env = raw.get("env") or {}
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ServiceValidationError("env must be a string-to-string mapping")
    secrets = raw.get("secrets") or []
    if not isinstance(secrets, list) or any(not isinstance(s, str) or not s for s in secrets):
        raise ServiceValidationError("secrets must be a list of Room secret names")
    if len(secrets) > 20:
        raise ServiceValidationError("at most 20 secrets per service")
    listen = dict(raw.get("listen") or {})
    if listen.get("protocol", "http") not in {"http", "tcp"}:
        raise ServiceValidationError("listen.protocol must be http or tcp")
    requested_port = listen.get("port", "auto")
    if requested_port != "auto" and (not isinstance(requested_port, int) or not 1 <= requested_port <= 65535):
        raise ServiceValidationError("listen.port must be auto or a TCP port")
    readiness = dict(raw.get("readiness") or {})
    readiness_type = readiness.get("type", "tcp")
    if readiness_type not in {"tcp", "http"}:
        raise ServiceValidationError("readiness.type must be tcp or http")
    if readiness_type == "http" and not str(readiness.get("path") or "/").startswith("/"):
        raise ServiceValidationError("readiness.path must start with /")
    timeout = int(readiness.get("timeout_sec", 45))
    if not 1 <= timeout <= 300:
        raise ServiceValidationError("readiness.timeout_sec must be between 1 and 300")
    dependencies = list(raw.get("depends_on") or [])
    if any(not isinstance(dep, str) or not _SLUG.fullmatch(dep) or dep == slug for dep in dependencies):
        raise ServiceValidationError("depends_on must contain distinct other service slugs")
    if len(set(dependencies)) != len(dependencies):
        raise ServiceValidationError("depends_on cannot contain duplicates")
    restart_policy = raw.get("restart_policy", "manual")
    if restart_policy not in {"manual", "on_workspace_restore"}:
        raise ServiceValidationError("invalid restart_policy")
    normalized = {
        "slug": slug, "title": title, "kind": kind, "argv": argv, "cwd": cwd,
        "env": env, "secrets": secrets,
        "listen": {"protocol": listen.get("protocol", "http"), "port": requested_port, "host": "127.0.0.1"},
        "readiness": {**readiness, "type": readiness_type, "timeout_sec": timeout},
        "default_path": str(raw.get("default_path") or "/"), "depends_on": dependencies,
        "restart_policy": restart_policy, "declared_by": str(raw.get("declared_by") or "agent"),
        "environment": normalize_environment(raw.get("environment")),
    }
    if not normalized["default_path"].startswith("/"):
        raise ServiceValidationError("default_path must start with /")
    return normalized


def config_hash(declaration: dict[str, Any]) -> str:
    body = json.dumps(declaration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


class WorkspaceServiceRegistry:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    @staticmethod
    def lock_for(workspace_id: UUID) -> asyncio.Lock:
        return _LOCKS.setdefault(workspace_id, asyncio.Lock())

    async def list(self, workspace_id: UUID) -> list[WorkspaceManagedProcess]:
        """Return only records declared through the service registry.

        ``workspace_managed_process`` also contains legacy Runs rows. They
        have no configuration hash (and frequently only a shell command), so
        presenting them as Browser services would make the new supervisor try
        to launch an incomplete declaration.
        """
        rows = await self.db.execute(
            select(WorkspaceManagedProcess).where(
                WorkspaceManagedProcess.workspace_id == workspace_id,
                WorkspaceManagedProcess.config_hash != "",
            )
            .order_by(WorkspaceManagedProcess.slug.asc())
        )
        return list(rows.scalars().all())

    async def get(self, workspace_id: UUID, slug: str) -> WorkspaceManagedProcess | None:
        rows = await self.db.execute(select(WorkspaceManagedProcess).where(
            WorkspaceManagedProcess.workspace_id == workspace_id,
            WorkspaceManagedProcess.slug == slug,
            WorkspaceManagedProcess.config_hash != "",
        ))
        return rows.scalar_one_or_none()

    async def _get_any(self, workspace_id: UUID, slug: str) -> WorkspaceManagedProcess | None:
        """Find a legacy row too, solely so declare can migrate it in place."""
        rows = await self.db.execute(select(WorkspaceManagedProcess).where(
            WorkspaceManagedProcess.workspace_id == workspace_id,
            WorkspaceManagedProcess.slug == slug,
        ))
        return rows.scalar_one_or_none()

    async def get_environment(self, workspace_id: UUID, slug: str) -> WorkspaceEnvironment | None:
        rows = await self.db.execute(select(WorkspaceEnvironment).where(
            WorkspaceEnvironment.workspace_id == workspace_id,
            WorkspaceEnvironment.slug == slug,
        ))
        return rows.scalar_one_or_none()

    async def environment_for(self, workspace_id: UUID, service: WorkspaceManagedProcess) -> WorkspaceEnvironment | None:
        if service.environment_id is None:
            return None
        rows = await self.db.execute(select(WorkspaceEnvironment).where(
            WorkspaceEnvironment.workspace_id == workspace_id,
            WorkspaceEnvironment.id == service.environment_id,
        ))
        return rows.scalar_one_or_none()

    async def _declare_environment(
        self, workspace_id: UUID, declaration: dict[str, Any]
    ) -> tuple[WorkspaceEnvironment, bool]:
        digest = config_hash(declaration)
        environment = await self.get_environment(workspace_id, declaration["slug"])
        changed = environment is None or environment.config_hash != digest
        if environment is None:
            environment = WorkspaceEnvironment(workspace_id=workspace_id, **declaration, config_hash=digest)
            self.db.add(environment)
            await self.db.flush()
            return environment, True
        if not changed:
            return environment, False
        active = await self.db.execute(select(WorkspaceManagedProcess.id).where(
            WorkspaceManagedProcess.environment_id == environment.id,
            WorkspaceManagedProcess.runtime_state.in_({"starting", "ready", "degraded"}),
        ))
        if active.first() is not None:
            raise ServiceValidationError("environment is in use by a running service; stop it before changing configuration")
        environment.profile = declaration["profile"]
        environment.root = declaration["root"]
        environment.manifests = declaration["manifests"]
        environment.config_hash = digest
        environment.state = "unprepared"
        environment.prepared_fingerprint = environment.runtime_fingerprint = None
        environment.runtime_env = {}
        environment.last_error = None
        environment.updated_at = datetime.now(timezone.utc)
        self.db.add(environment)
        return environment, True

    async def declare(self, workspace_id: UUID, raw: dict[str, Any], source: str = "agent") -> tuple[WorkspaceManagedProcess, bool]:
        declaration = normalize_declaration(raw)
        environment_declaration = declaration.pop("environment")
        digest = config_hash({**declaration, "environment": environment_declaration})
        # A new declaration with an old Run's slug upgrades that row in place.
        # This preserves the table's legacy unique ``(workspace_id, name)``
        # constraint without making the old shell command executable.
        service = await self._get_any(workspace_id, declaration["slug"])
        changed = service is None or service.config_hash != digest
        if service is not None and changed and service.runtime_state in {"starting", "ready", "degraded"}:
            raise ServiceValidationError("service is running; stop it or use explicit restart before changing configuration")
        environment, _environment_changed = await self._declare_environment(workspace_id, environment_declaration)
        if service is None:
            service = WorkspaceManagedProcess(
                workspace_id=workspace_id, name=declaration["slug"], command="",
                environment_id=environment.id, **declaration,
            )
        else:
            for key, value in declaration.items():
                setattr(service, key, value)
            service.name = declaration["slug"]  # legacy Runs compatibility
            service.environment_id = environment.id
        service.config_hash = digest
        service.schema_version = 1
        service.updated_at = datetime.now(timezone.utc)
        self.db.add(service)
        await self.db.flush()
        await self.event(service, "workspace.service.declared", source, {"changed": changed})
        return service, changed

    async def transition(self, service: WorkspaceManagedProcess, target: str, source: str, payload: dict[str, Any] | None = None) -> None:
        current = service.runtime_state
        if target != current and target not in _RUNTIME_TRANSITIONS.get(current, set()):
            raise ServiceValidationError(f"invalid service transition {current} -> {target}")
        service.runtime_state = target
        service.status = {"ready": "running", "degraded": "error"}.get(target, target)
        service.updated_at = datetime.now(timezone.utc)
        if target == "ready":
            service.last_ready_at = service.updated_at
            service.last_error = None
        if target == "starting":
            service.last_started_at = service.updated_at
        # Telemetry belongs next to the durable transition, so it covers both
        # Browser/API and agent-tool entry points without exposing identifiers
        # as Prometheus labels.
        from app.core.metrics import (
            record_workspace_service_failure,
            record_workspace_service_start_latency,
        )
        if target == "ready" and service.last_started_at is not None:
            started = service.last_started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            record_workspace_service_start_latency(
                (service.updated_at - started).total_seconds()
            )
        elif target == "error":
            record_workspace_service_failure(str((payload or {}).get("code") or "service_error"))
        self.db.add(service)
        await self.event(service, f"workspace.service.{target}", source, payload or {})

    async def event(self, service: WorkspaceManagedProcess, event_type: str, source: str, payload: dict[str, Any]) -> None:
        self.record_event(service, event_type, source, payload)
        await self.publish(service, event_type)

    def record_event(self, service: WorkspaceManagedProcess, event_type: str, source: str, payload: dict[str, Any]) -> None:
        # Do not persist argv/env or arbitrary stderr in the UI-safe ledger.
        safe_payload = {key: value for key, value in payload.items() if key not in {"argv", "env", "command", "secrets"}}
        self.db.add(WorkspaceServiceEvent(workspace_id=service.workspace_id, service_id=service.id,
                                          event_type=event_type, source=source, payload=safe_payload))

    async def publish(self, service: WorkspaceManagedProcess, event_type: str) -> None:
        # The catalog uses the room's existing WebSocket connection.  Keep this
        # frame deliberately small and free of launch configuration/log output;
        # a client that receives it refreshes the authoritative REST catalog.
        # Lifecycle cascades call this after their commit, so that refresh
        # reads the new state.
        room_id = await self.db.scalar(
            select(Workspace.room_id).where(Workspace.id == service.workspace_id)
        )
        if room_id is None:
            return
        listen = service.listen or {}
        try:
            await manager.broadcast({
                "type": "workspace_service",
                "event_type": event_type,
                "room_id": str(room_id),
                "service": {
                    "id": str(service.id), "slug": service.slug, "title": service.title,
                    "kind": service.kind, "desired_state": service.desired_state,
                    "runtime_state": service.runtime_state,
                    "default_path": service.default_path,
                    "assigned_port": listen.get("assigned_port"),
                    "last_error": service.last_error,
                },
            }, room_id)
        except Exception:
            # A disconnected UI must never prevent a supervisor state change.
            pass


async def reconcile_service_runtime(
    room_id: UUID, workspace: Workspace, registry: WorkspaceServiceRegistry, db: AsyncSession
) -> None:
    """Make durable state truthful after either backend or workspaced restarts."""
    if workspace.status != "running":
        return
    # Snapshot the pid probed for each service so a slow, concurrent probe can
    # be recognised as stale when its verdict is applied below.
    probed_pid: dict[str, int] = {}
    records = []
    for service in await registry.list(workspace.id):
        port = (service.listen or {}).get("assigned_port")
        # A supervisor restart can happen between a successful launch and the
        # next catalog read.  Keep reconciling a desired-running service even
        # if a failed duplicate Start temporarily marked it as ``error``.
        if service.desired_state == "running" and service.pid and port:
            probed_pid[str(service.id)] = int(service.pid)
            records.append({
                "service_id": str(service.id), "slug": service.slug, "pid": service.pid,
                "pgid": service.pgid, "port": port, "log_path": service.log_path,
            })
    if not records:
        return
    try:
        from app.core import workspaced_client as wsd
        response = await wsd.reconcile_workspace_services(str(room_id), records)
    except httpx.HTTPError:
        # A temporary supervisor outage must not turn a running service into a
        # durable error; the next catalog/status request will retry.
        return
    alive_by_id = {str(item.get("service_id")): bool(item.get("alive")) for item in response.get("services", [])}
    # Apply verdicts under the same per-workspace lock that start/restart hold,
    # re-reading fresh rows first.  Without this, an aggressive catalog poll can
    # probe a service's *old* pid, and by the time its verdict commits a
    # concurrent start/restart has already published a new, healthy pid — the
    # verdict then demotes a live service to ``error`` and, because the winning
    # session's ``last_error = None`` is a no-op it never rewrites, strands a
    # stale "process is no longer running" message on a service that is actually
    # ``ready`` (observed exactly this way after a workspace restart).  Holding
    # the lock serialises against the in-flight start; the probed-pid guard
    # drops any verdict whose target pid the row no longer carries.
    async with registry.lock_for(workspace.id):
        # Re-read committed rows without ``expire_all`` — expiring here makes a
        # later synchronous attribute access lazy-load under async SQLAlchemy and
        # raise MissingGreenlet.  ``populate_existing`` refreshes the mapped
        # instances from this awaited query instead, so ``service.pid`` below is
        # the freshly committed value a concurrent start may have just written.
        fresh = await db.execute(
            select(WorkspaceManagedProcess)
            .where(
                WorkspaceManagedProcess.workspace_id == workspace.id,
                WorkspaceManagedProcess.config_hash != "",
            )
            .execution_options(populate_existing=True)
        )
        changed = False
        for service in fresh.scalars().all():
            alive = alive_by_id.get(str(service.id))
            if alive is None or (service.pid or None) != probed_pid.get(str(service.id)):
                continue
            if alive is True and service.desired_state == "running" and service.runtime_state == "error":
                # ``error -> ready`` is intentionally two legal transitions;
                # the live process is confirmed by workspaced after its own
                # in-memory runtime map has been rebuilt.
                await registry.transition(service, "starting", "reconcile", {"code": "runtime_recovered"})
                await registry.transition(service, "ready", "reconcile", {"code": "runtime_recovered"})
                changed = True
            elif alive is False and service.runtime_state in {"starting", "ready", "degraded"}:
                service.pid = service.pgid = None
                service.last_error = "service process is no longer running"
                await registry.transition(service, "error", "reconcile", {"code": "orphan_process"})
                changed = True
        if changed:
            await db.commit()
