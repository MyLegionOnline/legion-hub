"""Durable workspace-service tool for agent-managed web/API/worker processes.

User-visible servers must be declared here instead of being launched through a
detached shell. The backend registry is authoritative; this tool calls the
internal workspaced supervisor only after persisting a validated declaration.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.future import select

from app.core.secrets.room_masking import (
    bg_secrets,
    forget_bg_secrets,
    mask_room_secrets,
    remember_bg_secrets,
)
from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_MAX_REQUEST_BODY = 16_000
_DEFAULT_LOG_LIMIT = 4_000
_MAX_LOG_LIMIT = 16_000


class WorkspaceServiceToolError(ValueError):
    """An error that can be returned to an agent with a concrete recovery."""


class WorkspaceServiceTool(Tool):
    name = "workspace_service"
    description = (
        "Declare and operate a durable user-visible workspace service (web, API, or worker). "
        "Use this instead of bash(background=true), nohup, or manually choosing a port for "
        "an application a user should open. Call action=list FIRST to see services already "
        "declared in this workspace — if one already matches the app you are about to expose, "
        "reuse its exact slug (start/restart it) instead of declaring a new one; declaring under "
        "a different slug for the same app creates a duplicate entry in the user's Browser tab. "
        "Declare with argv/cwd/listen/readiness, then start, wait_ready, or request to verify it. "
        "Returns compact state; use logs for a bounded tail."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "declare", "prepare", "start", "stop", "restart", "status", "logs", "wait_ready", "request"],
                "description": "Operation to perform. Use list first to check for an existing service before declare.",
            },
            "slug": {"type": "string", "description": "Declared service slug."},
            "service": {
                "type": "object",
                "description": "Required only for declare; shell command strings are not accepted.",
                "properties": {
                    "slug": {"type": "string"}, "title": {"type": "string"},
                    "kind": {"type": "string", "enum": ["web", "api", "worker"]},
                    "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "cwd": {"type": "string", "description": "Path relative to the workspace root."},
                    "env": {"type": "object", "additionalProperties": {"type": "string"}},
                    "secrets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Names of Room secrets (from the Room's Secrets & variables tab) to "
                            "resolve into this service's env each time it starts. You only ever see "
                            "the name, never the value. Room variables are already always in env; "
                            "this is only for secrets. Not usable for a deploy SSH key — those exist "
                            "only for a single bash exec, not a long-running service."
                        ),
                    },
                    "listen": {
                        "type": "object",
                        "properties": {
                            "protocol": {"type": "string", "enum": ["http", "tcp"]},
                            "port": {
                                "oneOf": [{"type": "string", "enum": ["auto"]}, {"type": "integer"}],
                                "description": (
                                    "Prefer \"auto\". If set, argv MUST forward the assigned port "
                                    "via the $PORT env var (e.g. --port $PORT); never hardcode a "
                                    "port that only happens to match the tool's own default (like "
                                    "Vite's 5173) without also passing --port $PORT, or the process "
                                    "will ignore the reserved port and become unrecoverable."
                                ),
                            },
                        },
                    },
                    "readiness": {"type": "object"}, "default_path": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "restart_policy": {"type": "string", "enum": ["manual", "on_workspace_restore"]},
                    "environment": {
                        "type": "object",
                        "description": "Required reproducible dependency environment; selected profiles never accept a shell bootstrap command.",
                        "properties": {
                            "slug": {"type": "string"},
                            "profile": {"type": "string", "enum": ["python-pip", "node-npm", "node-pnpm", "node-yarn", "none"]},
                            "root": {"type": "string", "description": "Project path relative to workspace root."},
                            "manifests": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["slug", "profile", "root", "manifests"],
                    },
                },
                "required": ["slug", "title", "kind", "argv", "cwd", "listen", "readiness", "environment"],
            },
            "path": {
                "type": "string",
                "default": "/",
                "description": (
                    "HTTP path for request, e.g. \"/article/first-article/\". "
                    "A leading slash is added automatically if missing."
                ),
            },
            "query": {"type": "string", "description": "Query string without leading ?."},
            "cursor": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LOG_LIMIT, "default": _DEFAULT_LOG_LIMIT},
            "log_target": {"type": "string", "enum": ["service", "environment"], "default": "service"},
            "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 300, "default": 45},
        },
        "required": ["action"],
    }
    # ChatMaster may use this only for a direct user request to open an
    # already-existing app. It cannot edit project files through this tool;
    # substantive implementation still belongs to an assigned agent task.
    roles: frozenset[str] = frozenset({"agent", "cm"})
    role_descriptions = {
        "cm": (
            "Handle a direct user request to open, start, stop, or inspect an already-existing "
            "workspace app without creating a task. Declare it with argv/cwd/readiness, then start it. "
            "Do not use this to implement or modify a project."
        ),
    }
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.workspace_services import ServiceValidationError, services_enabled

        if not services_enabled():
            return (
                "Error: workspace services are disabled on this Legion deployment "
                "(backend setting WORKSPACE_SERVICES_ENABLED). This is not a room variable or "
                "secret, and you cannot change it — do not retry workspace_service. To run a "
                "dev server or other long-running process, use bash with background=true "
                "(check it with bash_output, stop it with bash_kill) and preview_fetch to reach "
                "its port."
            )
        action = str(tool_input.get("action") or "").strip()
        allowed = {"list", "declare", "prepare", "start", "stop", "restart", "status", "logs", "wait_ready", "request"}
        if action not in allowed:
            return "Error: action must be list, declare, prepare, start, stop, restart, status, logs, wait_ready, or request"
        try:
            if action == "list":
                return await self._list(ctx)
            if action == "declare":
                return await self._declare(ctx, tool_input)
            return await self._operate(ctx, action, tool_input)
        except (WorkspaceServiceToolError, ServiceValidationError) as exc:
            return f"Error: workspace_service {action} failed: {exc}"
        except httpx.HTTPError as exc:
            return (
                f"Error: workspace_service {action} could not reach the service supervisor: {exc}. "
                "Check that the workspace is running, then retry once."
            )

    async def _workspace(self, ctx: ToolContext):
        from app.core.workspace_lifecycle import _get_or_create_workspace

        return await _get_or_create_workspace(ctx.room_id, ctx.db)

    @staticmethod
    def _summary(service, *, detail: str | None = None) -> str:
        lines = [
            f"slug: {service.slug}",
            f"state: {service.runtime_state}",
            f"health: {'ready' if service.runtime_state == 'ready' else service.runtime_state}",
            f"browser_label: {service.title}",
        ]
        if service.last_error:
            lines.append(f"error: {service.last_error}")
        if detail:
            lines.append(detail)
        return "\n".join(lines)

    @staticmethod
    def _slug(tool_input: dict[str, Any]) -> str:
        slug = str(tool_input.get("slug") or "").strip()
        if not slug:
            raise WorkspaceServiceToolError("slug is required; call declare first or pass an existing service slug")
        return slug

    async def _declare(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.workspace_services import WorkspaceServiceRegistry

        raw = tool_input.get("service")
        if not isinstance(raw, dict):
            raise WorkspaceServiceToolError("service is required for declare")
        slug = tool_input.get("slug")
        if slug and str(slug) != str(raw.get("slug") or ""):
            raise WorkspaceServiceToolError("slug must equal service.slug")
        if "command" in raw:
            raise WorkspaceServiceToolError("command is not supported; provide argv as a list of executable arguments")
        workspace = await self._workspace(ctx)
        registry = WorkspaceServiceRegistry(ctx.db)
        service, changed = await registry.declare(
            workspace.id, {**raw, "declared_by": "agent"}, source=f"agent:{ctx.session_id}"
        )
        await ctx.db.commit()
        return self._summary(service, detail=f"declaration: {'updated' if changed else 'unchanged'}")

    async def _list(self, ctx: ToolContext) -> str:
        from app.core.workspace_services import WorkspaceServiceRegistry

        workspace = await self._workspace(ctx)
        registry = WorkspaceServiceRegistry(ctx.db)
        services = await registry.list(workspace.id)
        if not services:
            return "No services declared yet in this workspace."
        lines = [f"{len(services)} service(s) declared in this workspace:"]
        for service in services:
            port = (service.listen or {}).get("assigned_port")
            lines.append(
                f"- slug: {service.slug} | title: {service.title} | kind: {service.kind} "
                f"| state: {service.runtime_state}" + (f" | port: {port}" if port else "")
            )
        return "\n".join(lines)

    async def _service(self, ctx: ToolContext, slug: str):
        from app.core.workspace_services import WorkspaceServiceRegistry

        workspace = await self._workspace(ctx)
        registry = WorkspaceServiceRegistry(ctx.db)
        service = await registry.get(workspace.id, slug)
        if service is None:
            raise WorkspaceServiceToolError(f"service {slug!r} is not declared; declare it before starting or inspecting it")
        return workspace, registry, service

    async def _operate(self, ctx: ToolContext, action: str, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd

        workspace, registry, service = await self._service(ctx, self._slug(tool_input))
        if action == "status":
            return self._summary(service)
        if action == "logs":
            return await self._logs(ctx, workspace, registry, service, tool_input)
        if action == "prepare":
            await self._ensure_workspace_started(ctx)
            return await self._prepare(ctx, workspace, registry, service)
        if action == "request":
            return await self._request(ctx, service, tool_input)
        if action == "stop":
            await self._stop(ctx, service, registry, wsd)
            return self._summary(service)
        if action == "restart":
            await self._stop(ctx, service, registry, wsd)
            await self._ensure_workspace_started(ctx)
            await self._start_tree(ctx, workspace, registry, service, wsd, set())
            return self._summary(
                service,
                detail="remediation: inspect logs, fix cwd/argv/readiness, then restart"
                if service.runtime_state == "error" else None,
            )
        if action == "start":
            await self._ensure_workspace_started(ctx)
            await self._start_tree(ctx, workspace, registry, service, wsd, set())
            return self._summary(
                service,
                detail="remediation: inspect logs, fix cwd/argv/readiness, then restart"
                if service.runtime_state == "error" else None,
            )
        if action == "wait_ready":
            return await self._wait_ready(ctx, service, registry, tool_input, wsd)
        raise AssertionError(f"unexpected action: {action}")

    async def _ensure_workspace_started(self, ctx: ToolContext) -> None:
        from app.core.tools._workspace_wake import ensure_workspace_running

        _ws, err = await ensure_workspace_running(ctx, reason="workspace_service")
        if err:
            raise WorkspaceServiceToolError(err.removeprefix("Error: "))

    async def _start_tree(self, ctx, workspace, registry, service, wsd, visiting: set[str]) -> None:
        if service.runtime_state == "ready":
            return
        if service.slug in visiting:
            raise WorkspaceServiceToolError(
                f"dependency cycle detected at {service.slug!r}; fix depends_on declarations before retrying"
            )
        visiting.add(service.slug)
        for dependency_slug in service.depends_on or []:
            dependency = await registry.get(workspace.id, dependency_slug)
            if dependency is None:
                raise WorkspaceServiceToolError(
                    f"dependency {dependency_slug!r} is not declared; declare it before starting {service.slug!r}"
                )
            await self._start_tree(ctx, workspace, registry, dependency, wsd, visiting)
            if dependency.runtime_state != "ready":
                raise WorkspaceServiceToolError(
                    f"dependency {dependency_slug!r} is not ready; inspect its logs and fix it before starting {service.slug!r}"
                )
        visiting.remove(service.slug)
        if service.runtime_state == "ready":
            return
        await registry.transition(service, "starting", f"agent:{ctx.session_id}")
        service.desired_state = "running"
        service.last_accessed_at = datetime.now(timezone.utc)
        workspace.last_activity_at = service.last_accessed_at
        ctx.db.add(workspace)
        await ctx.db.commit()
        from app.core.workspace_environments import WorkspaceEnvironmentError, prepare_service_environment
        try:
            runtime_env, _environment = await prepare_service_environment(
                str(ctx.room_id), workspace.id, service, registry, ctx.db,
            )
        except WorkspaceEnvironmentError as exc:
            service.last_error = f"environment preparation failed ({exc.code}): {exc}"
            await registry.transition(service, "error", f"agent:{ctx.session_id}", {"code": exc.code})
            await ctx.db.commit()
            return
        try:
            room_vars, resolved_secrets = await self._resolve_service_secrets(ctx, service)
        except WorkspaceServiceToolError as exc:
            service.last_error = f"secret resolution failed: {exc}"
            await registry.transition(service, "error", f"agent:{ctx.session_id}", {"code": "secret_not_available"})
            await ctx.db.commit()
            return
        result = await wsd.start_workspace_service(str(ctx.room_id), {
            "service_id": str(service.id), "slug": service.slug, "argv": service.argv,
            "cwd": f"/workspace/{service.cwd}",
            "env": {**service.env, **runtime_env, **room_vars, **resolved_secrets},
            "port": (service.listen or {}).get("port", "auto"),
            "readiness": service.readiness,
            "timeout_sec": int((service.readiness or {}).get("timeout_sec", 45)),
        })
        # Only the secret *values* are worth masking later — Room variables
        # are non-secret by definition (plan §2), and masking them too would
        # blindly redact any coincidental match in unrelated log output.
        remember_bg_secrets(f"service:{service.id}", resolved_secrets)
        listen = dict(service.listen or {})
        listen["assigned_port"] = result.get("port")
        service.listen = listen
        service.pid, service.pgid, service.log_path = result.get("pid"), result.get("pgid"), result.get("log_path")
        if result.get("state") == "ready":
            await registry.transition(service, "ready", f"agent:{ctx.session_id}")
        else:
            service.last_error = result.get("error") or "service failed readiness"
            await registry.transition(service, "error", f"agent:{ctx.session_id}", {
                "code": result.get("error_code", "readiness_failed")
            })
        await ctx.db.commit()

    async def _resolve_service_secrets(self, ctx: ToolContext, service) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve Room variables (always) and ``service.secrets`` (by name).

        internal design notes Stage 6 — mirrors ``_bash.py``'s
        ``_build_exec_env``: Room variables always ride along (non-secret,
        plan §3.1), declared secret *names* are resolved under the Room
        owner (no human actor at service-start time, same reasoning as bash).
        Returns ``(room_vars, resolved_secrets)`` so the caller can keep only
        the latter maskable.
        """
        from sqlalchemy.future import select

        from app.core.room_config import effective_variables
        from app.core.secrets import room_secrets
        from app.core.secrets.service import SecretService
        from app.models.room import Room

        room_vars = await effective_variables(ctx.db, ctx.room_id, None)
        if not service.secrets:
            return room_vars, {}

        room = (await ctx.db.execute(select(Room).where(Room.id == ctx.room_id))).scalar_one_or_none()
        if room is None:
            raise WorkspaceServiceToolError("room not found")

        secret_service = SecretService(ctx.db)
        try:
            resolved = await room_secrets.resolve_room_env(
                ctx.db,
                service=secret_service,
                room_id=ctx.room_id,
                actor_user_id=room.creator_id,
                names=service.secrets,
                # Explicit-name request from the declared service, same as
                # bash(secrets=[...]) — bypasses the scope allow-list, which
                # exists to gate *automatic* injection, not a named ask.
                scope=None,
                environment=None,
            )
        except room_secrets.RoomSecretNotAvailable as exc:
            raise WorkspaceServiceToolError(f"[{exc.code}] {exc}") from exc
        except room_secrets.RoomSecretError as exc:
            raise WorkspaceServiceToolError(str(exc)) from exc
        return room_vars, resolved

    async def _prepare(self, ctx, workspace, registry, service) -> str:
        from app.core.workspace_environments import WorkspaceEnvironmentError, prepare_service_environment

        try:
            _runtime_env, environment = await prepare_service_environment(
                str(ctx.room_id), workspace.id, service, registry, ctx.db,
            )
        except WorkspaceEnvironmentError as exc:
            raise WorkspaceServiceToolError(f"environment preparation failed ({exc.code}): {exc}") from exc
        if environment is None:
            return self._summary(service, detail="environment: legacy service without managed dependencies")
        return self._summary(
            service,
            detail=f"environment: {environment.slug}\nenvironment_state: {environment.state}",
        )

    async def _stop(self, ctx, service, registry, wsd) -> None:
        if service.runtime_state != "stopped":
            if service.runtime_state not in {"stopping", "error"}:
                await registry.transition(service, "stopping", f"agent:{ctx.session_id}")
            await wsd.stop_workspace_service(str(ctx.room_id), str(service.id), service.pid, service.pgid)
            await registry.transition(service, "stopped", f"agent:{ctx.session_id}")
        service.desired_state = "stopped"
        service.pid = service.pgid = None
        forget_bg_secrets(f"service:{service.id}")
        await ctx.db.commit()

    async def _wait_ready(self, ctx, service, registry, tool_input, wsd) -> str:
        if service.runtime_state == "ready":
            return self._summary(service)
        if not service.pid:
            raise WorkspaceServiceToolError("service is not running; call start before wait_ready")
        result = await wsd.wait_workspace_service_ready(
            str(ctx.room_id), str(service.id), service.readiness or {},
            max(1, min(int(tool_input.get("timeout_sec") or 45), 300)),
        )
        if result.get("ready"):
            if service.runtime_state == "starting":
                await registry.transition(service, "ready", f"agent:{ctx.session_id}")
                await ctx.db.commit()
            return self._summary(service)
        service.last_error = result.get("error") or "readiness check failed"
        if service.runtime_state == "starting":
            await registry.transition(service, "error", f"agent:{ctx.session_id}", {"code": "readiness_failed"})
        await ctx.db.commit()
        return self._summary(service, detail="remediation: inspect logs, fix cwd/argv/readiness, then restart")

    async def _logs(self, ctx, workspace, registry, service, tool_input) -> str:
        from app.core import workspaced_client as wsd
        from app.core.workspace_environments import WorkspaceEnvironmentError, environment_logs

        cursor = max(0, int(tool_input.get("cursor") or 0))
        limit = max(1, min(int(tool_input.get("limit") or _DEFAULT_LOG_LIMIT), _MAX_LOG_LIMIT))
        if tool_input.get("log_target") == "environment":
            environment = await registry.environment_for(workspace.id, service)
            if environment is None:
                raise WorkspaceServiceToolError("service has no managed environment")
            try:
                result = await environment_logs(str(ctx.room_id), environment, cursor)
            except WorkspaceEnvironmentError as exc:
                raise WorkspaceServiceToolError(f"could not read environment logs: {exc}") from exc
            environment.log_cursor = int(result.get("cursor", cursor))
            environment.updated_at = datetime.now(timezone.utc)
            ctx.db.add(environment)
            cursor_label = environment.log_cursor
        else:
            result = await wsd.get_workspace_service_logs(str(ctx.room_id), str(service.id), cursor, service.slug)
            cursor_label = int(result.get("cursor", cursor))
        content = mask_room_secrets(str(result.get("content") or ""), bg_secrets(f"service:{service.id}"))
        if len(content) > limit:
            content = content[-limit:] + f"\n[...log tail limited to {limit} chars]"
        if tool_input.get("log_target") != "environment":
            service.log_cursor = cursor_label
        service.last_accessed_at = datetime.now(timezone.utc)
        ctx.db.add(service)
        await ctx.db.commit()
        return _truncate(
            f"slug: {service.slug}\nstate: {service.runtime_state}\ncursor: {cursor_label}\n"
            f"rotated: {bool(result.get('rotated'))}\nlogs:\n{content or '(empty)'}"
        )

    async def _request(self, ctx, service, tool_input) -> str:
        from app.core import workspaced_client as wsd

        if (service.listen or {}).get("protocol", "http") != "http":
            raise WorkspaceServiceToolError("request is available only for HTTP services")
        port = (service.listen or {}).get("assigned_port")
        if not isinstance(port, int):
            raise WorkspaceServiceToolError("service has no assigned port; call start and wait_ready first")
        path = str(tool_input.get("path") or "/").strip() or "/"
        if not path.startswith("/"):
            path = "/" + path
        result = await wsd.proxy_fetch(str(ctx.room_id), port, path, str(tool_input.get("query") or ""))
        if result.get("error"):
            raise WorkspaceServiceToolError(f"request failed: {result['error']}")
        try:
            body = base64.b64decode(result.get("body_b64") or "").decode("utf-8", errors="replace")
        except Exception:
            body = "(binary or undecodable response)"
        if not body:
            body = "(empty body)"
        body = mask_room_secrets(body, bg_secrets(f"service:{service.id}"))
        if len(body) > _MAX_REQUEST_BODY:
            body = body[:_MAX_REQUEST_BODY] + f"\n[...body limited to {_MAX_REQUEST_BODY} chars]"
        service.last_accessed_at = datetime.now(timezone.utc)
        ctx.db.add(service)
        await ctx.db.commit()
        return _truncate(
            f"slug: {service.slug}\nstate: {service.runtime_state}\nstatus: {result.get('status_code', '?')}\n"
            f"content_type: {result.get('content_type', '')}\n\n{body}"
        )


REGISTRY.register(WorkspaceServiceTool())
