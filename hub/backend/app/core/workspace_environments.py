"""Backend orchestration for managed workspace dependency environments."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.core import workspaced_client as wsd


class WorkspaceEnvironmentError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _workspace_root(root: str) -> str:
    return "/workspace" if root == "." else f"/workspace/{root}"


async def prepare_service_environment(
    room_id: str,
    workspace_id,
    service,
    registry,
    db,
) -> tuple[dict[str, str], Any | None]:
    """Prepare ``service``'s declared environment and return launch env.

    Legacy rows have no environment reference and intentionally retain their
    old launch behaviour. New declarations cannot reach this path without a
    validated environment, so this is a compatibility bridge rather than a
    bypass for the new contract.
    """
    if getattr(service, "environment_id", None) is None:
        return {}, None
    environment = await registry.environment_for(workspace_id, service)
    if environment is None:
        raise WorkspaceEnvironmentError("environment_not_found", "declared environment is no longer available")
    environment.state = "preparing"
    environment.last_error = None
    environment.updated_at = datetime.now(timezone.utc)
    db.add(environment)
    await db.commit()
    try:
        result = await wsd.prepare_workspace_environment(str(room_id), {
            "environment_id": str(environment.id),
            "slug": environment.slug,
            "profile": environment.profile,
            "root": _workspace_root(environment.root),
            "manifests": environment.manifests or [],
            "timeout_sec": 600,
        })
    except httpx.HTTPError as exc:
        environment.state = "error"
        environment.last_error = str(exc)[:2_000]
        environment.updated_at = datetime.now(timezone.utc)
        db.add(environment)
        await db.commit()
        raise WorkspaceEnvironmentError("dependency_prepare_failed", str(exc)) from exc
    environment.state = str(result.get("state") or "error")
    environment.prepared_fingerprint = result.get("fingerprint")
    environment.runtime_fingerprint = result.get("runtime_fingerprint")
    environment.runtime_env = dict(result.get("runtime_env") or {})
    environment.log_path = result.get("log_path")
    environment.last_error = result.get("error")
    environment.updated_at = datetime.now(timezone.utc)
    db.add(environment)
    await db.commit()
    if environment.state != "ready":
        raise WorkspaceEnvironmentError(
            str(result.get("error_code") or "dependency_prepare_failed"),
            str(environment.last_error or "dependency preparation failed"),
        )
    return {str(key): str(value) for key, value in environment.runtime_env.items()}, environment


async def environment_logs(room_id: str, environment, cursor: int = 0) -> dict[str, Any]:
    if not environment.log_path:
        return {"content": "", "cursor": 0, "rotated": False, "log_path": None}
    try:
        return await wsd.get_workspace_environment_logs(
            str(room_id), str(environment.id), environment.slug, cursor,
        )
    except httpx.HTTPError as exc:
        raise WorkspaceEnvironmentError("environment_logs_unavailable", str(exc)) from exc
