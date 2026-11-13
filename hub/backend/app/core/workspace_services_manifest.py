"""Portable Browser-services manifest kept in the project repository.

A room's declared services live in the Hub database, so a project cloned into
another Legion used to arrive with an empty Browser tab. This module mirrors the
catalog into ``.legion/services.json`` (committed like ``.legion/run.json``) and
recreates it from that file after a Pull or the first adoption of a repository.

Rules:

* Only the launch declaration travels: no PIDs, assigned ports, logs, state or
  secret values. Secrets travel by name; each room supplies its own values.
* An imported service is marked ``declared_by="repo"``. The Browser tab shows
  its command and asks the user to confirm before the first start
  (``repo_approved`` afterwards). A changed manifest resets the approval.
* A service the room declared itself (agent or user) always wins over the
  manifest entry with the same slug.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.workspace_services import (
    ServiceValidationError,
    WorkspaceServiceRegistry,
    config_hash,
    normalize_declaration,
)
from app.models.workspace import WorkspaceManagedProcess

logger = logging.getLogger(__name__)

MANIFEST_PATH = ".legion/services.json"
MANIFEST_VERSION = 1
MANIFEST_COMMIT_MESSAGE = "legion: update Browser services manifest"
MAX_MANIFEST_SERVICES = 50

REPO_DECLARED = "repo"
REPO_APPROVED = "repo_approved"
REPO_SOURCES = frozenset({REPO_DECLARED, REPO_APPROVED})

# Env values are copied verbatim, so a key that names a credential is left out
# of the shared file: credentials belong in ``secrets`` (names only).
_SECRET_ENV_KEY = re.compile(
    r"SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE|CREDENTIAL|API_?KEY|ACCESS_?KEY|AUTH", re.IGNORECASE
)
_RUNNING_STATES = {"starting", "ready", "degraded", "stopping"}


def needs_repo_confirmation(service: WorkspaceManagedProcess) -> bool:
    return (service.declared_by or "") == REPO_DECLARED


def _export_entry(service: WorkspaceManagedProcess, environment) -> dict[str, Any]:
    listen = service.listen or {}
    readiness = dict(service.readiness or {})
    env = {
        key: value for key, value in sorted((service.env or {}).items())
        if not _SECRET_ENV_KEY.search(key)
    }
    return {
        "slug": service.slug,
        "title": service.title,
        "kind": service.kind,
        "argv": list(service.argv or []),
        "cwd": service.cwd,
        "env": env,
        "secrets": list(service.secrets or []),
        "listen": {"protocol": listen.get("protocol", "http"), "port": listen.get("port", "auto")},
        "readiness": readiness,
        "default_path": service.default_path or "/",
        "depends_on": list(service.depends_on or []),
        "restart_policy": service.restart_policy or "manual",
        "environment": {
            "slug": environment.slug,
            "profile": environment.profile,
            "root": environment.root,
            "manifests": list(environment.manifests or []),
        },
    }


async def build_manifest(db: AsyncSession, workspace_id: UUID) -> dict[str, Any]:
    registry = WorkspaceServiceRegistry(db)
    entries = []
    for service in await registry.list(workspace_id):
        # Legacy rows without a managed environment cannot be re-declared
        # elsewhere (the contract requires one), so they stay local.
        environment = await registry.environment_for(workspace_id, service)
        if environment is None:
            continue
        entries.append(_export_entry(service, environment))
    return {"version": MANIFEST_VERSION, "services": entries}


def render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


async def export_services_manifest(db: AsyncSession, room_id: UUID, workspace_id: UUID) -> bool:
    """Write and commit ``.legion/services.json`` when the catalog changed.

    Best effort: a busy git lock, a checked-out protected ``main`` or a missing
    workspace volume leave the file as it was and the next catalog change or
    push tries again. Returns True when a new version was written.
    """
    from app.core import workspaced_client as wsd

    try:
        manifest = await build_manifest(db, workspace_id)
        # Compare with the committed copy, not the working tree: a write whose
        # commit failed (busy lock, protected main) must be retried later.
        try:
            current = await wsd.read_workspace_file_at_rev(str(room_id), MANIFEST_PATH, "HEAD")
        except Exception:
            current = None
        if current is None and not manifest["services"]:
            return False
        body = render_manifest(manifest)
        if current is not None:
            try:
                if json.loads(current) == manifest:
                    return False
            except ValueError:
                pass
        await wsd.write_workspace_file(
            str(room_id), MANIFEST_PATH, body,
            commit_message=MANIFEST_COMMIT_MESSAGE, commit_only_path=True,
        )
        return True
    except Exception:
        logger.info("services manifest export skipped for room %s", room_id, exc_info=True)
        return False


def parse_manifest(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        raise ServiceValidationError(f"unsupported {MANIFEST_PATH} version")
    services = data.get("services")
    if not isinstance(services, list):
        raise ServiceValidationError(f"{MANIFEST_PATH}: services must be a list")
    if len(services) > MAX_MANIFEST_SERVICES:
        raise ServiceValidationError(f"{MANIFEST_PATH}: at most {MAX_MANIFEST_SERVICES} services")
    return [item for item in services if isinstance(item, dict)]


async def import_services_manifest(
    db: AsyncSession, room_id: UUID, workspace_id: UUID, *, source: str,
) -> dict[str, list[str]]:
    """Declare the services listed in the checked-out manifest.

    Returns ``{"added": [...], "updated": [...], "removed": [...], "skipped": [...]}``.
    Never starts anything. Running services and the room's own declarations
    are left untouched.
    """
    from app.core import workspaced_client as wsd

    report: dict[str, list[str]] = {"added": [], "updated": [], "removed": [], "skipped": []}
    try:
        text = await wsd.read_workspace_file(str(room_id), MANIFEST_PATH)
    except Exception:
        logger.info("services manifest import: could not read it in room %s", room_id, exc_info=True)
        return report
    if text is None:
        return report
    try:
        entries = parse_manifest(text)
    except (ValueError, ServiceValidationError):
        logger.warning("services manifest import: invalid %s in room %s", MANIFEST_PATH, room_id)
        return report

    registry = WorkspaceServiceRegistry(db)
    listed: set[str] = set()
    async with registry.lock_for(workspace_id):
        for raw in entries:
            slug = str(raw.get("slug") or "")
            listed.add(slug)
            candidate = {**raw, "declared_by": REPO_DECLARED}
            try:
                normalized = normalize_declaration(candidate)
            except (ServiceValidationError, TypeError, ValueError):
                report["skipped"].append(slug or "?")
                continue
            existing = await registry._get_any(workspace_id, slug)
            was_declared = existing is not None and bool(existing.config_hash)
            if existing is not None and existing.config_hash and existing.declared_by not in REPO_SOURCES:
                continue  # the room's own declaration wins
            if existing is not None and existing.config_hash == config_hash(normalized):
                continue
            if existing is not None and existing.runtime_state in _RUNNING_STATES:
                report["skipped"].append(slug)
                continue
            try:
                async with db.begin_nested():
                    await registry.declare(workspace_id, candidate, source=source)
            except ServiceValidationError:
                report["skipped"].append(slug)
                continue
            report["updated" if was_declared else "added"].append(slug)

        for service in await registry.list(workspace_id):
            if (
                service.declared_by in REPO_SOURCES
                and service.slug not in listed
                and service.runtime_state in {"stopped", "error"}
            ):
                await registry.delete(service)
                report["removed"].append(service.slug)
        await db.commit()

    if any(report[key] for key in ("added", "updated", "removed")):
        logger.info("services manifest import in room %s: %s", room_id, report)
    return report
