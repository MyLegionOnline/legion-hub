"""Browser services travel with the project in ``.legion/services.json``.

A colleague who clones a project built in another Legion gets the declared
services in the Browser tab instead of asking agents to rebuild them: the
catalog is exported to the repository and imported back after a Pull/adopt.
Imported services never start on their own and need one confirmation.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import workspace as ws_api
from app.core import workspace_services_manifest as manifest_mod
from app.core.workspace_services import config_hash, normalize_declaration


def _decl(**overrides):
    body = {
        "slug": "web", "title": "Landing", "kind": "web",
        "argv": ["npm", "run", "dev", "--", "--port", "$PORT"], "cwd": ".",
        "env": {}, "secrets": [], "listen": {"protocol": "http", "port": "auto"},
        "readiness": {"type": "http", "path": "/", "timeout_sec": 45},
        "default_path": "/", "depends_on": [], "restart_policy": "manual",
        "environment": {"slug": "node", "profile": "node-npm", "root": ".", "manifests": ["package-lock.json"]},
    }
    body.update(overrides)
    return body


def _row(declared_by="agent", runtime_state="stopped", **overrides):
    decl = normalize_declaration({**_decl(**overrides), "declared_by": declared_by})
    environment = decl.pop("environment")
    digest = config_hash({**decl, "environment": environment})
    return SimpleNamespace(
        id=uuid4(), workspace_id=uuid4(), environment_id=uuid4(), config_hash=digest,
        runtime_state=runtime_state, pid=4242, pgid=4242,
        **decl,
    ), SimpleNamespace(**environment)


class _Nested:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _db():
    db = MagicMock()
    db.begin_nested.return_value = _Nested()
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    return db


class _FakeRegistry:
    def __init__(self, rows):
        self.rows = {row.slug: row for row in rows}
        self.declared = []
        self.deleted = []

    def lock_for(self, _workspace_id):
        return _Nested()

    async def _get_any(self, _workspace_id, slug):
        return self.rows.get(slug)

    async def list(self, _workspace_id):
        return list(self.rows.values())

    async def delete(self, service):
        self.deleted.append(service.slug)

    async def declare(self, _workspace_id, raw, source="agent"):
        self.declared.append((raw["slug"], raw["declared_by"], source))
        return SimpleNamespace(slug=raw["slug"]), True


def _manifest(*services):
    return json.dumps({"version": 1, "services": list(services)})


async def _import(registry, text):
    with patch.object(manifest_mod, "WorkspaceServiceRegistry", return_value=registry), \
         patch("app.core.workspaced_client.read_workspace_file", AsyncMock(return_value=text)):
        db = _db()
        return await manifest_mod.import_services_manifest(db, uuid4(), uuid4(), source="git_pull:u"), db


# ── export ───────────────────────────────────────────────────────────────────

def test_export_entry_keeps_the_declaration_and_drops_runtime_and_credentials():
    row, environment = _row(
        env={"NODE_ENV": "development", "STRIPE_API_KEY": "sk_live_x", "DB_PASSWORD": "p"},
        secrets=["STRIPE_KEY"],
    )
    row.listen = {**row.listen, "assigned_port": 3017}
    entry = manifest_mod._export_entry(row, environment)

    assert entry["env"] == {"NODE_ENV": "development"}
    assert entry["secrets"] == ["STRIPE_KEY"]
    assert entry["listen"] == {"protocol": "http", "port": "auto"}
    assert "pid" not in entry and "declared_by" not in entry
    assert entry["environment"]["manifests"] == ["package-lock.json"]
    # What goes out must be importable as-is.
    normalize_declaration(entry)


@pytest.mark.asyncio
async def test_export_commits_only_the_manifest_when_the_catalog_changed():
    manifest = {"version": 1, "services": [{"slug": "web"}]}
    write = AsyncMock()
    with patch.object(manifest_mod, "build_manifest", AsyncMock(return_value=manifest)), \
         patch("app.core.workspaced_client.read_workspace_file_at_rev", AsyncMock(return_value=None)), \
         patch("app.core.workspaced_client.write_workspace_file", write):
        assert await manifest_mod.export_services_manifest(_db(), uuid4(), uuid4()) is True
    args, kwargs = write.await_args
    assert args[1] == ".legion/services.json"
    assert json.loads(args[2]) == manifest
    assert kwargs["commit_only_path"] is True


@pytest.mark.asyncio
async def test_export_skips_an_unchanged_or_empty_catalog():
    manifest = {"version": 1, "services": [{"slug": "web"}]}
    write = AsyncMock()
    with patch.object(manifest_mod, "build_manifest", AsyncMock(return_value=manifest)), \
         patch("app.core.workspaced_client.read_workspace_file_at_rev", AsyncMock(return_value=json.dumps(manifest))), \
         patch("app.core.workspaced_client.write_workspace_file", write):
        assert await manifest_mod.export_services_manifest(_db(), uuid4(), uuid4()) is False
    with patch.object(manifest_mod, "build_manifest", AsyncMock(return_value={"version": 1, "services": []})), \
         patch("app.core.workspaced_client.read_workspace_file_at_rev", AsyncMock(return_value=None)), \
         patch("app.core.workspaced_client.write_workspace_file", write):
        assert await manifest_mod.export_services_manifest(_db(), uuid4(), uuid4()) is False
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_failure_never_breaks_the_caller():
    with patch.object(manifest_mod, "build_manifest", AsyncMock(return_value={"version": 1, "services": [{}]})), \
         patch("app.core.workspaced_client.read_workspace_file_at_rev", AsyncMock(return_value=None)), \
         patch("app.core.workspaced_client.write_workspace_file", AsyncMock(side_effect=RuntimeError("main protected"))):
        assert await manifest_mod.export_services_manifest(_db(), uuid4(), uuid4()) is False


# ── import ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_import_declares_new_services_as_repo_without_starting_them():
    registry = _FakeRegistry([])
    report, db = await _import(registry, _manifest(_decl(), _decl(slug="api", kind="api", title="API")))
    assert report["added"] == ["web", "api"]
    assert registry.declared == [("web", "repo", "git_pull:u"), ("api", "repo", "git_pull:u")]
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_import_leaves_the_rooms_own_declaration_alone():
    own, _env = _row(declared_by="agent", argv=["python", "app.py"])
    registry = _FakeRegistry([own])
    report, _db_ = await _import(registry, _manifest(_decl()))
    assert registry.declared == []
    assert report["added"] == [] and report["removed"] == []


@pytest.mark.asyncio
async def test_import_keeps_an_approval_while_the_manifest_is_unchanged():
    approved, _env = _row(declared_by="repo")
    approved.declared_by = "repo_approved"
    registry = _FakeRegistry([approved])
    report, _db_ = await _import(registry, _manifest(_decl()))
    assert registry.declared == []
    assert report["updated"] == []


@pytest.mark.asyncio
async def test_import_redeclares_a_changed_repo_service_so_it_needs_confirmation_again():
    approved, _env = _row(declared_by="repo")
    approved.declared_by = "repo_approved"
    registry = _FakeRegistry([approved])
    report, _db_ = await _import(registry, _manifest(_decl(argv=["npm", "start"])))
    assert registry.declared == [("web", "repo", "git_pull:u")]
    assert report["updated"] == ["web"]


@pytest.mark.asyncio
async def test_import_does_not_touch_a_running_service():
    running, _env = _row(declared_by="repo_approved", runtime_state="ready")
    registry = _FakeRegistry([running])
    report, _db_ = await _import(registry, _manifest(_decl(argv=["npm", "start"])))
    assert registry.declared == []
    assert report["skipped"] == ["web"]


@pytest.mark.asyncio
async def test_import_removes_stopped_repo_services_dropped_from_the_manifest():
    stale, _env = _row(declared_by="repo", slug="old")
    own, _env2 = _row(declared_by="agent", slug="mine")
    registry = _FakeRegistry([stale, own])
    report, _db_ = await _import(registry, _manifest(_decl()))
    assert report["removed"] == ["old"]
    assert registry.deleted == ["old"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "not json", json.dumps({"version": 99, "services": []})])
async def test_import_ignores_a_missing_or_foreign_manifest(text):
    registry = _FakeRegistry([])
    report, _db_ = await _import(registry, text)
    assert registry.declared == []
    assert not any(report.values())


@pytest.mark.asyncio
async def test_import_skips_an_invalid_entry_and_keeps_the_rest():
    registry = _FakeRegistry([])
    report, _db_ = await _import(registry, _manifest(_decl(cwd="../escape"), _decl(slug="api", title="API")))
    assert report["skipped"] == ["web"]
    assert report["added"] == ["api"]


# ── first start confirmation ─────────────────────────────────────────────────

def test_repo_service_needs_confirmation_before_the_first_start():
    service = SimpleNamespace(declared_by="repo")
    with pytest.raises(HTTPException) as exc:
        ws_api._require_repo_confirmation(service, None)
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException):
        ws_api._require_repo_confirmation(service, ws_api.ServiceStartIn(confirm_repo=False))

    ws_api._require_repo_confirmation(service, ws_api.ServiceStartIn(confirm_repo=True))
    assert service.declared_by == "repo_approved"
    # Approved once — later starts need nothing.
    ws_api._require_repo_confirmation(service, None)


def test_agent_service_starts_without_confirmation():
    service = SimpleNamespace(declared_by="agent")
    ws_api._require_repo_confirmation(service, None)
    assert service.declared_by == "agent"


@pytest.mark.asyncio
async def test_pull_imports_the_manifest_and_reports_new_services():
    from app.api.v1 import git as git_api

    token_value = MagicMock()
    token_value.reveal.return_value = "t"
    room = MagicMock(git_config={"repo_url": "https://github.com/o/r", "branch": "main"}, creator_id=uuid4())
    imported = AsyncMock(return_value={"added": ["web"], "updated": [], "removed": [], "skipped": []})
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "apply_userfiles_policy", new_callable=AsyncMock), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock,
                      return_value={"synced": 1, "errors": [], "head_changed": False}), \
         patch("app.core.git_sync.import_pulled_services", imported):
        result = await git_api.git_sync(
            room_id=uuid4(), request=MagicMock(), db=AsyncMock(), current_user=MagicMock(id=uuid4()),
        )
    imported.assert_awaited_once()
    assert result.services_added == ["web"]
