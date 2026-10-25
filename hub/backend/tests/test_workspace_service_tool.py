from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools._workspace_service import WorkspaceServiceTool


def _context() -> ToolContext:
    db = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=None, db=db)


def _service(**overrides):
    values = {
        "id": uuid4(), "slug": "catalog-web", "title": "Catalog", "kind": "web",
        "argv": ["python", "-m", "http.server", "$PORT"], "cwd": "catalog", "env": {},
        "listen": {"protocol": "http", "port": "auto", "assigned_port": 3010},
        "readiness": {"type": "http", "path": "/", "timeout_sec": 10},
        "depends_on": [], "runtime_state": "stopped", "desired_state": "stopped",
        "environment_id": None, "secrets": [],
        "pid": None, "pgid": None, "log_path": None, "log_cursor": 0, "last_error": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_workspace_service_is_registered_with_required_actions():
    tool = REGISTRY.get("workspace_service")
    assert tool is not None
    assert tool.roles == frozenset({"agent", "cm"})
    assert set(tool.input_schema["properties"]["action"]["enum"]) == {
        "list", "declare", "prepare", "start", "stop", "restart", "status", "logs", "wait_ready",
        "request",
    }
    assert "command" not in tool.input_schema["properties"]["service"]["properties"]


def test_chatmaster_receives_workspace_service_for_direct_browser_actions():
    assert "workspace_service" in {tool.name for tool in REGISTRY.tools_for("cm")}


@pytest.mark.asyncio
async def test_disabled_services_returns_actionable_error(monkeypatch):
    monkeypatch.delenv("WORKSPACE_SERVICES_ENABLED", raising=False)

    result = await WorkspaceServiceTool().execute(_context(), {"action": "status", "slug": "catalog-web"})

    assert result.startswith("Error:")
    assert "WORKSPACE_SERVICES_ENABLED" in result
    assert "cannot change" in result and "background=true" in result


@pytest.mark.asyncio
async def test_declare_rejects_legacy_shell_command_before_persisting(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    ctx = _context()

    result = await WorkspaceServiceTool().execute(ctx, {
        "action": "declare",
        "service": {"slug": "catalog-web", "command": "npm run dev"},
    })

    assert result.startswith("Error:")
    assert "argv" in result
    ctx.db.add.assert_not_called()


@pytest.mark.asyncio
async def test_status_is_compact_and_does_not_expose_port(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    tool = WorkspaceServiceTool()
    service = _service(runtime_state="ready")
    tool._service = AsyncMock(return_value=(SimpleNamespace(id=uuid4()), MagicMock(), service))

    result = await tool.execute(_context(), {"action": "status", "slug": "catalog-web"})

    assert "slug: catalog-web" in result
    assert "health: ready" in result
    assert "browser_label: Catalog" in result
    assert "3010" not in result


@pytest.mark.asyncio
async def test_request_normalizes_relative_path_before_contacting_supervisor(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    tool = WorkspaceServiceTool()
    tool._service = AsyncMock(return_value=(SimpleNamespace(id=uuid4()), MagicMock(), _service(runtime_state="ready")))
    ctx = _context()

    import base64
    from app.core import workspaced_client as wsd

    async def proxy_fetch(room_id, port, path, query):
        assert path == "/health"
        return {"status_code": 200, "content_type": "text/plain", "body_b64": base64.b64encode(b"ok").decode()}

    monkeypatch.setattr(wsd, "proxy_fetch", proxy_fetch)

    result = await tool.execute(ctx, {
        "action": "request", "slug": "catalog-web", "path": "health",
    })

    assert not result.startswith("Error:")
    assert "status: 200" in result


@pytest.mark.asyncio
async def test_start_tree_reports_missing_dependency_without_loop():
    tool = WorkspaceServiceTool()
    root = _service(depends_on=["catalog-api"])
    registry = MagicMock()
    registry.get = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="not declared"):
        await tool._start_tree(_context(), SimpleNamespace(id=uuid4()), registry, root, MagicMock(), set())


@pytest.mark.asyncio
async def test_start_tree_does_not_start_a_service_when_dependency_fails(monkeypatch):
    tool = WorkspaceServiceTool()
    dependency = _service(slug="catalog-api")
    root = _service(depends_on=["catalog-api"])
    registry = MagicMock()
    registry.get = AsyncMock(return_value=dependency)
    async def transition(item, state, *_args, **_kwargs):
        item.runtime_state = state
    registry.transition = transition
    supervisor = MagicMock()
    supervisor.start_workspace_service = AsyncMock(return_value={
        "state": "error", "error": "readiness failed", "error_code": "readiness_failed",
    })
    monkeypatch.setattr(
        "app.core.room_config.effective_variables", AsyncMock(return_value={})
    )

    with pytest.raises(ValueError, match="not ready"):
        await tool._start_tree(_context(), SimpleNamespace(id=uuid4()), registry, root, supervisor, set())

    assert supervisor.start_workspace_service.await_count == 1


@pytest.mark.asyncio
async def test_start_tree_prepares_managed_environment_before_launch(monkeypatch):
    tool = WorkspaceServiceTool()
    service = _service(environment_id=uuid4())
    registry = MagicMock()

    async def transition(item, state, *_args, **_kwargs):
        item.runtime_state = state

    registry.transition = transition
    supervisor = MagicMock()
    supervisor.start_workspace_service = AsyncMock(return_value={
        "state": "ready", "port": 3000, "pid": 42, "pgid": 42, "log_path": "/tmp/service.log",
    })
    prepared = SimpleNamespace(slug="catalog-python", state="ready")

    async def prepare(*_args, **_kwargs):
        return {"VIRTUAL_ENV": "/workspace/.legion/env", "LEGION_ENV_PATH_PREPEND": "/workspace/.legion/env/bin"}, prepared

    monkeypatch.setattr("app.core.workspace_environments.prepare_service_environment", prepare)
    monkeypatch.setattr(
        "app.core.room_config.effective_variables", AsyncMock(return_value={})
    )

    await tool._start_tree(_context(), SimpleNamespace(id=uuid4()), registry, service, supervisor, set())

    launch_spec = supervisor.start_workspace_service.await_args.args[1]
    assert launch_spec["env"]["VIRTUAL_ENV"] == "/workspace/.legion/env"
    assert service.runtime_state == "ready"


@pytest.mark.asyncio
async def test_start_tree_does_not_launch_when_environment_prepare_fails(monkeypatch):
    tool = WorkspaceServiceTool()
    service = _service(environment_id=uuid4())
    registry = MagicMock()

    async def transition(item, state, *_args, **_kwargs):
        item.runtime_state = state

    registry.transition = transition
    supervisor = MagicMock()
    supervisor.start_workspace_service = AsyncMock()

    from app.core.workspace_environments import WorkspaceEnvironmentError

    async def prepare(*_args, **_kwargs):
        raise WorkspaceEnvironmentError("dependency_prepare_failed", "Django is missing")

    monkeypatch.setattr("app.core.workspace_environments.prepare_service_environment", prepare)

    await tool._start_tree(_context(), SimpleNamespace(id=uuid4()), registry, service, supervisor, set())

    assert service.runtime_state == "error"
    assert "dependency_prepare_failed" in service.last_error
    supervisor.start_workspace_service.assert_not_awaited()
