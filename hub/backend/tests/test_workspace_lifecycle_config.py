"""Stage 0 contract tests for workspace lifecycle configuration."""

from __future__ import annotations

import importlib
import uuid
from unittest.mock import AsyncMock

import pytest

import app.core.workspace_lifecycle as lifecycle
import app.core.workspace_quota as workspace_quota


@pytest.fixture(autouse=True)
def _restore_lifecycle_module():
    """Undo ``importlib.reload`` after each test.

    A reload rebinds every class in the module (``WorkspaceStartError``,
    ``WorkspaceLifecycleConfig``, ...). Other test files hold the original
    objects from their module-level imports, so ``except``/``isinstance``
    against them would silently stop matching. Restoring the namespace keeps
    the suite order-independent.
    """
    snapshot = dict(lifecycle.__dict__)
    yield
    lifecycle.__dict__.clear()
    lifecycle.__dict__.update(snapshot)


def _reload(monkeypatch, **values):
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    return importlib.reload(lifecycle)


def test_defaults_document_stage_zero_policy(monkeypatch):
    for name in (
        "WORKSPACE_STARTING_COUNTS_TOWARD_QUOTA",
        "WORKSPACE_ALLOW_BACKGROUND_ROOMS",
        "WORKSPACE_IDLE_TIMEOUT_SEC",
        "WORKSPACE_START_TIMEOUT_SEC",
        "WORKSPACE_DRAIN_TIMEOUT_SEC",
        "WORKSPACE_STOP_TIMEOUT_SEC",
        "WORKSPACE_RECOVERY_TIMEOUT_SEC",
        "WORKSPACE_STATUS_TRANSPORT",
        "WORKSPACE_SERVICES_ENABLED",
        "WORKSPACE_PREVIEW_GATEWAY_ENABLED",
        "WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC",
        "WORKSPACE_SERVICE_PORT_RANGE",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = _reload(monkeypatch).load_workspace_lifecycle_config()

    assert lifecycle.QUOTA_SOURCE == "effective_user_limits.max_running"
    assert cfg.starting_counts_toward_quota is True
    assert cfg.allow_background_rooms is True
    assert cfg.idle_timeout_sec == 900
    assert cfg.start_timeout_sec == 120
    assert cfg.drain_timeout_sec == 30
    assert cfg.stop_timeout_sec == 30
    assert cfg.recovery_timeout_sec == 60
    assert cfg.status_transport == "websocket"
    assert cfg.stop_on_browser_disconnect is False
    assert cfg.auto_resume_workspace_switch_tasks is True
    assert cfg.force_stop_requires_confirmation is True
    assert cfg.auto_force_stop_stale_runtime is True
    assert cfg.bash_satisfies_read_before_edit is False
    assert cfg.services_enabled is False
    assert cfg.preview_gateway_enabled is False
    assert cfg.service_idle_timeout_sec == 900
    assert cfg.service_port_range == (3000, 3999)
    assert cfg.counted_states == ("starting", "running", "draining", "stopping")


def test_environment_values_are_resolved(monkeypatch):
    module = _reload(
        monkeypatch,
        WORKSPACE_STARTING_COUNTS_TOWARD_QUOTA="false",
        WORKSPACE_ALLOW_BACKGROUND_ROOMS="off",
        WORKSPACE_IDLE_TIMEOUT_SEC=600,
        WORKSPACE_START_TIMEOUT_SEC=90,
        WORKSPACE_DRAIN_TIMEOUT_SEC=15,
        WORKSPACE_STOP_TIMEOUT_SEC=20,
        WORKSPACE_RECOVERY_TIMEOUT_SEC=45,
        WORKSPACE_STATUS_TRANSPORT="polling",
        WORKSPACE_STATUS_POLL_INITIAL_SEC=2,
        WORKSPACE_STATUS_POLL_MAX_SEC=8,
        WORKSPACE_SERVICES_ENABLED="true",
        WORKSPACE_PREVIEW_GATEWAY_ENABLED="1",
        WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC=480,
        WORKSPACE_SERVICE_PORT_RANGE="4100-4199",
    )

    cfg = module.load_workspace_lifecycle_config()
    assert cfg.starting_counts_toward_quota is False
    assert cfg.allow_background_rooms is False
    assert cfg.idle_timeout_sec == 600
    assert cfg.start_timeout_sec == 90
    assert cfg.drain_timeout_sec == 15
    assert cfg.stop_timeout_sec == 20
    assert cfg.recovery_timeout_sec == 45
    assert cfg.status_transport == "polling"
    assert cfg.polling_initial_sec == 2
    assert cfg.polling_max_sec == 8
    assert cfg.services_enabled is True
    assert cfg.preview_gateway_enabled is True
    assert cfg.service_idle_timeout_sec == 480
    assert cfg.service_port_range == (4100, 4199)
    assert cfg.counted_states == ("running", "draining", "stopping")


def test_invalid_values_fall_back_to_safe_defaults(monkeypatch):
    module = _reload(
        monkeypatch,
        WORKSPACE_IDLE_TIMEOUT_SEC=0,
        WORKSPACE_STATUS_TRANSPORT="long-poll",
        WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC=0,
        WORKSPACE_SERVICE_PORT_RANGE="4000-3000",
    )
    cfg = module.load_workspace_lifecycle_config()
    assert cfg.idle_timeout_sec == 900
    assert cfg.status_transport == "websocket"
    assert cfg.service_idle_timeout_sec == 900
    assert cfg.service_port_range == (3000, 3999)


def test_config_rejects_inconsistent_direct_construction():
    with pytest.raises(ValueError, match="polling_initial_sec"):
        lifecycle.WorkspaceLifecycleConfig(
            polling_initial_sec=10,
            polling_max_sec=5,
        )

    with pytest.raises(ValueError, match="service_port_range"):
        lifecycle.WorkspaceLifecycleConfig(service_port_range=(0, 1))


@pytest.mark.asyncio
async def test_n_is_resolved_from_effective_user_quota(monkeypatch):
    get_limits = AsyncMock(return_value={"max_running": 7})
    monkeypatch.setattr(workspace_quota, "get_effective_limits", get_limits)
    user_id = uuid.uuid4()
    db = object()

    assert await lifecycle.resolve_max_running_workspaces(user_id, db) == 7
    get_limits.assert_awaited_once_with(user_id, db)


@pytest.mark.asyncio
async def test_effective_quota_requires_finite_max_running(monkeypatch):
    monkeypatch.setattr(
        workspace_quota,
        "get_effective_limits",
        AsyncMock(return_value={"max_running": None}),
    )
    with pytest.raises(ValueError, match="finite integer"):
        await lifecycle.resolve_max_running_workspaces(uuid.uuid4(), object())
