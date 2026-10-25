from __future__ import annotations

import pytest

from app.core.workspace_services import ServiceValidationError, config_hash, normalize_declaration
from app.core.workspace_checkpoint import should_restore_declared_service


def _web_service(**overrides):
    service = {
        "slug": "catalog-web",
        "title": "Catalog",
        "kind": "web",
        "argv": ["npm", "run", "dev", "--", "--port", "$PORT"],
        "cwd": "catalog",
        "listen": {"protocol": "http", "port": "auto"},
        "readiness": {"type": "http", "path": "/", "statuses": [200]},
        "environment": {
            "slug": "catalog-node", "profile": "node-npm", "root": "catalog",
            "manifests": ["package-lock.json"],
        },
    }
    service.update(overrides)
    return service


def test_service_declaration_normalizes_loopback_and_is_hash_stable():
    first = normalize_declaration(_web_service())
    second = normalize_declaration(_web_service())

    assert first["listen"] == {"protocol": "http", "port": "auto", "host": "127.0.0.1"}
    assert first["readiness"]["timeout_sec"] == 45
    assert config_hash(first) == config_hash(second)


def test_service_declaration_accepts_workspace_root():
    declaration = normalize_declaration(_web_service(
        cwd=".", environment={
            "slug": "catalog-node", "profile": "node-npm", "root": ".",
            "manifests": ["package-lock.json"],
        },
    ))

    assert declaration["cwd"] == "."


@pytest.mark.parametrize(
    "override, message",
    [
        ({"slug": "Catalog"}, "kebab-case"),
        ({"cwd": "../catalog"}, "workspace-relative"),
        ({"argv": []}, "non-empty"),
        ({"depends_on": ["catalog-web"]}, "depends_on"),
        ({"listen": {"port": 0}}, "TCP port"),
        ({"environment": {"slug": "catalog", "profile": "node-npm", "root": "catalog", "manifests": []}}, "package-lock"),
        ({"environment": {"slug": "catalog", "profile": "python-pip", "root": "catalog", "manifests": []}}, "requirements"),
    ],
)
def test_service_declaration_rejects_unsafe_or_ambiguous_values(override, message):
    with pytest.raises(ServiceValidationError, match=message):
        normalize_declaration(_web_service(**override))


def test_service_declaration_requires_managed_environment():
    declaration = _web_service()
    declaration.pop("environment")

    with pytest.raises(ServiceValidationError, match="environment is required"):
        normalize_declaration(declaration)


def test_restore_requires_matching_checkpoint_config_and_desired_state():
    declaration = normalize_declaration(_web_service(restart_policy="on_workspace_restore"))
    digest = config_hash(declaration)
    process = type("Service", (), {
        "restart_policy": "on_workspace_restore",
        "desired_state": "running",
        "config_hash": digest,
    })()

    assert should_restore_declared_service(
        process, {"desired_state": "running", "config_hash": digest}
    ) == (True, None)
    assert should_restore_declared_service(
        process, {"desired_state": "running", "config_hash": "changed"}
    ) == (False, "config_hash_mismatch")
    assert should_restore_declared_service(
        process, {"desired_state": "stopped", "config_hash": digest}
    ) == (False, "checkpoint_desired_state_stopped")
