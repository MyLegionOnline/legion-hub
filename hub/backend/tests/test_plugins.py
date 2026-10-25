"""Plugin loader (internal design notes stage 1): the core without plugins, a fake
plugin using every extension point, and the failures that must stop startup."""

from __future__ import annotations

import pytest
from fastapi import APIRouter

from app.plugins import host as plugin_host
from app.plugins import PLUGIN_API, PluginLoadError, get_registry, load_plugins, reset_registry


class _EntryPoint:
    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _install(monkeypatch, **plugins):
    eps = [_EntryPoint(n, t) for n, t in plugins.items()]
    monkeypatch.setattr(plugin_host, "entry_points", lambda group: eps)


class FakePlugin:
    name = "fake"
    api_version = PLUGIN_API

    def register(self, host):
        router = APIRouter()

        @router.get("/ping")
        async def ping():
            return {"ok": True}

        host.add_router(router, prefix="/internal/fake", tags=["fake"])
        host.set_limits_policy("policy")
        host.add_work_gate("gate")
        host.add_reconciler_step("step")
        host.add_migrations(["SELECT 1"])
        host.on("room.time", lambda **kw: None)
        host.add_features(billing=True)
        host.add_sign_in_gate("sign-in gate")


def test_core_without_plugins(monkeypatch):
    monkeypatch.delenv("LEGION_PLUGINS", raising=False)
    # Installed-but-unlisted plugins must not be loaded.
    _install(monkeypatch, fake=FakePlugin)
    registry = load_plugins()
    assert registry.loaded == []
    assert registry.routers == [] and registry.limits_policy is None
    assert registry.work_gates == [] and registry.reconciler_steps == []
    assert registry.migrations == [] and registry.features == {} and registry.sign_in_gates == []
    assert get_registry() is registry


def test_fake_plugin_registers_every_extension_point(monkeypatch):
    monkeypatch.setenv("LEGION_PLUGINS", " fake , fake")
    _install(monkeypatch, fake=FakePlugin)
    registry = load_plugins()
    assert registry.loaded == ["fake"]
    assert [(s.prefix, s.tags, s.plugin) for s in registry.routers] == [
        ("/internal/fake", ["fake"], "fake")
    ]
    assert registry.limits_policy == "policy"
    assert registry.work_gates == ["gate"] and registry.reconciler_steps == ["step"]
    assert registry.migrations == ["SELECT 1"]
    assert len(registry.handlers_for("room.time")) == 1
    assert registry.handlers_for("nope") == []
    assert registry.features == {"billing": True}
    assert registry.sign_in_gates == ["sign-in gate"]


def test_unknown_plugin_fails_startup(monkeypatch):
    monkeypatch.setenv("LEGION_PLUGINS", "ghost")
    _install(monkeypatch, fake=FakePlugin)
    with pytest.raises(PluginLoadError, match="ghost"):
        load_plugins()


def test_foreign_api_version_fails_startup(monkeypatch):
    class Old(FakePlugin):
        api_version = PLUGIN_API + 1

    monkeypatch.setenv("LEGION_PLUGINS", "old")
    _install(monkeypatch, old=Old)
    with pytest.raises(PluginLoadError, match="API"):
        load_plugins()


def test_import_and_register_errors_fail_startup(monkeypatch):
    class Broken(FakePlugin):
        def register(self, host):
            raise ValueError("boom")

    monkeypatch.setenv("LEGION_PLUGINS", "broken")
    _install(monkeypatch, broken=Broken)
    with pytest.raises(PluginLoadError, match="register"):
        load_plugins()

    monkeypatch.setenv("LEGION_PLUGINS", "bad_import")
    _install(monkeypatch, bad_import=ImportError("no module"))
    with pytest.raises(PluginLoadError, match="import"):
        load_plugins()


def test_two_limits_policies_conflict(monkeypatch):
    class Other(FakePlugin):
        name = "other"

    monkeypatch.setenv("LEGION_PLUGINS", "fake,other")
    _install(monkeypatch, fake=FakePlugin, other=Other)
    with pytest.raises(PluginLoadError, match="limits policy"):
        load_plugins()
