from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# The core refuses a plugin built against another API version.
PLUGIN_API = 1
PLUGIN_ENTRY_POINT_GROUP = "legion_hub.plugins"
PLUGINS_ENV = "LEGION_PLUGINS"


class PluginLoadError(RuntimeError):
    """A plugin named in LEGION_PLUGINS could not be loaded.

    Startup fails on purpose: a cloud Hub must not silently run without its
    billing plugin.
    """


@runtime_checkable
class HubPlugin(Protocol):
    name: str
    api_version: int

    def register(self, host: "PluginHost") -> None: ...


@dataclass
class RouterSpec:
    router: Any
    prefix: str = ""
    tags: list[str] | None = None
    plugin: str = ""


@dataclass
class PluginRegistry:
    """Everything loaded plugins have attached to the core."""

    loaded: list[str] = field(default_factory=list)
    routers: list[RouterSpec] = field(default_factory=list)
    limits_policy: Any = None
    limits_policy_owner: str = ""
    work_gates: list[Any] = field(default_factory=list)
    reconciler_steps: list[Any] = field(default_factory=list)
    migrations: list[str] = field(default_factory=list)
    required_migration_markers: list[str] = field(default_factory=list)
    event_handlers: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    market_credentials: list[Callable[[], dict[str, str]]] = field(default_factory=list)
    sign_in_gates: list[Any] = field(default_factory=list)

    def handlers_for(self, event: str) -> list[Callable[..., Any]]:
        return list(self.event_handlers.get(event, ()))


class PluginHost:
    """The surface a plugin sees inside ``register``."""

    def __init__(self, registry: PluginRegistry, plugin_name: str) -> None:
        self._registry = registry
        self._plugin = plugin_name

    def add_router(self, router: Any, prefix: str = "", tags: list[str] | None = None) -> None:
        self._registry.routers.append(
            RouterSpec(router=router, prefix=prefix, tags=tags, plugin=self._plugin)
        )

    def set_limits_policy(self, policy: Any) -> None:
        # Two plugins deciding the limits would be an ambiguity, not a merge.
        if self._registry.limits_policy is not None:
            raise PluginLoadError(
                f"plugin {self._plugin!r} sets a limits policy, but "
                f"{self._registry.limits_policy_owner!r} already did"
            )
        self._registry.limits_policy = policy
        self._registry.limits_policy_owner = self._plugin

    def add_work_gate(self, gate: Any) -> None:
        self._registry.work_gates.append(gate)

    def add_reconciler_step(self, step: Any) -> None:
        self._registry.reconciler_steps.append(step)

    def add_migrations(self, statements: list[str] | tuple[str, ...]) -> None:
        """Idempotent DDL, run by init_db after the core migrations.

        The plugin's SQLModel tables must be imported inside ``register`` so
        that create_all sees them.
        """
        self._registry.migrations.extend(statements)

    def require_migration_marker(self, name: str) -> None:
        """Fail startup if a security-critical plugin migration did not commit."""
        self._registry.required_migration_markers.append(name)

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self._registry.event_handlers.setdefault(event, []).append(handler)

    def add_sign_in_gate(self, gate: Any) -> None:
        """Decide who may sign in to this Hub with a Legion account.

        See ``app.core.sign_in_gates``: a gate refuses an account, or admits it
        and says a Hub user may be made for it now. A self-hosted Hub has no
        gates and lets in every account its own rules let in.
        """
        self._registry.sign_in_gates.append(gate)

    def add_features(self, **flags: Any) -> None:
        self._registry.features.update(flags)

    def add_market_credentials(self, provider: Callable[[], dict[str, str]]) -> None:
        """Headers sent with a skill package download (``market_client.fetch_package``).

        Market gives a paid package only to a Hub it trusts: the cloud plugin
        presents its service token here. The core sends nothing of its own, so a
        self-hosted Hub asks as an anonymous caller.
        """
        self._registry.market_credentials.append(provider)


_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    return _registry


def reset_registry() -> PluginRegistry:
    """Drop everything plugins attached (tests, and a fresh load)."""
    global _registry
    _registry = PluginRegistry()
    return _registry


def _configured_names(raw: str | None) -> list[str]:
    names: list[str] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if part and part not in names:
            names.append(part)
    return names


def load_plugins(names: list[str] | None = None) -> PluginRegistry:
    """Load the plugins named in LEGION_PLUGINS (or ``names``) into a fresh registry.

    Raises PluginLoadError for an unknown name, a failing import, a wrong
    ``api_version`` or an exception inside ``register``.
    """
    registry = reset_registry()
    wanted = names if names is not None else _configured_names(os.getenv(PLUGINS_ENV))
    if not wanted:
        return registry

    available = {ep.name: ep for ep in entry_points(group=PLUGIN_ENTRY_POINT_GROUP)}
    for name in wanted:
        ep = available.get(name)
        if ep is None:
            raise PluginLoadError(
                f"{PLUGINS_ENV} names plugin {name!r}, but no installed package "
                f"declares it in the {PLUGIN_ENTRY_POINT_GROUP!r} entry-point group "
                f"(found: {sorted(available) or 'none'})"
            )
        try:
            obj = ep.load()
            plugin: Any = obj() if isinstance(obj, type) else obj
        except Exception as exc:
            raise PluginLoadError(f"plugin {name!r} failed to import: {exc!r}") from exc

        version = getattr(plugin, "api_version", None)
        if version != PLUGIN_API:
            raise PluginLoadError(
                f"plugin {name!r} targets plugin API {version!r}, "
                f"this Hub provides API {PLUGIN_API}"
            )
        if not callable(getattr(plugin, "register", None)):
            raise PluginLoadError(f"plugin {name!r} has no register(host) method")

        try:
            plugin.register(PluginHost(registry, name))
        except PluginLoadError:
            raise
        except Exception as exc:
            raise PluginLoadError(f"plugin {name!r} failed in register(): {exc!r}") from exc
        registry.loaded.append(name)
        logger.info("plugin loaded: %s (api %s)", name, version)
    return registry
