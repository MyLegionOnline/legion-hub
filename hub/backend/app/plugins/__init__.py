"""Hub plugin system.

The open core knows nothing about the cloud (tariffs, hours, billing); a
private plugin package attaches that behaviour through the extension points of
``PluginHost`` (internal design notes section 3).

A plugin declares itself in the ``legion_hub.plugins`` entry-point group, but
only the names listed in ``LEGION_PLUGINS`` (comma separated) are loaded: an
arbitrary pip package with such an entry point must never execute inside the
Hub just by being installed.
"""

from app.plugins.host import (
    PLUGIN_API,
    PLUGIN_ENTRY_POINT_GROUP,
    HubPlugin,
    PluginHost,
    PluginLoadError,
    PluginRegistry,
    get_registry,
    load_plugins,
    reset_registry,
)

__all__ = [
    "PLUGIN_API",
    "PLUGIN_ENTRY_POINT_GROUP",
    "HubPlugin",
    "PluginHost",
    "PluginLoadError",
    "PluginRegistry",
    "get_registry",
    "load_plugins",
    "reset_registry",
]
