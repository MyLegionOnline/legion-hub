"""Hub ↔ connector compatibility: protocol version range and agent mode.

The connector lives in its own repository and is upgraded independently of
the Hub, so the Hub cannot assume the other end speaks its protocol. The
connector announces ``protocol_version`` in ``hello``
(``hub/docs/connector_protocol.md`` §4.1, "Versioning"); the Hub stores it in
``Connector.capabilities["protocol_version"]`` (no schema change) and every
dispatch is gated on :func:`dispatch_block_reason`.

Pure functions only — no DB, no sockets — so the ``Connector`` row, the API
schema and the dispatcher all agree on one definition.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# The range of ``protocol_version`` this Hub can talk to. Raise
# ``CONNECTOR_PROTOCOL`` when the Hub starts sending a frame or a required
# field older connectors do not understand; raise ``MIN_CONNECTOR_PROTOCOL``
# only when support for the previous version is dropped (see the
# "Versioning" section of connector_protocol.md for the support window).
MIN_CONNECTOR_PROTOCOL = 2
CONNECTOR_PROTOCOL = 2

# A connector that sends no ``protocol_version`` predates this check.
LEGACY_PROTOCOL = 1

# Mode an agent runs in when ``llm_config.connector_mode`` is unset, and the
# mode set assumed for a connector that did not list ``capabilities.modes``.
DEFAULT_MODE = "operator"

COMPAT_OK = "ok"
COMPAT_TOO_OLD = "protocol_too_old"
COMPAT_TOO_NEW = "protocol_too_new"
# No ``hello`` has been seen yet (freshly paired), so nothing to judge.
COMPAT_UNKNOWN = "unknown"

# Dispatch-failure reasons (``TASK_DISPATCH_FAILED`` / session close reason).
REASON_INCOMPATIBLE = "connector_incompatible"
REASON_MODE_UNSUPPORTED = "connector_mode_unsupported"

_MAJOR = re.compile(r"^\s*(\d+)")


def parse_protocol_version(raw: Any) -> int:
    """Major protocol number from a ``hello`` value; garbage counts as legacy.

    ``None``/absent is the old connector that never sent the field, so it is
    reported as :data:`LEGACY_PROTOCOL` rather than accepted silently.
    """
    if isinstance(raw, bool):
        return LEGACY_PROTOCOL
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        match = _MAJOR.match(raw)
        if match:
            return int(match.group(1))
    return LEGACY_PROTOCOL


def compatibility(version: Optional[str], capabilities: Optional[Mapping[str, Any]]) -> str:
    """``ok`` / ``protocol_too_old`` / ``protocol_too_new`` / ``unknown``."""
    caps = capabilities or {}
    if version is None and "protocol_version" not in caps:
        return COMPAT_UNKNOWN
    protocol = parse_protocol_version(caps.get("protocol_version"))
    if protocol < MIN_CONNECTOR_PROTOCOL:
        return COMPAT_TOO_OLD
    if protocol > CONNECTOR_PROTOCOL:
        return COMPAT_TOO_NEW
    return COMPAT_OK


def supported_modes(version: Optional[str], capabilities: Optional[Mapping[str, Any]]) -> Optional[set[str]]:
    """Modes the connector reported, or ``None`` when it has not said hello."""
    caps = capabilities or {}
    modes = caps.get("modes")
    if isinstance(modes, (list, tuple)):
        return {str(m) for m in modes}
    if version is None and "protocol_version" not in caps:
        return None
    return {DEFAULT_MODE}


def dispatch_block_reason(
    version: Optional[str], capabilities: Optional[Mapping[str, Any]], mode: str
) -> Optional[str]:
    """Why a task in ``mode`` must not be dispatched to this connector, if so.

    A connector that never said hello is not blocked here: it is offline, and
    the dispatch timeout already covers it.
    """
    if compatibility(version, capabilities) in (COMPAT_TOO_OLD, COMPAT_TOO_NEW):
        return REASON_INCOMPATIBLE
    modes = supported_modes(version, capabilities)
    if modes is not None and mode not in modes:
        return REASON_MODE_UNSUPPORTED
    return None


def agent_mode(llm_config: Optional[Mapping[str, Any]]) -> str:
    return str((llm_config or {}).get("connector_mode") or DEFAULT_MODE)
