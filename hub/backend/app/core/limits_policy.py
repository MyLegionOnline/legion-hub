"""Limits policy (internal design notes stage 2).

``workspace_quota.get_effective_limits`` / ``get_user_tier`` keep their
signatures (18 modules call them) and delegate here. Who decides the limits:

* a plugin that called ``PluginHost.set_limits_policy`` (the cloud: tariffs,
  ``legion_cloud.tariffs``);
* otherwise the self-hosted ``config`` policy: built-in limits, then
  ``legion.toml``, then ``LEGION_LIMIT_*``.
"""

from __future__ import annotations

import difflib
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import workspace_quota as wq
from app.plugins import get_registry

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "LEGION_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/legion/legion.toml"
LIMIT_ENV_PREFIX = "LEGION_LIMIT_"
SELFHOSTED_TIER = "selfhosted"

# Counters the runtime reserves slots against: they must stay finite integers.
FINITE_KEYS = frozenset({"max_running", "mcp_personal_max_running"})
LIST_KEYS = frozenset({"market_paid_tiers"})
_UNLIMITED_WORDS = frozenset({"", "none", "null", "unlimited"})


class LimitsConfigError(RuntimeError):
    """legion.toml or a LEGION_LIMIT_* variable is wrong; startup must stop."""


@runtime_checkable
class LimitsPolicy(Protocol):
    async def effective_limits(self, user_id: UUID, db: AsyncSession) -> dict: ...

    async def user_tier(self, user_id: UUID, db: AsyncSession) -> str: ...


class ConfigLimitsPolicy:
    """The same limits for every user of a self-hosted Hub, from configuration.

    Built-in defaults are ``SELFHOSTED_LIMITS``: no caps,
    except the counters that must be finite. ``legion.toml`` ``[limits]``
    overrides them and ``LEGION_LIMIT_<KEY>`` overrides the file.
    """

    def __init__(self, limits: Mapping[str, Any]) -> None:
        self._limits = dict(limits)

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "ConfigLimitsPolicy":
        env = os.environ if environ is None else environ
        limits = {key: wq.SELFHOSTED_LIMITS[key] for key in wq._LIMIT_KEYS}
        limits["market_paid_tiers"] = list(limits["market_paid_tiers"] or [])

        explicit = path or env.get(CONFIG_PATH_ENV)
        config_file = Path(explicit or DEFAULT_CONFIG_PATH)
        if config_file.is_file():
            for key, raw in _read_limits_table(config_file).items():
                limits[_check_key(key, f"{config_file} [limits]")] = _coerce(key, raw, str(config_file))
        elif explicit:
            raise LimitsConfigError(f"{CONFIG_PATH_ENV} points to {explicit!r}, which is not a file")

        for name, raw in env.items():
            if not name.startswith(LIMIT_ENV_PREFIX):
                continue
            if not raw.strip():
                # compose passes an unset ${LEGION_LIMIT_X:-} as "": the file decides.
                continue
            key = name[len(LIMIT_ENV_PREFIX):].lower()
            limits[_check_key(key, name)] = _coerce(key, raw, name)
        return cls(limits)

    async def effective_limits(self, user_id: UUID, db: AsyncSession) -> dict:
        limits = dict(self._limits)
        limits["market_paid_tiers"] = list(limits["market_paid_tiers"])
        limits["plan_tier"] = SELFHOSTED_TIER
        limits["override_active"] = False
        return limits

    async def user_tier(self, user_id: UUID, db: AsyncSession) -> str:
        return SELFHOSTED_TIER


def _read_limits_table(config_file: Path) -> dict:
    try:
        with config_file.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise LimitsConfigError(f"{config_file}: invalid TOML: {exc}") from exc
    table = data.get("limits", {})
    if not isinstance(table, dict):
        raise LimitsConfigError(f"{config_file}: [limits] must be a table")
    return table


def _check_key(key: str, source: str) -> str:
    if key in wq._LIMIT_KEYS:
        return key
    hint = difflib.get_close_matches(key, wq._LIMIT_KEYS, n=1)
    suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
    raise LimitsConfigError(
        f"{source}: unknown limit {key!r}.{suggestion} "
        f"Valid limits: {', '.join(wq._LIMIT_KEYS)}"
    )


def _coerce(key: str, raw: Any, source: str) -> Any:
    """Validate one value; None means no cap. ``source`` names the origin in errors."""
    if key in LIST_KEYS:
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.split(",") if part.strip()]
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise LimitsConfigError(f"{source}: {key} must be a list of strings")
        return raw

    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in _UNLIMITED_WORDS:
            if key in FINITE_KEYS:
                raise LimitsConfigError(
                    f"{source}: {key} must be a whole number; it cannot be unlimited"
                )
            return None
        try:
            raw = int(text)
        except ValueError:
            raise LimitsConfigError(
                f"{source}: {key} must be a whole number or 'unlimited', got {raw!r}"
            ) from None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise LimitsConfigError(
            f"{source}: {key} must be a whole number or 'unlimited', got {raw!r}"
        )
    if raw < 0:
        raise LimitsConfigError(f"{source}: {key} must not be negative, got {raw}")
    return raw


_default_policy: Optional[LimitsPolicy] = None


def get_limits_policy() -> LimitsPolicy:
    """The policy in force: the plugin's, else the configured default."""
    plugin_policy = get_registry().limits_policy
    if plugin_policy is not None:
        return plugin_policy
    global _default_policy
    if _default_policy is None:
        _default_policy = ConfigLimitsPolicy.load()
    return _default_policy


def reset_limits_policy() -> None:
    """Forget the cached default (tests, config reload)."""
    global _default_policy
    _default_policy = None
