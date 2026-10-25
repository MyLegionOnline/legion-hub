"""Controlled rollout policy for managed stdio personal MCPs.

The policy is deliberately environment-driven so operators can move through
internal -> canary -> limited rollout without changing application code.  A
global kill switch only blocks new imports/starts; existing rows remain
readable and can still be stopped or deleted for safe rollback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.personal_mcp import PersonalMcpRuntime


ROLLOUT_STAGES = frozenset({"off", "internal", "canary", "limited"})


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _ids(*names: str) -> frozenset[str]:
    values: set[str] = set()
    for name in names:
        values.update(item.strip().lower() for item in os.getenv(name, "").split(",") if item.strip())
    return frozenset(values)


@dataclass(frozen=True)
class PersonalMcpRolloutPolicy:
    enabled: bool
    stage: str
    user_ids: frozenset[str]
    max_users: int
    max_packages: int

    @property
    def launches_enabled(self) -> bool:
        return self.enabled and self.stage != "off"

    def allows_user(self, owner_id: UUID | str | None) -> bool:
        if not self.launches_enabled or owner_id is None:
            return False
        # `limited` may be opened by an explicit user list or by a bounded cap
        # for a small plan cohort. Internal/canary always require a list.
        if self.stage in {"internal", "canary"}:
            return str(owner_id).lower() in self.user_ids
        return not self.user_ids or str(owner_id).lower() in self.user_ids


def rollout_policy() -> PersonalMcpRolloutPolicy:
    stage = os.getenv("PERSONAL_MCP_ROLLOUT_STAGE", "off").strip().lower()
    if stage not in ROLLOUT_STAGES:
        stage = "off"
    user_ids = _ids(
        "PERSONAL_MCP_ROLLOUT_USER_IDS",
        f"PERSONAL_MCP_{stage.upper()}_USER_IDS" if stage != "off" else "PERSONAL_MCP_ROLLOUT_USER_IDS",
    )
    defaults = {"off": (0, 0), "internal": (25, 20), "canary": (100, 100), "limited": (1000, 500)}
    default_users, default_packages = defaults[stage]
    return PersonalMcpRolloutPolicy(
        enabled=_bool("PERSONAL_MCP_MANAGED_STDIO_ENABLED", False),
        stage=stage,
        user_ids=user_ids,
        max_users=_int("PERSONAL_MCP_ROLLOUT_MAX_USERS", default_users),
        max_packages=_int("PERSONAL_MCP_ROLLOUT_MAX_PACKAGES", default_packages),
    )


def package_name(launch_spec: dict[str, Any] | None) -> str:
    value = (launch_spec or {}).get("package")
    if isinstance(value, dict):
        value = value.get("registry_name") or value.get("requested")
    if not isinstance(value, str):
        return "unknown"
    return value.split("@", 1)[0].strip().lower()[:214] or "unknown"


async def rollout_usage(db: AsyncSession) -> tuple[set[str], set[str]]:
    result = await db.execute(
        select(PersonalMcpRuntime.owner_id, PersonalMcpRuntime.launch_spec).where(
            PersonalMcpRuntime.actual_status != "deleted"
        )
    )
    owners: set[str] = set()
    packages: set[str] = set()
    for owner_id, launch_spec in result.all():
        owners.add(str(owner_id).lower())
        packages.add(package_name(launch_spec if isinstance(launch_spec, dict) else {}))
    return owners, packages


async def rollout_capacity(
    owner_id: UUID,
    launch_spec: dict[str, Any],
    db: AsyncSession,
) -> dict[str, Any] | None:
    """Return a stable denial descriptor, or ``None`` when admission is safe."""

    policy = rollout_policy()
    if not policy.allows_user(owner_id):
        return {
            "reason": "user_not_in_rollout",
            "stage": policy.stage,
            "max_users": policy.max_users,
            "max_packages": policy.max_packages,
        }
    owners, packages = await rollout_usage(db)
    if policy.max_users and str(owner_id).lower() not in owners and len(owners) >= policy.max_users:
        return {
            "reason": "user_cap",
            "stage": policy.stage,
            "max_users": policy.max_users,
            "current_users": len(owners),
        }
    package = package_name(launch_spec)
    if policy.max_packages and package not in packages and len(packages) >= policy.max_packages:
        return {
            "reason": "package_cap",
            "stage": policy.stage,
            "max_packages": policy.max_packages,
            "current_packages": len(packages),
        }
    return None


__all__ = [
    "PersonalMcpRolloutPolicy",
    "ROLLOUT_STAGES",
    "package_name",
    "rollout_capacity",
    "rollout_policy",
    "rollout_usage",
]
