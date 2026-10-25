"""Controlled rollout policy for verified HTTP MCP discovery.

The flag is evaluated at request time so an operator can move from the
internal cohort to canary or roll back without deleting bindings or durable
snapshots.  Cohort identifiers are configuration only; they never become
metric labels or user-visible diagnostics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID


ROLLOUT_STAGES = frozenset({"off", "internal", "canary", "on"})


def _ids(*names: str) -> frozenset[str]:
    values: set[str] = set()
    for name in names:
        values.update(
            value.strip().lower()
            for value in os.getenv(name, "").split(",")
            if value.strip()
        )
    return frozenset(values)


@dataclass(frozen=True)
class HttpMcpRolloutPolicy:
    stage: str
    user_ids: frozenset[str]

    @property
    def enabled(self) -> bool:
        return self.stage != "off"

    def allows_user(self, owner_id: UUID | str | None) -> bool:
        if not self.enabled or owner_id is None:
            return False
        if self.stage in {"internal", "canary"}:
            return str(owner_id).lower() in self.user_ids
        return not self.user_ids or str(owner_id).lower() in self.user_ids


def rollout_policy() -> HttpMcpRolloutPolicy:
    """Read the flag and cohort on every call for safe live rollback."""

    stage = os.getenv("HTTP_MCP_VERIFIED_DISCOVERY_V1", "on").strip().lower()
    if stage not in ROLLOUT_STAGES:
        stage = "off"
    user_ids = _ids(
        "HTTP_MCP_ROLLOUT_USER_IDS",
        f"HTTP_MCP_{stage.upper()}_USER_IDS" if stage != "off" else "HTTP_MCP_ROLLOUT_USER_IDS",
    )
    return HttpMcpRolloutPolicy(stage=stage, user_ids=user_ids)


def enabled_for(owner_id: UUID | str | None) -> bool:
    return rollout_policy().allows_user(owner_id)


__all__ = ["HttpMcpRolloutPolicy", "ROLLOUT_STAGES", "enabled_for", "rollout_policy"]
