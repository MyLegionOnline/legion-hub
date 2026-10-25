"""Resumable, value-safe migration from legacy JSON secret fields.

This module deliberately has no HTTP surface.  It is used by the operator-only
``scripts/secrets_migrate.py`` command during the short ``dual_read`` window.
Every successful source field is moved in one database transaction with its
encrypted secret, binding, JSON rewrite, audit events, and checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.secret_keyring import SecretKeyring
from app.models.agent import Agent
from app.models.market import UserInstalledItem
from app.models.room import Room
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretKind,
    SecretMigrationCheckpoint,
    SecretPurpose,
    SecretResourceType,
    UserSecret,
)

from .crypto import SecretDecryptionError, SecretValue, decrypt_secret
from .store import BindingTarget, PostgresEncryptedSecretStore


def _fingerprint(value: SecretValue) -> str:
    """Return a one-way source fingerprint without exposing its plaintext."""

    return sha256(value.reveal().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacyCandidate:
    """One migratable legacy JSON value; its repr intentionally hides it."""

    source_table: str
    resource_id: UUID
    owner_id: UUID
    json_path: str
    kind: SecretKind
    purpose: SecretPurpose
    label: str
    provider: str | None
    value: SecretValue = field(repr=False)
    resource_type: SecretResourceType = SecretResourceType.AGENT
    injection_config: dict[str, Any] | None = None
    resource: Any = field(repr=False, compare=False, default=None)
    config_attr: str = field(repr=False, compare=False, default="")
    rewrite: Callable[[dict[str, Any], UUID], dict[str, Any]] = field(
        repr=False, compare=False, default=lambda config, binding_id: config
    )
    source_value: SecretValue | None = field(repr=False, compare=False, default=None)

    @property
    def source_hash(self) -> str:
        return _fingerprint(self.source_value or self.value)

    def rewritten_config(self, binding_id: UUID) -> dict[str, Any]:
        return self.rewrite(deepcopy(getattr(self.resource, self.config_attr) or {}), binding_id)


@dataclass(frozen=True)
class MigrationIssue:
    """A source that requires an operator decision, without its raw value."""

    source_table: str
    resource_id: UUID
    json_path: str
    error_code: str
    value: SecretValue = field(repr=False)
    resource: Any = field(repr=False, compare=False, default=None)

    @property
    def source_hash(self) -> str:
        return _fingerprint(self.value)


@dataclass
class MigrationReport:
    discovered: int = 0
    migrated: int = 0
    skipped: int = 0
    failed: int = 0
    decrypt_failed: int = 0
    by_path: dict[str, int] = field(default_factory=dict)
    issues: list[dict[str, str]] = field(default_factory=list)

    def add_issue(self, issue: MigrationIssue) -> None:
        self.failed += 1
        self.issues.append(
            {
                "source_table": issue.source_table,
                "resource_id": str(issue.resource_id),
                "json_path": issue.json_path,
                "error_code": issue.error_code,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "migrated": self.migrated,
            "skipped": self.skipped,
            "failed": self.failed,
            "decrypt_failed": self.decrypt_failed,
            "by_path": dict(sorted(self.by_path.items())),
            "issues": self.issues,
        }


def _string_value(raw: Any) -> SecretValue | None:
    if not isinstance(raw, str) or not raw:
        return None
    return SecretValue(raw)


def _set_top_level_binding(key: str, legacy_key: str) -> Callable[[dict[str, Any], UUID], dict[str, Any]]:
    def rewrite(config: dict[str, Any], binding_id: UUID) -> dict[str, Any]:
        config.pop(legacy_key, None)
        config[key] = str(binding_id)
        return config

    return rewrite


def _set_media_binding(config: dict[str, Any], binding_id: UUID) -> dict[str, Any]:
    media = dict(config.get("media") or {})
    media.pop("api_key", None)
    media["secret_binding_id"] = str(binding_id)
    config["media"] = media
    return config


def _set_mcp_auth_binding(config: dict[str, Any], binding_id: UUID) -> dict[str, Any]:
    headers = dict(config.get("headers") or {})
    headers.pop("Authorization", None)
    if headers:
        config["headers"] = headers
    else:
        config.pop("headers", None)
    config["auth"] = {"mode": "bearer", "secret_binding_id": str(binding_id)}
    return config


def _set_mcp_env_binding(name: str) -> Callable[[dict[str, Any], UUID], dict[str, Any]]:
    def rewrite(config: dict[str, Any], binding_id: UUID) -> dict[str, Any]:
        env = dict(config.get("env") or {})
        env.pop(name, None)
        if env:
            config["env"] = env
        else:
            config.pop("env", None)
        existing = config.get("env_bindings")
        bindings = [
            binding
            for binding in existing
            if isinstance(binding, dict) and binding.get("name") != name
        ] if isinstance(existing, list) else []
        bindings.append({"name": name, "binding_id": str(binding_id)})
        config["env_bindings"] = bindings
        config.pop("secret_env_migration_allowlist", None)
        return config

    return rewrite


async def discover_legacy_sources(db: AsyncSession, *, limit: int | None = None) -> tuple[list[LegacyCandidate], list[MigrationIssue]]:
    """Inventory known legacy fields without logging or returning their values."""

    candidates: list[LegacyCandidate] = []
    issues: list[MigrationIssue] = []

    agents = (await db.execute(select(Agent))).scalars().all()
    for agent in agents:
        config = agent.llm_config if isinstance(agent.llm_config, dict) else {}
        value = _string_value(config.get("api_key"))
        if value is not None:
            candidates.append(
                LegacyCandidate(
                    "agent", agent.id, agent.owner_id, "llm_config.api_key",
                    SecretKind.LLM_API_KEY, SecretPurpose.LLM_PRIMARY,
                    "Migrated LLM credential", str(config.get("provider"))[:64] if config.get("provider") else None,
                    value, SecretResourceType.AGENT, resource=agent, config_attr="llm_config",
                    rewrite=_set_top_level_binding("primary_secret_binding_id", "api_key"),
                )
            )
        media = config.get("media")
        value = _string_value(media.get("api_key")) if isinstance(media, dict) else None
        if value is not None:
            candidates.append(
                LegacyCandidate(
                    "agent", agent.id, agent.owner_id, "llm_config.media.api_key",
                    SecretKind.LLM_API_KEY, SecretPurpose.LLM_MEDIA,
                    "Migrated media credential", str(media.get("provider") or config.get("provider"))[:64] if (media.get("provider") or config.get("provider")) else None,
                    value, SecretResourceType.AGENT, resource=agent, config_attr="llm_config", rewrite=_set_media_binding,
                )
            )

    rooms = (await db.execute(select(Room))).scalars().all()
    for room in rooms:
        config = room.git_config if isinstance(room.git_config, dict) else {}
        for key, kind, purpose, binding_key, label in (
            ("access_token", SecretKind.GIT_ACCESS_TOKEN, SecretPurpose.GIT_HTTPS, "access_token_secret_binding_id", "Migrated Git credential"),
            ("webhook_secret", SecretKind.GIT_WEBHOOK_SECRET, SecretPurpose.GIT_WEBHOOK, "webhook_secret_binding_id", "Migrated Git webhook secret"),
        ):
            value = _string_value(config.get(key))
            if value is not None:
                candidates.append(
                    LegacyCandidate(
                        "room", room.id, room.creator_id, f"git_config.{key}", kind, purpose,
                        label, str(config.get("provider"))[:64] if config.get("provider") else None,
                        value, SecretResourceType.ROOM, resource=room, config_attr="git_config",
                        rewrite=_set_top_level_binding(binding_key, key),
                    )
                )

    items = (await db.execute(select(UserInstalledItem).where(UserInstalledItem.type == "mcp"))).scalars().all()
    for item in items:
        config = item.custom_config if isinstance(item.custom_config, dict) else {}
        headers = config.get("headers")
        raw_authorization = headers.get("Authorization") if isinstance(headers, dict) else None
        source_value = _string_value(raw_authorization)
        if source_value is not None and raw_authorization.startswith("Bearer ") and raw_authorization[7:]:
            candidates.append(
                LegacyCandidate(
                    "user_installed_item", item.id, item.user_id, "custom_config.headers.Authorization",
                    SecretKind.MCP_HEADER, SecretPurpose.MCP_AUTH_HEADER, "Migrated MCP authorization",
                    None, SecretValue(raw_authorization[7:]), SecretResourceType.INSTALLED_MCP, {"mode": "bearer"}, item,
                    "custom_config", _set_mcp_auth_binding, source_value=source_value,
                )
            )
        elif source_value is not None:
            issues.append(
                MigrationIssue(
                    "user_installed_item", item.id, "custom_config.headers.Authorization",
                    "mcp_auth_requires_bearer", source_value, item,
                )
            )
        env = config.get("env")
        allowlist = config.get("secret_env_migration_allowlist")
        allowed_names = set(allowlist) if isinstance(allowlist, list) and all(isinstance(name, str) for name in allowlist) else set()
        if isinstance(env, dict):
            for name, raw in env.items():
                value = _string_value(raw)
                if value is None:
                    continue
                if name not in allowed_names or not name.isidentifier() or len(name) > 128:
                    issues.append(MigrationIssue("user_installed_item", item.id, f"custom_config.env.{name}", "mcp_env_requires_manual_opt_in", value, item))
                    continue
                candidates.append(
                    LegacyCandidate(
                        "user_installed_item", item.id, item.user_id, f"custom_config.env.{name}",
                        SecretKind.MCP_ENV, SecretPurpose.MCP_ENV, f"Migrated MCP environment {name}",
                        None, value, SecretResourceType.INSTALLED_MCP, {"name": name}, item,
                        "custom_config", _set_mcp_env_binding(name),
                    )
                )

    if limit is not None:
        return candidates[:limit], issues[: max(0, limit - len(candidates[:limit]))]
    return candidates, issues


class LegacySecretMigrator:
    """Apply a bounded, checkpointed batch against one database session."""

    def __init__(self, db: AsyncSession, *, keyring: SecretKeyring):
        self.db = db
        self.store = PostgresEncryptedSecretStore(db, keyring=keyring)
        self.keyring = keyring

    async def _checkpoint_exists(self, source_table: str, resource_id: UUID, json_path: str, source_hash: str) -> bool:
        row = await self.db.scalar(
            select(SecretMigrationCheckpoint.id).where(
                SecretMigrationCheckpoint.source_table == source_table,
                SecretMigrationCheckpoint.resource_id == resource_id,
                SecretMigrationCheckpoint.json_path == json_path,
                SecretMigrationCheckpoint.source_hash == source_hash,
                SecretMigrationCheckpoint.state == "completed",
            )
        )
        return row is not None

    async def _record_issue(self, issue: MigrationIssue) -> None:
        existing = await self.db.scalar(
            select(SecretMigrationCheckpoint.id).where(
                SecretMigrationCheckpoint.source_table == issue.source_table,
                SecretMigrationCheckpoint.resource_id == issue.resource_id,
                SecretMigrationCheckpoint.json_path == issue.json_path,
                SecretMigrationCheckpoint.source_hash == issue.source_hash,
            )
        )
        if existing is not None:
            return
        if issue.resource is not None:
            issue.resource.status = "disabled"
            self.db.add(issue.resource)
        self.db.add(
            SecretMigrationCheckpoint(
                source_table=issue.source_table,
                resource_id=issue.resource_id,
                json_path=issue.json_path,
                source_hash=issue.source_hash,
                state="failed",
                error_code=issue.error_code,
            )
        )
        self.db.add(
            SecretAudit(
                actor_type=SecretActorType.MIGRATION,
                action=SecretAuditAction.MIGRATE,
                outcome=SecretAuditOutcome.FAILED,
                resource_type=SecretResourceType.INSTALLED_MCP,
                resource_id=issue.resource_id,
                error_code=issue.error_code,
            )
        )
        await self.db.flush()

    async def _migrate_candidate(self, candidate: LegacyCandidate) -> tuple[UUID, UUID] | None:
        if await self._checkpoint_exists(candidate.source_table, candidate.resource_id, candidate.json_path, candidate.source_hash):
            return None
        async with self.db.begin_nested():
            secret = await self.store.create(
                owner_id=candidate.owner_id,
                kind=candidate.kind,
                label=candidate.label,
                provider=candidate.provider,
                value=candidate.value,
                actor_type=SecretActorType.MIGRATION,
            )
            binding = await self.store.bind(
                owner_id=candidate.owner_id,
                secret_id=secret.id,
                target=BindingTarget(
                    candidate.resource_type,
                    candidate.resource_id,
                    candidate.purpose,
                    candidate.injection_config,
                ),
                actor_type=SecretActorType.MIGRATION,
            )
            setattr(candidate.resource, candidate.config_attr, candidate.rewritten_config(binding.id))
            flag_modified(candidate.resource, candidate.config_attr)
            self.db.add(candidate.resource)
            self.db.add(
                SecretMigrationCheckpoint(
                    source_table=candidate.source_table,
                    resource_id=candidate.resource_id,
                    json_path=candidate.json_path,
                    source_hash=candidate.source_hash,
                    state="completed",
                    secret_id=secret.id,
                    binding_id=binding.id,
                )
            )
            self.db.add(
                SecretAudit(
                    actor_type=SecretActorType.MIGRATION,
                    actor_id=candidate.owner_id,
                    action=SecretAuditAction.MIGRATE,
                    outcome=SecretAuditOutcome.SUCCESS,
                    secret_id=secret.id,
                    resource_type=candidate.resource_type,
                    resource_id=candidate.resource_id,
                    purpose=candidate.purpose,
                )
            )
            await self.db.flush()
            return secret.id, binding.id

    async def _record_candidate_failure(self, candidate: LegacyCandidate) -> None:
        """Persist a value-free failure marker after a rolled-back savepoint."""

        if await self._checkpoint_exists(
            candidate.source_table,
            candidate.resource_id,
            candidate.json_path,
            candidate.source_hash,
        ):
            return
        existing = await self.db.scalar(
            select(SecretMigrationCheckpoint.id).where(
                SecretMigrationCheckpoint.source_table == candidate.source_table,
                SecretMigrationCheckpoint.resource_id == candidate.resource_id,
                SecretMigrationCheckpoint.json_path == candidate.json_path,
                SecretMigrationCheckpoint.source_hash == candidate.source_hash,
            )
        )
        if existing is None:
            self.db.add(
                SecretMigrationCheckpoint(
                    source_table=candidate.source_table,
                    resource_id=candidate.resource_id,
                    json_path=candidate.json_path,
                    source_hash=candidate.source_hash,
                    state="failed",
                    error_code="migration_failed",
                )
            )
            self.db.add(
                SecretAudit(
                    actor_type=SecretActorType.MIGRATION,
                    actor_id=candidate.owner_id,
                    action=SecretAuditAction.MIGRATE,
                    outcome=SecretAuditOutcome.FAILED,
                    resource_type=candidate.resource_type,
                    resource_id=candidate.resource_id,
                    purpose=candidate.purpose,
                    error_code="migration_failed",
                )
            )
            await self.db.flush()

    async def run(self, *, batch_size: int, apply: bool, sample_size: int = 3) -> MigrationReport:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        candidates, issues = await discover_legacy_sources(self.db, limit=batch_size)
        report = MigrationReport(discovered=len(candidates) + len(issues))
        for candidate in candidates:
            report.by_path[candidate.json_path] = report.by_path.get(candidate.json_path, 0) + 1
        for issue in issues:
            report.add_issue(issue)
        if not apply:
            return report

        for issue in issues:
            await self._record_issue(issue)

        secret_ids: list[UUID] = []
        for candidate in candidates:
            try:
                result = await self._migrate_candidate(candidate)
            except Exception:
                # The checkpoint/audit records only a stable code.  Do not log
                # exception text because database drivers may include values.
                await self._record_candidate_failure(candidate)
                report.add_issue(
                    MigrationIssue(
                        candidate.source_table,
                        candidate.resource_id,
                        candidate.json_path,
                        "migration_failed",
                        candidate.value,
                    )
                )
                continue
            if result is None:
                report.skipped += 1
            else:
                report.migrated += 1
                secret_ids.append(result[0])

        for secret_id in secret_ids[:sample_size]:
            checkpoint = await self.db.scalar(
                select(SecretMigrationCheckpoint).where(SecretMigrationCheckpoint.secret_id == secret_id)
            )
            if checkpoint is None:
                report.decrypt_failed += 1
                continue
            row = await self.db.get(UserSecret, secret_id)
            try:
                if row is None:
                    raise SecretDecryptionError("missing")
                value = decrypt_secret(
                    row.ciphertext,
                    secret_id=row.id,
                    owner_id=row.owner_id,
                    kind=row.kind,
                    key_id=row.key_id,
                    aad_version=row.aad_version,
                    keyring=self.keyring,
                )
                value = None
            except SecretDecryptionError:
                report.decrypt_failed += 1
        return report
