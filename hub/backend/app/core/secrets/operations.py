"""Value-safe operator operations for the SecretStore cutover.

These helpers intentionally return counters and stable identifiers only.  They
are used by the command-line tools that run outside the HTTP request path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.secret_keyring import SecretKeyring
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretMigrationCheckpoint,
    UserSecret,
)

from .crypto import SecretDecryptionError, decrypt_secret, encrypt_secret
from .migration import LegacyCandidate, discover_legacy_sources


@dataclass
class VerificationReport:
    """A value-free health snapshot suitable for cron output and reports."""

    secrets_by_state: dict[str, int] = field(default_factory=dict)
    bindings: int = 0
    stale_bindings: int = 0
    owner_mismatches: int = 0
    legacy_candidates: int = 0
    legacy_issues: int = 0
    failed_checkpoints: int = 0
    decrypt_checked: int = 0
    decrypt_failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def healthy(self) -> bool:
        return not any(
            (
                self.stale_bindings,
                self.owner_mismatches,
                self.legacy_issues,
                self.failed_checkpoints,
                self.decrypt_failed,
            )
        )


@dataclass
class CutoverReport:
    """Outcome of the final legacy-field cleanup."""

    discovered: int = 0
    removed: int = 0
    blocked: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RotationReport:
    """Value-free progress report for a key rotation."""

    discovered: int = 0
    rotated: int = 0
    already_current: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _completed_checkpoint(
    db: AsyncSession, candidate: LegacyCandidate
) -> SecretMigrationCheckpoint | None:
    return await db.scalar(
        select(SecretMigrationCheckpoint).where(
            SecretMigrationCheckpoint.source_table == candidate.source_table,
            SecretMigrationCheckpoint.resource_id == candidate.resource_id,
            SecretMigrationCheckpoint.json_path == candidate.json_path,
            SecretMigrationCheckpoint.source_hash == candidate.source_hash,
            SecretMigrationCheckpoint.state == "completed",
        )
    )


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


async def _candidate_is_migrated(
    db: AsyncSession, candidate: LegacyCandidate
) -> tuple[UUID, UUID] | None:
    """Return its binding only when the matching migration is intact."""

    checkpoint = await _completed_checkpoint(db, candidate)
    if checkpoint is None or checkpoint.secret_id is None or checkpoint.binding_id is None:
        return None
    binding = await db.get(SecretBinding, checkpoint.binding_id)
    if (
        binding is None
        or binding.secret_id != checkpoint.secret_id
        or binding.owner_id != candidate.owner_id
        or binding.resource_type != candidate.resource_type
        or binding.resource_id != candidate.resource_id
        or binding.purpose != candidate.purpose
    ):
        return None
    return binding.id, checkpoint.secret_id


async def remove_migrated_plaintext(
    db: AsyncSession, *, apply: bool
) -> CutoverReport:
    """Remove only legacy fields with an intact encrypted replacement.

    Unknown formats and incomplete migrations block the whole apply.  This
    prevents a cutover command from becoming a data-loss tool.
    """

    candidates, migration_issues = await discover_legacy_sources(db)
    report = CutoverReport(discovered=len(candidates) + len(migration_issues))
    for issue in migration_issues:
        report.blocked += 1
        report.issues.append(
            {
                "source_table": issue.source_table,
                "resource_id": str(issue.resource_id),
                "json_path": issue.json_path,
                "error_code": issue.error_code,
            }
        )

    ready: list[tuple[LegacyCandidate, UUID, UUID]] = []
    for candidate in candidates:
        migrated = await _candidate_is_migrated(db, candidate)
        if migrated is None:
            report.blocked += 1
            report.issues.append(
                {
                    "source_table": candidate.source_table,
                    "resource_id": str(candidate.resource_id),
                    "json_path": candidate.json_path,
                    "error_code": "missing_completed_migration",
                }
            )
        else:
            binding_id, secret_id = migrated
            ready.append((candidate, binding_id, secret_id))

    if not apply or report.blocked:
        return report

    for candidate, binding_id, secret_id in ready:
        setattr(candidate.resource, candidate.config_attr, candidate.rewritten_config(binding_id))
        flag_modified(candidate.resource, candidate.config_attr)
        db.add(candidate.resource)
        db.add(
            SecretAudit(
                actor_type=SecretActorType.MIGRATION,
                actor_id=candidate.owner_id,
                action=SecretAuditAction.MIGRATE,
                outcome=SecretAuditOutcome.SUCCESS,
                secret_id=secret_id,
                resource_type=candidate.resource_type,
                resource_id=candidate.resource_id,
                purpose=candidate.purpose,
            )
        )
        report.removed += 1
    await db.flush()
    return report


async def verify_secret_store(
    db: AsyncSession,
    *,
    keyring: SecretKeyring | None = None,
    decrypt_sample_size: int = 3,
) -> VerificationReport:
    """Check binding integrity and a bounded ciphertext sample without output."""

    if decrypt_sample_size < 0:
        raise ValueError("decrypt_sample_size cannot be negative")
    report = VerificationReport()
    secrets = (await db.execute(select(UserSecret).order_by(UserSecret.created_at))).scalars().all()
    secret_by_id = {secret.id: secret for secret in secrets}
    for secret in secrets:
        state = _enum_value(secret.state)
        report.secrets_by_state[state] = report.secrets_by_state.get(state, 0) + 1

    bindings = (await db.execute(select(SecretBinding))).scalars().all()
    report.bindings = len(bindings)
    for binding in bindings:
        secret = secret_by_id.get(binding.secret_id)
        if secret is None or _enum_value(secret.state) != "active":
            report.stale_bindings += 1
        elif secret.owner_id != binding.owner_id:
            report.owner_mismatches += 1

    candidates, issues = await discover_legacy_sources(db)
    report.legacy_candidates = len(candidates)
    report.legacy_issues = len(issues)
    report.failed_checkpoints = int(
        await db.scalar(
            select(func.count())
            .select_from(SecretMigrationCheckpoint)
            .where(SecretMigrationCheckpoint.state == "failed")
        )
        or 0
    )

    if keyring is not None:
        for secret in secrets[:decrypt_sample_size]:
            report.decrypt_checked += 1
            try:
                value = decrypt_secret(
                    secret.ciphertext,
                    secret_id=secret.id,
                    owner_id=secret.owner_id,
                    kind=secret.kind,
                    key_id=secret.key_id,
                    aad_version=secret.aad_version,
                    keyring=keyring,
                )
                value = None
            except SecretDecryptionError:
                report.decrypt_failed += 1
    return report


async def rotate_secret_keyring(
    db: AsyncSession,
    *,
    keyring: SecretKeyring,
    new_key_id: str,
    apply: bool,
) -> RotationReport:
    """Re-encrypt all stored ciphertext using an already-active keyring key."""

    if new_key_id != keyring.active_key_id or new_key_id not in keyring.keys:
        raise ValueError("new_key_id must be the active key in the loaded keyring")
    rows = (await db.execute(select(UserSecret).order_by(UserSecret.created_at))).scalars().all()
    report = RotationReport(discovered=len(rows))
    for secret in rows:
        if secret.key_id == new_key_id:
            report.already_current += 1
            continue
        try:
            value = decrypt_secret(
                secret.ciphertext,
                secret_id=secret.id,
                owner_id=secret.owner_id,
                kind=secret.kind,
                key_id=secret.key_id,
                aad_version=secret.aad_version,
                keyring=keyring,
            )
            if apply:
                ciphertext, key_id, _ = encrypt_secret(
                    value,
                    secret_id=secret.id,
                    owner_id=secret.owner_id,
                    kind=secret.kind,
                    keyring=keyring,
                    key_id=new_key_id,
                )
                secret.ciphertext = ciphertext
                secret.key_id = key_id
                secret.version += 1
                secret.updated_at = datetime.now(timezone.utc)
                db.add(secret)
                db.add(
                    SecretAudit(
                        actor_type=SecretActorType.SYSTEM,
                        action=SecretAuditAction.ROTATE,
                        outcome=SecretAuditOutcome.SUCCESS,
                        secret_id=secret.id,
                    )
                )
            value = None
            report.rotated += 1
        except SecretDecryptionError:
            report.failed += 1
            if apply:
                db.add(
                    SecretAudit(
                        actor_type=SecretActorType.SYSTEM,
                        action=SecretAuditAction.ROTATE,
                        outcome=SecretAuditOutcome.FAILED,
                        secret_id=secret.id,
                        error_code="decrypt_failed",
                    )
                )
    if apply:
        await db.flush()
    return report
