from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.core.secret_keyring import SecretKeyring
from app.core.secrets.crypto import SecretValue, encrypt_secret
from app.core.secrets.migration import LegacyCandidate
from app.core.secrets.operations import (
    remove_migrated_plaintext,
    rotate_secret_keyring,
    verify_secret_store,
)
from app.models.secret import SecretKind, SecretState, UserSecret


try:
    import nacl  # noqa: F401
except ImportError:
    HAS_NACL = False
else:
    HAS_NACL = True


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _OperationsSession:
    def __init__(self, *rows, count: int = 0):
        self.rows = iter(rows)
        self.count = count
        self.added = []
        self.flushed = False

    async def execute(self, _statement):
        return _Result(next(self.rows))

    async def scalar(self, _statement):
        return self.count or None

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flushed = True


def _keyring(*, active_key_id: str = "current", extra: bool = False) -> SecretKeyring:
    keys = {active_key_id: os.urandom(32)}
    if extra:
        keys["old"] = os.urandom(32)
    return SecretKeyring(active_key_id=active_key_id, keys=keys, path="test")


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NACL, reason="PyNaCl is installed in the production image")
async def test_verify_reports_active_and_stale_bindings_without_values(monkeypatch):
    keyring = _keyring()
    secret = UserSecret(
        id=uuid4(), owner_id=uuid4(), kind=SecretKind.LLM_API_KEY, label="Test", key_id="current", ciphertext=b"x"
    )
    secret.ciphertext, secret.key_id, secret.aad_version = encrypt_secret(
        SecretValue("test-only-value"),
        secret_id=secret.id,
        owner_id=secret.owner_id,
        kind=secret.kind,
        keyring=keyring,
    )
    secret.state = SecretState.ACTIVE
    session = _OperationsSession([secret], [])

    async def empty_discovery(_db):
        return [], []

    monkeypatch.setattr("app.core.secrets.operations.discover_legacy_sources", empty_discovery)
    report = await verify_secret_store(session, keyring=keyring, decrypt_sample_size=1)

    assert report.secrets_by_state == {"active": 1}
    assert report.decrypt_checked == 1
    assert report.decrypt_failed == 0
    assert report.healthy
    assert "test-only-value" not in repr(report)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NACL, reason="PyNaCl is installed in the production image")
async def test_rotation_reencrypts_with_active_key_without_exposing_value():
    keyring = _keyring(extra=True)
    secret = UserSecret(
        id=uuid4(), owner_id=uuid4(), kind=SecretKind.LLM_API_KEY, label="Test", key_id="old", ciphertext=b"x"
    )
    secret.ciphertext, secret.key_id, secret.aad_version = encrypt_secret(
        SecretValue("rotation-test-value"),
        secret_id=secret.id,
        owner_id=secret.owner_id,
        kind=secret.kind,
        keyring=keyring,
        key_id="old",
    )
    session = _OperationsSession([secret])

    report = await rotate_secret_keyring(
        session, keyring=keyring, new_key_id="current", apply=True
    )

    assert report.rotated == 1
    assert report.failed == 0
    assert secret.key_id == "current"
    assert secret.version == 2
    assert "rotation-test-value" not in repr(report)


@pytest.mark.asyncio
async def test_cutover_refuses_uncheckpointed_plaintext(monkeypatch):
    candidate = LegacyCandidate(
        source_table="agent",
        resource_id=uuid4(),
        owner_id=uuid4(),
        json_path="llm_config.api_key",
        kind=SecretKind.LLM_API_KEY,
        purpose="llm_primary",
        label="Test",
        provider=None,
        value=SecretValue("do-not-remove-without-checkpoint"),
    )

    async def discovery(_db):
        return [candidate], []

    monkeypatch.setattr("app.core.secrets.operations.discover_legacy_sources", discovery)
    report = await remove_migrated_plaintext(_OperationsSession(), apply=False)

    assert report.removed == 0
    assert report.blocked == 1
    assert report.issues[0]["error_code"] == "missing_completed_migration"
    assert "do-not-remove-without-checkpoint" not in repr(report)
