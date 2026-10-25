"""Binding a new LLM key replaces the old one instead of stacking on it.

``bind`` upserts on (secret, resource, purpose), which is correct for MCP
servers that legitimately draw several secrets at once.  An agent's LLM slots
hold exactly one key each, so switching providers used to leave the previous
secret bound to the agent as well.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.secrets.store import BindingTarget, PostgresEncryptedSecretStore
from app.models.secret import (
    SecretActorType,
    SecretAuditAction,
    SecretPurpose,
    SecretResourceType,
)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.deleted = []
        self.flushes = 0

    async def execute(self, _statement):
        return _FakeResult(self._rows)

    async def delete(self, row):
        self.deleted.append(row)

    async def flush(self):
        self.flushes += 1


def _store(rows):
    store = PostgresEncryptedSecretStore.__new__(PostgresEncryptedSecretStore)
    store.db = _FakeSession(rows)
    return store


def _binding(secret_id=None):
    return SimpleNamespace(id=uuid4(), secret_id=secret_id or uuid4())


def _target(purpose, resource_id):
    return BindingTarget(
        resource_type=SecretResourceType.AGENT,
        resource_id=resource_id,
        purpose=purpose,
    )


async def _displace(store, keep, target, audits):
    async def _record_audit(**kwargs):
        audits.append(kwargs)

    store._audit = _record_audit
    await store._displace_single_bindings(
        owner_id=uuid4(),
        keep=keep,
        target=target,
        request_id=None,
        source_ip=None,
        actor_type=SecretActorType.USER,
    )


@pytest.mark.asyncio
async def test_switching_the_primary_key_unbinds_the_previous_secret():
    stale = _binding()
    store = _store([stale])
    keep = _binding()
    audits: list[dict] = []

    await _displace(store, keep, _target(SecretPurpose.LLM_PRIMARY, uuid4()), audits)

    assert store.db.deleted == [stale]
    assert store.db.flushes == 1
    assert [a["action"] for a in audits] == [SecretAuditAction.UNBIND]
    assert audits[0]["secret_id"] == stale.secret_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "purpose",
    [SecretPurpose.LLM_PRIMARY, SecretPurpose.LLM_MEDIA, SecretPurpose.LLM_VISION],
)
async def test_every_llm_slot_holds_one_secret(purpose):
    stale_a, stale_b = _binding(), _binding()
    store = _store([stale_a, stale_b])

    await _displace(store, _binding(), _target(purpose, uuid4()), [])

    assert store.db.deleted == [stale_a, stale_b]


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", [SecretPurpose.MCP_ENV, SecretPurpose.MCP_AUTH_HEADER])
async def test_mcp_purposes_keep_their_multiple_secrets(purpose):
    store = _store([_binding(), _binding()])

    await _displace(store, _binding(), _target(purpose, uuid4()), [])

    assert store.db.deleted == []
    assert store.db.flushes == 0


@pytest.mark.asyncio
async def test_rebinding_the_same_secret_deletes_nothing():
    store = _store([])
    audits: list[dict] = []

    await _displace(store, _binding(), _target(SecretPurpose.LLM_PRIMARY, uuid4()), audits)

    assert store.db.deleted == []
    assert store.db.flushes == 0
    assert audits == []
