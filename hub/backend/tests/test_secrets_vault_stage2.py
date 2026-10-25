from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.core.secrets.crypto import SecretDecryptionError, SecretValue, decrypt_secret, encrypt_secret
from app.core.secrets.redaction import redact_secrets, sanitize_response
from app.core.secret_keyring import SecretKeyring
from app.models.secret import SecretKind
from app.schemas.agent import AgentOut
from app.schemas.room import RoomOut


def test_secret_value_and_recursive_redaction_never_include_plaintext():
    value = SecretValue("super-secret-value")

    assert str(value) == "<secret>"
    assert repr(value) == "<secret>"
    assert "super-secret-value" not in repr(value)
    assert redact_secrets({"nested": [{"Authorization": value}]}).get("nested") == [
        {"Authorization": "[REDACTED]"}
    ]
    assert sanitize_response({"api_key": "super-secret-value", "secret_binding_id": "binding"}) == {
        "secret_binding_id": "binding"
    }


def test_legacy_agent_and_room_responses_drop_plaintext_fields():
    agent = AgentOut(
        id=uuid4(),
        owner_id=uuid4(),
        nickname="agent",
        agent_uid="agent-1",
        agent_type="internal",
        status="offline",
        version="1",
        update_policy="manual",
        config={"nested": {"PASSWORD": "raw-password"}},
        llm_config={"provider": "openai", "api_key": "raw-api-key"},
        token="legacy-agent-token",
    )
    assert "api_key" not in agent.llm_config
    assert "PASSWORD" not in agent.config["nested"]
    assert agent.token is None

    room = RoomOut(
        id=uuid4(),
        creator_id=uuid4(),
        name="room",
        git_config={"repo_url": "https://example.test/repo", "access_token": "raw-token"},
    )
    assert room.git_config == {"repo_url": "https://example.test/repo"}


def _keyring(*, active_key_id: str = "test") -> SecretKeyring:
    return SecretKeyring(active_key_id=active_key_id, keys={active_key_id: os.urandom(32)}, path="test")


@pytest.mark.crypto
def test_aead_binds_identity_and_uses_a_fresh_nonce():
    pytest.importorskip("nacl")
    keyring = _keyring()
    secret_id = uuid4()
    owner_id = uuid4()
    value = SecretValue("value")

    first, key_id, aad_version = encrypt_secret(
        value,
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.LLM_API_KEY,
        keyring=keyring,
    )
    second, _, _ = encrypt_secret(
        value,
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.LLM_API_KEY,
        keyring=keyring,
    )
    assert first != second
    assert decrypt_secret(
        first,
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.LLM_API_KEY,
        key_id=key_id,
        aad_version=aad_version,
        keyring=keyring,
    ).reveal() == "value"

    with pytest.raises(SecretDecryptionError):
        decrypt_secret(
            first,
            secret_id=secret_id,
            owner_id=uuid4(),
            kind=SecretKind.LLM_API_KEY,
            key_id=key_id,
            aad_version=aad_version,
            keyring=keyring,
        )


@pytest.mark.crypto
def test_aead_rotation_reads_old_key_and_writes_active_key():
    pytest.importorskip("nacl")
    old_id = "old"
    new_id = "new"
    keyring = SecretKeyring(
        active_key_id=new_id,
        keys={old_id: os.urandom(32), new_id: os.urandom(32)},
        path="test",
    )
    secret_id = uuid4()
    owner_id = uuid4()
    ciphertext, key_id, aad_version = encrypt_secret(
        SecretValue("old-value"),
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.CUSTOM,
        keyring=keyring,
        key_id=old_id,
    )
    assert key_id == old_id
    assert decrypt_secret(
        ciphertext,
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.CUSTOM,
        key_id=old_id,
        aad_version=aad_version,
        keyring=keyring,
    ).reveal() == "old-value"

    _, active_key_id, _ = encrypt_secret(
        SecretValue("new-value"),
        secret_id=secret_id,
        owner_id=owner_id,
        kind=SecretKind.CUSTOM,
        keyring=keyring,
    )
    assert active_key_id == new_id
