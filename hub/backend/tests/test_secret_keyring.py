from __future__ import annotations

import base64
import json
import os

import pytest

from app.core.secret_keyring import SecretKeyringError, load_keyring, read_protected_file


def _payload(key_id: str = "2026-08-04") -> dict:
    return {
        "active_key_id": key_id,
        "keys": {key_id: base64.b64encode(b"k" * 32).decode("ascii")},
    }


def _write(path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def test_load_keyring_validates_and_keeps_key_bytes_in_memory(tmp_path):
    path = tmp_path / "keyring.json"
    _write(path, _payload())

    keyring = load_keyring(path)

    assert keyring.active_key_id == "2026-08-04"
    assert keyring.keys["2026-08-04"] == b"k" * 32
    assert str(path) in keyring.path


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"active_key_id": "missing", "keys": {}},
        {"active_key_id": "2026-08-04", "keys": {"2026-08-04": "not-base64"}},
        {"active_key_id": "2026-08-04", "keys": {"2026-08-04": base64.b64encode(b"short").decode()}},
    ],
)
def test_load_keyring_rejects_malformed_documents(tmp_path, payload):
    path = tmp_path / "keyring.json"
    _write(path, payload)

    with pytest.raises(SecretKeyringError):
        load_keyring(path)


@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX mode bits")
def test_load_keyring_rejects_group_or_world_access(tmp_path, monkeypatch):
    # The dev container sets this to tolerate Docker Desktop's bind-mount mode.
    monkeypatch.delenv("SECRETS_DEV_ALLOW_INSECURE_KEYRING_PERMISSIONS", raising=False)
    path = tmp_path / "keyring.json"
    _write(path, _payload())
    path.chmod(0o644)

    with pytest.raises(SecretKeyringError, match="permissions"):
        load_keyring(path)


def test_protected_file_does_not_accept_empty_value(tmp_path):
    path = tmp_path / "empty.secret"
    path.write_text("\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(SecretKeyringError, match="empty"):
        read_protected_file(path, name="TEST_SECRET")
