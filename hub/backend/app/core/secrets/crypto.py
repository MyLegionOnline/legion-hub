"""Cryptographic primitives used by the encrypted SecretStore.

This module deliberately delegates the primitive and nonce handling to
PyNaCl/libsodium.  It contains only the Legion-specific AAD and safe value
wrapper; it does not implement encryption, padding, or nonce generation.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from app.core.secret_keyring import SecretKeyring
from app.models.secret import SecretKind


AAD_VERSION = 1
MAX_SECRET_BYTES = 64 * 1024


class SecretCryptoError(RuntimeError):
    """Base class for controlled SecretStore cryptographic failures."""


class SecretDecryptionError(SecretCryptoError):
    """Ciphertext, AAD, or key material did not authenticate."""


class SecretValue:
    """A value whose normal string representations never reveal plaintext."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError("secret value must be text")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("secret value must be valid Unicode") from exc
        if not encoded:
            raise ValueError("secret value must not be empty")
        if len(encoded) > MAX_SECRET_BYTES:
            raise ValueError(f"secret value exceeds {MAX_SECRET_BYTES} UTF-8 bytes")
        self._value = value

    def reveal(self) -> str:
        """Return plaintext only at the final outbound-call boundary."""

        return self._value

    def __str__(self) -> str:
        return "<secret>"

    def __repr__(self) -> str:
        return "<secret>"

    def __bool__(self) -> bool:
        return True


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def build_aad(
    *,
    secret_id: UUID,
    owner_id: UUID,
    kind: SecretKind | str,
    key_id: str,
    schema_version: int = AAD_VERSION,
) -> bytes:
    """Build canonical authenticated metadata for one ciphertext row."""

    document = {
        "schema_version": schema_version,
        "secret_id": str(secret_id),
        "owner_id": str(owner_id),
        "kind": _enum_value(kind),
        "key_id": key_id,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _aead(key: bytes):
    try:
        from nacl.secret import Aead
    except ImportError as exc:  # pragma: no cover - deployment dependency check
        raise SecretCryptoError("PyNaCl is required for SecretStore encryption") from exc
    return Aead(key)


def encrypt_secret(
    value: SecretValue,
    *,
    secret_id: UUID,
    owner_id: UUID,
    kind: SecretKind | str,
    keyring: SecretKeyring,
    key_id: str | None = None,
) -> tuple[bytes, str, int]:
    """Encrypt one value with the active (or explicitly selected) key.

    PyNaCl's ``Aead`` prepends a fresh random nonce to the returned blob, so
    the nonce and authenticated ciphertext remain in the same DB field.
    """

    selected_key_id = key_id or keyring.active_key_id
    try:
        key = keyring.keys[selected_key_id]
    except KeyError as exc:
        raise SecretCryptoError("secret encryption key is not present in keyring") from exc
    aad = build_aad(
        secret_id=secret_id,
        owner_id=owner_id,
        kind=kind,
        key_id=selected_key_id,
    )
    try:
        ciphertext = bytes(_aead(key).encrypt(value.reveal().encode("utf-8"), aad=aad))
    except (TypeError, ValueError) as exc:
        raise SecretCryptoError("secret encryption failed") from exc
    return ciphertext, selected_key_id, AAD_VERSION


def decrypt_secret(
    ciphertext: bytes,
    *,
    secret_id: UUID,
    owner_id: UUID,
    kind: SecretKind | str,
    key_id: str,
    aad_version: int,
    keyring: SecretKeyring,
) -> SecretValue:
    """Authenticate and decrypt one row without exposing failure details."""

    if aad_version != AAD_VERSION:
        raise SecretDecryptionError("unsupported secret AAD version")
    try:
        key = keyring.keys[key_id]
    except KeyError as exc:
        raise SecretDecryptionError("secret key is not available") from exc
    aad = build_aad(
        secret_id=secret_id,
        owner_id=owner_id,
        kind=kind,
        key_id=key_id,
        schema_version=aad_version,
    )
    try:
        plaintext = bytes(_aead(key).decrypt(bytes(ciphertext), aad=aad)).decode("utf-8")
    except Exception as exc:  # libsodium exposes several backend-specific error types
        # Never include the ciphertext, plaintext, or low-level exception in the
        # application error.  The store writes only the stable error code.
        raise SecretDecryptionError("secret ciphertext failed authentication") from None
    try:
        return SecretValue(plaintext)
    except (TypeError, ValueError):
        raise SecretDecryptionError("secret plaintext is invalid") from None
