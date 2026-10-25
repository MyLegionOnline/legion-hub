"""Protected-file backed configuration for system credentials."""

from __future__ import annotations

import os
from enum import StrEnum

from app.core.secret_keyring import SecretKeyringError, read_protected_file


def strict_secrets_enabled() -> bool:
    raw = os.getenv("SECRETS_STRICT_CONFIG", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class SecretStoreMode(StrEnum):
    """Runtime policy for legacy plaintext secret reads."""

    LEGACY = "legacy"
    DUAL_READ = "dual_read"
    STRICT = "strict"


def secret_store_mode() -> SecretStoreMode:
    """Return the configured migration mode without exposing secret values."""

    raw = os.getenv("SECRETS_STORE_MODE", "dual_read").strip().lower()
    try:
        return SecretStoreMode(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SECRETS_STORE_MODE must be one of: legacy, dual_read, strict"
        ) from exc


def allow_legacy_secret_read() -> bool:
    return secret_store_mode() in {SecretStoreMode.LEGACY, SecretStoreMode.DUAL_READ}


def allow_legacy_secret_write() -> bool:
    raw = os.getenv("SECRETS_ALLOW_LEGACY_WRITE", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def configured_secret(name: str, default: str | None = None) -> str | None:
    """Resolve ``NAME_FILE`` first, then ``NAME``, with an optional dev fallback."""

    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        try:
            return read_protected_file(file_path, name=name)
        except SecretKeyringError as exc:
            raise RuntimeError(f"required protected setting {name} is invalid") from exc

    value = os.getenv(name)
    if value is not None and value != "":
        return value
    if strict_secrets_enabled():
        raise RuntimeError(f"required protected setting {name} is not configured")
    return default
