"""Fail-closed validation for the runtime secret keyring.

The keyring is deliberately kept separate from the database and is read only
at process startup. Encryption/decryption belongs to the SecretStore stage;
stage 1 only establishes a safe bootstrap boundary.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_KEYRING_PATH = "/run/secrets/legion_secret_keyring"
MAX_KEYRING_BYTES = 64 * 1024
KEY_BYTES = 32
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SecretKeyringError(RuntimeError):
    """Raised when a keyring cannot be trusted or parsed."""


@dataclass(frozen=True)
class SecretKeyring:
    """Validated key metadata and key bytes held in process memory."""

    active_key_id: str
    keys: dict[str, bytes]
    path: str


def _reject(reason: str) -> SecretKeyringError:
    return SecretKeyringError(f"secret keyring rejected: {reason}")


def _dev_allows_broad_keyring_permissions() -> bool:
    """Allow Docker Desktop's bind-mounted mode only in explicit dev mode."""

    raw = os.getenv("SECRETS_DEV_ALLOW_INSECURE_KEYRING_PERMISSIONS", "false").strip().lower()
    if raw in {"0", "false", "no", "off", ""}:
        return False
    if raw not in {"1", "true", "yes", "on"}:
        raise SecretKeyringError(
            "SECRETS_DEV_ALLOW_INSECURE_KEYRING_PERMISSIONS has an invalid value"
        )
    strict_raw = os.getenv("SECRETS_STRICT_CONFIG", "false").strip().lower()
    if strict_raw in {"1", "true", "yes", "on"}:
        raise _reject("development keyring permission override is forbidden in strict mode")
    return True


def _check_file_stat(
    file_stat: os.stat_result, *, allow_broad_permissions: bool = False
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise _reject("file is not regular")

    if os.name != "nt":
        mode = stat.S_IMODE(file_stat.st_mode)
        if not mode & stat.S_IRUSR:
            raise _reject("file is not owner-readable")
        if mode & 0o077 and not allow_broad_permissions:
            raise _reject("file permissions are broader than owner-only")
        geteuid = getattr(os, "geteuid", None)
        if (
            geteuid is not None
            and file_stat.st_uid != geteuid()
            and not allow_broad_permissions
        ):
            raise _reject("file owner does not match the backend process")


def _open_checked(path: Path) -> int:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(path, flags)
    except FileNotFoundError as exc:
        raise _reject("file is missing") from exc
    except OSError as exc:
        raise _reject("file cannot be opened") from exc


def read_protected_file(path: str | os.PathLike[str], *, name: str) -> str:
    """Read a small Docker-secret-style file after checking its metadata."""

    file_path = Path(path)
    fd = _open_checked(file_path)
    try:
        _check_file_stat(os.fstat(fd))
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            raw = stream.read(MAX_KEYRING_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)

    if len(raw) > MAX_KEYRING_BYTES:
        raise _reject(f"{name} file is too large")
    try:
        value = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise _reject(f"{name} file is not UTF-8") from exc
    if not value:
        raise _reject(f"{name} file is empty")
    return value


def _read_keyring_bytes(path: Path, *, allow_broad_permissions: bool = False) -> bytes:
    fd = _open_checked(path)
    try:
        _check_file_stat(os.fstat(fd), allow_broad_permissions=allow_broad_permissions)
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            raw = stream.read(MAX_KEYRING_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > MAX_KEYRING_BYTES:
        raise _reject("file is too large")
    return raw


def _decode_key(key_id: str, encoded: Any) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise _reject(f"key {key_id!r} is not base64 text")
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise _reject(f"key {key_id!r} is not valid base64") from exc
    if len(decoded) != KEY_BYTES:
        raise _reject(f"key {key_id!r} is not {KEY_BYTES} bytes")
    return decoded


def _parse_keyring(raw: bytes, *, path: str) -> SecretKeyring:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _reject("file is not valid UTF-8 JSON") from exc

    if not isinstance(document, dict):
        raise _reject("root must be an object")
    if set(document) != {"active_key_id", "keys"}:
        raise _reject("root must contain only active_key_id and keys")

    active_key_id = document.get("active_key_id")
    keys = document.get("keys")
    if not isinstance(active_key_id, str) or not _KEY_ID_RE.fullmatch(active_key_id):
        raise _reject("active_key_id is invalid")
    if not isinstance(keys, dict) or not keys:
        raise _reject("keys must be a non-empty object")

    decoded_keys: dict[str, bytes] = {}
    for key_id, encoded in keys.items():
        if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id):
            raise _reject("a key id is invalid")
        decoded_keys[key_id] = _decode_key(key_id, encoded)
    if active_key_id not in decoded_keys:
        raise _reject("active_key_id is not present in keys")
    return SecretKeyring(active_key_id=active_key_id, keys=decoded_keys, path=path)


def load_keyring(path: str | os.PathLike[str] | None = None) -> SecretKeyring:
    """Load and validate a keyring without printing or logging key material."""

    configured_path = path or os.getenv("SECRET_KEYRING_PATH") or DEFAULT_KEYRING_PATH
    file_path = Path(configured_path)
    if file_path.is_symlink():
        raise _reject("symlink is not allowed")
    return _parse_keyring(
        _read_keyring_bytes(
            file_path,
            allow_broad_permissions=_dev_allows_broad_keyring_permissions(),
        ),
        path=str(file_path),
    )


def keyring_required() -> bool:
    """Return whether the deployment opted into the fail-closed bootstrap."""

    raw = os.getenv("SECRETS_KEYRING_REQUIRED", "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise SecretKeyringError("SECRETS_KEYRING_REQUIRED has an invalid value")
