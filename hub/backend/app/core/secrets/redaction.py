"""Defence-in-depth recursive secret redaction for logs and API serializers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .crypto import SecretValue


REDACTED = "[REDACTED]"
_EXACT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "authorization",
    "password",
    "token",
    "secret",
    "ciphertext",
    "nonce",
    "private_key",
    "privatekey",
}
_SAFE_REFERENCE_KEYS = {
    "secret_id",
    "secret_binding_id",
    "secretbindingid",
    "key_id",
    "keyid",
}


def _normalise_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def is_sensitive_key(key: Any) -> bool:
    normalised = _normalise_key(key)
    compact = normalised.replace("_", "")
    if (
        normalised in _SAFE_REFERENCE_KEYS
        or compact in _SAFE_REFERENCE_KEYS
        or normalised.endswith("_secret_id")
        or normalised.endswith("_binding_id")
    ):
        return False
    if normalised in _EXACT_SENSITIVE_KEYS or compact in _EXACT_SENSITIVE_KEYS:
        return True
    return (
        normalised.endswith("_api_key")
        or normalised.endswith("_access_token")
        or normalised.endswith("_authorization")
        or normalised.endswith("_password")
        or normalised.endswith("_private_key")
        or normalised.endswith("_secret")
        or normalised.startswith("secret_")
    )


def redact_secrets(value: Any, *, replacement: Any = REDACTED) -> Any:
    """Recursively redact values in mappings, sequences, and model objects."""

    if isinstance(value, SecretValue):
        return replacement
    if isinstance(value, Mapping):
        return {
            key: replacement if is_sensitive_key(key) else redact_secrets(item, replacement=replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item, replacement=replacement) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item, replacement=replacement) for item in value)
    if isinstance(value, set):
        return {redact_secrets(item, replacement=replacement) for item in value}
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return redact_secrets(value.model_dump(), replacement=replacement)
    return value


def sanitize_response(value: Any) -> Any:
    """Remove sensitive mapping keys entirely from public representations."""

    if isinstance(value, SecretValue):
        return None
    if isinstance(value, Mapping):
        return {
            key: sanitize_response(item)
            for key, item in value.items()
            if not is_sensitive_key(key)
        }
    if isinstance(value, list):
        return [sanitize_response(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_response(item) for item in value)
    if isinstance(value, set):
        return {sanitize_response(item) for item in value}
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return sanitize_response(value.model_dump())
    return value


_TEXT_SECRET_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(sorted(_EXACT_SENSITIVE_KEYS)) + r")\b\s*[:=]\s*"
    r"(?:Bearer\s+)?['\"]?([^\s'\"]+)"
)


def redact_secrets_in_text(text: str) -> str:
    """Best-effort redaction of ``key: value`` / ``key=value`` secret-looking
    spans inside free-form text (e.g. a raw tool error message).

    Unlike :func:`redact_secrets`, this has no structure to key off of — it
    is a heuristic, not a guarantee. Use it only for user-facing summaries
    that are already truncated/low-detail by design (see
    ``safe_tool_error_text`` in ``app.core.tools.registry``), never as the
    sole protection for a full raw payload.
    """

    def _sub(match: "re.Match[str]") -> str:
        return f"{match.group(1)}={REDACTED}"

    return _TEXT_SECRET_PATTERN.sub(_sub, text)


def assert_no_secret_fields(value: Any, *, path: str = "config") -> None:
    """Reject plaintext-looking config keys on new public writes."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_sensitive_key(key):
                raise ValueError(f"{path}.{key} must reference a SecretStore binding")
            assert_no_secret_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            assert_no_secret_fields(item, path=f"{path}[{index}]")
