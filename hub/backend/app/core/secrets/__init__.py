"""Typed, encrypted user-secret storage for Hub."""

from .crypto import (
    AAD_VERSION,
    MAX_SECRET_BYTES,
    SecretDecryptionError,
    SecretValue,
    build_aad,
    decrypt_secret,
    encrypt_secret,
)
from .redaction import redact_secrets, sanitize_response
from .service import SecretService
from .runtime import (
    MCP_UPSTREAM_AUTH_HEADER,
    McpSecretUnavailable,
    binding_id_from_config,
    legacy_secret_value,
    resolve_agent_secret,
    resolve_for_use,
    resolve_mcp_upstream_auth,
)
from .store import (
    BindingMeta,
    BindingTarget,
    SecretAccessDenied,
    SecretNotFound,
    SecretStoreError,
    SecretUnavailable,
    SecretMeta,
    SecretStore,
    PostgresEncryptedSecretStore,
)
from .lease import (
    ConsumedSecretLease,
    IssuedSecretLease,
    ManagedMcpSecretLeaseService,
    SecretLeaseDenied,
    SecretLeaseError,
    SecretLeaseExpired,
    SecretLeaseService,
)

__all__ = [
    "AAD_VERSION",
    "MAX_SECRET_BYTES",
    "SecretDecryptionError",
    "SecretValue",
    "build_aad",
    "decrypt_secret",
    "encrypt_secret",
    "redact_secrets",
    "sanitize_response",
    "SecretService",
    "SecretMeta",
    "BindingMeta",
    "BindingTarget",
    "SecretAccessDenied",
    "SecretNotFound",
    "SecretStoreError",
    "SecretUnavailable",
    "SecretStore",
    "PostgresEncryptedSecretStore",
    "ConsumedSecretLease",
    "IssuedSecretLease",
    "ManagedMcpSecretLeaseService",
    "SecretLeaseDenied",
    "SecretLeaseError",
    "SecretLeaseExpired",
    "SecretLeaseService",
    "binding_id_from_config",
    "MCP_UPSTREAM_AUTH_HEADER",
    "McpSecretUnavailable",
    "legacy_secret_value",
    "resolve_agent_secret",
    "resolve_for_use",
    "resolve_mcp_upstream_auth",
]
