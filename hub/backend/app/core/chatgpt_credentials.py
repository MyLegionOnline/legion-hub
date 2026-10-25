"""Signed process-local references keep OAuth bundles out of API-key transports."""
import hashlib
import hmac
import secrets
from uuid import UUID
from app.core.secrets.crypto import SecretValue

PROVIDER = "openai_chatgpt"
PREFIX = "legion-chatgpt:"
_REFERENCE_KEY = secrets.token_bytes(32)

def runtime_reference(owner_id: UUID, secret_id: UUID) -> SecretValue:
    payload = f"{owner_id}:{secret_id}"
    signature = hmac.new(_REFERENCE_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return SecretValue(f"{PREFIX}{payload}:{signature}")

def parse_reference(value: str) -> tuple[UUID, UUID]:
    try:
        prefix, owner, secret, signature = value.split(":")
        expected = hmac.new(_REFERENCE_KEY, f"{owner}:{secret}".encode(), hashlib.sha256).hexdigest()
        if prefix != PREFIX[:-1] or not hmac.compare_digest(signature, expected):
            raise ValueError
        return UUID(owner), UUID(secret)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Reconnect this ChatGPT provider from Settings") from None
