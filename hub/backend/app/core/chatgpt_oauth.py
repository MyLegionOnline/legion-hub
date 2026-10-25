"""Public-client SIWC with Hub-owned encrypted, serialized token renewal."""
from __future__ import annotations
import json
import secrets
import time
from uuid import uuid4
import httpx
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.chatgpt_credentials import PROVIDER, parse_reference
from app.core.secrets.crypto import SecretValue, decrypt_secret
from app.core.secrets.runtime import service_for
from app.core.secrets.store import SecretUnavailable
from app.models.secret import SecretState, UserSecret

ISSUER = "https://auth.openai.com"
AUTHORIZE = f"{ISSUER}/api/accounts/authorize"
TOKEN = f"{ISSUER}/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"

class ChatGPTError(RuntimeError):
    """Safe message plus structured diagnostics; never stringify credential bodies."""
    def __init__(self, message, *, code=None, status=None, request_id=None, response_body=None, retryable=False):
        super().__init__(message)
        self.code = code
        self.status_code = status
        self.request_id = request_id
        self.response_body = response_body
        self.retryable = retryable

TERMINAL_REFRESH_ERRORS = {
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused",
}

def registration_mapping(bundle):
    return {key: bundle.get(key) for key in ("client_id", "sub", "email")}


async def token_request(fields: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            response = await client.post(TOKEN, data={**fields, "resource": RESOURCE})
        if response.status_code != 200:
            error = response.json().get("error")
            code = error.get("code") if isinstance(error, dict) else error
            if code in TERMINAL_REFRESH_ERRORS:
                raise ChatGPTError("ChatGPT authorization expired or was revoked. Reconnect in Settings.", code=code)
            raise ChatGPTError(f"ChatGPT token exchange failed (HTTP {response.status_code}). Try again later.", code=code)
        result = response.json()
        if not result.get("access_token") or result.get("token_type", "").lower() != "bearer":
            raise ChatGPTError("OpenAI returned an incomplete token response")
        return result
    except (httpx.HTTPError, ValueError, TypeError):
        raise ChatGPTError("ChatGPT authorization service is unavailable. Try again later.") from None

async def verify_identity(token: str, client_id: str, nonce: str | None = None) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            discovery = await client.get(f"{ISSUER}/.well-known/openid-configuration")
            discovery.raise_for_status()
            config = discovery.json()
            jwks_url = config["jwks_uri"]
            if not jwks_url.startswith(f"{ISSUER}/") or config["issuer"] != ISSUER:
                raise ValueError
            response = await client.get(jwks_url)
            response.raise_for_status()
        claims = jwt.decode(token, response.json(), algorithms=["RS256"], audience=client_id,
                            issuer=ISSUER, options={"require_exp": True, "require_sub": True, "require_aud": True, "require_iss": True,
                                                   "verify_at_hash": False})
        if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise ValueError
        return claims
    except Exception:
        raise ChatGPTError("Could not verify the ChatGPT account. Start authorization again.") from None

def saved_tokens(result: dict, *, client_id: str, subject: str, email: str | None, previous: dict | None = None) -> dict:
    previous = previous or {}
    granted = result.get("scope", previous.get("scope", ""))
    if DIRECT_SCOPE not in granted.split():
        raise ChatGPTError("Permission to use your ChatGPT plan was not granted. Reconnect and allow plan usage.")
    if not result.get("refresh_token"):
        raise ChatGPTError("OpenAI did not return a renewable session. Reconnect in Settings.")
    return {"client_id": client_id, "sub": subject, "email": email, "scope": granted,
            "access_token": result["access_token"], "refresh_token": result["refresh_token"],
            "id_token": result.get("id_token", previous.get("id_token")),
            "expires_at": time.time() + int(result["expires_in"]),
            "earliest_refresh_at": result.get("earliest_refresh_at")}

def read_bundle(db: AsyncSession, row: UserSecret) -> dict:
    from app.core.secrets.store import _is_expired
    if row.provider != PROVIDER or row.state != SecretState.ACTIVE or _is_expired(row.expires_at):
        raise SecretUnavailable("ChatGPT connection is disabled or disconnected")
    value = decrypt_secret(row.ciphertext, secret_id=row.id, owner_id=row.owner_id, kind=row.kind,
                           key_id=row.key_id, aad_version=row.aad_version, keyring=service_for(db).store.keyring)
    try:
        bundle = json.loads(value.reveal())
        if not bundle.get("refresh_token") or not bundle.get("access_token") or DIRECT_SCOPE not in bundle.get("scope", "").split():
            raise ValueError
        return bundle
    except (ValueError, TypeError, AttributeError):
        raise SecretUnavailable("Reconnect this ChatGPT provider in Settings") from None

async def access_token(reference: str) -> str:
    """Resolve every request; a DB row lock serializes refreshes across all workers."""
    from app.db.session import engine
    owner_id, secret_id = parse_reference(reference)
    terminal_error = None
    async with AsyncSession(engine, expire_on_commit=False) as db:
        async with db.begin():
            row = (await db.execute(select(UserSecret).where(
                UserSecret.id == secret_id, UserSecret.owner_id == owner_id).with_for_update())).scalar_one_or_none()
            if row is None:
                raise SecretUnavailable("ChatGPT connection is unavailable")
            bundle = read_bundle(db, row)
            if bundle["expires_at"] <= time.time() + 60:
                try:
                    result = await token_request({"grant_type": "refresh_token", "client_id": bundle["client_id"],
                                                  "refresh_token": bundle["refresh_token"]})
                except ChatGPTError as exc:
                    if exc.code not in TERMINAL_REFRESH_ERRORS:
                        raise
                    terminal_error = exc
                    mapping = registration_mapping(bundle)
                    await service_for(db).store.replace_value(owner_id=owner_id, secret_id=secret_id,
                        value=SecretValue(json.dumps(mapping)), request_id=uuid4())
                    row.state = SecretState.DISABLED
                if terminal_error is None:
                    if result.get("id_token"):
                        identity = await verify_identity(result["id_token"], bundle["client_id"])
                        if identity["sub"] != bundle["sub"]:
                            raise ChatGPTError("ChatGPT account changed during renewal. Reconnect in Settings.")
                    bundle = saved_tokens(result, client_id=bundle["client_id"], subject=bundle["sub"],
                                          email=bundle.get("email"), previous=bundle)
                    await service_for(db).store.replace_value(owner_id=owner_id, secret_id=secret_id,
                                                             value=SecretValue(json.dumps(bundle)), request_id=uuid4())
    # Raise only AFTER committing removal of an unusable rotating session.
    if terminal_error is not None:
        raise terminal_error
    return bundle["access_token"]

async def model_catalog(reference: str) -> list[dict]:
    bearer = await access_token(reference)
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            response = await client.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {bearer}"})
        if response.status_code != 200:
            raise ChatGPTError(f"ChatGPT model catalog unavailable (HTTP {response.status_code}). Check your account access.")
        return [{"id": m["slug"], "name": m.get("display_name", m["slug"])}
                for m in response.json()["models"] if m.get("visibility") == "list"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        raise ChatGPTError("Could not load models for this ChatGPT account") from None

async def revoke(bundle: dict) -> bool:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.get(f"{ISSUER}/.well-known/openid-configuration")
            response.raise_for_status()
            endpoint = response.json()["revocation_endpoint"]
            if not endpoint.startswith(f"{ISSUER}/"):
                return False
            response = await client.post(endpoint, data={"token": bundle["refresh_token"],
                                         "token_type_hint": "refresh_token", "client_id": bundle["client_id"]})
        return response.status_code == 200
    except (httpx.HTTPError, ValueError, KeyError):
        return False
