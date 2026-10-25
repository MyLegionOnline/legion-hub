"""Account-scoped SIWC setup and one-use, ten-minute local-helper pairing."""
from __future__ import annotations
import base64
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.core import chatgpt_oauth as oauth
from app.core.chatgpt_credentials import PROVIDER, runtime_reference
from app.core.secrets.crypto import SecretValue, decrypt_secret, encrypt_secret
from app.core.secrets.runtime import service_for
from app.core.secrets.store import SecretStoreError
from app.db.session import get_session
from app.models.chatgpt_auth import ChatGPTAuthorization, ChatGPTHost
from app.models.provider_connection import ProviderConnection
from app.models.secret import SecretKind, SecretState, UserSecret
from app.models.user import User

router = APIRouter()
def now():
    return datetime.now(timezone.utc)
def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

async def owned(db, owner, connection_id):
    row = await db.get(ProviderConnection, connection_id)
    if row is None or row.user_id != owner or row.provider != PROVIDER:
        raise HTTPException(404, "ChatGPT connection not found")
    return row

def unpack(db, row):
    return json.loads(decrypt_secret(row.ciphertext, secret_id=row.id, owner_id=row.owner_id,
                      kind=SecretKind.CUSTOM, key_id=row.key_id, aad_version=row.aad_version,
                      keyring=service_for(db).store.keyring).reveal())

async def pending(db, ticket):
    row = (await db.execute(select(ChatGPTAuthorization).where(
        ChatGPTAuthorization.ticket_hash == digest(ticket)).with_for_update())).scalar_one_or_none()
    if row is None or row.expires_at.replace(tzinfo=timezone.utc) < now():
        raise HTTPException(410, "Setup link expired or was already used. Start again in Settings.")
    return row, unpack(db, row)

def registration(db, secret):
    if secret is None or secret.provider != PROVIDER or secret.state == SecretState.DELETED:
        return {}
    return json.loads(decrypt_secret(secret.ciphertext, secret_id=secret.id, owner_id=secret.owner_id,
                      kind=secret.kind, key_id=secret.key_id, aad_version=secret.aad_version,
                      keyring=service_for(db).store.keyring).reveal())

@router.post("/{connection_id}/setup")
async def setup(connection_id: UUID, db: AsyncSession = Depends(get_session),
                user: User = Depends(deps.get_current_user)):
    connection = await owned(db, user.id, connection_id)
    # A single durable host identity survives container recreation, browsers, and replicas.
    await db.execute(text("SELECT pg_advisory_xact_lock(hashtext('chatgpt_host'))"))
    host = await db.get(ChatGPTHost, 1)
    if host is None:
        host = ChatGPTHost()
        db.add(host)
    await db.execute(delete(ChatGPTAuthorization).where(
        (ChatGPTAuthorization.expires_at < now()) | (ChatGPTAuthorization.connection_id == connection_id)))
    existing = await db.get(UserSecret, connection.secret_id) if connection.secret_id else None
    if existing is not None and existing.owner_id != user.id:
        raise HTTPException(404, "ChatGPT connection not found")
    previous = registration(db, existing)
    ticket = secrets.token_urlsafe(32)
    row = ChatGPTAuthorization(owner_id=user.id, connection_id=connection.id,
                              ticket_hash=digest(ticket), expires_at=now() + timedelta(minutes=10),
                              ciphertext=b"", key_id="")
    payload = {"state": secrets.token_urlsafe(32), "nonce": secrets.token_urlsafe(32),
               "verifier": secrets.token_urlsafe(48), "host_id": host.host_id,
               "client_id": previous.get("client_id"), "sub": previous.get("sub"), "usage_denied": previous.get("usage_denied", False)}
    row.ciphertext, row.key_id, row.aad_version = encrypt_secret(
        SecretValue(json.dumps(payload)), secret_id=row.id, owner_id=user.id,
        kind=SecretKind.CUSTOM, keyring=service_for(db).store.keyring)
    db.add(row)
    await db.commit()
    # Only a short-lived setup capability is returned, never an OpenAI credential.
    return {"ticket": ticket, "expires_in": 600}

@router.get("/helper.py")
async def helper_source(user: User = Depends(deps.get_current_user)):
    source = (Path(__file__).parents[2] / "core" / "chatgpt_helper.py").read_text(encoding="utf-8")
    return Response(source, media_type="text/x-python", headers={"Cache-Control": "no-store"})

class Start(BaseModel):
    ticket: str = Field(min_length=40, max_length=100)
    port: int = Field(ge=1024, le=65535)

@router.post("/helper/start")
async def helper_start(body: Start, db: AsyncSession = Depends(get_session)):
    row, payload = await pending(db, body.ticket)
    if row.redirect_uri is not None:
        raise HTTPException(409, "This setup attempt has already started")
    row.redirect_uri = f"http://127.0.0.1:{body.port}/auth/callback"
    params = {"client_id": payload["client_id"] or "dynamic_agent_client", "ext_agent_host_id": payload["host_id"],
              "response_type": "code", "redirect_uri": row.redirect_uri, "scope": oauth.SCOPES,
              "resource": oauth.RESOURCE, "state": payload["state"], "nonce": payload["nonce"],
              "code_challenge_method": "S256",
              "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(payload["verifier"].encode()).digest()).rstrip(b"=").decode()}
    if not payload["client_id"]:
        params["agent_name_hint"] = "Legion"
    elif payload.get("usage_denied"):
        params["prompt"] = "consent"
    await db.commit()
    return {"authorize_url": oauth.AUTHORIZE + "?" + urlencode(params), "state": payload["state"]}

class Complete(BaseModel):
    ticket: str = Field(min_length=40, max_length=100)
    state: str = Field(max_length=200)
    code: str | None = Field(default=None, max_length=4096)
    client_id: str | None = Field(default=None, max_length=200)
    error: str | None = Field(default=None, max_length=200)

@router.post("/helper/complete")
async def helper_complete(body: Complete, db: AsyncSession = Depends(get_session)):
    row, payload = await pending(db, body.ticket)
    if not secrets.compare_digest(body.state, payload["state"]) or not row.redirect_uri:
        raise HTTPException(400, "Invalid OAuth transaction")
    connection = await owned(db, row.owner_id, row.connection_id)
    # Burn a valid attempt before the external exchange. A rollback must never permit code replay.
    owner_id, redirect_uri = row.owner_id, row.redirect_uri
    await db.delete(row)
    await db.commit()
    if body.error or not body.code:
        raise HTTPException(400, "ChatGPT authorization was declined. Start again in Settings.")
    client_id = payload["client_id"] or body.client_id
    if not client_id or not client_id.startswith("oaiapp_") or (payload["client_id"] and body.client_id not in (None, payload["client_id"])):
        raise HTTPException(400, "OpenAI returned an invalid client registration")
    try:
        result = await oauth.token_request({"grant_type": "authorization_code", "client_id": client_id,
                                           "code": body.code, "code_verifier": payload["verifier"], "redirect_uri": redirect_uri})
        identity = await oauth.verify_identity(result["id_token"], client_id, payload["nonce"])
        if payload["sub"] and identity["sub"] != payload["sub"]:
            raise oauth.ChatGPTError("This connection belongs to another ChatGPT account. Add a separate provider to switch accounts.")
        usage_denied = oauth.DIRECT_SCOPE not in result.get("scope", "").split()
        if usage_denied:
            # Keep the verified registration for explicit reconsent without enabling inference.
            bundle = {"client_id": client_id, "sub": identity["sub"], "email": identity.get("email"),
                      "scope": result.get("scope", ""), "id_token": result["id_token"], "usage_denied": True}
        else:
            bundle = oauth.saved_tokens(result, client_id=client_id, subject=identity["sub"], email=identity.get("email"))
    except oauth.ChatGPTError as exc:
        raise HTTPException(400, str(exc)) from None
    # Serialize completion against concurrent reconnects and deletion.
    connection = (await db.execute(select(ProviderConnection).where(
        ProviderConnection.id == connection.id, ProviderConnection.user_id == owner_id).with_for_update())).scalar_one_or_none()
    if connection is None:
        raise HTTPException(410, "This provider was removed")
    store = service_for(db).store
    if connection.secret_id:
        secret = await db.get(UserSecret, connection.secret_id)
        if secret is None or secret.owner_id != owner_id or secret.provider != PROVIDER:
            raise HTTPException(409, "This provider changed during authorization. Start again.")
        if secret.state == SecretState.DELETED:
            meta = await store.create(owner_id=owner_id, kind=SecretKind.LLM_API_KEY, provider=PROVIDER,
                                      label="ChatGPT subscription", value=SecretValue(json.dumps(bundle)), request_id=uuid4())
            connection.secret_id = meta.id
        else:
            await store.replace_value(owner_id=owner_id, secret_id=secret.id, value=SecretValue(json.dumps(bundle)), request_id=uuid4())
    else:
        meta = await store.create(owner_id=owner_id, kind=SecretKind.LLM_API_KEY, provider=PROVIDER,
                                  label="ChatGPT subscription", value=SecretValue(json.dumps(bundle)), request_id=uuid4())
        connection.secret_id = meta.id
    connection.status = "error" if usage_denied else "idle"
    connection.error_msg = "Permission to use your ChatGPT plan was not granted. Continue with ChatGPT to allow plan usage." if usage_denied else None
    if usage_denied:
        connection.models = []
        connection.selected_model = None
    await db.commit()
    return {"connected": not usage_denied, "error": connection.error_msg}

@router.get("/{connection_id}/status")
async def status(connection_id: UUID, db: AsyncSession = Depends(get_session),
                 user: User = Depends(deps.get_current_user)):
    connection = await owned(db, user.id, connection_id)
    secret = await db.get(UserSecret, connection.secret_id) if connection.secret_id else None
    bundle = registration(db, secret) if secret and secret.owner_id == user.id else {}
    pending_attempt = (await db.execute(select(ChatGPTAuthorization).where(
        ChatGPTAuthorization.connection_id == connection_id, ChatGPTAuthorization.owner_id == user.id,
        ChatGPTAuthorization.expires_at > now()))).scalar_one_or_none()
    connected = bool(secret and secret.state == SecretState.ACTIVE and bundle.get("refresh_token"))
    if pending_attempt is not None:
        phase = "authorizing" if pending_attempt.redirect_uri else "awaiting_helper"
    else:
        phase = "connected" if connected else "disconnected"
    return {"pending": pending_attempt is not None, "phase": phase,
            "connected": connected, "email": bundle.get("email"), "error": connection.error_msg}

@router.post("/{connection_id}/models")
async def models(connection_id: UUID, db: AsyncSession = Depends(get_session),
                 user: User = Depends(deps.get_current_user)):
    connection = await owned(db, user.id, connection_id)
    if not connection.secret_id:
        raise HTTPException(409, "Connect your ChatGPT account first")
    reference = runtime_reference(user.id, connection.secret_id).reveal()
    await db.commit()
    try:
        catalog = await oauth.model_catalog(reference)
    except (oauth.ChatGPTError, SecretStoreError) as exc:
        raise HTTPException(409, str(exc)) from None
    connection = await owned(db, user.id, connection_id)
    connection.models = catalog
    connection.status = "ok" if catalog else "error"
    connection.error_msg = None if catalog else "No eligible models are available for this account"
    if connection.selected_model not in {m["id"] for m in catalog}:
        connection.selected_model = catalog[0]["id"] if catalog else None
    await db.commit()
    return {"models": catalog, "secret_id": str(connection.secret_id), "selected_model": connection.selected_model}

async def disconnect_connection(db, connection):
    await db.execute(delete(ChatGPTAuthorization).where(ChatGPTAuthorization.connection_id == connection.id))
    if not connection.secret_id:
        return True
    secret = (await db.execute(select(UserSecret).where(UserSecret.id == connection.secret_id,
        UserSecret.owner_id == connection.user_id).with_for_update())).scalar_one_or_none()
    if secret is None or secret.provider != PROVIDER or secret.state == SecretState.DELETED:
        return True
    bundle = registration(db, secret)
    confirmed = await oauth.revoke(bundle) if bundle.get("refresh_token") else True
    # Keep only the registration mapping for later reauthorization, never bearer or refresh tokens.
    mapping = {key: bundle.get(key) for key in ("client_id", "sub", "email")}
    await service_for(db).store.replace_value(owner_id=connection.user_id, secret_id=secret.id,
                                             value=SecretValue(json.dumps(mapping)), request_id=uuid4())
    secret.state = SecretState.DISABLED
    connection.models = []
    connection.selected_model = None
    connection.status = "idle"
    connection.error_msg = None
    return confirmed

@router.post("/{connection_id}/disconnect")
async def disconnect(connection_id: UUID, db: AsyncSession = Depends(get_session),
                     user: User = Depends(deps.get_current_user)):
    connection = await owned(db, user.id, connection_id)
    confirmed = await disconnect_connection(db, connection)
    await db.commit()
    return {"disconnected": True, "remote_revocation_confirmed": confirmed}
