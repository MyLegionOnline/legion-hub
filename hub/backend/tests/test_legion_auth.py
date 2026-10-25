"""The Hub signs in through a Legion account (internal design notes stage 2).

Unit part: the OIDC flow cookie and PKCE, id_token checks against a JWKS,
the login page config, and the password form / registration switches.

The credential push from Market and the DB-backed sign-in flows belong to the
cloud plugin and are tested there.
"""

from __future__ import annotations

import os
import time
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwk, jwt

from app.core import legion_auth
from app.core.legion_auth import LegionAuthError

ISSUER = "https://market.legion.test"
CLIENT_ID = "legion-hub"
PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _legion_settings(monkeypatch):
    monkeypatch.setattr(legion_auth, "LEGION_ISSUER", ISSUER)
    monkeypatch.setattr(legion_auth, "LEGION_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(legion_auth, "LEGION_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(legion_auth, "LEGION_ACCOUNTS_TOKEN", "test-accounts-token")
    monkeypatch.setattr(legion_auth, "LEGION_REDIRECT_URI", "https://hub.legion.test/auth/legion/callback")
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "legion")
    legion_auth.set_market_available(None)
    legion_auth.reset_jwks_cache()
    legion_auth._attempts.clear()
    yield
    legion_auth.set_market_available(None)
    legion_auth.reset_jwks_cache()


# ---------- a signing key and the JWKS Market would publish -------------------

class Signer:
    def __init__(self, kid: str = "k1"):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = kid
        self.pem = key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode()
        public_pem = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        self.jwk = {**jwk.construct(public_pem, "RS256").to_dict(), "kid": kid, "use": "sig", "alg": "RS256"}

    def id_token(self, *, sub: str, email: str, nonce: str, tv: int = 0, verified: bool = True, **overrides) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER, "aud": CLIENT_ID, "sub": sub, "iat": now, "exp": now + 900, "tv": tv,
            "email": email, "email_verified": verified, "nonce": nonce, "auth_time": now,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.pem, algorithm="RS256", headers={"kid": self.kid})

    def publish(self, monkeypatch, *, counter: list | None = None):
        async def fetch():
            if counter is not None:
                counter.append(1)
            legion_auth._jwks_cache["keys"] = {self.kid: self.jwk}
            legion_auth._jwks_cache["fetched_at"] = time.monotonic()

        monkeypatch.setattr(legion_auth, "_fetch_jwks", fetch)


@pytest.fixture(scope="module")
def signer() -> Signer:
    return Signer()


# ---------- unit: flow cookie, PKCE, id_token ------------------------------------

def test_start_builds_an_authorize_request_with_pkce_and_state():
    url, cookie = legion_auth.start_flow("login")
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == f"{ISSUER}/oauth/authorize"
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    assert q["response_type"] == "code" and q["client_id"] == CLIENT_ID
    assert q["code_challenge_method"] == "S256" and "prompt" not in q
    flow = legion_auth.read_flow(cookie, q["state"])
    assert legion_auth.pkce_challenge(flow["verifier"]) == q["code_challenge"]
    assert flow["nonce"] == q["nonce"]

    register_url, _ = legion_auth.start_flow("register")
    assert parse_qs(urlparse(register_url).query)["prompt"] == ["create"]


def test_flow_cookie_must_match_the_state_and_be_ours():
    _, cookie = legion_auth.start_flow("login")
    with pytest.raises(LegionAuthError) as missing:
        legion_auth.read_flow(None, "x")
    assert missing.value.code == "flow_missing"
    with pytest.raises(LegionAuthError) as mismatch:
        legion_auth.read_flow(cookie, "another-state")
    assert mismatch.value.code == "state_mismatch"
    forged = jwt.encode({"state": "s", "nonce": "n", "verifier": "v", "exp": int(time.time()) + 60},
                        "not-the-hub-key", algorithm="HS256")
    with pytest.raises(LegionAuthError) as bad:
        legion_auth.read_flow(forged, "s")
    assert bad.value.code == "flow_expired"


@pytest.mark.asyncio
async def test_id_token_is_checked_against_the_jwks(signer, monkeypatch):
    signer.publish(monkeypatch)
    sub = str(uuid4())
    claims = await legion_auth.verify_id_token(signer.id_token(sub=sub, email="a@b.test", nonce="n1"), "n1")
    assert claims["sub"] == sub and claims["tv"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("override, nonce, code", [
    ({"aud": "someone-else"}, "n1", "bad_token"),
    ({"iss": "https://evil.test"}, "n1", "bad_token"),
    ({"exp": int(time.time()) - 10}, "n1", "bad_token"),
    ({}, "another-nonce", "nonce_mismatch"),
    ({"tv": "0"}, "n1", "bad_token"),
])
async def test_bad_id_tokens_are_refused(signer, monkeypatch, override, nonce, code):
    signer.publish(monkeypatch)
    token = signer.id_token(sub=str(uuid4()), email="a@b.test", nonce="n1", **override)
    with pytest.raises(LegionAuthError) as refused:
        await legion_auth.verify_id_token(token, nonce)
    assert refused.value.code == code


@pytest.mark.asyncio
async def test_a_token_signed_by_another_key_is_refused(signer, monkeypatch):
    signer.publish(monkeypatch)
    stranger = Signer(kid=signer.kid)  # same kid, different key
    with pytest.raises(LegionAuthError):
        await legion_auth.verify_id_token(stranger.id_token(sub=str(uuid4()), email="a@b.test", nonce="n"), "n")


@pytest.mark.asyncio
async def test_an_unknown_kid_refetches_the_jwks_once(signer, monkeypatch):
    fetched: list = []
    signer.publish(monkeypatch, counter=fetched)
    token = signer.id_token(sub=str(uuid4()), email="a@b.test", nonce="n")
    await legion_auth.verify_id_token(token, "n")
    await legion_auth.verify_id_token(token, "n")
    assert len(fetched) == 1
    rotated = Signer(kid="k2")
    with pytest.raises(LegionAuthError) as unknown:
        await legion_auth.verify_id_token(rotated.id_token(sub=str(uuid4()), email="a@b.test", nonce="n"), "n")
    assert unknown.value.code == "unknown_key"
    assert len(fetched) == 1  # the refetch is rate-limited


def test_login_page_config_follows_the_mode_and_market_health(monkeypatch):
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "local")
    assert legion_auth.auth_config() == {
        "mode": "local", "legion": None, "password_login": True, "registration": "local",
    }
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "legion")
    up = legion_auth.auth_config()
    assert up["legion"]["available"] and not up["password_login"] and up["registration"] == "legion"
    assert up["legion"]["account_url"] == f"{ISSUER}/account"
    assert up["legion"]["billing_url"] == f"{ISSUER}/account/billing"
    assert up["legion"]["pricing_url"] == f"{ISSUER}/pricing"
    # The documents of Market's services, linked from the Hub (legal plan L4).
    assert up["legion"]["legal_urls"] == {
        "terms": f"{ISSUER}/terms",
        "privacy": f"{ISSUER}/privacy",
        "acceptable_use": f"{ISSUER}/acceptable-use",
    }
    legion_auth.set_market_available(False)
    down = legion_auth.auth_config()
    assert down["password_login"] and down["registration"] == "unavailable"
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "both")
    legion_auth.set_market_available(True)
    assert legion_auth.auth_config()["password_login"]


# ---------- the auth API without a database -------------------------------------

def _app(session_factory=None) -> FastAPI:
    from app.api.v1 import auth
    from app.db.session import get_session

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1/auth")
    if session_factory is not None:
        async def _session():
            async with session_factory() as db:
                yield db
        app.dependency_overrides[get_session] = _session
    else:
        async def _no_db():
            yield None
        app.dependency_overrides[get_session] = _no_db
    return app


def test_the_password_form_is_refused_while_legion_is_up():
    client = TestClient(_app())
    response = client.post("/api/v1/auth/login", data={"username": "a@b.test", "password": "x"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "legion_required"


def test_registration_goes_through_legion_and_waits_while_it_is_down():
    client = TestClient(_app())
    body = {"email": "a@example.com", "password": "secret-secret", "full_name": "A"}
    assert client.post("/api/v1/auth/register", json=body).status_code == 403
    legion_auth.set_market_available(False)
    assert client.post("/api/v1/auth/register", json=body).status_code == 503


def test_start_redirects_to_legion_with_the_flow_cookie():
    client = TestClient(_app())
    response = client.get("/api/v1/auth/legion/start?intent=register", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith(f"{ISSUER}/oauth/authorize?")
    assert "prompt=create" in response.headers["location"]
    cookie = response.headers["set-cookie"]
    assert "legion_oidc=" in cookie and "HttpOnly" in cookie and "Path=/api/v1/auth/legion" in cookie


class _CredentialDb:
    """``db.get(LegionCredential, id)`` answering one row or None."""

    def __init__(self, token_version=None):
        self.row = None if token_version is None else type("Row", (), {"token_version": token_version})()

    async def get(self, model, key):
        return self.row


@pytest.mark.asyncio
@pytest.mark.parametrize("account, stored, claims, revoked", [
    (None, 3, {}, False),               # a local user: Legion has no say
    (uuid4(), None, {}, False),         # no credential copy pushed yet
    (uuid4(), 1, {"tv": 1}, False),
    (uuid4(), 2, {"tv": 1}, True),      # signed out everywhere after this token
    (uuid4(), 1, {}, True),             # a token from before tv was put in it
    (uuid4(), 0, {}, False),
])
async def test_a_hub_session_ends_with_sign_out_everywhere(account, stored, claims, revoked):
    assert await legion_auth.session_revoked(_CredentialDb(stored), account, claims) is revoked


@pytest.mark.asyncio
async def test_a_hub_session_token_carries_the_account_token_version():
    from app.core import security

    assert await legion_auth.current_token_version(_CredentialDb(4), None) is None
    assert await legion_auth.current_token_version(_CredentialDb(4), uuid4()) == 4
    assert await legion_auth.current_token_version(_CredentialDb(None), uuid4()) == 0
    assert security.decode_access_token(security.create_access_token("u", token_version=4))["tv"] == 4
    assert "tv" not in security.decode_access_token(security.create_access_token("u"))


