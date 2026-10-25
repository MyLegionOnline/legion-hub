"""Who may register on a self-hosted Hub (internal design notes stage 4.3): no database."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import registration
from app.core.registration import RegistrationConfigError, RegistrationRefused


@pytest.fixture(autouse=True)
def _fresh_mode(monkeypatch):
    monkeypatch.delenv("LEGION_CONFIG", raising=False)
    monkeypatch.delenv("LEGION_REGISTRATION", raising=False)
    registration.reset_mode()
    yield
    registration.reset_mode()


def _toml(tmp_path, body: str) -> str:
    path = tmp_path / "legion.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


# --- the configured mode --------------------------------------------------------

def test_registration_is_closed_by_default(tmp_path):
    assert registration.load_mode(None, {}) == "closed"


def test_file_sets_the_mode_and_env_overrides_it(tmp_path):
    path = _toml(tmp_path, '[registration]\nmode = "invite"\n')
    assert registration.load_mode(path, {}) == "invite"
    assert registration.load_mode(path, {"LEGION_REGISTRATION": "OPEN"}) == "open"


def test_bad_values_stop_the_startup(tmp_path):
    with pytest.raises(RegistrationConfigError, match="Did you mean 'invite'"):
        registration.load_mode(_toml(tmp_path, '[registration]\nmode = "invit"\n'), {})
    with pytest.raises(RegistrationConfigError, match="unknown key 'mood'"):
        registration.load_mode(_toml(tmp_path, '[registration]\nmood = "open"\n'), {})
    with pytest.raises(RegistrationConfigError, match="must be a table"):
        registration.load_mode(_toml(tmp_path, 'registration = "open"\n'), {})
    with pytest.raises(RegistrationConfigError):
        registration.load_mode(None, {"LEGION_REGISTRATION": "everyone"})
    with pytest.raises(RegistrationConfigError, match="not a file"):
        registration.load_mode(None, {"LEGION_CONFIG": str(tmp_path / "nope.toml")})


def test_other_sections_of_the_file_are_left_alone(tmp_path):
    path = _toml(tmp_path, '[limits]\nmax_running = 4\n\n[registration]\nmode = "open"\n')
    assert registration.load_mode(path, {}) == "open"


# --- the decision ---------------------------------------------------------------

def _db():
    return SimpleNamespace(bind=None)  # no Postgres: the advisory lock is skipped


async def _admit(monkeypatch, *, users: int, mode: str, token=None, consume=None):
    async def count(_db):
        return users

    monkeypatch.setattr(registration, "user_count", count)
    monkeypatch.setattr(registration, "get_mode", lambda: mode)
    if consume is not None:
        monkeypatch.setattr(registration, "_consume_invite", consume)
    return await registration.admit(_db(), email="a@example.com", invite_token=token)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["closed", "invite", "open"])
async def test_the_first_user_is_the_owner_in_any_mode(monkeypatch, mode):
    assert await _admit(monkeypatch, users=0, mode=mode) == (True, None)


@pytest.mark.asyncio
async def test_closed_refuses_everyone_after_the_owner(monkeypatch):
    with pytest.raises(RegistrationRefused) as exc:
        await _admit(monkeypatch, users=1, mode="closed", token="lgi_whatever")
    assert (exc.value.status, exc.value.code) == (403, "registration_closed")


@pytest.mark.asyncio
async def test_open_admits_anyone_but_not_as_owner(monkeypatch):
    assert await _admit(monkeypatch, users=3, mode="open") == (False, None)


@pytest.mark.asyncio
async def test_invite_mode_needs_a_token(monkeypatch):
    for token in (None, "", "   "):
        with pytest.raises(RegistrationRefused) as exc:
            await _admit(monkeypatch, users=1, mode="invite", token=token)
        assert exc.value.code == "invite_required"


@pytest.mark.asyncio
async def test_invite_mode_spends_the_invitation(monkeypatch):
    seen = []

    async def consume(_db, token, email):
        seen.append((token, email))
        return "invite-id"

    result = await _admit(monkeypatch, users=1, mode="invite", token="lgi_abc", consume=consume)
    assert result == (False, "invite-id")
    assert seen == [("lgi_abc", "a@example.com")]


# --- tokens ---------------------------------------------------------------------

def test_tokens_are_stored_only_as_hashes():
    token = registration.new_token()
    assert token.startswith("lgi_") and len(token) > 20
    assert registration.hash_token(token) != token
    assert registration.hash_token(f"  {token}  ") == registration.hash_token(token)
    assert registration.new_token() != token


def test_invite_status():
    from datetime import datetime, timedelta, timezone

    from app.models.registration_invite import RegistrationInvite

    now = datetime.now(timezone.utc)

    def invite(**kw):
        return RegistrationInvite(token_hash="h", created_by=__import__("uuid").uuid4(),
                                  expires_at=kw.pop("expires_at", now + timedelta(days=1)), **kw)

    assert registration.invite_status(invite()) == "pending"
    assert registration.invite_status(invite(expires_at=now - timedelta(seconds=1))) == "expired"
    assert registration.invite_status(invite(used_at=now)) == "used"
    assert registration.invite_status(invite(revoked_at=now)) == "revoked"
    # Revoking wins over every other state.
    assert registration.invite_status(invite(revoked_at=now, used_at=now)) == "revoked"
