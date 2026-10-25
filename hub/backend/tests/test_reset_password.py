"""``python -m app.reset_password``: the owner's way back in on a self-hosted Hub."""

import asyncio
import io

import pytest

from app import reset_password as rp
from app.core import security
from app.models.user import User


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Db:
    """Answers the lookup the way Postgres would for ``lower(email) = lower(:email)``."""

    def __init__(self, users):
        self.users = users
        self.commits = 0

    async def execute(self, statement):
        wanted = next(iter(statement.compile().params.values()))
        return _Result([u for u in self.users if u.email.lower() == wanted])

    async def commit(self):
        self.commits += 1


def _user(email, **extra):
    return User(email=email, hashed_password=security.get_password_hash("old password"), **extra)


def test_sets_a_password_that_signs_in():
    owner = _user("Owner@Example.com", is_owner=True)
    db = _Db([owner, _user("other@example.com")])

    asyncio.run(rp.reset_password(db, " owner@example.com ", "new password 1"))

    assert security.verify_password("new password 1", owner.hashed_password)
    assert not security.verify_password("old password", owner.hashed_password)
    assert db.commits == 1


def test_unknown_email_changes_nothing():
    db = _Db([_user("owner@example.com")])
    with pytest.raises(rp.ResetError, match="No user"):
        asyncio.run(rp.reset_password(db, "nobody@example.com", "new password 1"))
    assert db.commits == 0


def test_case_twins_need_the_exact_spelling():
    upper, lower = _user("A@example.com"), _user("a@example.com")
    db = _Db([upper, lower])
    with pytest.raises(rp.ResetError, match="Several users"):
        asyncio.run(rp.reset_password(db, "A@EXAMPLE.COM", "new password 1"))

    asyncio.run(rp.reset_password(db, "a@example.com", "new password 1"))
    assert security.verify_password("new password 1", lower.hashed_password)
    assert security.verify_password("old password", upper.hashed_password)


def test_short_password_is_refused():
    db = _Db([_user("owner@example.com")])
    with pytest.raises(rp.ResetError, match="at least"):
        asyncio.run(rp.reset_password(db, "owner@example.com", "short"))
    assert db.commits == 0


def test_refused_on_a_hub_with_legion_accounts(monkeypatch):
    from app.core import legion_auth

    monkeypatch.setattr(legion_auth, "AUTH_MODE", "legion")
    with pytest.raises(rp.ResetError, match="Legion Market"):
        rp.check_local_auth()
    assert rp.main(["owner@example.com", "--password-stdin"]) == 1


def test_main_reads_the_password_from_stdin(monkeypatch):
    seen = {}

    async def fake_run(email, password):
        seen.update(email=email, password=password)

    monkeypatch.setattr(rp, "_run", fake_run)
    monkeypatch.setattr("sys.stdin", io.StringIO("new password 1\r\n"))

    assert rp.main(["owner@example.com", "--password-stdin"]) == 0
    assert seen == {"email": "owner@example.com", "password": "new password 1"}


def test_interactive_passwords_must_match(monkeypatch):
    answers = iter(["new password 1", "new password 2"])
    monkeypatch.setattr(rp.getpass, "getpass", lambda prompt: next(answers))
    with pytest.raises(rp.ResetError, match="do not match"):
        rp.read_password(from_stdin=False)
