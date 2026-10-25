"""git_author — who a room's commits are authored by."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core import git_author
from app.core import workspaced_client as wsd


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    git_author._synced.clear()
    git_author._github_identity.clear()
    calls: list[tuple[str, str, str]] = []

    async def fake_set(room_id, name, email):
        calls.append((room_id, name, email))
        return {"ok": True}

    monkeypatch.setattr(wsd, "set_git_author", fake_set)
    return calls


class _Db:
    def __init__(self, user):
        self.user = user

    async def get(self, _model, _id):
        return self.user


@pytest.mark.asyncio
async def test_room_without_remote_uses_hub_account(_reset):
    room = SimpleNamespace(id=uuid4(), git_config={}, creator_id=uuid4())
    user = SimpleNamespace(full_name="Ada  Lovelace", email="ada@example.com")
    await git_author.sync_room_author(room, _Db(user))
    await git_author.sync_room_author(room, _Db(user))  # cached: one PUT
    assert _reset == [(str(room.id), "Ada Lovelace", "ada@example.com")]


@pytest.mark.asyncio
async def test_github_room_without_token_is_left_alone(_reset):
    room = SimpleNamespace(
        id=uuid4(), git_config={"repo_url": "https://github.com/o/r"}, creator_id=uuid4()
    )
    user = SimpleNamespace(full_name="Ada", email="ada@example.com")
    await git_author.sync_room_author(room, _Db(user))
    assert _reset == []


@pytest.mark.asyncio
async def test_github_token_uses_noreply_email(_reset, monkeypatch):
    async def fake_identity(_cfg, token):
        assert token == "tok"
        return ("Ada", "42+ada@users.noreply.github.com")

    monkeypatch.setattr(git_author, "github_identity", fake_identity)
    rid = uuid4()
    await git_author.sync_from_token(rid, {"repo_url": "https://github.com/o/r"}, "tok")
    assert _reset == [(str(rid), "Ada", "42+ada@users.noreply.github.com")]
