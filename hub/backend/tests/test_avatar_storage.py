"""
Stage 2 regression test for internal design notes — storage backend module.

Covers the acceptance criteria of stage 2:
  * `DatabaseAvatarBackend` put/get/delete are no-ops and
    `get_download_url` returns None (the API layer reads/writes the ORM
    `AgentAvatar` row directly — R2/R3);
  * `S3AvatarBackend` put/get/delete round-trip through a mocked
    `aioboto3` session by the same pattern used elsewhere in the project
    (`__aenter__`/`__aexit__` on an AsyncMock), object key shape is
    `avatars/{owner_id}/{agent_id}` (R2);
  * `get_avatar_storage_backend()` returns the right singleton per
    `AVATAR_STORAGE_BACKEND` and raises `ValueError` on an unknown value,
    mirroring `get_skill_storage_backend`.

The module is not yet wired into any API route (that happens in stage 3),
so no DB or live S3 is required — these are pure unit tests, not gated on
`INIT_DB_TEST_URL` / `DATABASE_URL`.
"""
import os
from uuid import uuid4

import pytest

from app.core.avatar_storage import (
    AvatarStorageBackend,
    DatabaseAvatarBackend,
    S3AvatarBackend,
    avatar_key,
    get_avatar_storage_backend,
    reset_avatar_storage_backend,
)
from unittest.mock import AsyncMock, MagicMock, patch


# ── avatar_key ──────────────────────────────────────────────────────────────


def test_avatar_key_shape():
    owner = uuid4()
    agent = uuid4()
    assert avatar_key(owner, agent) == f"avatars/{owner}/{agent}"


# ── DatabaseAvatarBackend ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_backend_put_get_delete_are_noops():
    backend = DatabaseAvatarBackend()
    # put/get/delete must not raise and must not touch any external state.
    await backend.put("avatars/x/y", b"ignored")
    assert await backend.get("avatars/x/y") is None
    await backend.delete("avatars/x/y")  # no error


@pytest.mark.asyncio
async def test_db_backend_get_download_url_is_none():
    backend = DatabaseAvatarBackend()
    assert await backend.get_download_url("avatars/x/y") is None


# ── S3AvatarBackend ─────────────────────────────────────────────────────────


def _mock_s3_client():
    """Build an AsyncMock S3 client that supports `async with ... as client`."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    # head_bucket succeeds by default → _ensure_bucket is a no-op.
    client.head_bucket = AsyncMock(return_value={})
    client.create_bucket = AsyncMock()
    client.put_object = AsyncMock()
    client.get_object = AsyncMock()
    client.delete_object = AsyncMock()
    client.generate_presigned_url = AsyncMock(return_value="https://presigned/url")
    return client


def _patch_session(client):
    """Patch `S3AvatarBackend._session` to return a session whose `.client()`
    returns an async-context-manager wrapping `client`."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.client = MagicMock(return_value=ctx)
    return patch("app.core.avatar_storage.S3AvatarBackend._session", return_value=session)


@pytest.mark.asyncio
async def test_s3_backend_put_writes_to_bucket():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    with _patch_session(client):
        await backend.put("avatars/o/a", b"\x89PNG\r\n...")
    client.put_object.assert_awaited_once()
    call = client.put_object.call_args
    assert call.kwargs["Bucket"] == backend.bucket
    assert call.kwargs["Key"] == "avatars/o/a"
    assert call.kwargs["Body"] == b"\x89PNG\r\n..."


@pytest.mark.asyncio
async def test_s3_backend_get_returns_object_bytes():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    body = MagicMock()
    body.read = AsyncMock(return_value=b"avatar-bytes")
    client.get_object = AsyncMock(return_value={"Body": body})
    with _patch_session(client):
        data = await backend.get("avatars/o/a")
    assert data == b"avatar-bytes"
    client.get_object.assert_awaited_once()
    assert client.get_object.call_args.kwargs["Key"] == "avatars/o/a"


@pytest.mark.asyncio
async def test_s3_backend_get_returns_none_on_missing_object():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    client.get_object = AsyncMock(side_effect=Exception("NoSuchKey"))
    with _patch_session(client):
        data = await backend.get("avatars/o/missing")
    assert data is None


@pytest.mark.asyncio
async def test_s3_backend_delete_removes_object():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    with _patch_session(client):
        await backend.delete("avatars/o/a")
    client.delete_object.assert_awaited_once()
    assert client.delete_object.call_args.kwargs["Key"] == "avatars/o/a"


@pytest.mark.asyncio
async def test_s3_backend_delete_swallows_errors():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    client.delete_object = AsyncMock(side_effect=Exception("boom"))
    with _patch_session(client):
        # must not raise
        await backend.delete("avatars/o/a")


@pytest.mark.asyncio
async def test_s3_backend_get_download_url_returns_presigned_url():
    backend = S3AvatarBackend()
    client = _mock_s3_client()
    with _patch_session(client):
        url = await backend.get_download_url("avatars/o/a", expires_in=300)
    assert url == "https://presigned/url"
    client.generate_presigned_url.assert_called_once()
    assert client.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 300


def test_s3_backend_bucket_default():
    # Ensure the default bucket matches stage 0 env wiring.
    backend = S3AvatarBackend()
    assert backend.bucket == os.getenv("AVATAR_S3_BUCKET", "legion-avatars")


# ── get_avatar_storage_backend ──────────────────────────────────────────────


def test_get_backend_defaults_to_db(monkeypatch):
    monkeypatch.delenv("AVATAR_STORAGE_BACKEND", raising=False)
    reset_avatar_storage_backend()
    backend = get_avatar_storage_backend()
    assert isinstance(backend, DatabaseAvatarBackend)


def test_get_backend_returns_s3_when_configured(monkeypatch):
    monkeypatch.setenv("AVATAR_STORAGE_BACKEND", "s3")
    reset_avatar_storage_backend()
    backend = get_avatar_storage_backend()
    assert isinstance(backend, S3AvatarBackend)


def test_get_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("AVATAR_STORAGE_BACKEND", "ftp")
    reset_avatar_storage_backend()
    with pytest.raises(ValueError, match="Unknown AVATAR_STORAGE_BACKEND"):
        get_avatar_storage_backend()


def test_get_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("AVATAR_STORAGE_BACKEND", "DB")
    reset_avatar_storage_backend()
    assert isinstance(get_avatar_storage_backend(), DatabaseAvatarBackend)


def test_get_backend_caches_singleton(monkeypatch):
    monkeypatch.delenv("AVATAR_STORAGE_BACKEND", raising=False)
    reset_avatar_storage_backend()
    first = get_avatar_storage_backend()
    second = get_avatar_storage_backend()
    assert first is second


def test_reset_backend_clears_singleton(monkeypatch):
    monkeypatch.delenv("AVATAR_STORAGE_BACKEND", raising=False)
    reset_avatar_storage_backend()
    first = get_avatar_storage_backend()
    reset_avatar_storage_backend()
    second = get_avatar_storage_backend()
    assert first is not second


# ── ABC contract ────────────────────────────────────────────────────────────


def test_avatar_storage_backend_is_abstract():
    with pytest.raises(TypeError):
        AvatarStorageBackend()  # type: ignore[abstract]
