"""
Stage 3 regression test for internal design notes — avatar API endpoints.

Covers the acceptance criteria of stage 3:
  * POST /agents/{id}/avatar happy path (PNG/JPEG/WebP) stores an AgentAvatar
    row, flips `has_avatar`/`avatar_updated_at`, returns the agent;
  * unsupported MIME -> 415; oversized upload -> 413; corrupt image -> 422;
  * quota exceed -> 413 with a clear message;
  * re-upload releases the previous avatar's quota before reserving the new;
  * GET returns raw bytes with the stored media type and private Cache-Control;
  * GET on an agent without an avatar -> 404;
  * DELETE removes the row, S3 object (in s3 mode) and releases quota, and
    clears `has_avatar`;
  * all three endpoints 404 (not 403) on a foreign/missing agent — R7;
  * `_purge_agent_avatar` is wired into `_delete_agent_dependents` so deleting
    an agent cleans its avatar too.

The route functions are called directly with a mocked `db` AsyncMock and a
`SimpleNamespace` user, mirroring tests/test_skills_tab.py — no live Postgres
or HTTP stack is required. `normalize_avatar` is patched so the tests do not
depend on Pillow being installed locally (Pillow is present in the Docker
image; a dedicated test below covers the real normalization path when Pillow
is importable).
"""
from __future__ import annotations

import base64
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import Response

from app.api.v1 import agents as agents_api
from app.core.avatar_storage import DatabaseAvatarBackend, avatar_key
from app.models.agent_avatar import AgentAvatar

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff\xe0"
WEBP_MAGIC = b"RIFF" + b"\x00" * 4 + b"WEBP"


class _FakeUpload:
    """Minimal stand-in for FastAPI's UploadFile: one async .read()."""

    def __init__(self, data: bytes, filename: str = "avatar.png") -> None:
        self._data = data
        self.filename = filename

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:n], self._data[n:]
        return data


def _first(value):
    result = MagicMock()
    result.scalars.return_value.first.return_value = value
    return result


def _rowcount(n: int):
    result = MagicMock()
    result.rowcount = n
    return result


def _agent(user_id, **overrides):
    base = dict(
        id=uuid4(),
        owner_id=user_id,
        nickname="exo",
        has_avatar=False,
        avatar_updated_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_owned(agent):
    """Patch `_get_owned_agent` to return `agent` (or raise 404 if None)."""
    if agent is None:
        async def _raise(agent_id, user_id, db):
            raise HTTPException(status_code=404, detail="Agent not found")
        return patch.object(agents_api, "_get_owned_agent", new=AsyncMock(side_effect=_raise))
    return patch.object(agents_api, "_get_owned_agent", new=AsyncMock(return_value=agent))


# ── POST /agents/{id}/avatar ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_avatar_png_happy_path(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    normalized = b"\x89PNG\r\n\x1a\nNORMALIZED"

    db = AsyncMock()
    db.get = AsyncMock(return_value=None)  # no existing AgentAvatar
    db.add = MagicMock()
    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: normalized)
    monkeypatch.setattr(agents_api, "_get_or_create_avatar_quota", AsyncMock())
    monkeypatch.setattr(agents_api, "_reserve_avatar_quota", AsyncMock())
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )

    with _patch_owned(agent):
        result = await agents_api.upload_agent_avatar(
            agent_id=agent.id,
            file=_FakeUpload(PNG_MAGIC + b"payload"),
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    assert result is agent
    assert agent.has_avatar is True
    assert agent.avatar_updated_at is not None

    added = [c.args[0] for c in db.add.call_args_list]
    avatar_rows = [a for a in added if isinstance(a, AgentAvatar)]
    assert len(avatar_rows) == 1
    assert avatar_rows[0].agent_id == agent.id
    assert avatar_rows[0].mime == "image/png"
    assert avatar_rows[0].size_bytes == len(normalized)
    assert avatar_rows[0].content == base64.b64encode(normalized).decode("ascii")
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_upload_avatar_accepts_jpeg_and_webp(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.add = MagicMock()
    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: b"png")
    monkeypatch.setattr(agents_api, "_get_or_create_avatar_quota", AsyncMock())
    monkeypatch.setattr(agents_api, "_reserve_avatar_quota", AsyncMock())
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )

    for magic in (JPEG_MAGIC, WEBP_MAGIC):
        agent.has_avatar = False
        with _patch_owned(agent):
            await agents_api.upload_agent_avatar(
                agent_id=agent.id,
                file=_FakeUpload(magic + b"data"),
                db=db,
                current_user=SimpleNamespace(id=user_id),
            )
    # both uploads succeeded (no exception) and flipped the flag
    assert agent.has_avatar is True


@pytest.mark.asyncio
async def test_upload_avatar_rejects_unsupported_mime(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: b"x")
    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.upload_agent_avatar(
                agent_id=agent.id,
                file=_FakeUpload(b"GIF89a" + b"x"),  # GIF is not allowed
                db=AsyncMock(),
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_upload_avatar_413_on_oversized_upload(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    too_big = PNG_MAGIC + b"x" * (agents_api.AVATAR_MAX_UPLOAD_BYTES + 1)
    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.upload_agent_avatar(
                agent_id=agent.id,
                file=_FakeUpload(too_big),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_upload_avatar_422_on_undecodable_image(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)

    def boom(raw):
        raise ValueError("not an image")

    monkeypatch.setattr(agents_api, "normalize_avatar", boom)
    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.upload_agent_avatar(
                agent_id=agent.id,
                file=_FakeUpload(PNG_MAGIC + b"garbage"),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_avatar_413_when_quota_exceeded(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)

    async def _quota_exceeded(db, user_id, delta, freed=0):
        raise HTTPException(status_code=413, detail="quota exceeded")

    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: b"png")
    monkeypatch.setattr(agents_api, "_get_or_create_avatar_quota", AsyncMock())
    monkeypatch.setattr(agents_api, "_reserve_avatar_quota", _quota_exceeded)
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )

    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.upload_agent_avatar(
                agent_id=agent.id,
                file=_FakeUpload(PNG_MAGIC + b"data"),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 413
    assert "quota" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_upload_avatar_replacement_releases_old_quota(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    existing = AgentAvatar(
        agent_id=agent.id, mime="image/png", content="old",
        size_bytes=1000, updated_at=datetime(2026, 1, 1),
    )
    normalized = b"\x89PNG\r\n\x1a\nNEW"

    db = AsyncMock()
    db.get = AsyncMock(return_value=existing)  # existing avatar present
    db.add = MagicMock()
    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: normalized)
    monkeypatch.setattr(agents_api, "_get_or_create_avatar_quota", AsyncMock())
    reserve = AsyncMock()
    monkeypatch.setattr(agents_api, "_reserve_avatar_quota", reserve)
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )

    with _patch_owned(agent):
        await agents_api.upload_agent_avatar(
            agent_id=agent.id,
            file=_FakeUpload(PNG_MAGIC + b"data"),
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    # _reserve_avatar_quota must be told the previous size as `freed`
    assert reserve.await_count == 1
    args = reserve.await_args
    assert args.args[2] == len(normalized)  # delta
    assert args.kwargs["freed"] == 1000
    # the existing row is mutated in place, not duplicated
    added_avatars = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], AgentAvatar)]
    assert added_avatars == [existing]
    assert existing.size_bytes == len(normalized)
    assert existing.content == base64.b64encode(normalized).decode("ascii")


@pytest.mark.asyncio
async def test_upload_avatar_s3_backend_puts_object(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id)
    normalized = b"\x89PNG\r\n\x1a\nNORMALIZED"

    s3_backend = MagicMock()
    s3_backend.put = AsyncMock()
    # MagicMock is not a DatabaseAvatarBackend instance → S3 path is taken
    monkeypatch.setattr(agents_api, "get_avatar_storage_backend", lambda: s3_backend)
    monkeypatch.setattr(agents_api, "normalize_avatar", lambda raw: normalized)
    monkeypatch.setattr(agents_api, "_get_or_create_avatar_quota", AsyncMock())
    monkeypatch.setattr(agents_api, "_reserve_avatar_quota", AsyncMock())

    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.add = MagicMock()

    with _patch_owned(agent):
        await agents_api.upload_agent_avatar(
            agent_id=agent.id,
            file=_FakeUpload(PNG_MAGIC + b"data"),
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    s3_backend.put.assert_awaited_once_with(avatar_key(agent.owner_id, agent.id), normalized)
    added_avatars = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], AgentAvatar)]
    assert added_avatars[0].content == ""  # placeholder in S3 mode


@pytest.mark.asyncio
async def test_upload_avatar_404_on_foreign_agent():
    with _patch_owned(None):
        with pytest.raises(HTTPException) as exc:
            await agents_api.upload_agent_avatar(
                agent_id=uuid4(),
                file=_FakeUpload(PNG_MAGIC + b"x"),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=uuid4()),
            )
    assert exc.value.status_code == 404


# ── GET /agents/{id}/avatar ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_avatar_returns_raw_bytes(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=True)
    payload = b"\x89PNG\r\n\x1a\nNORMALIZED"
    row = AgentAvatar(
        agent_id=agent.id, mime="image/png",
        content=base64.b64encode(payload).decode("ascii"),
        size_bytes=len(payload), updated_at=datetime.utcnow(),
    )
    db = AsyncMock()
    db.get = AsyncMock(return_value=row)
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )

    with _patch_owned(agent):
        response = await agents_api.get_agent_avatar(
            agent_id=agent.id,
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    assert isinstance(response, Response)
    assert response.body == payload
    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "private, max-age=86400"


@pytest.mark.asyncio
async def test_get_avatar_404_when_none_set(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=False)
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)  # no AgentAvatar row
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )
    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.get_agent_avatar(
                agent_id=agent.id,
                db=db,
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_avatar_404_on_foreign_agent():
    with _patch_owned(None):
        with pytest.raises(HTTPException) as exc:
            await agents_api.get_agent_avatar(
                agent_id=uuid4(),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=uuid4()),
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_avatar_s3_backend_404_when_object_missing(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=True)
    row = AgentAvatar(
        agent_id=agent.id, mime="image/png", content="",
        size_bytes=10, updated_at=datetime.utcnow(),
    )
    s3_backend = MagicMock()
    s3_backend.get = AsyncMock(return_value=None)  # object gone
    monkeypatch.setattr(agents_api, "get_avatar_storage_backend", lambda: s3_backend)

    db = AsyncMock()
    db.get = AsyncMock(return_value=row)

    with _patch_owned(agent):
        with pytest.raises(HTTPException) as exc:
            await agents_api.get_agent_avatar(
                agent_id=agent.id,
                db=db,
                current_user=SimpleNamespace(id=user_id),
            )
    assert exc.value.status_code == 404
    s3_backend.get.assert_awaited_once_with(avatar_key(agent.owner_id, agent.id))


# ── DELETE /agents/{id}/avatar ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_avatar_purges_row_releases_quota_clears_flag(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=True)
    row = AgentAvatar(
        agent_id=agent.id, mime="image/png", content="x",
        size_bytes=500, updated_at=datetime.utcnow(),
    )
    db = AsyncMock()
    db.get = AsyncMock(return_value=row)
    db.add = MagicMock()
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )
    release = AsyncMock()
    monkeypatch.setattr(agents_api, "_release_avatar_quota", release)

    with _patch_owned(agent):
        await agents_api.delete_agent_avatar(
            agent_id=agent.id,
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    db.delete.assert_awaited_with(row)
    release.assert_awaited_once()
    assert release.await_args.args[1] == user_id
    assert release.await_args.args[2] == 500
    assert agent.has_avatar is False
    assert agent.avatar_updated_at is not None
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_delete_avatar_noop_when_none_set(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=False)
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)  # no row
    db.add = MagicMock()
    monkeypatch.setattr(
        agents_api, "get_avatar_storage_backend", lambda: DatabaseAvatarBackend()
    )
    release = AsyncMock()
    monkeypatch.setattr(agents_api, "_release_avatar_quota", release)

    with _patch_owned(agent):
        await agents_api.delete_agent_avatar(
            agent_id=agent.id,
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    db.delete.assert_not_awaited()
    release.assert_not_awaited()
    assert agent.has_avatar is False
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_delete_avatar_s3_backend_removes_object(monkeypatch):
    user_id = uuid4()
    agent = _agent(user_id, has_avatar=True)
    row = AgentAvatar(
        agent_id=agent.id, mime="image/png", content="",
        size_bytes=500, updated_at=datetime.utcnow(),
    )
    s3_backend = MagicMock()
    s3_backend.delete = AsyncMock()
    monkeypatch.setattr(agents_api, "get_avatar_storage_backend", lambda: s3_backend)
    monkeypatch.setattr(agents_api, "_release_avatar_quota", AsyncMock())

    db = AsyncMock()
    db.get = AsyncMock(return_value=row)  # AgentAvatar row; owner_id passed by caller

    with _patch_owned(agent):
        await agents_api.delete_agent_avatar(
            agent_id=agent.id,
            db=db,
            current_user=SimpleNamespace(id=user_id),
        )

    s3_backend.delete.assert_awaited_once_with(avatar_key(agent.owner_id, agent.id))


@pytest.mark.asyncio
async def test_delete_avatar_404_on_foreign_agent():
    with _patch_owned(None):
        with pytest.raises(HTTPException) as exc:
            await agents_api.delete_agent_avatar(
                agent_id=uuid4(),
                db=AsyncMock(),
                current_user=SimpleNamespace(id=uuid4()),
            )
    assert exc.value.status_code == 404


# ── _delete_agent_dependents integration ────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_agent_dependents_purges_avatar(monkeypatch):
    """Deleting an agent must clean its avatar (row + S3 object + quota)."""
    user_id = uuid4()
    agent_id = uuid4()
    row = AgentAvatar(
        agent_id=agent_id, mime="image/png", content="x",
        size_bytes=800, updated_at=datetime.utcnow(),
    )

    db = AsyncMock()
    db.get = AsyncMock(return_value=row)
    db.execute = AsyncMock(return_value=_rowcount(0))

    purge = AsyncMock()
    monkeypatch.setattr(agents_api, "_purge_agent_avatar", purge)

    await agents_api._delete_agent_dependents(db, agent_id)

    purge.assert_awaited_once_with(db, agent_id)


# ── _reserve_avatar_quota ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reserve_avatar_quota_413_when_rowcount_zero(monkeypatch):
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_rowcount(0))
    with pytest.raises(HTTPException) as exc:
        await agents_api._reserve_avatar_quota(db, uuid4(), delta=10, freed=0)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_reserve_avatar_quota_ok_when_rowcount_one():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_rowcount(1))
    await agents_api._reserve_avatar_quota(db, uuid4(), delta=10, freed=0)
    db.execute.assert_awaited_once()


# ── normalize_avatar (real Pillow path, gated) ───────────────────────────────


try:
    from PIL import Image  # noqa: F401
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this env")
def test_normalize_avatar_downscales_and_reencodes_png():
    from io import BytesIO
    from PIL import Image
    from app.core.avatar_storage import normalize_avatar

    buf = BytesIO()
    Image.new("RGBA", (1024, 768), (255, 0, 0, 128)).save(buf, format="PNG")
    raw = buf.getvalue()

    out = normalize_avatar(raw, max_dim=512)
    with Image.open(BytesIO(out)) as img:
        assert img.format == "PNG"
        assert img.width <= 512
        assert img.height <= 512
        # aspect ratio preserved (contain)
        assert img.width / img.height == 1024 / 768


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this env")
def test_normalize_avatar_strips_metadata():
    from io import BytesIO
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    from app.core.avatar_storage import normalize_avatar

    buf = BytesIO()
    img = Image.new("RGB", (100, 100), (0, 0, 255))
    # attach a fake EXIF IFD to prove re-encode drops metadata
    img.info["exif"] = ImageFileDirectory_v2().tobytes()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    out = normalize_avatar(raw, max_dim=512)
    with Image.open(BytesIO(out)) as out_img:
        assert "exif" not in out_img.info


def test_allowed_avatar_mime_only_png_jpeg_webp():
    from app.core.avatar_storage import allowed_avatar_mime

    assert allowed_avatar_mime("image/png")
    assert allowed_avatar_mime("image/jpeg")
    assert allowed_avatar_mime("image/webp")
    assert not allowed_avatar_mime("image/gif")
    assert not allowed_avatar_mime(None)
    assert not allowed_avatar_mime("application/octet-stream")
