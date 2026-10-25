"""
Storage backend abstraction for per-agent custom avatar images.

AVATAR_STORAGE_BACKEND=db  → DatabaseAvatarBackend (default): bytes live in
                              the AgentAvatar.content column (base64), the API
                              layer reads/writes the ORM model directly —
                              put/get/delete are no-ops, exactly like
                              DatabaseSkillBackend.
AVATAR_STORAGE_BACKEND=s3  → S3AvatarBackend: objects in MinIO / AWS S3 / R2
                              under avatars/{owner_id}/{agent_id}

This is a deliberate copy of the skill_storage.py contract (R2), not a shared
base class: separate env namespace (`AVATAR_*`, not `SKILL_*`), a different
key shape (one object per agent, no per-file path), a different default
bucket. The two features are functionally unrelated and a common ancestor
would couple them for ~90 lines of savings.

Quotas are *not* handled here — the storage backend is user-quota-agnostic,
mirroring skill_storage.py. Quota checks live next to the upload endpoint
(stage 3) via `UserQuota.avatars_used_bytes` (added in stage 1).
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.core.secret_config import configured_secret


def avatar_key(owner_id: UUID, agent_id: UUID) -> str:
    """S3 object key for an agent avatar: avatars/{owner_id}/{agent_id}.

    Each agent has at most one avatar, so the key has no version/path
    component — a re-upload overwrites the same object.
    """
    return f"avatars/{owner_id}/{agent_id}"


class AvatarStorageBackend(ABC):
    """Abstract interface for avatar image content storage."""

    @abstractmethod
    async def put(self, key: str, content: bytes) -> None:
        """Store raw image bytes under the given key."""

    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve content by key. Returns None if not found."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove content by key. No-op if key does not exist."""

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        """Return a presigned download URL (TTL in seconds). None for non-S3 backends."""
        return None


class DatabaseAvatarBackend(AvatarStorageBackend):
    """
    Default backend: content lives in the AgentAvatar.content column
    (base64-encoded). The API layer reads and writes the ORM model directly,
    so put/get/delete are no-ops — same contract as DatabaseSkillBackend.
    """

    async def put(self, key: str, content: bytes) -> None:
        pass  # content already set on the ORM model by the caller

    async def get(self, key: str) -> Optional[bytes]:
        return None  # caller reads from ORM model directly

    async def delete(self, key: str) -> None:
        pass  # content is removed when the ORM record is deleted


class S3AvatarBackend(AvatarStorageBackend):
    """
    S3-compatible backend: content in MinIO / AWS S3 / Cloudflare R2.

    Env variables:
      AVATAR_S3_BUCKET       — bucket name (default: legion-avatars)
      AVATAR_S3_ENDPOINT_URL — e.g. http://minio:9000 (falls back to
                               S3_ENDPOINT_URL; omit both for AWS S3)
      AVATAR_S3_ACCESS_KEY   — access key id (dev default: minio)
      AVATAR_S3_SECRET_KEY   — secret access key (dev default: minio123)
      S3_REGION              — AWS region (default: us-east-1)

    Note the credentials are read from the AVATAR_-prefixed names only — they
    do *not* fall back to the shared S3_ACCESS_KEY/S3_SECRET_KEY, same as
    S3SkillBackend.  An S3 deployment must set the AVATAR_* pair explicitly
    (or the `_FILE` variants, see `configured_secret`).
    """

    def __init__(self) -> None:
        self.bucket = os.getenv("AVATAR_S3_BUCKET", "legion-avatars")
        self._endpoint = os.getenv("AVATAR_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
        self._access_key = configured_secret("AVATAR_S3_ACCESS_KEY", "minio")
        self._secret_key = configured_secret("AVATAR_S3_SECRET_KEY", "minio123")
        self._region = os.getenv("S3_REGION", "us-east-1")

    def _session(self):
        import aioboto3  # deferred import — not needed for db backend
        return aioboto3.Session()

    def _client_kwargs(self) -> dict:
        kwargs: dict = {
            "aws_access_key_id": self._access_key,
            "aws_secret_access_key": self._secret_key,
            "region_name": self._region,
        }
        if self._endpoint:
            kwargs["endpoint_url"] = self._endpoint
        return kwargs

    async def _ensure_bucket(self, client) -> None:
        try:
            await client.head_bucket(Bucket=self.bucket)
        except Exception:
            await client.create_bucket(Bucket=self.bucket)

    async def put(self, key: str, content: bytes) -> None:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            await self._ensure_bucket(client)
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                # Always PNG: normalize_avatar re-encodes every upload, so the
                # object is directly usable through a presigned URL.
                ContentType="image/png",
            )

    async def get(self, key: str) -> Optional[bytes]:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                obj = await client.get_object(Bucket=self.bucket, Key=key)
                return await obj["Body"].read()
        except Exception:
            return None

    async def delete(self, key: str) -> None:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                await client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            pass

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )


_backend: Optional[AvatarStorageBackend] = None


def get_avatar_storage_backend() -> AvatarStorageBackend:
    """Return the configured avatar storage backend singleton.

    Reads AVATAR_STORAGE_BACKEND (default 'db') once; unknown values raise
    ValueError, mirroring get_skill_storage_backend.
    """
    global _backend
    if _backend is None:
        backend_name = os.getenv("AVATAR_STORAGE_BACKEND", "db").lower()
        if backend_name == "db":
            _backend = DatabaseAvatarBackend()
        elif backend_name == "s3":
            _backend = S3AvatarBackend()
        else:
            raise ValueError(
                f"Unknown AVATAR_STORAGE_BACKEND={backend_name!r}. "
                "Supported values: 'db', 's3'."
            )
    return _backend


def reset_avatar_storage_backend() -> None:
    """Clear the cached singleton. For tests that switch env between cases."""
    global _backend
    _backend = None


# ── image normalization (stage 3, R4) ────────────────────────────────────────

AVATAR_MAX_DIMENSION_PX = int(os.getenv("AVATAR_MAX_DIMENSION_PX", "512"))

_ALLOWED_AVATAR_MIME = frozenset({"image/png", "image/jpeg", "image/webp"})


def allowed_avatar_mime(mime: Optional[str]) -> bool:
    return mime in _ALLOWED_AVATAR_MIME


def normalize_avatar(raw: bytes, max_dim: int = AVATAR_MAX_DIMENSION_PX) -> bytes:
    """Downscale to fit ``max_dim``x``max_dim`` (contain) and re-encode as PNG.

    Strips EXIF/metadata (R4): Pillow re-saves only pixel data. Transparency is
    preserved (RGBA). Raises whatever Pillow raises when the bytes cannot be
    decoded (``UnidentifiedImageError``/``OSError``/``ValueError``, or
    ``DecompressionBombError`` on a pixel bomb) — the caller has already
    magic-byte-sniffed the MIME, so this only happens on truncated/corrupt or
    hostile input, and the upload endpoint turns any of them into a 422.
    """
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(raw)) as img:
        img.load()
        width, height = img.width, img.height
        if width > max_dim or height > max_dim:
            scale = min(max_dim / width, max_dim / height)
            new_w = max(1, round(width * scale))
            new_h = max(1, round(height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        mode = "RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"
        out = img.convert(mode)
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
