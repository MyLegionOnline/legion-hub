"""
Log storage backends for agent session turn logs.

LOG_STORAGE_BACKEND=disabled  → no-ops (default, works without Redis/S3)
LOG_STORAGE_BACKEND=redis+s3  → Redis buffer during session, S3 archive after flush
"""

import json
import logging
import os
from typing import Optional

from app.core.secret_config import configured_secret

logger = logging.getLogger(__name__)


class RedisLogBuffer:
    """Append-only turn buffer in Redis. TTL protects against zombie keys."""

    def __init__(self, url: str, ttl: int) -> None:
        self._url = url
        self._ttl = ttl
        self._redis = None

    async def _get(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"session_log:{session_id}"

    async def append(self, session_id: str, turn: dict) -> None:
        try:
            r = await self._get()
            k = self._key(session_id)
            await r.rpush(k, json.dumps(turn, default=str))
            await r.expire(k, self._ttl)
        except Exception as exc:
            logger.warning("RedisLogBuffer.append failed: %s", exc)

    async def get_all(self, session_id: str) -> list[dict]:
        try:
            r = await self._get()
            items = await r.lrange(self._key(session_id), 0, -1)
            return [json.loads(item) for item in items]
        except Exception as exc:
            logger.warning("RedisLogBuffer.get_all failed: %s", exc)
            return []

    async def delete(self, session_id: str) -> None:
        try:
            r = await self._get()
            await r.delete(self._key(session_id))
        except Exception as exc:
            logger.warning("RedisLogBuffer.delete failed: %s", exc)


class NoopLogBuffer:
    async def append(self, session_id: str, turn: dict) -> None:
        pass

    async def get_all(self, session_id: str) -> list[dict]:
        return []

    async def delete(self, session_id: str) -> None:
        pass


class S3LogArchive:
    """Write/read/delete JSONL files in the legion-logs S3 bucket."""

    def __init__(self) -> None:
        self.bucket = os.getenv("LOG_S3_BUCKET", "legion-logs")
        self._endpoint = os.getenv("LOG_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
        self._access_key = configured_secret("LOG_S3_ACCESS_KEY", "minio")
        self._secret_key = configured_secret("LOG_S3_SECRET_KEY", "minio123")
        self._region = os.getenv("S3_REGION", "us-east-1")

    def _session(self):
        import aioboto3
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
            try:
                await client.create_bucket(Bucket=self.bucket)
            except Exception:
                pass

    async def flush(self, key: str, turns: list[dict]) -> int:
        """Write turns as JSONL to S3. Returns bytes written."""
        body = "\n".join(json.dumps(t, default=str) for t in turns) + "\n"
        data = body.encode("utf-8")
        async with self._session().client("s3", **self._client_kwargs()) as client:
            await self._ensure_bucket(client)
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/x-ndjson",
            )
        return len(data)

    async def get_download_url(self, key: str, expires_in: int = 900) -> str:
        async with self._session().client("s3", **self._client_kwargs()) as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )

    async def delete(self, key: str) -> None:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                await client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            logger.warning("S3LogArchive.delete failed for %s: %s", key, exc)

    async def get_size(self, key: str) -> int:
        try:
            async with self._session().client("s3", **self._client_kwargs()) as client:
                obj = await client.head_object(Bucket=self.bucket, Key=key)
                return obj.get("ContentLength", 0)
        except Exception:
            return 0


class NoopLogArchive:
    async def flush(self, key: str, turns: list[dict]) -> int:
        return 0

    async def get_download_url(self, key: str, expires_in: int = 900) -> Optional[str]:
        return None

    async def delete(self, key: str) -> None:
        pass

    async def get_size(self, key: str) -> int:
        return 0


_buffer: Optional[object] = None
_archive: Optional[object] = None


def get_log_buffer():
    global _buffer
    if _buffer is None:
        backend = os.getenv("LOG_STORAGE_BACKEND", "disabled").lower()
        if backend == "redis+s3":
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            ttl = int(os.getenv("LOG_REDIS_TTL", "172800"))
            _buffer = RedisLogBuffer(url, ttl)
        else:
            _buffer = NoopLogBuffer()
    return _buffer


def get_log_archive():
    global _archive
    if _archive is None:
        backend = os.getenv("LOG_STORAGE_BACKEND", "disabled").lower()
        if backend == "redis+s3":
            _archive = S3LogArchive()
        else:
            _archive = NoopLogArchive()
    return _archive
