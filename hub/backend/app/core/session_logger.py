"""
SessionLogger: fire-and-forget turn logging for agent sessions.

Appends each LLM call/tool call to Redis during execution, then flushes
the full log to S3 when the session ends. Never raises — all errors are
logged as warnings so they don't interrupt the agent loop.
"""

import logging
import traceback as tb
from datetime import datetime, timezone
from uuid import UUID

from app.core.secrets.room_masking import mask_room_secrets_deep

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionLogger:
    def __init__(self, session_id: UUID, user_id: UUID, buffer) -> None:
        self.session_id = str(session_id)
        self.user_id = str(user_id)
        self._buffer = buffer

    async def log_turn(
        self,
        turn: int,
        messages_sent: list,
        response: str,
        tool_calls: list,
        tool_results: list,
        usage: dict,
        model: str,
        error: str | None = None,
    ) -> None:
        turn_data = {
            "turn": turn,
            "ts": _now(),
            "type": "llm_call",
            "messages_sent": messages_sent,
            "response": response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "usage": usage,
            "model": model,
            "error": error,
        }
        # Second line of defence (plan §4.5): a Room secret used earlier in
        # this call is still tracked on the contextvar here, so any value
        # that slipped past the tool-output mask gets caught before it's
        # durably persisted to Redis/S3.
        await self._buffer.append(self.session_id, mask_room_secrets_deep(turn_data))

    async def log_error(self, error: str, traceback: str) -> None:
        turn_data = {
            "turn": None,
            "ts": _now(),
            "type": "session_error",
            "error": error,
            "traceback": traceback,
        }
        await self._buffer.append(self.session_id, turn_data)

    async def flush_to_s3(self, archive) -> tuple[str, int]:
        """Read all turns from Redis, write JSONL to S3. Returns (key, bytes_written)."""
        try:
            turns = await self._buffer.get_all(self.session_id)
            if not turns:
                return "", 0

            dt = datetime.now(timezone.utc)
            key = f"logs/{self.user_id}/{dt.year}/{dt.month:02d}/{self.session_id}.jsonl"

            size = await archive.flush(key, turns)
            await self._buffer.delete(self.session_id)
            logger.info("Flushed %d turns (%d bytes) to %s", len(turns), size, key)
            return key, size
        except Exception as exc:
            logger.warning("SessionLogger.flush_to_s3 failed for %s: %s", self.session_id, exc)
            return "", 0

    async def mark_lost(self) -> None:
        await self._buffer.delete(self.session_id)
