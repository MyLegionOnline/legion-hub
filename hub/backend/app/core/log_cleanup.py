"""
Background cleanup for stale agent sessions (log_status='recording' older than 2 hours).
Tries to flush Redis data to S3; marks as 'lost' if Redis is empty.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.models.agent_session import AgentSession

logger = logging.getLogger(__name__)

_STALE_HOURS = 2
_INTERVAL_SECONDS = 600  # run every 10 minutes


async def cleanup_stale_sessions(db: AsyncSession) -> None:
    from app.core.log_storage import get_log_buffer, get_log_archive
    from app.core.session_logger import SessionLogger

    cutoff = datetime.utcnow() - timedelta(hours=_STALE_HOURS)
    result = await db.execute(
        select(AgentSession)
        .where(AgentSession.log_status == "recording")
        .where(AgentSession.started_at < cutoff)
    )
    stale = result.scalars().all()
    if not stale:
        return

    logger.info("log_cleanup: found %d stale sessions", len(stale))
    buffer = get_log_buffer()
    archive = get_log_archive()

    for session in stale:
        user_id = session.room_id  # fallback; ideally we'd have creator_id
        sl = SessionLogger(session_id=session.id, user_id=user_id, buffer=buffer)
        turns = await buffer.get_all(str(session.id))
        if turns:
            log_key, log_bytes = await sl.flush_to_s3(archive)
            session.log_key = log_key or None
            session.log_size_bytes = log_bytes
            session.log_status = "flushed" if log_key else "flush_failed"
        else:
            await sl.mark_lost()
            session.log_status = "lost"
        if not session.ended_at:
            session.ended_at = datetime.utcnow()
        db.add(session)

    await db.commit()
    logger.info("log_cleanup: processed %d stale sessions", len(stale))


async def run_cleanup_loop(engine) -> None:
    """Infinite loop; intended to run as a background asyncio task."""
    from sqlalchemy.orm import sessionmaker as _sm
    async_session = _sm(engine, class_=AsyncSession, expire_on_commit=False)
    while True:
        try:
            await asyncio.sleep(_INTERVAL_SECONDS)
            async with async_session() as db:
                await cleanup_stale_sessions(db)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("log_cleanup loop error: %s", exc)
