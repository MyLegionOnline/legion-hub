"""Stage 0 of internal design notes — close user-facing
sessions that outlived their conversation.

Until the lifecycle fixes of stage 1 land, a `user_cm_*` / `user_agent_*` session
is never closed: the plan-accept endpoint does not close it, the p2 prompt forbids
`close_session`, and the watchdog only knows about proxy pairs. Every later user
message is appended to the same `context_snapshot.messages` list, so the prompt
grows without bound (observed: one session open for 8 days, 185 messages,
3.18M cumulative input tokens).

This script closes such sessions with `close_reason='stale_cleanup'`, but only
after dumping the full transcript to disk — the transcript is the only place
where a week of project decisions currently lives, and it must be reviewable
before it stops being reachable through the chat flow.

Run inside the backend container:

    python -m scripts.close_stale_user_sessions --dry-run
    python -m scripts.close_stale_user_sessions --older-than-hours 24 --out /tmp/stale

Options:
    --older-than-hours N  only sessions started more than N hours ago (default 24)
    --out DIR             transcript dump directory (default ./reports/stale_sessions)
    --dry-run             report what would happen, change nothing
    --no-events           skip the SESSION_CLOSED room event
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.session import engine

USER_KINDS = ("user_agent_p1", "user_agent_p2", "user_cm_p1", "user_cm_p2")
CLOSE_REASON = "stale_cleanup"
SUMMARY_MAX_CHARS = 2000


def _messages(session: Any) -> list[dict[str, Any]]:
    snap = session.context_snapshot
    if isinstance(snap, str):
        try:
            snap = json.loads(snap or "{}")
        except ValueError:
            snap = {}
    if not isinstance(snap, dict):
        return []
    msgs = snap.get("messages")
    return msgs if isinstance(msgs, list) else []


def _build_summary(messages: list[dict[str, Any]]) -> str:
    """Compose a deterministic summary from the last assistant replies.

    No LLM call: this runs as a one-off cleanup and must work even when no
    provider key is reachable. Stage 4 replaces this with a real memory flush.
    """
    replies = [
        str(m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "assistant" and str(m.get("content") or "").strip()
    ]
    if not replies:
        return f"Session closed by the sanitation pass (stage 0). Messages in the feed: {len(messages)}."
    tail = replies[-3:]
    body = "\n\n---\n\n".join(tail)
    header = (
        f"Session closed by the sanitation pass (stage 0). Messages in the feed: {len(messages)}. "
        f"Last assistant replies:\n\n"
    )
    return (header + body)[:SUMMARY_MAX_CHARS]


def _dump_path(out_dir: str, session: Any, room_name: str) -> str:
    safe_room = "".join(c if c.isalnum() or c in "-_" else "_" for c in room_name)[:40]
    return os.path.join(out_dir, f"{session.started_at:%Y%m%d}_{safe_room}_{session.id}.json")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-than-hours", type=float, default=24.0)
    parser.add_argument("--out", default=os.path.join("reports", "stale_sessions"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.older_than_hours)

    from app.models.room import Room
    from app.models.session.base import Session

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        rows = (
            await db.execute(
                select(Session, Room.name)
                .join(Room, Room.id == Session.room_id)
                .where(
                    Session.kind.in_(USER_KINDS),
                    Session.closed_at.is_(None),
                    Session.started_at < cutoff,
                )
                .order_by(Session.started_at)
            )
        ).all()

        print(
            f"open user sessions older than {args.older_than_hours}h "
            f"(cutoff {cutoff.isoformat()}): {len(rows)}"
        )
        if not rows:
            return

        if not args.dry_run:
            os.makedirs(args.out, exist_ok=True)

        closed = 0
        for session, room_name in rows:
            messages = _messages(session)
            age_days = (datetime.now(timezone.utc) - session.started_at).days
            print(
                f"  {session.id}  kind={session.kind:<12} room={room_name!r:<20} "
                f"age={age_days}d msgs={len(messages)} "
                f"tokens_in={session.tokens_in} cost=${float(session.cost_usd or 0):.2f}"
            )

            if args.dry_run:
                continue

            # 0.2 — dump before closing, never after.
            dump = {
                "session_id": str(session.id),
                "room_id": str(session.room_id),
                "room_name": room_name,
                "kind": session.kind,
                "started_at": session.started_at.isoformat(),
                "closed_by": "close_stale_user_sessions.py",
                "tokens_in": session.tokens_in,
                "tokens_out": session.tokens_out,
                "cost_usd": float(session.cost_usd or 0),
                "message_count": len(messages),
                "messages": messages,
            }
            path = _dump_path(args.out, session, room_name)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(dump, fh, ensure_ascii=False, indent=2)
            print(f"      transcript → {path}")

            session.closed_at = datetime.now(timezone.utc)
            session.close_reason = CLOSE_REASON
            if not session.summary:
                session.summary = _build_summary(messages)
            db.add(session)
            closed += 1

        if args.dry_run:
            print("dry-run: nothing changed")
            return

        await db.commit()
        print(f"closed {closed} session(s) with close_reason={CLOSE_REASON!r}")

        if args.no_events:
            return

        from app.core.events import emit_event

        for session, _room_name in rows:
            await emit_event(
                room_id=session.room_id,
                event_type="SESSION_CLOSED",
                content=(
                    f"Session {session.id} closed "
                    f"(kind={session.kind}, reason={CLOSE_REASON})"
                ),
                db=db,
                metadata={
                    "session_id": str(session.id),
                    "kind": session.kind,
                    "close_reason": CLOSE_REASON,
                },
            )


if __name__ == "__main__":
    asyncio.run(main())
