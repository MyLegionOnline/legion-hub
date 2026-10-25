"""One-off audit: count Task.state x dispatch_status x open-session
combinations that the task lifecycle plan (internal design notes,
"Part 3", stage 1) flagged as symptoms of the missing single dispatch owner.
Read-only — does not modify any data.

Run inside the backend container:
    python -m scripts.task_dispatch_audit
"""
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.session import engine


async def main() -> None:
    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        # 1. Claimed for dispatch but the workflow state was never flipped to
        #    IN_PROGRESS — looks "assigned" while a proxy pair may actually
        #    be running.
        rows = (
            await db.execute(
                text(
                    "SELECT id, room_id, updated_at FROM task "
                    "WHERE state = 'ASSIGNED' AND dispatch_status = 'dispatched' "
                    "ORDER BY updated_at DESC"
                )
            )
        ).all()
        print(f"1) state=assigned AND dispatch_status=dispatched: {len(rows)}")
        for task_id, room_id, updated_at in rows:
            print(f"   task={task_id} room={room_id} updated_at={updated_at}")

        # 2. Ready to dispatch but an open proxy session already exists for
        #    the task — a manual Start racing with an in-flight auto-dispatch
        #    would produce this.
        rows = (
            await db.execute(
                text(
                    "SELECT DISTINCT t.id, t.room_id FROM task t "
                    "JOIN cm_agent_proxy_session cps ON cps.task_id = t.id "
                    "JOIN session s ON s.id = cps.session_id "
                    "WHERE t.state = 'ASSIGNED' AND t.dispatch_status = 'ready' "
                    "AND s.closed_at IS NULL"
                )
            )
        ).all()
        print(f"2) state=assigned AND dispatch_status=ready AND has open session: {len(rows)}")
        for task_id, room_id in rows:
            print(f"   task={task_id} room={room_id}")

        # 3. In progress with no open proxy session — either a stale state
        #    (crashed worker never closed it out) or a legacy approve_and_start
        #    task, which never creates cm_agent_proxy_session rows at all.
        rows = (
            await db.execute(
                text(
                    "SELECT t.id, t.room_id, t.updated_at FROM task t "
                    "WHERE t.state = 'IN_PROGRESS' AND NOT EXISTS ("
                    "  SELECT 1 FROM cm_agent_proxy_session cps "
                    "  JOIN session s ON s.id = cps.session_id "
                    "  WHERE cps.task_id = t.id AND s.closed_at IS NULL"
                    ") ORDER BY t.updated_at DESC"
                )
            )
        ).all()
        print(f"3) state=in_progress AND no open session: {len(rows)}")
        for task_id, room_id, updated_at in rows:
            print(f"   task={task_id} room={room_id} updated_at={updated_at}")

        # 4. Marked blocked but has no dependency rows at all — the blocking
        #    reason can no longer be explained by task_deps.
        rows = (
            await db.execute(
                text(
                    "SELECT t.id, t.room_id FROM task t "
                    "WHERE t.dispatch_status = 'blocked' AND NOT EXISTS ("
                    "  SELECT 1 FROM task_dependency td WHERE td.task_id = t.id"
                    ")"
                )
            )
        ).all()
        print(f"4) dispatch_status=blocked AND no dependency rows: {len(rows)}")
        for task_id, room_id in rows:
            print(f"   task={task_id} room={room_id}")


if __name__ == "__main__":
    asyncio.run(main())
