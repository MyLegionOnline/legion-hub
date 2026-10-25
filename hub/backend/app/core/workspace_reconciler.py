"""
Workspace reconciler: syncs workspace.status with Docker reality.

Workspaced's idle-reaper stops containers without notifying the backend,
so workspace_usage rows can be left open and workspace.status can stay
'running' after the container is gone.  This reconciler runs every 60 s,
calls workspaced for each apparently-running workspace, and closes any
orphaned usage records.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.models.room import Room
from app.models.workspace import Workspace, WorkspaceUsage
from app.db.release import release_db

logger = logging.getLogger(__name__)

_INTERVAL_SEC = 60
#: A tick this late means the process was not watching in between (a
#: suspended host, a stalled event loop).
_BLIND_TICK_SEC = 5 * _INTERVAL_SEC


async def _split_unobserved_intervals(async_session) -> None:
    """After a window nobody watched, end open intervals where use stopped.

    Runs before the tick so the work-ended sweep and the open sweep see the
    split rows (billing_interval.close_unobserved_intervals).
    """
    from app.core.billing_interval import close_unobserved_intervals

    try:
        async with async_session() as db:
            await close_unobserved_intervals(db)
    except Exception as exc:
        logger.warning("reconciler: unobserved-gap billing split failed: %s", exc)


async def _reset_orphaned_processes(db: AsyncSession, workspace_id) -> list:
    """Mark a vanished workspace's processes dead, mirroring graceful stop.

    Returns the services that were reset; the caller publishes their ``stopped``
    event after it commits.
    """
    from app.core.workspace_services import WorkspaceServiceRegistry
    from app.models.workspace import WorkspaceBgProc, WorkspaceManagedProcess

    registry = WorkspaceServiceRegistry(db)
    stopped_services = []

    now = datetime.now(timezone.utc)
    bg_rows = (
        await db.execute(
            select(WorkspaceBgProc).where(
                WorkspaceBgProc.workspace_id == workspace_id,
                WorkspaceBgProc.status == "running",
            )
        )
    ).scalars().all()
    for process in bg_rows:
        process.status = "failed"
        db.add(process)
    managed_rows = (
        await db.execute(
            select(WorkspaceManagedProcess).where(
                WorkspaceManagedProcess.workspace_id == workspace_id,
                WorkspaceManagedProcess.status.in_(["starting", "running"]),
            )
        )
    ).scalars().all()
    for process in managed_rows:
        is_service = bool(process.argv)
        process.status = "paused" if (
            process.restart_policy == "on_workspace_restore" if is_service
            else process.resume_policy == "restart"
        ) else "stopped"
        if is_service:
            process.runtime_state = "stopped"
            registry.record_event(process, "workspace.service.stopped", "reconciler", {})
            stopped_services.append(process)
            # A container that vanished under a running service is not a
            # graceful stop: count it so a pattern of them is visible.
            from app.core.metrics import record_workspace_service_failure
            record_workspace_service_failure("orphan_process")
        process.pid = None
        process.pgid = None
        process.updated_at = now
        db.add(process)
    return stopped_services


#: How long a billing interval may stay open after its container is gone while
#: the room still looks busy. This is a safety cap, not a policy: a session row
#: that never gets closed (a crashed dispatch, a bug) would otherwise bill a
#: user indefinitely, and the quota check reads open intervals as "still
#: running". Generous on purpose — real work outliving its container is normal
#: under hibernation, a row still open a day later is not.
STALE_OPEN_INTERVAL_SEC = 24 * 3600


async def _close_finished_billing_intervals(db: AsyncSession) -> None:
    """Close billing intervals whose room has stopped working.

    Counterpart to the guard in ``workspace_lifecycle._close_workspace_usage``:
    once the interval is allowed to outlive its container, something has to end
    it when the work finally stops — the container stop already happened and
    will not come back to close it. This runs every tick, so the interval ends
    within one tick of the last task/session finishing.
    """
    from app.core.billing_interval import close_usage
    from app.core.workspace_lifecycle import room_has_live_work

    open_rows = (
        await db.execute(
            select(WorkspaceUsage, Workspace)
            .join(Workspace, Workspace.id == WorkspaceUsage.workspace_id)
            .where(
                WorkspaceUsage.stopped_at.is_(None),
                # A running container is billed the ordinary way; its interval
                # is closed by the stop path, not here.
                Workspace.status != "running",
            )
        )
    ).all()
    if not open_rows:
        return

    now = datetime.now(timezone.utc)
    closed = 0
    for usage, ws in open_rows:
        started = usage.started_at
        if started is not None and started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_sec = (now - started).total_seconds() if started else 0.0

        stale = age_sec >= STALE_OPEN_INTERVAL_SEC
        if not stale and await room_has_live_work(ws.room_id, db):
            continue

        # Dated by the last evidence of use, not by this tick: after a window
        # nobody watched (backend down, a stuck row) `now` would bill the gap.
        billed_h = await close_usage(
            usage, ws, db,
            stop_reason="stale_open" if stale else "work_ended",
            now=now,
        )
        closed += 1
        if stale:
            logger.warning(
                "reconciler: force-closed a billing interval open for %.1f h "
                "(billed %.1f h) on room %s — its room still reports live work "
                "with no container; check for a session row that never closed",
                age_sec / 3600.0, billed_h, ws.room_id,
            )
    if closed:
        await db.commit()
        logger.info("reconciler: closed %d finished billing interval(s)", closed)


async def _reconcile_tick(db: AsyncSession) -> None:
    from app.core import workspaced_client as wsd
    from app.core.metrics import workspace_running_total

    res = await db.execute(select(Workspace).where(Workspace.status == "running"))
    running_in_db = res.scalars().all()

    actually_running = 0
    externally_stopped: list = []
    reset_services: list = []
    for ws in running_in_db:
        # Stage 1.2 (internal design notes §8.2): one workspaced probe per
        # room, sequential.  Holding the tick's transaction across all of them
        # kept one connection idle in transaction ~50 s at 25 rooms on the A3
        # ladder.  Committing here also makes each heartbeat durable as soon as
        # it is observed (expire_on_commit=False keeps `ws` usable).
        await release_db(db, reason="reconciler/status_probe")
        try:
            status_data = await wsd.get_workspace_status(str(ws.room_id))
            if status_data.get("status") == "running":
                actually_running += 1
                # A successful status probe is also a durable lease heartbeat.
                # Without persisting it, the recovery pass below sees a
                # healthy container as stale after 60s and force-stops it
                # while an active ChatMaster task is still running.
                ws.last_heartbeat_at = datetime.now(timezone.utc)
                db.add(ws)
                continue
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.debug("reconciler: workspaced unreachable for room_id=%s: %s", ws.room_id, exc)

        # Container is gone — sync DB state
        logger.info("reconciler: room_id=%s container stopped externally, syncing", ws.room_id)

        usage_res = await db.execute(
            select(WorkspaceUsage).where(
                WorkspaceUsage.workspace_id == ws.id,
                WorkspaceUsage.stopped_at.is_(None),
            )
        )
        usage = usage_res.scalar_one_or_none()
        if usage:
            from app.core.billing_interval import close_usage, fold_container_time
            from app.core.workspace_lifecycle import room_has_live_work

            # A vanished container is no more the end of the room's work than
            # a graceful stop is: while work continues the interval stays open
            # and _close_finished_billing_intervals ends it with the work.
            if await room_has_live_work(ws.room_id, db):
                fold_container_time(usage, datetime.now(timezone.utc))
                db.add(usage)
            else:
                # The container may have vanished long before this tick saw
                # it (backend/Docker down, idle auto-stop never ran): close at
                # the last evidence of use, not at `now` (economics.md §2).
                await close_usage(usage, ws, db, stop_reason="reconciler")

        ws.status = "stopped"
        ws.container_id = None
        db.add(ws)
        # The container took every process with it.  Graceful stop_workspace
        # cascades this reset (see workspace_lifecycle); an externally-vanished
        # container must too, or managed services stay ``ready`` with a dead
        # pid — the Browser tab then offers "Open" (which 502s against the
        # gone container) instead of "Start", and the workspace can never
        # be revived from that tab.
        reset_services.extend(await _reset_orphaned_processes(db, ws.id))
        externally_stopped.append(ws.room_id)

    await db.commit()
    from app.core.workspace_services import WorkspaceServiceRegistry

    for process in reset_services:
        await WorkspaceServiceRegistry(db).publish(process, "workspace.service.stopped")
    # This path bypasses the lifecycle broadcast, so the sidebar dot of every
    # member would keep showing a room that is no longer billed as running.
    if externally_stopped:
        from app.core.workspace_lifecycle import publish_room_runtime

        for room_id in externally_stopped:
            await publish_room_runtime(room_id, "stopped")
    # Lifecycle stage 2: idle is based on backend activity, not browser
    # connectivity.  The helper emits the warning event before stopping.
    try:
        from app.core.workspace_lifecycle import (
            auto_stop_idle_workspaces,
            recover_stale_workspaces,
        )

        await auto_stop_idle_workspaces(db)
        await recover_stale_workspaces(db)
    except Exception as exc:
        logger.warning("reconciler: idle auto-stop failed: %s", exc)

    try:
        # Open before close: work that began without a container (file/git
        # tools, a ChatMaster session) is billed from its start, back-dated.
        from app.core.billing_interval import open_intervals_for_working_rooms

        await open_intervals_for_working_rooms(db)
    except Exception as exc:
        await db.rollback()
        logger.warning("reconciler: billing interval open sweep failed: %s", exc)
    # Steps attached by plugins (the cloud plugin's queued plan periods, its
    # hour quota: out of hours work stops whether or not it has a container,
    # internal design notes T8).
    from app.core.work_gates import run_reconciler_steps

    await run_reconciler_steps(db)
    try:
        # internal design notes T3: room files re-measured every 10 min
        # while a room works; at the quota the owner's tasks pause.
        from app.core.room_storage import enforce_storage_quota

        await enforce_storage_quota(db)
    except Exception as exc:
        await db.rollback()
        logger.warning("reconciler: storage quota sweep failed: %s", exc)
    try:
        # internal design notes T0.16: free space of the rooms' volume to
        # Prometheus; below 5 % free the fastest-growing room stops.
        from app.core.disk_watch import watch_data_volume

        await watch_data_volume(db)
    except Exception as exc:
        await db.rollback()
        logger.warning("reconciler: disk watchdog failed: %s", exc)
    try:
        # Host memory, reserved memory and
        # OOM kills to Prometheus, to size [capacity] on the VPS.
        from app.core.host_metrics import publish_host_metrics

        await publish_host_metrics(db)
    except Exception as exc:
        await db.rollback()
        logger.warning("reconciler: host metrics failed: %s", exc)
    try:
        await _close_finished_billing_intervals(db)
    except Exception as exc:
        logger.warning("reconciler: billing interval sweep failed: %s", exc)
    workspace_running_total.set(actually_running)
    logger.debug("reconciler: %d/%d workspaces actually running", actually_running, len(running_in_db))

    # Room deletion marks the DB row before calling the external daemon.  If
    # the backend restarts in that window, finish the runtime destruction on
    # the next tick. The destroy endpoint is idempotent and the room remains
    # marked until the user request can complete its DB cleanup.
    pending = (
        await db.execute(
            select(Room.id).where(Room.deletion_started_at.is_not(None))
        )
    ).all()
    for (room_id,) in pending:
        try:
            await wsd.destroy_workspace(str(room_id))
            logger.info("reconciler: destroyed pending room workspace room_id=%s", room_id)
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("reconciler: pending room cleanup failed room_id=%s: %s", room_id, exc)

    # Also remove legacy containers and volume subdirectories left behind by
    # room deletions that happened before this lifecycle existed. Workspaced
    # returns both container identities (including stopped containers) and
    # data-only room directories from the shared volume.
    try:
        discovered = await wsd.list_workspace_ids()
        known = {
            str(room_id)
            for (room_id,) in (
                await db.execute(select(Room.id))
            ).all()
        }
        for discovered_room_id in discovered:
            if discovered_room_id in known:
                continue
            # Production room ids are UUIDs.  Do not let a filesystem entry
            # such as ``lost+found`` or a future workspaced metadata directory
            # become a deletion target during orphan cleanup.
            try:
                UUID(discovered_room_id)
            except (TypeError, ValueError):
                logger.warning(
                    "reconciler: ignoring non-room workspace identity=%s",
                    discovered_room_id,
                )
                continue
            try:
                await wsd.destroy_workspace(discovered_room_id)
                logger.info(
                    "reconciler: removed orphaned workspace room_id=%s",
                    discovered_room_id,
                )
            except (httpx.HTTPError, OSError) as exc:
                logger.warning(
                    "reconciler: orphan cleanup failed room_id=%s: %s",
                    discovered_room_id,
                    exc,
                )
    except (httpx.HTTPError, httpx.RequestError, OSError) as exc:
        logger.debug("reconciler: orphan discovery unavailable: %s", exc)


async def run_reconciler_loop(engine) -> None:
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    last_tick_at: datetime | None = None

    while True:
        try:
            await asyncio.sleep(_INTERVAL_SEC)
            # Wall clock on purpose: a suspended host does not advance the
            # monotonic one, and that sleep is exactly the window nobody saw.
            now = datetime.now(timezone.utc)
            blind = last_tick_at is None or (
                (now - last_tick_at).total_seconds() > _BLIND_TICK_SEC
            )
            last_tick_at = now
            if blind:
                await _split_unobserved_intervals(async_session)
            async with async_session() as db:
                await _reconcile_tick(db)
            from app.core.bg_wake import wake_finished_processes

            await wake_finished_processes(async_session)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("workspace reconciler error: %s", exc)
