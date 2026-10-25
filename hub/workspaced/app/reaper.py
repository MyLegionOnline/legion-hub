from __future__ import annotations

import asyncio
import logging
import time

import app.config as config

logger = logging.getLogger("workspaced.reaper")


async def exited_sweeper(docker_manager) -> None:
    """Remove workspace containers left ``exited`` — at startup, then periodically.

    Always on, unlike ``idle_reaper``: it never touches a running container,
    so it cannot step on the backend's idle policy.
    """
    while True:
        try:
            removed = await docker_manager.remove_exited()
            if removed:
                logger.info("Removed %d exited workspace container(s): %s", len(removed), removed)
        except Exception:
            logger.exception("Exited-container sweep failed")
        await asyncio.sleep(config.EXITED_SWEEP_INTERVAL_SEC)


async def idle_reaper(docker_manager) -> None:
    while True:
        await asyncio.sleep(config.REAPER_INTERVAL_SEC)
        try:
            running = await docker_manager.list_running()
            now = time.time()

            for room_id in running:
                last = docker_manager.last_activity(room_id)
                idle_sec = now - last if last else float("inf")

                if idle_sec > config.IDLE_TIMEOUT_SEC:
                    logger.info(
                        "Reaping idle workspace room_id=%s idle=%.0fs",
                        room_id, idle_sec,
                    )
                    await docker_manager.stop(room_id)
        except Exception:
            logger.exception("Reaper tick failed")
