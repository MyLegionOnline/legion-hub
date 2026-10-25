"""The database side of deleting a room (api/v1/rooms.py ``delete_room``).

The caller has marked ``room.deletion_started_at`` and destroyed the room's
container and volume in workspaced. Also used by the 90th-day purge of an
ended plan (legion_cloud/data_lifecycle.py, internal design notes §5.5).
"""

from __future__ import annotations

import logging

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.agent_memory import AgentMemory
from app.models.room import Room, RoomAgentLink, RoomUserLink

logger = logging.getLogger(__name__)


async def _delete_stored_objects(room_id, db: AsyncSession) -> None:
    """What the room's rows point to outside the database: tool-output payloads
    of the context artifacts and the archived session logs (internal design notes
    stage 13: erasing a user must not leave their rooms' text in the object
    store). A store that is down is logged and skipped; the rows go regardless."""
    from app.models.agent_session import AgentSession
    from app.models.context_artifact import ContextArtifact

    payload_keys = (
        await db.execute(select(ContextArtifact.storage_key).where(ContextArtifact.room_id == room_id))
    ).scalars().all()
    log_keys = (
        await db.execute(
            select(AgentSession.log_key).where(AgentSession.room_id == room_id, AgentSession.log_key.is_not(None))
        )
    ).scalars().all()
    if payload_keys:
        from app.core.context_observations import _payload_store

        store = _payload_store()
        for key in payload_keys:
            try:
                await store.delete(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("room %s: deleting context payload failed (%s)", room_id, type(exc).__name__)
    if log_keys:
        from app.core.log_storage import get_log_archive

        archive = get_log_archive()
        for key in log_keys:
            try:
                await archive.delete(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("room %s: deleting session log failed (%s)", room_id, type(exc).__name__)


async def delete_room_rows(room: Room, db: AsyncSession) -> None:
    """Delete the room and every row hanging off it; commits."""
    room_id = room.id
    await _delete_stored_objects(room_id, db)
    from app.models.message import Message
    from app.models.task import Task, TaskDependency, TaskHandoff
    from app.models.agent_session import AgentSession
    from app.models.project_memory import ProjectMemory
    from app.models.goal import RoomGoal
    from app.models.goal_plan import GoalPlan, GoalPlanStep
    from app.models.session.base import Session
    from app.models.session.cm_self_plan import CmSelfPlanSession
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.session.user_cm import UserCmSession
    from app.models.session.user_agent import UserAgentSession
    from app.models.workspace import (
        Workspace, WorkspaceUsage, WorkspaceExec,
        WorkspaceBgProc, WorkspaceManagedProcess, WorkspaceCheckpoint, WorkspaceSwitchRequest,
        WorkspaceServiceEvent, WorkspaceEnvironment,
    )
    from app.models.context_artifact import ContextArtifact

    from app.models.promotion import Promotion

    await db.execute(delete(Promotion).where(Promotion.room_id == room_id))
    await db.execute(delete(Message).where(Message.room_id == room_id))

    task_ids_result = await db.execute(select(Task.id).where(Task.room_id == room_id))
    task_ids = [row[0] for row in task_ids_result.all()]
    if task_ids:
        await db.execute(delete(TaskDependency).where(TaskDependency.task_id.in_(task_ids)))
        await db.execute(delete(TaskDependency).where(TaskDependency.depends_on_id.in_(task_ids)))
        await db.execute(delete(AgentSession).where(AgentSession.task_id.in_(task_ids)))
        await db.execute(delete(TaskHandoff).where(TaskHandoff.source_task_id.in_(task_ids)))
        await db.execute(delete(TaskHandoff).where(TaskHandoff.target_task_id.in_(task_ids)))

    session_ids_result = await db.execute(select(Session.id).where(Session.room_id == room_id))
    session_ids = [row[0] for row in session_ids_result.all()]
    if session_ids:
        await db.execute(delete(CmAgentProxySession).where(CmAgentProxySession.id.in_(session_ids)))
        await db.execute(delete(CmSelfPlanSession).where(CmSelfPlanSession.id.in_(session_ids)))
        await db.execute(delete(UserCmSession).where(UserCmSession.id.in_(session_ids)))
        await db.execute(delete(UserAgentSession).where(UserAgentSession.id.in_(session_ids)))
        await db.execute(delete(ContextArtifact).where(ContextArtifact.session_id.in_(session_ids)))
    await db.execute(delete(AgentSession).where(AgentSession.room_id == room_id))
    await db.execute(delete(Session).where(Session.room_id == room_id))

    await db.execute(delete(Task).where(Task.room_id == room_id))
    await db.execute(delete(ProjectMemory).where(ProjectMemory.room_id == room_id))
    await db.execute(delete(AgentMemory).where(AgentMemory.room_id == room_id))

    goal_ids_result = await db.execute(select(RoomGoal.id).where(RoomGoal.room_id == room_id))
    goal_ids = [row[0] for row in goal_ids_result.all()]
    if goal_ids:
        plan_ids_result = await db.execute(select(GoalPlan.id).where(GoalPlan.goal_id.in_(goal_ids)))
        plan_ids = [row[0] for row in plan_ids_result.all()]
        if plan_ids:
            await db.execute(delete(GoalPlanStep).where(GoalPlanStep.plan_id.in_(plan_ids)))
        await db.execute(delete(GoalPlan).where(GoalPlan.goal_id.in_(goal_ids)))
    await db.execute(delete(RoomGoal).where(RoomGoal.room_id == room_id))
    await db.execute(delete(RoomUserLink).where(RoomUserLink.room_id == room_id))
    await db.execute(delete(RoomAgentLink).where(RoomAgentLink.room_id == room_id))

    ws_ids_result = await db.execute(select(Workspace.id).where(Workspace.room_id == room_id))
    ws_ids = [row[0] for row in ws_ids_result.all()]
    if ws_ids:
        await db.execute(delete(WorkspaceUsage).where(WorkspaceUsage.workspace_id.in_(ws_ids)))
        await db.execute(delete(WorkspaceExec).where(WorkspaceExec.workspace_id.in_(ws_ids)))
        await db.execute(delete(WorkspaceBgProc).where(WorkspaceBgProc.workspace_id.in_(ws_ids)))
        # Databases created via create_all lack ON DELETE CASCADE on these FKs:
        # events -> processes -> environments, in that order.
        await db.execute(delete(WorkspaceServiceEvent).where(WorkspaceServiceEvent.workspace_id.in_(ws_ids)))
        await db.execute(delete(WorkspaceManagedProcess).where(WorkspaceManagedProcess.workspace_id.in_(ws_ids)))
        await db.execute(delete(WorkspaceEnvironment).where(WorkspaceEnvironment.workspace_id.in_(ws_ids)))
        await db.execute(delete(WorkspaceCheckpoint).where(WorkspaceCheckpoint.workspace_id.in_(ws_ids)))
    await db.execute(delete(WorkspaceSwitchRequest).where(WorkspaceSwitchRequest.from_room_id == room_id))
    await db.execute(delete(WorkspaceSwitchRequest).where(WorkspaceSwitchRequest.to_room_id == room_id))
    await db.execute(delete(Workspace).where(Workspace.room_id == room_id))

    # internal design notes §2.3 / §7.5: room_variable rows cascade on
    # the FK, but Room secret bindings must be detached explicitly. The
    # encrypted user_secret rows stay with their individual owners.
    from app.core.secrets.room_secrets import unbind_all_for_room

    await unbind_all_for_room(db, room_id)

    await db.delete(room)
    await db.commit()
