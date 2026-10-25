"""Erasing a user's data from the Hub (internal design notes stage 13, owner decision O18).

Deleting a Legion account must really remove what the person left in a Hub we
run: rooms with their files and dialogues, agents, personal MCPs, provider
keys, secrets, skills, connectors. Nothing here is specific to the cloud: the
cloud plugin decides *when* (Market says the account is gone, or the operator
asks) and attaches its own tables through the ``user_erasure`` event; the work
is the core's, and a self-hosted owner can use it for a user of their Hub.

How it runs
-----------
``request_erasure`` disables the user at once and files a ``user_erasure`` row.
``run_due_erasures`` (the cloud plugin calls it from the reconciler) takes the
open rows and calls :func:`erase_user`, repeating it with a back-off until it
finishes. :func:`erase_user` is idempotent: it can stop anywhere and start
again. Order: containers and volumes first (a failure of workspaced or the MCP
host raises, the step is retried), then object-store payloads, then database
rows, last the row of the user itself, which becomes an anonymous shell.

What stays, and why, is written down per table in :data:`HANDLED_TABLES`; a test
(``tests/test_user_erasure.py``) fails when a table that points at a user is
not listed there, so the erasure cannot silently fall behind the schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.plugins import get_registry

logger = logging.getLogger(__name__)

#: Plugin event: ``async handler(db, user_id, account_id) -> dict | None``. Runs
#: after the core steps; erases what the plugin keeps about the user and may
#: return counters for the report. An exception makes the pass retry.
USER_ERASURE_EVENT = "user_erasure"

#: How long the owner promised (Privacy §6): past it a request is overdue.
ERASURE_DEADLINE = timedelta(hours=24)

SOURCE_MARKET = "market"
SOURCE_OPERATOR = "operator"

ERASE = "erase"
ANONYMIZE = "anonymize"
KEEP = "keep"

#: Every table that points at a user, with the decision. ``erase`` — the rows
#: are deleted (room tables: with their room); ``anonymize`` — the row stays
#: without what identifies; ``keep`` — a record of payment or spending that
#: says nothing about who the user was.
HANDLED_TABLES: dict[str, tuple[str, str]] = {
    # The user and the sign-in copies
    "user": (ANONYMIZE, "an anonymous shell the billing records point to"),
    "legion_credential": (ANONYMIZE, "tombstone: no address, no hash, status erased; a late old push cannot revive it"),
    "legion_link": (ERASE, "the encrypted Legion API key"),
    "registration_invite": (ERASE, "invitations the user issued; the address on one they used is blanked"),
    "user_erasure": (KEEP, "the record that the data is gone: counters only"),
    # What the user owns
    "agent": (ERASE, "all agents, the Router included, with memory and avatars"),
    "connector": (ERASE, "external-agent installations with their tokens, grants and queued frames"),
    "connector_room_grant": (ERASE, "goes with the connector and the room"),
    "skill": (ERASE, "uploaded skills with their files (the builtin ones belong to nobody)"),
    "skillfile": (ERASE, "files of the skills above"),
    "user_installed_item": (ERASE, "installed skills and MCPs"),
    "user_default_provision": (ERASE, "ledger of starter items offered"),
    "agent_skill": (ERASE, "assignments; go with the agent"),
    "mcp_connection_snapshot": (ERASE, "verification result of an installed HTTP MCP"),
    "mcp_oauth_flow": (ERASE, "unfinished MCP sign-ins"),
    "personal_mcp_runtime": (ERASE, "personal MCP runners, their containers and volumes"),
    "personal_mcp_event": (ERASE, "log of a runner"),
    "personal_mcp_import_preview": (ERASE, "uploaded MCP config previews"),
    "provider_connection": (ERASE, "LLM provider connections"),
    "user_secret": (ERASE, "secrets, ciphertext included"),
    "secret_binding": (ERASE, "bindings of secrets to rooms, MCPs, agents"),
    "secret_lease": (ERASE, "leases of secrets"),
    "secret_audit": (ERASE, "audit of secret operations: carries an IP address"),
    # Rooms are single-user: all of them go, whole
    "room": (ERASE, "rooms the user created, with container, volume, archive, dialogue, tasks"),
    "roomuserlink": (ERASE, "membership; go with the room"),
    "message": (ERASE, "dialogue; goes with the room"),
    "room_mcp_server": (ERASE, "MCPs attached to a room"),
    "room_variable": (ERASE, "room variables; go with the room"),
    "room_environment_unlock": (ERASE, "go with the room"),
    "promotion": (ERASE, "go with the room"),
    "task_interjection": (ERASE, "go with the task"),
    "user_agent_session": (ERASE, "agent conversation sessions; go with the room"),
    "user_cm_session": (ERASE, "Chat Master conversation sessions; go with the room"),
    "workspace": (ERASE, "workspace of a room"),
    "workspace_switch_request": (ERASE, "go with the room"),
    "workspace_managed_process": (ERASE, "go with the workspace"),
    # Records of payment and spending: no sign of who the user was
    "userquota": (KEEP, "plan, period and quota counters; the plan period is a payment record"),
    "mcp_credit_wallet": (KEEP, "balance of paid credits"),
    "mcp_credit_ledger": (KEEP, "spending and grants of credits"),
    # The operator's own audit
    "admin_action": (ANONYMIZE, "what the operator did stays; addresses and names leave the payload"),
}

#: Columns that hold a user id and are not foreign keys; the completeness test
#: looks for them by name too.
USER_COLUMN_NAMES = frozenset({
    "user_id", "owner_id", "creator_id", "quota_owner_id", "created_by", "used_by", "added_by",
    "granted_by", "sender_user_id", "target_user_id", "author_user_id", "declared_by", "actor_id",
    "invited_by", "account_id",
})


def user_reference_tables(metadata: sa.MetaData) -> dict[str, list[str]]:
    """Tables with a foreign key to ``user.id`` or a column named like one that
    holds a user id, with those columns. The completeness test's input."""
    found: dict[str, list[str]] = {}
    for table in metadata.tables.values():
        for column in table.columns:
            fk_to_user = any(fk.column.table.name == "user" for fk in column.foreign_keys)
            if fk_to_user or column.name in USER_COLUMN_NAMES:
                found.setdefault(table.name, []).append(column.name)
    return found


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def placeholder_email(user_id: UUID) -> str:
    return f"erased+{user_id}@invalid"


# ---------- The request -----------------------------------------------------------------


async def request_erasure(
    db: AsyncSession, user: Any, *, source: str, account_id: Optional[UUID] = None
) -> Any:
    """Close the user's door and file the request; the row of a user already
    asked about is returned as it is (a repeated push never makes a second one).
    Commit is the caller's."""
    from app.core.legion_auth import NO_PASSWORD
    from app.models.user_erasure import UserErasure

    user.is_active = False
    user.hashed_password = NO_PASSWORD
    db.add(user)
    row = (await db.execute(select(UserErasure).where(UserErasure.user_id == user.id))).scalars().first()
    if row is None:
        row = UserErasure(
            user_id=user.id,
            legion_account_id=account_id if account_id is not None else user.legion_account_id,
            source=source,
        )
        db.add(row)
        await db.flush()
        logger.info("user_erasure: user %s queued for erasure (%s)", user.id, source)
    return row


def is_overdue(row: Any, now: Optional[datetime] = None) -> bool:
    """Not done 24 hours after it was asked (Privacy §6)."""
    if row.completed_at is not None:
        return False
    return (now or _now()) - _aware(row.requested_at) > ERASURE_DEADLINE


# ---------- The steps ---------------------------------------------------------------------


async def _erase_rooms(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    """Every room the user created, whole: container and volume, archive, then
    the rows. Rooms are single-user, so there is no one else's conversation to
    keep. A failure of workspaced raises: the room stays marked and is retried."""
    from app.core import workspaced_client as wsd
    from app.core.room_delete import delete_room_rows
    from app.models.message import Message
    from app.models.room import Room, RoomUserLink

    rooms = (await db.execute(select(Room).where(Room.creator_id == user_id))).scalars().all()
    for room in rooms:
        room_id = room.id
        if room.deletion_started_at is None:
            room.deletion_started_at = _now()
            db.add(room)
            await db.commit()
        await wsd.destroy_workspace(str(room_id))
        await wsd.delete_workspace_archive(str(room_id))
        await delete_room_rows(room, db)
    # A link in a room that is not theirs, and lines they wrote there: the
    # room is not erased, the user's mark on it is.
    links = await db.execute(delete(RoomUserLink).where(RoomUserLink.user_id == user_id))
    await db.execute(update(Message).where(Message.sender_user_id == user_id).values(sender_user_id=None))
    await db.commit()
    return {"rooms": len(rooms), "foreign_room_links": int(links.rowcount or 0)}


async def _erase_agents(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    from app.api.v1.agents import _delete_agent_dependents
    from app.models.agent import Agent

    agents = (await db.execute(select(Agent).where(Agent.owner_id == user_id))).scalars().all()
    for agent in agents:
        await _delete_agent_dependents(db, agent.id)
        await db.delete(agent)
    await db.commit()
    return {"agents": len(agents)}


async def _erase_personal_mcps(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    """Runners first (container and volume at the MCP host), then their rows."""
    from app.core.personal_mcp_host import host_client
    from app.models.personal_mcp import PersonalMcpEvent, PersonalMcpImportPreview, PersonalMcpRuntime

    runtimes = (
        await db.execute(select(PersonalMcpRuntime).where(PersonalMcpRuntime.owner_id == user_id))
    ).scalars().all()
    deleted_runners = 0
    for runtime in runtimes:
        if runtime.actual_status == "deleted":
            continue
        # PersonalMcpHostError propagates: the container may still exist, retry.
        await host_client.delete(user_id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
        deleted_runners += 1
    await db.execute(delete(PersonalMcpEvent).where(PersonalMcpEvent.owner_id == user_id))
    await db.execute(delete(PersonalMcpImportPreview).where(PersonalMcpImportPreview.owner_id == user_id))
    await db.execute(delete(PersonalMcpRuntime).where(PersonalMcpRuntime.owner_id == user_id))
    await db.commit()
    return {"personal_mcps": len(runtimes), "personal_mcp_runners_removed": deleted_runners}


async def _erase_installed_items(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    """Installed skills and MCPs with everything that hangs off them, then the
    skills the user uploaded (stored files too)."""
    from app.core.skill_install import delete_stored_files
    from app.models.market import AgentSkill, RoomMcpServer, UserDefaultProvision, UserInstalledItem
    from app.models.mcp_connection import McpConnectionSnapshot
    from app.models.mcp_oauth import McpOAuthFlow
    from app.models.secret import SecretBinding
    from app.models.skill import Skill

    item_ids = (
        await db.execute(select(UserInstalledItem.id).where(UserInstalledItem.user_id == user_id))
    ).scalars().all()
    if item_ids:
        await db.execute(delete(AgentSkill).where(AgentSkill.user_installed_item_id.in_(item_ids)))
        await db.execute(delete(RoomMcpServer).where(RoomMcpServer.user_installed_item_id.in_(item_ids)))
        await db.execute(delete(McpConnectionSnapshot).where(McpConnectionSnapshot.installed_item_id.in_(item_ids)))
        await db.execute(delete(McpOAuthFlow).where(McpOAuthFlow.installed_item_id.in_(item_ids)))
        await db.execute(delete(SecretBinding).where(SecretBinding.resource_id.in_(item_ids)))
    await db.execute(delete(McpOAuthFlow).where(McpOAuthFlow.user_id == user_id))
    await db.execute(delete(McpConnectionSnapshot).where(McpConnectionSnapshot.owner_id == user_id))
    await db.execute(delete(RoomMcpServer).where(RoomMcpServer.added_by == user_id))
    await db.execute(delete(UserInstalledItem).where(UserInstalledItem.user_id == user_id))
    await db.execute(delete(UserDefaultProvision).where(UserDefaultProvision.user_id == user_id))
    await db.flush()
    skills = (await db.execute(select(Skill).where(Skill.owner_id == user_id))).scalars().all()
    for skill in skills:
        await delete_stored_files(skill, db)
        await db.delete(skill)
    await db.commit()
    return {"installed_items": len(item_ids), "skills": len(skills)}


async def _revoke_connectors(user_id: UUID, db: AsyncSession) -> list[UUID]:
    """Revoke the user's connectors and close their live sockets. Their rows go later."""
    from app.models.connector import Connector
    from app.websockets.manager import manager

    ids = (await db.execute(select(Connector.id).where(Connector.owner_id == user_id))).scalars().all()
    if ids:
        await db.execute(
            update(Connector).where(Connector.id.in_(ids), Connector.revoked_at.is_(None)).values(revoked_at=_now())
        )
        await db.commit()
    for connector_id in ids:
        await manager.close_connector(connector_id)
    return list(ids)


async def _erase_connectors(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    from app.models.connector import Connector, ConnectorRoomGrant

    # A socket opened since the door was closed cannot be (a revoked token is
    # refused), but closing again costs nothing.
    ids = await _revoke_connectors(user_id, db)
    if ids:
        # Frames waiting for a connector and its grants go with it (ON DELETE CASCADE).
        await db.execute(delete(ConnectorRoomGrant).where(ConnectorRoomGrant.connector_id.in_(ids)))
        await db.execute(delete(Connector).where(Connector.id.in_(ids)))
    await db.execute(delete(ConnectorRoomGrant).where(ConnectorRoomGrant.granted_by == user_id))
    await db.commit()
    return {"connectors": len(ids)}


async def _erase_keys_and_secrets(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    from app.models.legion_link import LegionLink
    from app.models.provider_connection import ProviderConnection
    from app.models.secret import SecretAudit, SecretBinding, SecretLease, UserSecret

    secret_ids = (await db.execute(select(UserSecret.id).where(UserSecret.owner_id == user_id))).scalars().all()
    await db.execute(delete(ProviderConnection).where(ProviderConnection.user_id == user_id))
    await db.execute(delete(SecretLease).where(SecretLease.owner_id == user_id))
    await db.execute(delete(SecretBinding).where(SecretBinding.owner_id == user_id))
    audit = delete(SecretAudit).where(SecretAudit.actor_id == user_id)
    if secret_ids:
        audit = delete(SecretAudit).where(
            sa.or_(SecretAudit.actor_id == user_id, SecretAudit.secret_id.in_(secret_ids))
        )
    await db.execute(audit)
    await db.execute(delete(UserSecret).where(UserSecret.owner_id == user_id))
    link = await db.execute(delete(LegionLink).where(LegionLink.user_id == user_id))
    await db.commit()
    return {"secrets": len(secret_ids), "legion_links": int(link.rowcount or 0)}


async def _scrub_invitations(user_id: UUID, db: AsyncSession) -> dict[str, int]:
    from app.models.registration_invite import RegistrationInvite

    made = await db.execute(delete(RegistrationInvite).where(RegistrationInvite.created_by == user_id))
    used = await db.execute(
        update(RegistrationInvite).where(RegistrationInvite.used_by == user_id).values(email=None)
    )
    await db.commit()
    return {"invitations_deleted": int(made.rowcount or 0), "invitations_blanked": int(used.rowcount or 0)}


def _scrub_payload(payload: Any, needles: tuple[str, ...]) -> Any:
    """The operator's audit payload without addresses, names or the user's own email."""
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if "email" in lowered or "name" in lowered or lowered in {"note", "reason", "comment"}:
                continue
            out[key] = _scrub_payload(value, needles)
        return out
    if isinstance(payload, list):
        return [_scrub_payload(v, needles) for v in payload]
    if isinstance(payload, str) and any(n and n in payload.lower() for n in needles):
        return "[erased]"
    return payload


async def _scrub_admin_actions(user_id: UUID, emails: tuple[str, ...], db: AsyncSession) -> dict[str, int]:
    from app.models.admin_action import AdminAction

    rows = (await db.execute(select(AdminAction).where(AdminAction.target_user_id == user_id))).scalars().all()
    needles = tuple(e.lower() for e in emails if e)
    for row in rows:
        if row.payload:
            row.payload = _scrub_payload(row.payload, needles)
            db.add(row)
    await db.commit()
    return {"admin_actions_scrubbed": len(rows)}


async def _tombstone_credentials(account_id: Optional[UUID], db: AsyncSession) -> None:
    """The Hub's copy of the Legion credentials becomes a tombstone."""
    if account_id is None:
        return
    from app.models.legion_credential import LegionCredential

    row = await db.get(LegionCredential, account_id)
    if row is None:
        return
    if row.status != "erased":
        row.token_version = (row.token_version or 0) + 1
    row.email = f"deleted-{account_id.hex}@deleted.invalid"
    row.password_hash = None
    row.status = "erased"
    row.updated_at = _now()
    db.add(row)
    await db.commit()


# ---------- The whole erasure --------------------------------------------------------------


async def erase_user(user_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """Erase everything the user left in the Hub; the user row becomes an
    anonymous shell. Idempotent. Returns counters (no personal data). A failure
    of an external service raises; call it again."""
    from app.core.legion_auth import NO_PASSWORD
    from app.models.user import User

    user = await db.get(User, user_id)
    if user is None:
        return {}
    account_id = user.legion_account_id
    emails = (user.email or "", user.full_name or "")

    # The door first: nothing the user does races the erasure.
    if user.is_active or user.hashed_password != NO_PASSWORD:
        user.is_active = False
        user.hashed_password = NO_PASSWORD
        db.add(user)
        await db.commit()
    # External agents too: their sockets are open now and would keep acting
    # in the rooms while those are being erased.
    await _revoke_connectors(user_id, db)

    report: dict[str, Any] = {}
    steps: tuple[Callable[[UUID, AsyncSession], Awaitable[dict[str, int]]], ...] = (
        _erase_rooms,
        _erase_agents,
        _erase_personal_mcps,
        _erase_installed_items,
        _erase_connectors,
        _erase_keys_and_secrets,
        _scrub_invitations,
    )
    for step in steps:
        report.update(await step(user_id, db))
    report.update(await _scrub_admin_actions(user_id, emails, db))

    for handler in get_registry().handlers_for(USER_ERASURE_EVENT):
        extra = await handler(db, user_id, account_id)
        if isinstance(extra, dict):
            report.update(extra)

    await _tombstone_credentials(account_id, db)

    user = await db.get(User, user_id)
    user.email = placeholder_email(user_id)
    user.full_name = None
    user.two_factor_enabled = False
    user.is_active = False
    user.hashed_password = NO_PASSWORD
    user.erased_at = user.erased_at or _now()
    db.add(user)
    await db.commit()
    logger.info("user_erasure: user %s erased: %s", user_id, report)
    return report


# ---------- The repeat loop ----------------------------------------------------------------

_BACKOFF_BASE_SEC = 60
_BACKOFF_MAX_SEC = 900
BATCH = 1


def _backoff(attempts: int) -> timedelta:
    return timedelta(seconds=min(_BACKOFF_BASE_SEC * 2 ** max(attempts - 1, 0), _BACKOFF_MAX_SEC))


async def run_one(request_id: UUID, db: AsyncSession) -> bool:
    """Run one request. True when it finished; a failure is recorded on the row."""
    from app.models.user_erasure import UserErasure

    row = await db.get(UserErasure, request_id)
    if row is None or row.completed_at is not None:
        return row is not None
    now = _now()
    row.started_at = row.started_at or now
    row.attempts = (row.attempts or 0) + 1
    user_id = row.user_id
    db.add(row)
    await db.commit()
    try:
        report = await erase_user(user_id, db)
    except Exception as exc:  # noqa: BLE001 - any failure is retried
        await db.rollback()
        row = await db.get(UserErasure, request_id)
        row.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        row.next_attempt_at = _now() + _backoff(row.attempts)
        db.add(row)
        await db.commit()
        logger.warning("user_erasure: attempt %d for user %s failed: %s", row.attempts, user_id, type(exc).__name__)
        return False
    row = await db.get(UserErasure, request_id)
    row.completed_at = _now()
    row.last_error = None
    row.report = report
    db.add(row)
    await db.commit()
    return True


async def run_due_erasures(db: AsyncSession, now: Optional[datetime] = None) -> dict[str, int]:
    """Take the open requests whose time has come. The reconciler step."""
    from app.models.user_erasure import UserErasure

    now = now or _now()
    ids = (
        await db.execute(
            select(UserErasure.id)
            .where(UserErasure.completed_at.is_(None), UserErasure.next_attempt_at <= now)
            .order_by(UserErasure.requested_at)
            .limit(BATCH)
        )
    ).scalars().all()
    done = failed = 0
    for request_id in ids:
        if await run_one(request_id, db):
            done += 1
        else:
            failed += 1
    return {"erased": done, "failed": failed}


async def overdue_count(db: AsyncSession, now: Optional[datetime] = None) -> int:
    from app.models.user_erasure import UserErasure

    cutoff = (now or _now()) - ERASURE_DEADLINE
    return int(
        (
            await db.execute(
                select(sa.func.count(UserErasure.id)).where(
                    UserErasure.completed_at.is_(None), UserErasure.requested_at < cutoff
                )
            )
        ).scalar_one()
    )
