"""internal design notes stage 13 (O18): erasing a user's data from the Hub.

Unit part: the completeness guard. Every table that points at a user must have
a decision in ``user_erasure.HANDLED_TABLES``; a new table with a user's data
and no decision fails here, so the erasure cannot fall behind the schema.

DB part (PostgreSQL, marked ``db``): a user with something in every table that
points at them is erased with workspaced, the MCP host and the object stores
mocked. Run against a scratch database only:

    DATABASE_URL=postgresql+asyncpg://.../legion_erasure_test pytest -m db tests/test_user_erasure.py

They skip unless the database name contains "test".
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlmodel import SQLModel

from app.core import user_erasure

UTC = timezone.utc


# ---------- unit: the completeness guard -------------------------------------------------


def test_every_table_that_points_at_a_user_has_a_decision():
    import app.models  # noqa: F401

    referencing = user_erasure.user_reference_tables(SQLModel.metadata)
    undecided = sorted(set(referencing) - set(user_erasure.HANDLED_TABLES))
    assert not undecided, (
        "These tables hold a user id but have no entry in user_erasure.HANDLED_TABLES "
        f"(erase / anonymize / keep, with the reason): {undecided}"
    )


def test_no_decision_names_a_table_that_does_not_exist():
    import app.models  # noqa: F401

    stale = sorted(set(user_erasure.HANDLED_TABLES) - set(SQLModel.metadata.tables))
    assert not stale, f"HANDLED_TABLES names tables that are gone: {stale}"


def test_decisions_are_one_of_the_three():
    for table, (decision, reason) in user_erasure.HANDLED_TABLES.items():
        assert decision in {user_erasure.ERASE, user_erasure.ANONYMIZE, user_erasure.KEEP}, table
        assert reason.strip(), f"{table}: say why"


def test_payment_and_spending_records_are_the_only_ones_kept():
    kept = {t for t, (d, _) in user_erasure.HANDLED_TABLES.items() if d == user_erasure.KEEP}
    assert kept == {"user_erasure", "userquota", "mcp_credit_wallet", "mcp_credit_ledger"}


def test_the_operators_audit_payload_loses_addresses_and_names():
    payload = {
        "plan": "pro", "email": "person@example.com", "note": "asked by mail from person@example.com",
        "nested": {"display_name": "Gina", "hours": 3, "text": "mailto:person@example.com"},
    }
    scrubbed = user_erasure._scrub_payload(payload, ("person@example.com",))
    assert scrubbed == {"plan": "pro", "nested": {"hours": 3, "text": "[erased]"}}


def test_overdue_means_older_than_24_hours_and_not_done():
    from types import SimpleNamespace

    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    fresh = SimpleNamespace(requested_at=now - timedelta(hours=23), completed_at=None)
    late = SimpleNamespace(requested_at=now - timedelta(hours=25), completed_at=None)
    done = SimpleNamespace(requested_at=now - timedelta(hours=25), completed_at=now)
    assert [user_erasure.is_overdue(r, now) for r in (fresh, late, done)] == [False, True, False]


# ---------- DB integration (requires PostgreSQL) --------------------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")


def _is_scratch_db(url: str) -> bool:
    from sqlalchemy.engine import make_url

    try:
        return "test" in (make_url(url).database or "")
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _is_scratch_db(DB_URL), reason="needs DATABASE_URL pointing at a *test* database")


@pytest_asyncio.fixture
async def engine() -> AsyncGenerator:
    from sqlalchemy.ext.asyncio import create_async_engine

    import app.models  # noqa: F401

    eng = create_async_engine(DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def sessions(engine):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def clean_requests(sessions):
    """The scratch database outlives the run and the step takes the oldest open request first."""
    from sqlalchemy import delete

    from app.models.user_erasure import UserErasure

    async with sessions() as db:
        await db.execute(delete(UserErasure))
        await db.commit()


@pytest.fixture
def outside(monkeypatch):
    """workspaced, the MCP host and the object stores: they record what they were asked to delete."""
    from app.core import context_observations, log_storage
    from app.core import personal_mcp_host
    from app.core import workspaced_client

    calls = {"destroy": [], "archive": [], "runner": [], "payload": [], "log": []}

    async def destroy(room_id):
        calls["destroy"].append(room_id)
        return {}

    async def archive(room_id):
        calls["archive"].append(room_id)
        return {}

    async def runner(owner_id, item_id, *, correlation_id=None):
        calls["runner"].append((owner_id, item_id))
        return {}

    class Store:
        async def delete(self, key):
            calls["payload"].append(key)

    class Archive:
        async def delete(self, key):
            calls["log"].append(key)

    monkeypatch.setattr(workspaced_client, "destroy_workspace", destroy)
    monkeypatch.setattr(workspaced_client, "delete_workspace_archive", archive)
    monkeypatch.setattr(personal_mcp_host.host_client, "delete", runner)
    monkeypatch.setattr(context_observations, "_payload_store", lambda: Store())
    monkeypatch.setattr(log_storage, "get_log_archive", lambda: Archive())
    return calls


async def _seed_user(sessions, *, email=None) -> dict:
    """A user with something in every table that points at them."""
    import app.models as m
    from app.models.connector import Connector, ConnectorOutbox, ConnectorRoomGrant
    from app.models.legion_credential import LegionCredential
    from app.models.legion_link import LegionLink
    from app.models.mcp_connection import McpConnectionSnapshot
    from app.models.mcp_credits import McpCreditLedger, McpCreditWallet
    from app.models.mcp_oauth import McpOAuthFlow
    from app.models.personal_mcp import PersonalMcpEvent, PersonalMcpImportPreview, PersonalMcpRuntime
    from app.models.provider_connection import ProviderConnection
    from app.models.registration_invite import RegistrationInvite
    from app.models.secret import (
        SecretAudit, SecretBinding, SecretLease, SecretKind, SecretResourceType, SecretPurpose,
        SecretActorType, SecretAuditAction, SecretAuditOutcome, SecretState, UserSecret,
    )
    from app.models.session import UserAgentSession
    from app.models.context_artifact import ContextArtifact

    account_id = uuid4()
    email = email or f"{uuid4().hex[:10]}@example.com"
    ids: dict = {"account_id": account_id, "email": email}
    now = datetime.now(UTC)
    async with sessions() as db:
        user = m.User(id=uuid4(), email=email, full_name="Gina Person", hashed_password="hash",
                      legion_account_id=account_id)
        db.add(user)
        db.add(m.UserQuota(user_id=user.id, plan_tier="pro", plan_expires_at=now + timedelta(days=20)))
        db.add(LegionCredential(account_id=account_id, email=email, password_hash="hash", token_version=3))
        await db.flush()
        ids["user_id"] = user.id

        agent = m.Agent(nickname="helper", agent_uid=f"helper-{uuid4().hex[:6]}", agent_type="internal",
                        owner_id=user.id, token=uuid4().hex)
        router = m.Agent(nickname="Jev", agent_uid=f"jev-{uuid4().hex[:6]}", agent_type="internal",
                         owner_id=user.id, token=uuid4().hex)
        db.add_all([agent, router])
        await db.flush()
        ids["agent_id"] = agent.id

        room = m.Room(name="my room", creator_id=user.id, created_at=now)
        db.add(room)
        await db.flush()
        ids["room_id"] = room.id
        db.add(m.RoomUserLink(room_id=room.id, user_id=user.id))
        db.add(m.RoomAgentLink(room_id=room.id, agent_id=agent.id))
        db.add(m.Message(room_id=room.id, content="a private sentence", sender_user_id=user.id))
        db.add(m.Workspace(room_id=room.id, quota_owner_id=user.id, created_at=now,
                           archived_at=now - timedelta(days=1)))
        task = m.Task(room_id=room.id, title="a task")
        db.add(task)
        await db.flush()
        session = UserAgentSession(room_id=room.id, kind="user_agent_p1", user_id=user.id, agent_id=agent.id,
                                   task_id=task.id)
        db.add(session)
        await db.flush()
        db.add(m.AgentSession(agent_id=agent.id, room_id=room.id, task_id=task.id, log_key="logs/u/1.jsonl"))
        db.add(ContextArtifact(
            room_id=room.id, session_id=session.id, tool_name="read_file", args_hash="h", source_fingerprint="f",
            storage_key=f"context-observations/{room.id}/x.txt",
        ))

        secret = UserSecret(owner_id=user.id, kind=SecretKind.LLM_API_KEY, label="my key", provider="openai",
                            ciphertext=b"sealed", key_id="k1", state=SecretState.ACTIVE)
        db.add(secret)
        await db.flush()
        db.add(ProviderConnection(user_id=user.id, provider="openai", secret_id=secret.id))
        db.add(SecretBinding(secret_id=secret.id, owner_id=user.id, resource_type=SecretResourceType.ROOM,
                             resource_id=room.id, purpose=SecretPurpose.ROOM_ENV))
        db.add(SecretLease(owner_id=user.id, resource_type=SecretResourceType.ROOM, resource_id=room.id,
                           binding_ids=[], env_names=[], token_hash=uuid4().hex * 2, expires_at=now + timedelta(hours=1)))
        db.add(SecretAudit(actor_type=SecretActorType.USER, actor_id=user.id, action=SecretAuditAction.CREATE,
                           secret_id=secret.id, outcome=SecretAuditOutcome.SUCCESS, source_ip="203.0.113.7"))
        db.add(LegionLink(user_id=user.id, ciphertext=b"sealed", key_id="k1", key_prefix="lgn_abcd",
                          account_id=account_id, email=email))

        # A custom skill with a file, an MCP with a runner, and everything hanging off them.
        skill = m.Skill(owner_id=user.id, name=f"my-skill-{uuid4().hex[:6]}", source="upload", skill_md="body")
        db.add(skill)
        await db.flush()
        db.add(m.SkillFile(skill_id=skill.id, path="notes.md", content="private"))
        ids["skill_id"] = skill.id
        skill_item = m.UserInstalledItem(user_id=user.id, type="skill", is_custom=True, name="my skill",
                                         skill_id=skill.id)
        mcp_item = m.UserInstalledItem(user_id=user.id, type="mcp", is_custom=True, name="my mcp")
        db.add_all([skill_item, mcp_item])
        await db.flush()
        db.add(m.AgentSkill(agent_id=agent.id, user_installed_item_id=skill_item.id))
        db.add(m.RoomMcpServer(room_id=room.id, user_installed_item_id=mcp_item.id, added_by=user.id))
        db.add(McpConnectionSnapshot(installed_item_id=mcp_item.id, owner_id=user.id, tools_snapshot=[]))
        db.add(McpOAuthFlow(state=uuid4().hex, code_verifier="v", installed_item_id=mcp_item.id, user_id=user.id,
                            redirect_uri="https://x/cb", issuer="https://i", client_id="c",
                            expires_at=now + timedelta(minutes=5)))
        db.add(SecretBinding(secret_id=secret.id, owner_id=user.id, resource_type=SecretResourceType.INSTALLED_MCP,
                             resource_id=mcp_item.id, purpose=SecretPurpose.MCP_ENV))
        db.add(m.UserDefaultProvision(user_id=user.id, market_item_id=uuid4()))
        preview = PersonalMcpImportPreview(owner_id=user.id, candidates=[], rejected=[],
                                           expires_at=now + timedelta(hours=1))
        db.add(preview)
        await db.flush()
        db.add(PersonalMcpRuntime(installed_item_id=mcp_item.id, owner_id=user.id, preview_id=preview.id,
                                  candidate_id=uuid4(), actual_status="running", correlation_id="corr-1",
                                  launch_spec={"package": "x", "env_names": []}))
        ids["mcp_item_id"] = mcp_item.id
        await db.flush()
        db.add(PersonalMcpEvent(installed_item_id=mcp_item.id, owner_id=user.id, step="ready",
                                message="up", correlation_id="corr-1"))

        connector = Connector(owner_id=user.id, label="my vps")
        db.add(connector)
        await db.flush()
        ids["connector_id"] = connector.id
        db.add(ConnectorRoomGrant(connector_id=connector.id, room_id=room.id, granted_by=user.id))
        db.add(ConnectorOutbox(connector_id=connector.id, session_id=session.id, seq=1, kind="dispatch", payload={}))

        db.add(RegistrationInvite(token_hash=uuid4().hex, created_by=user.id, expires_at=now + timedelta(days=1)))
        other_user = m.User(id=uuid4(), email=f"{uuid4().hex[:10]}@example.com", hashed_password="x")
        db.add(other_user)
        await db.flush()
        ids["other_user_id"] = other_user.id
        db.add(RegistrationInvite(token_hash=uuid4().hex, created_by=other_user.id, email=email,
                                  expires_at=now + timedelta(days=1), used_by=user.id, used_at=now))
        db.add(m.AdminAction(actor="admin_center:op", action="set_plan", target_user_id=user.id,
                             payload={"tier": "pro", "email": email, "note": f"asked by {email}"}))

        db.add(McpCreditWallet(user_id=user.id, period_started_at=now, period_ends_at=now + timedelta(days=7)))
        await db.flush()
        db.add(McpCreditLedger(user_id=user.id, kind="grant", amount_mcr=500, balance_after_mcr=500))
        await db.commit()
    return ids


async def _count(sessions, model, *where) -> int:
    from sqlalchemy import func, select

    async with sessions() as db:
        return (await db.execute(select(func.count()).select_from(model).where(*where))).scalar_one()


@needs_db
@pytest.mark.db
@pytest.mark.asyncio
async def test_erase_user_removes_everything_and_leaves_a_shell(sessions, outside, monkeypatch):
    import app.models as m
    from app.models.connector import Connector
    from app.models.legion_credential import LegionCredential
    from app.models.legion_link import LegionLink
    from app.models.mcp_credits import McpCreditLedger, McpCreditWallet
    from app.models.personal_mcp import PersonalMcpEvent, PersonalMcpRuntime
    from app.models.provider_connection import ProviderConnection
    from app.models.registration_invite import RegistrationInvite
    from app.models.secret import SecretAudit, SecretBinding, SecretLease, UserSecret
    from app.models.admin_action import AdminAction
    from app.models.skill import Skill
    from sqlalchemy import select

    from app.websockets.manager import manager

    closed: list = []

    async def _close(connector_id, code=1008):
        async with sessions() as db:
            # Revoked before the socket is closed, so a reconnect is refused.
            closed.append((connector_id, (await db.get(Connector, connector_id)).revoked_at is not None))
        return True

    monkeypatch.setattr(manager, "close_connector", _close)
    ids = await _seed_user(sessions)
    uid = ids["user_id"]

    async with sessions() as db:
        report = await user_erasure.erase_user(uid, db)

    # The live connector socket was closed at the door, before the rooms went.
    assert closed and closed[0] == (ids["connector_id"], True)
    assert report["rooms"] == 1 and report["agents"] == 2 and report["personal_mcps"] == 1
    # Containers, volumes, archives, runners and stored objects were asked to go.
    assert outside["destroy"] == [str(ids["room_id"])] and outside["archive"] == [str(ids["room_id"])]
    assert outside["runner"] == [(uid, ids["mcp_item_id"])]
    assert outside["payload"] == [f"context-observations/{ids['room_id']}/x.txt"]
    assert outside["log"] == ["logs/u/1.jsonl"]

    gone = [
        (m.Room, m.Room.creator_id == uid), (m.RoomUserLink, m.RoomUserLink.user_id == uid),
        (m.Message, m.Message.room_id == ids["room_id"]), (m.Workspace, m.Workspace.quota_owner_id == uid),
        (m.Task, m.Task.room_id == ids["room_id"]), (m.AgentSession, m.AgentSession.room_id == ids["room_id"]),
        (m.Agent, m.Agent.owner_id == uid), (m.UserInstalledItem, m.UserInstalledItem.user_id == uid),
        (m.AgentSkill, m.AgentSkill.agent_id == ids["agent_id"]), (m.RoomMcpServer, m.RoomMcpServer.added_by == uid),
        (m.UserDefaultProvision, m.UserDefaultProvision.user_id == uid), (Skill, Skill.owner_id == uid),
        (m.SkillFile, m.SkillFile.skill_id == ids["skill_id"]),
        (UserSecret, UserSecret.owner_id == uid), (SecretBinding, SecretBinding.owner_id == uid),
        (SecretLease, SecretLease.owner_id == uid), (SecretAudit, SecretAudit.actor_id == uid),
        (ProviderConnection, ProviderConnection.user_id == uid), (LegionLink, LegionLink.user_id == uid),
        (Connector, Connector.owner_id == uid), (PersonalMcpRuntime, PersonalMcpRuntime.owner_id == uid),
        (PersonalMcpEvent, PersonalMcpEvent.owner_id == uid),
        (RegistrationInvite, RegistrationInvite.created_by == uid),
    ]
    for model, clause in gone:
        assert await _count(sessions, model, clause) == 0, model.__tablename__

    async with sessions() as db:
        user = await db.get(m.User, uid)
        # An anonymous shell: nothing that says who it was, nobody can sign in.
        assert user.email == f"erased+{uid}@invalid" and user.full_name is None
        assert user.is_active is False and user.hashed_password == "!legion" and user.erased_at is not None
        # The copy of the Legion credentials is a tombstone.
        cred = await db.get(LegionCredential, ids["account_id"])
        assert (cred.password_hash, cred.status, cred.token_version) == (None, "erased", 4)
        assert ids["email"] not in cred.email
        # The invitation another user issued to this one no longer names an address.
        invite = (await db.execute(
            select(RegistrationInvite).where(RegistrationInvite.used_by == uid)
        )).scalars().one()
        assert invite.email is None
        # The operator's audit stays, without addresses.

        action = (await db.execute(select(AdminAction).where(AdminAction.target_user_id == uid))).scalars().one()
        assert action.payload == {"tier": "pro"}
        # Records of payment and spending are in place.
        assert (await db.execute(select(m.UserQuota).where(m.UserQuota.user_id == uid))).scalars().one().plan_tier == "pro"
        assert await db.get(McpCreditWallet, uid) is not None
        assert (await db.execute(select(McpCreditLedger).where(McpCreditLedger.user_id == uid))).scalars().one().amount_mcr == 500
        # Somebody else's data is untouched.
        assert await db.get(m.User, ids["other_user_id"]) is not None


@needs_db
@pytest.mark.db
@pytest.mark.asyncio
async def test_erasing_twice_is_harmless_and_a_failure_resumes(sessions, outside, monkeypatch):
    """workspaced fails once in the middle: the pass raises, the next one finishes."""
    from app.core import workspaced_client
    import app.models as m

    ids = await _seed_user(sessions)
    uid = ids["user_id"]
    real = workspaced_client.destroy_workspace
    state = {"failed": False}

    async def flaky(room_id):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("workspaced is down")
        return await real(room_id)

    monkeypatch.setattr(workspaced_client, "destroy_workspace", flaky)
    async with sessions() as db:
        with pytest.raises(RuntimeError):
            await user_erasure.erase_user(uid, db)
    # The door is already shut, the room is marked and still there.
    async with sessions() as db:
        user = await db.get(m.User, uid)
        assert user.is_active is False and user.erased_at is None
        assert await _count(sessions, m.Room, m.Room.creator_id == uid) == 1
    async with sessions() as db:
        await user_erasure.erase_user(uid, db)
    assert await _count(sessions, m.Room, m.Room.creator_id == uid) == 0
    async with sessions() as db:
        second = await user_erasure.erase_user(uid, db)
    assert second["rooms"] == 0 and second["agents"] == 0


@needs_db
@pytest.mark.db
@pytest.mark.asyncio
async def test_the_request_runs_to_completion_and_is_filed_once(sessions, outside, clean_requests):
    import app.models as m
    from app.models.user_erasure import UserErasure
    from sqlalchemy import select

    ids = await _seed_user(sessions)
    uid = ids["user_id"]
    async with sessions() as db:
        user = await db.get(m.User, uid)
        first = await user_erasure.request_erasure(db, user, source=user_erasure.SOURCE_MARKET)
        again = await user_erasure.request_erasure(db, user, source=user_erasure.SOURCE_OPERATOR)
        await db.commit()
        assert first.id == again.id and again.source == "market"
        assert user.is_active is False and user.hashed_password == "!legion"

    async with sessions() as db:
        assert await user_erasure.overdue_count(db) == 0
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 1, "failed": 0}
    async with sessions() as db:
        row = (await db.execute(select(UserErasure).where(UserErasure.user_id == uid))).scalars().one()
        assert row.completed_at is not None and row.last_error is None and row.attempts == 1
        # Counters only: no address, no name in the report.
        assert row.report["rooms"] == 1 and ids["email"] not in str(row.report)
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 0, "failed": 0}


@needs_db
@pytest.mark.db
@pytest.mark.asyncio
async def test_a_failing_request_waits_and_goes_overdue(sessions, outside, clean_requests, monkeypatch):
    import app.models as m
    from app.core import workspaced_client
    from app.models.user_erasure import UserErasure
    from sqlalchemy import select, update

    ids = await _seed_user(sessions)
    uid = ids["user_id"]
    monkeypatch.setattr(workspaced_client, "destroy_workspace", AsyncMock(side_effect=RuntimeError("down")))
    async with sessions() as db:
        await user_erasure.request_erasure(db, await db.get(m.User, uid), source=user_erasure.SOURCE_OPERATOR)
        await db.commit()
    async with sessions() as db:
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 0, "failed": 1}
    async with sessions() as db:
        row = (await db.execute(select(UserErasure).where(UserErasure.user_id == uid))).scalars().one()
        assert row.completed_at is None and row.attempts == 1 and "RuntimeError" in row.last_error
        assert row.next_attempt_at > datetime.now(UTC)
        # Not due yet: the next pass leaves it for later.
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 0, "failed": 0}
        assert await user_erasure.overdue_count(db) == 0
        await db.execute(update(UserErasure).where(UserErasure.id == row.id)
                         .values(requested_at=datetime.now(UTC) - timedelta(hours=30)))
        await db.commit()
        assert await user_erasure.overdue_count(db) == 1


@needs_db
@pytest.mark.db
@pytest.mark.asyncio
async def test_market_can_push_an_erased_account_again_after_a_hub_restore(sessions, outside, clean_requests):
    """Disaster recovery (PROD_RUNBOOK.md): the restored Hub still has a user Market already erased,
    and Market pushes every erased account again. The push closes and erases the user that came back
    from the backup, and a repeat of it, at any time, changes nothing."""
    import app.models as m
    from app.core import legion_auth
    from app.models.user_erasure import UserErasure
    from sqlalchemy import select

    ids = await _seed_user(sessions)
    uid, account_id = ids["user_id"], ids["account_id"]
    push = {"email": f"deleted-{account_id.hex}@deleted.invalid", "password_hash": None,
            "token_version": 99, "status": "erased"}

    async with sessions() as db:
        assert await legion_auth.store_credentials(db, account_id, push) is True
        await db.commit()
        user = await db.get(m.User, uid)
        assert user.is_active is False and user.hashed_password == "!legion"
    async with sessions() as db:
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 1, "failed": 0}

    # The same push once more, now that the data is gone: no second request, nothing to run.
    async with sessions() as db:
        assert await legion_auth.store_credentials(db, account_id, push) is True
        await db.commit()
        rows = (await db.execute(select(UserErasure).where(UserErasure.user_id == uid))).scalars().all()
        assert len(rows) == 1 and rows[0].completed_at is not None and rows[0].attempts == 1
        assert (await user_erasure.run_due_erasures(db)) == {"erased": 0, "failed": 0}
