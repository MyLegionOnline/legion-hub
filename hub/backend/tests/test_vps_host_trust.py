"""internal design notes §3.5/§10 (Stage 6) — the TOFU ask-user card for
`host_key_unknown`/`host_key_mismatch` and the exactly-one-retry it earns.

Same shape as tests/test_vps_connection_inject.py: real Postgres (gated on
INIT_DB_TEST_URL + PyNaCl), the upstream Gateway call faked via
`mcp_gateway._gateway_post` so nothing here touches the network or vps-mcp
itself — only the Room/server-inventory path plus the retry wiring in
`execute_mcp_tool`.
"""

import json
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL

pytest.importorskip("nacl")


def _keyring():
    from app.core.secret_keyring import SecretKeyring

    return SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


_SHARED_KEYRING = _keyring()


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=_SHARED_KEYRING))


def _install_runtime_keyring() -> None:
    from app.core.secrets import runtime as secrets_runtime

    secrets_runtime.configure_runtime_keyring(_SHARED_KEYRING)


if TEST_DB_URL:
    _install_runtime_keyring()


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_room(db, room_id, creator_id, *, vps_access_mode="full"):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=creator_id, email=f"vht-{creator_id}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="vht", creator_id=creator_id, vps_access_mode=vps_access_mode))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    await db.commit()


async def _add_server(db, room_id, owner_id, **overrides):
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.models.secret import SecretKind, SecretPurpose

    svc = _service(db)
    kwargs = dict(host="203.0.113.10", root_password="root-password-value")
    kwargs.update(overrides)
    await vps_inventory.add_server(
        db, service=svc, room_id=room_id, actor_user_id=owner_id, name="prod", **kwargs,
    )
    await room_secrets.bind_room_secret(
        db, service=svc, room_id=room_id, actor_user_id=owner_id,
        env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
        purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
    )
    await db.commit()


def _vps_binding():
    from app.core.mcp_gateway import McpToolBinding

    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="legion-vps-mcp",
        tool_name="vps_exec",
        custom_config=None,
        isolation=None,
    )


def _host_key_unknown_error(fingerprint="SHA256:AAAAbbbbCCCCddddEEEEffffGGGGhhhh0000===="):
    return json.dumps({
        "error": "host_key_unknown",
        "message": "The server's host key is not yet trusted",
        "fingerprint": fingerprint,
    })


def _host_key_mismatch_error(expected="SHA256:old", actual="SHA256:new"):
    return json.dumps({
        "error": "host_key_mismatch",
        "message": "The server's host key does not match the trusted fingerprint",
        "expected_fingerprint": expected,
        "actual_fingerprint": actual,
    })


def _sequenced_gateway_post(*responses):
    """Each call returns the next `(isError, text)` pair; the last repeats."""
    calls: list[dict] = []

    async def fake_post(path, user_id, room_id, payload, **kwargs):
        calls.append(payload)
        is_error, text = responses[min(len(calls) - 1, len(responses) - 1)]
        return {"isError": is_error, "content": [{"type": "text", "text": text}]}

    return fake_post, calls


@pytest.mark.asyncio
async def test_host_key_unknown_allowed_pins_fingerprint_and_retries_once():
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _add_server(db, room_id, owner_id)

        binding = _vps_binding()
        fake_post, calls = _sequenced_gateway_post(
            (True, _host_key_unknown_error("SHA256:freshfingerprint")),
            (False, json.dumps({"ok": True})),
        )

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_vps_host_trust", AsyncMock(return_value=True)) as confirm,
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {"server": "prod", "command": ["nginx", "-t"]},
                db=db, request_id=uuid4(),
            )

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        confirm.assert_awaited_once()
        assert confirm.await_args.kwargs["code"] == "host_key_unknown"
        assert len(calls) == 2
        # The retry carries the newly pinned fingerprint, not the empty one
        # the first attempt was made with.
        assert calls[1]["arguments"]["_legion_conn"]["host_fingerprint"] == "SHA256:freshfingerprint"

        servers = await vps_inventory.list_servers(db, room_id)
        assert next(s for s in servers if s["name"] == "PROD")["fingerprint"] == "SHA256:freshfingerprint"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_host_key_unknown_denied_returns_typed_error_and_pins_nothing():
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _add_server(db, room_id, owner_id)

        binding = _vps_binding()
        fake_post, calls = _sequenced_gateway_post(
            (True, _host_key_unknown_error()),
        )

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_vps_host_trust", AsyncMock(return_value=False)) as confirm,
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {"server": "prod", "command": ["nginx", "-t"]},
                db=db, request_id=uuid4(),
            )

        assert result == "Error: [host_key_unknown] the server's host key was not trusted and the operator did not accept it"
        confirm.assert_awaited_once()
        assert len(calls) == 1  # no retry on a Deny

        servers = await vps_inventory.list_servers(db, room_id)
        assert next(s for s in servers if s["name"] == "PROD")["fingerprint"] is None
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_host_key_mismatch_denied_leaves_old_fingerprint_untouched():
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _add_server(db, room_id, owner_id)
        await vps_inventory.set_fingerprint(
            db, room_id=room_id, actor_user_id=owner_id, name="prod", fingerprint="SHA256:old-trusted",
        )

        binding = _vps_binding()
        fake_post, calls = _sequenced_gateway_post(
            (True, _host_key_mismatch_error(expected="SHA256:old-trusted", actual="SHA256:new-and-suspicious")),
        )

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_vps_host_trust", AsyncMock(return_value=False)) as confirm,
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {"server": "prod", "command": ["nginx", "-t"]},
                db=db, request_id=uuid4(),
            )

        assert result.startswith("Error: [host_key_mismatch]")
        confirm.assert_awaited_once()
        assert confirm.await_args.kwargs["details"]["expected_fingerprint"] == "SHA256:old-trusted"
        assert confirm.await_args.kwargs["details"]["actual_fingerprint"] == "SHA256:new-and-suspicious"
        assert len(calls) == 1

        servers = await vps_inventory.list_servers(db, room_id)
        assert next(s for s in servers if s["name"] == "PROD")["fingerprint"] == "SHA256:old-trusted"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_host_key_mismatch_allowed_repins_and_retries_once():
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _add_server(db, room_id, owner_id)
        await vps_inventory.set_fingerprint(
            db, room_id=room_id, actor_user_id=owner_id, name="prod", fingerprint="SHA256:old-trusted",
        )

        binding = _vps_binding()
        fake_post, calls = _sequenced_gateway_post(
            (True, _host_key_mismatch_error(expected="SHA256:old-trusted", actual="SHA256:new-and-legit")),
            (False, json.dumps({"ok": True})),
        )

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_vps_host_trust", AsyncMock(return_value=True)),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {"server": "prod", "command": ["nginx", "-t"]},
                db=db, request_id=uuid4(),
            )

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        assert len(calls) == 2
        assert calls[1]["arguments"]["_legion_conn"]["host_fingerprint"] == "SHA256:new-and-legit"

        servers = await vps_inventory.list_servers(db, room_id)
        assert next(s for s in servers if s["name"] == "PROD")["fingerprint"] == "SHA256:new-and-legit"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_host_trust_card_carries_no_secret_value():
    """§10 acceptance — the AGENT_QUESTION event's content/metadata never
    includes the resolved credential, only the tool-supplied fingerprint(s).
    """
    from app.core import mcp_gateway
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _add_server(db, room_id, owner_id)

        binding = _vps_binding()
        fake_post, _ = _sequenced_gateway_post(
            (True, _host_key_unknown_error()),
        )

        emitted: list[dict] = []

        async def fake_emit_event(**kwargs):
            emitted.append(kwargs)

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "emit_event", AsyncMock(side_effect=fake_emit_event)),
            # No one answers the card in this test — force the timeout path
            # (a Deny, per §10) instantly instead of waiting the real
            # `MCP_POLICY_APPROVAL_TIMEOUT_SECONDS` (default 300s).
            patch.dict(os.environ, {"MCP_POLICY_APPROVAL_TIMEOUT_SECONDS": "0"}),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {"server": "prod", "command": ["nginx", "-t"]},
                db=db, request_id=uuid4(),
            )

        assert result.startswith("Error: [host_key_unknown]")
        question_events = [e for e in emitted if e.get("event_type") == "AGENT_QUESTION"]
        assert len(question_events) == 1
        blob = json.dumps(question_events[0], default=str)
        assert "everyday-password-value" not in blob
        assert "root-password-value" not in blob
        break

    await engine.dispose()
