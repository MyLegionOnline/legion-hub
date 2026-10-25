"""internal design notes Stage 3 tests — `_legion_conn` injection in execute_mcp_tool.

Gated on INIT_DB_TEST_URL and PyNaCl, same as tests/test_vps_inventory.py.
The upstream Gateway call itself is faked (``mcp_gateway._gateway_post``) so
these tests never touch the network — only the Room/secret resolve path
that ``vps_inventory.py`` and ``execute_mcp_tool`` run against real Postgres.
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


# One keyring for the whole module: ``_service()`` below and
# ``execute_mcp_tool``'s VPS-injection path (which builds its own
# SecretService via ``secrets_runtime.service_for`` — mirroring the existing
# room-mcp-auth path) must decrypt with the *same* key, so both are pointed
# at this one instance instead of each minting their own random key.
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


async def _make_room(db, room_id, creator_id, *, vps_access_mode="readonly"):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=creator_id, email=f"vci-{creator_id}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="vci", creator_id=creator_id, vps_access_mode=vps_access_mode))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    await db.commit()


async def _installed_item(db, *, user_id):
    from app.models.market import UserInstalledItem

    item = UserInstalledItem(user_id=user_id, type="mcp", name="test-http-mcp")
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item.id


def _http_link_binding(item_id):
    from app.core.mcp_gateway import McpToolBinding

    return McpToolBinding(
        installed_item_id=item_id,
        market_item_id=None,
        market_slug=None,
        tool_name="deploy",
        custom_config=None,
        isolation=None,
    )


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


def _fake_gateway_post(response_text: str):
    captured: dict = {}

    async def fake_post(path, user_id, room_id, payload, **kwargs):
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": response_text}]}

    return fake_post, captured


@pytest.mark.asyncio
async def test_execute_mcp_tool_injects_conn_and_strips_model_supplied_one():
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", port=2222, user="legion",
            root_password="root-password-value",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await db.commit()

        binding = _vps_binding()
        fake_post, captured = _fake_gateway_post('{"ok": true}')

        with patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)):
            result = await execute_mcp_tool(
                binding, owner_id, room_id,
                {
                    "server": "prod",
                    "command": ["nginx", "-t"],
                    # a model trying to smuggle its own connection — must be
                    # discarded, never forwarded (plan §3.3 step 1).
                    "_legion_conn": {"host": "evil.example", "auth": {"kind": "password", "password": "x"}},
                },
                db=db, request_id=uuid4(),
            )

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        sent_args = captured["payload"]["arguments"]
        assert sent_args["server"] == "prod"
        conn = sent_args["_legion_conn"]
        assert conn["host"] == "203.0.113.10"
        assert conn["port"] == 2222
        assert conn["user"] == "legion"
        assert conn["auth"] == {"kind": "password", "password": "everyday-password-value"}
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_execute_mcp_tool_unknown_server_returns_typed_error():
    """A bare ``{"server": "missing"}`` call has no ``command``, so
    ``mcp_policy.classify_call`` can only mark it ``unknown`` -- under
    ``vps_access_mode="full"`` that's a ``require_approval`` verdict, not an
    outright ``allow`` (policy's ``decide()``), so a granted human approval
    is also faked here, mirroring ``test_mcp_policy.py``'s
    ``test_require_approval_allowed_by_human_proceeds``.
    """
    from app.core import mcp_gateway
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, vps_access_mode="full")
        binding = _vps_binding()
        fake_post, _ = _fake_gateway_post('{"ok": true}')

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_mcp_policy_approval", AsyncMock(return_value=True)),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, {"server": "missing"}, db=db, request_id=uuid4(),
            )
        assert result == "Error: [server_not_found] MISSING"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_execute_mcp_tool_vps_gated_by_protected_environment():
    """8b.1 — the protected-environment gate must actually run on the live
    ``execute_mcp_tool`` path, not only when a test calls
    ``vps_inventory.resolve_connection(..., environment=...)`` directly.
    Before this fix, `execute_mcp_tool` never passed `environment` through,
    so `_assert_environment_unlocked` silently no-op'd for every VPS call.
    """
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
            environment="production",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
            environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True,
        )
        await db.commit()

        binding = _vps_binding()
        fake_post, captured = _fake_gateway_post('{"ok": true}')
        arguments = {"server": "prod", "environment": "production", "command": ["nginx", "-t"]}

        # Deny is simulated directly (rather than letting the real ask-user
        # card time out) — the point under test is that the gate engages at
        # all on this path, not the interactive-confirm mechanics, which
        # `_bash.py`'s own tests already cover.
        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_mcp_environment_unlock", AsyncMock(return_value=False)),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, dict(arguments), db=db, request_id=uuid4(),
            )
        assert result == "Error: [environment_locked] production"
        assert "payload" not in captured

        # A real (non-interactive) unlock grant — same as a human clicking
        # "Allow" on the card — must let the identical call through, with
        # `environment` still stripped before the arguments reach upstream.
        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        await db.commit()

        with patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, dict(arguments), db=db, request_id=uuid4(),
            )
        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        sent_args = captured["payload"]["arguments"]
        assert "environment" not in sent_args
        assert sent_args["_legion_conn"]["host"] == "203.0.113.10"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_execute_mcp_tool_http_link_mcp_gated_by_protected_environment():
    """8b.1 — same gate, HTTP-link MCP side: `execute_mcp_tool` never passed
    `environment` into `resolve_room_mcp_auth_header` either, so a Room
    member's `room_mcp_auth_header` binding under a protected environment
    was resolved (and forwarded as the upstream bearer) with no unlock ever
    required.
    """
    from app.core import mcp_gateway
    from app.core.mcp_gateway import execute_mcp_tool
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        item_id = await _installed_item(db, user_id=owner_id)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_MCP_TOKEN", value="bearer-token-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_AUTH_HEADER, environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True,
        )
        await db.commit()

        binding = _http_link_binding(item_id)
        arguments = {"environment": "production", "path": "/deploy"}
        captured: dict = {}

        async def fake_post(path, user_id, room_id, payload, **kwargs):
            captured["payload"] = payload
            captured["headers"] = kwargs.get("extra_headers") or {}
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_mcp_environment_unlock", AsyncMock(return_value=False)),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, dict(arguments), db=db, request_id=uuid4(),
            )
        # A denied confirm surfaces the same typed `environment_locked` error
        # the VPS path does, and the call never reaches the upstream Gateway
        # at all — the Room-supplied bearer is additive, but not so additive
        # that a protected environment can be silently skipped.
        assert result == "Error: [environment_locked] production"
        assert "payload" not in captured

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        await db.commit()

        with patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, dict(arguments), db=db, request_id=uuid4(),
            )
        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        assert captured["headers"].get(mcp_gateway.MCP_UPSTREAM_AUTH_HEADER) == "Bearer bearer-token-value"
        sent_args = captured["payload"]["arguments"]
        assert "environment" not in sent_args
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_execute_mcp_tool_masks_credential_in_result():
    """Even if the upstream server echoed the credential back, the value
    never reaches the agent unmasked (room_masking, applied via the same
    ``room_mcp_secret`` dict the HTTP-auth-header path already masks with).

    ``{"server": "prod"}`` has no ``command``, so this is also an
    ``unknown``-risk, ``require_approval`` call under ``vps_access_mode=
    "full"`` -- the human approval is faked the same way as in
    ``test_execute_mcp_tool_unknown_server_returns_typed_error``.
    """
    from app.core import mcp_gateway, vps_inventory
    from app.core.mcp_gateway import execute_mcp_tool
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, vps_access_mode="full")
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await db.commit()

        binding = _vps_binding()
        fake_post, _ = _fake_gateway_post(
            json.dumps({"stdout": "connected with everyday-password-value"})
        )

        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(mcp_gateway, "_confirm_mcp_policy_approval", AsyncMock(return_value=True)),
        ):
            result = await execute_mcp_tool(
                binding, owner_id, room_id, {"server": "prod"}, db=db, request_id=uuid4(),
            )

        assert "everyday-password-value" not in result
        assert "MASKED" in result
        break

    await engine.dispose()