"""internal design notes Stages 14/16 — what the Hub fills in for the VPS tools.

Three defects a live deploy run exposed, one test each:

* ``vps_list_servers`` was unusable — the "every VPS tool needs a 'server'"
  guard answered the one tool an agent calls *because* it does not know the
  names, and nothing injected the ``_legion_servers`` its schema requires;
* ``vps_upload`` promised in its own schema that the Hub fills
  ``content_base64`` from ``workspace_path`` and never did, so the model
  filled it by hand (a literal ``"inject"`` reached a server);
* nothing told an agent that the Room has a repository, or that the server
  holds a read-only credential for it — so the agent invented a transport.

Gated on INIT_DB_TEST_URL and PyNaCl, same harness as
``tests/test_vps_connection_inject.py``; the upstream Gateway call is faked,
only the Room/secret resolve path runs for real.
"""

import base64
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


if TEST_DB_URL:
    from app.core.secrets import runtime as _secrets_runtime

    _secrets_runtime.configure_runtime_keyring(_SHARED_KEYRING)


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_room(db, room_id, creator_id, *, vps_access_mode="readonly", git_config=None):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=creator_id, email=f"vru-{creator_id}@example.com", hashed_password="x"))
    db.add(Room(
        id=room_id, name="vru", creator_id=creator_id,
        vps_access_mode=vps_access_mode, git_config=git_config or {},
    ))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    await db.commit()


async def _server(db, room_id, owner_id, *, with_everyday_secret=True):
    """One inventory entry plus the credential a call would resolve."""
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.models.secret import SecretKind, SecretPurpose

    svc = _service(db)
    await vps_inventory.add_server(
        db, service=svc, room_id=room_id, actor_user_id=owner_id,
        name="prod", host="203.0.113.10", root_password="root-password-value",
    )
    if with_everyday_secret:
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
    await db.commit()


def _binding(tool_name: str):
    from app.core.mcp_gateway import McpToolBinding

    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="legion-vps-mcp",
        tool_name=tool_name,
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
async def test_list_servers_needs_no_server_and_carries_the_repository():
    from app.core import mcp_gateway
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _server(db, room_id, owner_id)

        fake_post, captured = _fake_gateway_post('{"servers": []}')
        with patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)):
            result = await execute_mcp_tool(
                _binding("vps_list_servers"), owner_id, room_id, {},
                db=db, request_id=uuid4(),
            )

        assert "server_not_found" not in result
        servers = captured["payload"]["arguments"]["_legion_servers"]
        assert [s["name"] for s in servers] == ["PROD"]
        assert servers[0]["host"] == "203.0.113.10"
        # The two facts a deploy needs, which no tool used to state.
        assert servers[0]["project_dir"].startswith("/srv/legion/")
        assert servers[0]["repo"] is None
        assert "no external repository" in servers[0]["repo_note"]
        # Inventory metadata only: no credential of any kind travels with it.
        assert "password" not in json.dumps(servers).lower()
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_upload_reads_the_workspace_file_itself():
    from app.core import mcp_gateway, workspace_fs
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, vps_access_mode="full")
        await _server(db, room_id, owner_id)

        fake_post, captured = _fake_gateway_post('{"bytes": 10}')
        with (
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
            patch.object(workspace_fs, "fs_read_file", AsyncMock(return_value="server {}\n")),
        ):
            await execute_mcp_tool(
                _binding("vps_upload"), owner_id, room_id,
                {
                    "server": "prod",
                    "workspace_path": "deploy/site.conf",
                    "remote_path": "/srv/legion/site.conf",
                    # A model-supplied value is overwritten, never trusted:
                    # this is the exact string a live run wrote to a server.
                    "content_base64": "inject",
                },
                db=db, request_id=uuid4(),
            )

        sent = captured["payload"]["arguments"]
        assert base64.b64decode(sent["content_base64"]).decode() == "server {}\n"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_upload_of_a_bundle_says_use_git_and_never_reaches_the_gateway():
    from app.core import mcp_gateway, workspace_fs
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, vps_access_mode="full")
        await _server(db, room_id, owner_id)

        gateway = AsyncMock()
        with (
            patch.object(mcp_gateway, "_gateway_post", gateway),
            patch.object(workspace_fs, "fs_read_file", AsyncMock(return_value="x" * (3 * 1024 * 1024))),
        ):
            result = await execute_mcp_tool(
                _binding("vps_upload"), owner_id, room_id,
                {
                    "server": "prod",
                    "workspace_path": "dist/bundle.js",
                    "remote_path": "/srv/legion/bundle.js",
                },
                db=db, request_id=uuid4(),
            )

        assert "[use_git_transport]" in result
        # Not just a refusal: the next action, and an explicit "not chunks".
        assert "chunks" in result
        gateway.assert_not_awaited()
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_upload_of_a_missing_path_is_free_and_says_which_path():
    from app.core import mcp_gateway, workspace_fs
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, vps_access_mode="full")
        await _server(db, room_id, owner_id)

        gateway = AsyncMock()
        with (
            patch.object(mcp_gateway, "_gateway_post", gateway),
            patch.object(workspace_fs, "fs_read_file", AsyncMock(return_value=None)),
        ):
            result = await execute_mcp_tool(
                _binding("vps_upload"), owner_id, room_id,
                {
                    "server": "prod",
                    "workspace_path": "deploy/nope.conf",
                    "remote_path": "/srv/legion/nope.conf",
                },
                db=db, request_id=uuid4(),
            )

        assert "[path_not_found]" in result and "deploy/nope.conf" in result
        gateway.assert_not_awaited()
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_repo_status_separates_no_repository_from_no_credential():
    from app.core import mcp_gateway
    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session
    from app.models.room import Room

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        await _server(db, room_id, owner_id)

        gateway = AsyncMock()
        with patch.object(mcp_gateway, "_gateway_post", gateway):
            no_repo = await execute_mcp_tool(
                _binding("vps_repo_status"), owner_id, room_id,
                {"server": "prod"}, db=db, request_id=uuid4(),
            )
        assert "[repo_not_configured]" in no_repo
        assert "Git" in no_repo

        room = await db.get(Room, room_id)
        room.git_config = {"repo_url": "https://github.com/acme/site", "branch": "main"}
        await db.commit()

        with patch.object(mcp_gateway, "_gateway_post", gateway):
            no_cred = await execute_mcp_tool(
                _binding("vps_repo_status"), owner_id, room_id,
                {"server": "prod"}, db=db, request_id=uuid4(),
            )
        assert "[repo_credential_missing]" in no_cred
        # Neither case is worth a billed call: both are fixed by a person.
        gateway.assert_not_awaited()
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_room_config_tool_names_the_repository():
    """The blind spot outside VPS work too: the workspace checkout has no git
    remote (syncing carries the URL in the call body), so ``git remote -v``
    there is empty and an agent concludes the Room has no repository.
    """
    from app.core.tools.registry import REGISTRY, ToolContext
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(
            db, room_id, owner_id,
            git_config={"repo_url": "https://github.com/acme/site", "branch": "main"},
        )
        tool = REGISTRY.get("room_config")
        ctx = ToolContext(room_id=room_id, session_id=uuid4(), task_id=None, db=db)
        out = await tool.execute(ctx, {})
        assert "github.com/acme/site" in out
        assert "branch=main" in out
        break

    await engine.dispose()
