"""internal design notes §3.9 / §11 (Stage 7) — auto-backup before a system write.

The operator approves a `write_system` call on the understanding that it can
be undone. These tests pin the two halves of that promise: a copy really is
taken, and a copy that cannot be taken cancels the write instead of letting
it through unbacked.

Everything below the Gateway boundary is faked: `_billed_mcp_call` is the
single seam through which both the read-back and the backup write leave the
Hub, so patching it gives an exact transcript of what the backup did.
`test_mcp_policy.py`'s DB-gated wiring tests cover the surrounding
`execute_mcp_tool` path; the end-to-end case here is gated the same way.
"""

import base64
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_gateway
from app.core.mcp_gateway import McpToolBinding, _backup_before_write, _BackupDeferred

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

NGINX_CONF = "server {\n  listen 80;\n}\n"
NGINX_B64 = base64.b64encode(NGINX_CONF.encode()).decode()


def _vps_binding(tool_name: str = "vps_write_file", *, market_item_id=None) -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=market_item_id,
        market_slug="legion-vps-mcp",
        tool_name=tool_name,
        custom_config=None,
        isolation=None,
    )


def _stdio_binding(tool_name: str) -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="mcp-ssh-manager",
        tool_name=tool_name,
        custom_config=None,
        isolation=None,
        managed_stdio=True,
    )


WRITE_ARGS = {
    "server": "prod",
    "path": "/etc/nginx/sites-enabled/site.conf",
    "content_base64": base64.b64encode(b"server {\n  listen 443;\n}\n").decode(),
    "sudo": True,
    "mode": "644",
    "_legion_conn": {"host": "10.0.0.1"},
}


async def _run_backup(binding, arguments, calls, *, db=None):
    """Invoke the backup with `_billed_mcp_call` replaced by `calls`."""

    with patch.object(mcp_gateway, "_billed_mcp_call", AsyncMock(side_effect=calls)) as spy:
        error = await _backup_before_write(
            binding, arguments,
            user_id=uuid4(), room_id=uuid4(), db=db,
            upstream_headers={}, custom_config=None, request_id=uuid4(),
        )
    return error, spy


# ── the happy path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_existing_file_is_copied_to_a_timestamped_bak():
    read_reply = {"server": "prod", "path": WRITE_ARGS["path"], "content_base64": NGINX_B64, "bytes": 21}
    error, spy = await _run_backup(_vps_binding(), WRITE_ARGS, [read_reply, {"bytes": 21}])

    assert error is None
    read_call, write_call = spy.await_args_list
    assert read_call.args[1] == "vps_read_file"
    assert write_call.args[1] == "vps_write_file"

    backup_args = write_call.args[2]
    assert backup_args["path"].startswith(WRITE_ARGS["path"] + ".bak_")
    # `<path>.bak_<YYYYMMDD_HHMMSS>` — 15 characters after the prefix.
    assert len(backup_args["path"].removeprefix(WRITE_ARGS["path"] + ".bak_")) == 15
    assert backup_args["content_base64"] == NGINX_B64
    assert backup_args["_legion_conn"] == WRITE_ARGS["_legion_conn"]


@pytest.mark.asyncio
async def test_read_back_does_not_forward_the_writes_own_extras():
    """`vps_read_file` declares `additionalProperties: false`: forwarding the
    write's `sudo`/`mode`/`owner` would make every backup an invalid_input."""

    read_reply = {"content_base64": NGINX_B64}
    _, spy = await _run_backup(_vps_binding(), WRITE_ARGS, [read_reply, {"bytes": 1}])

    read_args = spy.await_args_list[0].args[2]
    assert set(read_args) == {"server", "path", "max_bytes", "_legion_conn"}
    assert read_args["max_bytes"] == mcp_gateway._BACKUP_MAX_BYTES


@pytest.mark.asyncio
async def test_vps_upload_is_backed_up_through_vps_write_file():
    """`vps_upload` addresses its target as `remote_path` and requires a
    `workspace_path` it has no business re-deriving for a backup — §3.9 names
    `vps_read_file`/`vps_write_file` as the pair for exactly this reason."""

    upload_args = {
        "server": "prod",
        "workspace_path": "deploy/site.conf",
        "remote_path": "/etc/nginx/site.conf",
        "content_base64": NGINX_B64,
        "_legion_conn": {"host": "10.0.0.1"},
    }
    error, spy = await _run_backup(
        _vps_binding("vps_upload"), upload_args, [{"content_base64": NGINX_B64}, {"bytes": 1}],
    )

    assert error is None
    assert spy.await_args_list[1].args[1] == "vps_write_file"
    backup_args = spy.await_args_list[1].args[2]
    assert backup_args["path"].startswith("/etc/nginx/site.conf.bak_")
    assert "workspace_path" not in backup_args


# ── nothing to back up ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_file_is_not_a_backup_failure():
    """§3.9: fail-closed "except the case 'the file does not exist'" — creating a
    new file is not an overwrite and has nothing to preserve."""

    absent = 'Error: {"error": "path_not_found", "message": "The requested path does not exist"}'
    error, spy = await _run_backup(_vps_binding(), WRITE_ARGS, [absent])

    assert error is None
    assert spy.await_count == 1  # no pointless empty backup written


@pytest.mark.asyncio
async def test_empty_existing_file_has_nothing_to_preserve():
    """An existing but empty file loses nothing on overwrite — and
    `vps_write_file` rejects an empty body, so a 0-byte `.bak` would fail
    and block the write for no gain."""

    error, spy = await _run_backup(_vps_binding(), WRITE_ARGS, [{"content_base64": "", "bytes": 0}])
    assert error is None
    assert spy.await_count == 1


@pytest.mark.asyncio
async def test_plain_text_absence_from_a_third_party_mcp_is_also_recognized():
    error, _ = await _run_backup(
        _vps_binding(), WRITE_ARGS, ["Error: open /etc/x: no such file or directory"],
    )
    assert error is None


# ── fail-closed ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unreadable_file_cancels_the_write():
    denied = 'Error: {"error": "sudo_not_permitted", "message": "sudo was refused"}'
    error, spy = await _run_backup(_vps_binding(), WRITE_ARGS, [denied])

    assert error.startswith("Error: [backup_failed]")
    assert spy.await_count == 1


@pytest.mark.asyncio
async def test_unwritable_backup_cancels_the_write():
    full = 'Error: {"error": "disk_full", "message": "no space left on the device"}'
    error, _ = await _run_backup(
        _vps_binding(), WRITE_ARGS, [{"content_base64": NGINX_B64}, full],
    )
    assert error.startswith("Error: [backup_failed]")


@pytest.mark.asyncio
async def test_read_back_without_contents_cancels_the_write():
    error, _ = await _run_backup(_vps_binding(), WRITE_ARGS, [{"server": "prod", "bytes": 12}])
    assert error.startswith("Error: [backup_failed]")


@pytest.mark.asyncio
async def test_unidentifiable_target_cancels_the_write():
    error, spy = await _run_backup(_vps_binding(), {"server": "prod", "content": "x"}, [])
    assert error.startswith("Error: [backup_failed]")
    assert spy.await_count == 0


# ── host-key trust is not a backup failure ────────────────────────────────


@pytest.mark.asyncio
async def test_host_trust_reply_defers_instead_of_cancelling():
    """A first-contact server answers the backup read with `host_key_unknown`.
    That is the TOFU card's job (§3.5), not a reason to refuse the write:
    `execute_mcp_tool` lets the real call raise the card and re-runs the
    backup on the one retry an operator's "Allow" earns."""

    unknown = 'Error: {"error": "host_key_unknown", "fingerprint": "SHA256:abc"}'
    with pytest.raises(_BackupDeferred):
        await _run_backup(_vps_binding(), WRITE_ARGS, [unknown])


# ── an SSH MCP the user installed themselves ──────────────────────────────


class _FakeDb:
    """Just enough of AsyncSession for `_backup_read_tool`'s single `get`."""

    def __init__(self, tools_snapshot):
        self._runtime = type("R", (), {"tools_snapshot": tools_snapshot})()

    async def get(self, model, key):
        return self._runtime


@pytest.mark.asyncio
async def test_third_party_read_tool_is_taken_from_the_verified_snapshot():
    """§3.9 — the backup must not depend on a flag the package sets about
    itself, so the sibling read tool comes from the snapshot the Hub
    verified, matched against the write tool actually being called."""

    db = _FakeDb({"names": ["ssh_exec", "ssh_read_file", "ssh_write_file"]})
    args = {"host": "prod", "path": "/etc/hosts", "content": NGINX_CONF}
    error, spy = await _run_backup(
        _stdio_binding("ssh_write_file"), args,
        [{"content": NGINX_CONF}, {"ok": True}], db=db,
    )

    assert error is None
    assert spy.await_args_list[0].args[1] == "ssh_read_file"
    write_call = spy.await_args_list[1]
    assert write_call.args[1] == "ssh_write_file"
    assert write_call.args[2]["path"].startswith("/etc/hosts.bak_")
    assert write_call.args[2]["content"] == NGINX_CONF
    assert write_call.args[2]["host"] == "prod"


@pytest.mark.asyncio
async def test_third_party_mcp_with_no_read_tool_cancels_the_write():
    db = _FakeDb({"names": ["ssh_exec", "ssh_write_file"]})
    error, spy = await _run_backup(
        _stdio_binding("ssh_write_file"), {"path": "/etc/hosts", "content": "x"}, [], db=db,
    )
    assert error.startswith("Error: [backup_unsupported]")
    assert spy.await_count == 0


@pytest.mark.asyncio
async def test_base64_read_is_recoded_for_a_plain_text_write_tool():
    db = _FakeDb({"names": ["ssh_read_file", "ssh_put"]})
    args = {"path": "/etc/hosts", "content": "new"}
    error, spy = await _run_backup(
        _stdio_binding("ssh_put"), args, [{"content_base64": NGINX_B64}, {"ok": True}], db=db,
    )
    assert error is None
    assert spy.await_args_list[1].args[2]["content"] == NGINX_CONF


@pytest.mark.asyncio
async def test_binary_contents_cannot_be_squeezed_into_a_text_field():
    db = _FakeDb({"names": ["ssh_read_file", "ssh_put"]})
    binary = base64.b64encode(bytes(range(256))).decode()
    error, _ = await _run_backup(
        _stdio_binding("ssh_put"), {"path": "/usr/bin/app", "content": "x"},
        [{"content_base64": binary}, {"ok": True}], db=db,
    )
    assert error.startswith("Error: [backup_failed]")


# ── execute_mcp_tool wiring — real Postgres, upstream call faked ──────────
#
# Same gating and the same shared-keyring setup as
# tests/test_vps_connection_inject.py: the Room's server inventory has to
# resolve for real, because the backup runs *after* `_legion_conn` injection
# and carries the resolved connection into its own two calls.

pytestmark_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL
    pytest.importorskip("nacl")


def _keyring():
    from app.core.secret_keyring import SecretKeyring

    return SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


if TEST_DB_URL:
    _SHARED_KEYRING = _keyring()

    from app.core.secrets import runtime as _secrets_runtime

    _secrets_runtime.configure_runtime_keyring(_SHARED_KEYRING)


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=_SHARED_KEYRING))


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _room_with_server(db, room_id, creator_id):
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.models.room import Room, RoomUserLink
    from app.models.secret import SecretKind, SecretPurpose
    from app.models.user import User

    db.add(User(id=creator_id, email=f"bk-{creator_id}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="bk", creator_id=creator_id, vps_access_mode="full"))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    await db.commit()

    svc = _service(db)
    await vps_inventory.add_server(
        db, service=svc, room_id=room_id, actor_user_id=creator_id,
        name="prod", host="203.0.113.10", port=22, user="legion",
        root_password="root-password-value",
    )
    await room_secrets.bind_room_secret(
        db, service=svc, room_id=room_id, actor_user_id=creator_id,
        env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
        purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
    )
    await db.commit()


def _transcript_gateway(replies: dict[str, str]):
    """Fake `_gateway_post` that answers per tool name and records the order."""

    seen: list[tuple[str, dict]] = []

    async def fake_post(path, user_id, room_id, payload, **kwargs):
        seen.append((payload["tool"], payload["arguments"]))
        text = replies.get(payload["tool"], '{"ok": true}')
        if text.startswith("Error: "):
            return {"isError": True, "content": [{"type": "text", "text": text[len("Error: "):]}]}
        return {"content": [{"type": "text", "text": text}]}

    return fake_post, seen


@pytest.mark.asyncio
@pytestmark_db
async def test_system_write_leaves_a_bak_next_to_the_original():
    """The acceptance criterion, end to end: overwriting an existing
    `site.conf` under /etc leaves `site.conf.bak_<ts>` behind."""

    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _room_with_server(db, room_id, owner_id)
        fake_post, seen = _transcript_gateway({
            "vps_read_file": '{"content_base64": "%s"}' % NGINX_B64,
        })
        with (
            patch.object(mcp_gateway, "_confirm_mcp_policy_approval", AsyncMock(return_value=True)),
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
        ):
            result = await execute_mcp_tool(
                _vps_binding(), owner_id, room_id,
                {k: v for k, v in WRITE_ARGS.items() if k != "_legion_conn"},
                db=db, request_id=uuid4(),
            )

        assert not result.startswith("Error:")
        assert [tool for tool, _ in seen] == ["vps_read_file", "vps_write_file", "vps_write_file"]
        backup_args, real_args = seen[1][1], seen[2][1]
        assert backup_args["path"].startswith(WRITE_ARGS["path"] + ".bak_")
        assert backup_args["content_base64"] == NGINX_B64
        # The real write still lands on the original path, unchanged.
        assert real_args["path"] == WRITE_ARGS["path"]
        assert real_args["content_base64"] == WRITE_ARGS["content_base64"]
        break

    await engine.dispose()


@pytest.mark.asyncio
@pytestmark_db
async def test_failed_backup_stops_the_write_from_reaching_the_server():
    """Fail-closed, end to end: the card was answered "Allow", the backup
    cannot be made, and the overwrite never happens."""

    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _room_with_server(db, room_id, owner_id)
        fake_post, seen = _transcript_gateway({
            "vps_read_file": 'Error: {"error": "sudo_not_permitted", "message": "sudo was refused"}',
        })
        with (
            patch.object(mcp_gateway, "_confirm_mcp_policy_approval", AsyncMock(return_value=True)),
            patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)),
        ):
            result = await execute_mcp_tool(
                _vps_binding(), owner_id, room_id,
                {k: v for k, v in WRITE_ARGS.items() if k != "_legion_conn"},
                db=db, request_id=uuid4(),
            )

        assert result.startswith("Error: [backup_failed]")
        assert [tool for tool, _ in seen] == ["vps_read_file"]
        break

    await engine.dispose()


@pytest.mark.asyncio
@pytestmark_db
async def test_write_into_the_agents_own_directory_is_not_backed_up():
    """Only `write_system` buys a backup (§3.7): a `write_safe` path is
    allowed outright and must not cost two extra calls."""

    from app.core.mcp_gateway import execute_mcp_tool
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _room_with_server(db, room_id, owner_id)
        fake_post, seen = _transcript_gateway({})
        with patch.object(mcp_gateway, "_gateway_post", AsyncMock(side_effect=fake_post)):
            result = await execute_mcp_tool(
                _vps_binding(), owner_id, room_id,
                {k: v for k, v in WRITE_ARGS.items() if k != "_legion_conn"}
                | {"path": "/srv/legion/app/site.conf"},
                db=db, request_id=uuid4(),
            )

        assert not result.startswith("Error:")
        assert [tool for tool, _ in seen] == ["vps_write_file"]
        break

    await engine.dispose()
