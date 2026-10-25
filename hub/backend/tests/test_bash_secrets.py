"""bash tool: Room variables/secrets injection into exec env (Stage 4).

internal design notes Stage 4 — `secrets: []` and `environment`
params on the bash tool, Room variables always in env, deterministic
`secret_not_available` errors, masking of resolved secret values in the
returned stdout/stderr, and the same values surviving into a later
bash_output(bg_id=...) read for background commands.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.secrets import room_masking, room_secrets
from app.core.tools import REGISTRY, ToolContext


def _db_returning(*objs):
    """AsyncMock db whose .execute() returns each of ``objs`` in turn, then the last forever."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _result(obj):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=obj)
        r.scalars.return_value.all.return_value = []
        return r

    objs = list(objs)

    async def _execute(_q):
        obj = objs.pop(0) if len(objs) > 1 else objs[0]
        return _result(obj)

    db.execute = AsyncMock(side_effect=_execute)
    return db


def _make_ctx(*db_objs):
    return ToolContext(
        room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=_db_returning(*db_objs)
    )


# ── _build_exec_env ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_variables_always_in_env_without_secrets():
    tool = REGISTRY.get("bash")
    ctx = _make_ctx(None)

    with patch(
        "app.core.room_config.effective_variables",
        new_callable=AsyncMock,
        return_value={"APP_ENV": "staging"},
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(ctx, None, None)

    assert err is None
    assert env == {"APP_ENV": "staging"}
    assert secrets == {}
    assert secret_files == []


@pytest.mark.asyncio
async def test_requested_secret_merges_over_variables():
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={"APP_ENV": "staging"},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            return_value=({"DEPLOY_TOKEN": "supersecretvalue"}, {}),
        ) as mock_resolve,
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(ctx, ["DEPLOY_TOKEN"], None)

    assert err is None
    assert env == {"APP_ENV": "staging", "DEPLOY_TOKEN": "supersecretvalue"}
    assert secrets == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert secret_files == []
    # Resolved under the room's creator, per plan §2.3 — bash has no human actor.
    assert mock_resolve.await_args.kwargs["actor_user_id"] == room.creator_id
    assert mock_resolve.await_args.kwargs["names"] == ["DEPLOY_TOKEN"]


@pytest.mark.asyncio
async def test_missing_secret_is_a_deterministic_error():
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            side_effect=room_secrets.RoomSecretNotAvailable("DEPLOY_TOKEN"),
        ),
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(ctx, ["DEPLOY_TOKEN"], None)

    assert env == {} and secrets == {} and secret_files == []
    assert err == "Error: [secret_not_available] DEPLOY_TOKEN"


@pytest.mark.asyncio
async def test_deploy_ssh_key_becomes_secret_file_not_env():
    """internal design notes Stage 5: a deploy key never touches env."""
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            return_value=({}, {"SSH_DEPLOY_KEY": "-----BEGIN...-----"}),
        ) as mock_resolve,
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(
            ctx, ["SSH_DEPLOY_KEY"], None
        )

    assert err is None
    assert "SSH_DEPLOY_KEY" not in env
    assert secrets == {"SSH_DEPLOY_KEY": "-----BEGIN...-----"}
    assert secret_files == [
        {
            "path": "/run/legion/secrets/SSH_DEPLOY_KEY",
            "content": "-----BEGIN...-----",
            "mode": "0600",
        }
    ]
    assert env["GIT_SSH_COMMAND"] == (
        "ssh -i /run/legion/secrets/SSH_DEPLOY_KEY -o IdentitiesOnly=yes "
        "-o StrictHostKeyChecking=accept-new"
    )
    assert mock_resolve.await_args.kwargs["names"] == ["SSH_DEPLOY_KEY"]


@pytest.mark.asyncio
async def test_deploy_ssh_key_rejected_with_background():
    tool = REGISTRY.get("bash")
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(ws, room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            return_value=({}, {"SSH_DEPLOY_KEY": "-----BEGIN...-----"}),
        ),
    ):
        result = await tool.execute(
            ctx,
            {"cmd": "git push", "background": True, "secrets": ["SSH_DEPLOY_KEY"]},
        )

    assert result == "Error: deploy-key secrets cannot be used with background=true"


# ── Stage 12: protected-environment gate raises an ask-user confirmation ────


class _FakePendingQuestion:
    """Stand-in for ask_user_registry.PendingQuestion — event pre-set, canned answer."""

    def __init__(self, selected_options):
        import asyncio

        self.event = asyncio.Event()
        self.event.set()
        self.answer = {"selected_options": selected_options, "free_text": None}


@pytest.mark.asyncio
async def test_environment_locked_denied_keeps_it_locked():
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    # Two db.execute calls: one Room lookup inside _build_exec_env, one more
    # inside _confirm_environment_unlock (only reached on "Allow" — here it
    # isn't, but _db_returning's pop-until-last-object semantics tolerate it).
    ctx = _make_ctx(room, room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            side_effect=room_secrets.EnvironmentLocked("production"),
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.ask_user_registry.register",
            return_value=_FakePendingQuestion(["Deny"]),
        ) as mock_register,
        patch("app.core.ask_user_registry.expire") as mock_expire,
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(
            ctx, ["DEPLOY_TOKEN"], "production"
        )

    assert env == {} and secrets == {} and secret_files == []
    assert err == "Error: [environment_locked] production"
    mock_register.assert_called_once()
    mock_expire.assert_called_once()


@pytest.mark.asyncio
async def test_environment_locked_timeout_keeps_it_locked():
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(room, room)

    class _NeverAnswered:
        def __init__(self):
            import asyncio

            self.event = asyncio.Event()  # never set → wait_for times out
            self.answer = None

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            side_effect=room_secrets.EnvironmentLocked("production"),
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch("app.core.ask_user_registry.register", return_value=_NeverAnswered()),
        patch("app.core.ask_user_registry.expire"),
        patch.dict("os.environ", {"ENV_UNLOCK_CONFIRM_TIMEOUT_SECONDS": "0"}),
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(
            ctx, ["DEPLOY_TOKEN"], "production"
        )

    assert err == "Error: [environment_locked] production"


@pytest.mark.asyncio
async def test_environment_locked_approved_grants_unlock_and_retries():
    tool = REGISTRY.get("bash")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(room, room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            side_effect=[
                room_secrets.EnvironmentLocked("production"),
                ({"DEPLOY_TOKEN": "supersecretvalue"}, {}),
            ],
        ) as mock_resolve,
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.ask_user_registry.register",
            return_value=_FakePendingQuestion(["Allow"]),
        ),
        patch("app.core.ask_user_registry.expire"),
        patch(
            "app.core.secrets.room_environment_gate.grant_unlock",
            new_callable=AsyncMock,
        ) as mock_grant,
    ):
        env, secrets, secret_files, err = await tool._build_exec_env(
            ctx, ["DEPLOY_TOKEN"], "production"
        )

    assert err is None
    assert env == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert secrets == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert mock_resolve.await_count == 2
    mock_grant.assert_awaited_once()
    assert mock_grant.await_args.kwargs["room_id"] == ctx.room_id
    assert mock_grant.await_args.kwargs["environment"] == "production"
    assert mock_grant.await_args.kwargs["actor_user_id"] == room.creator_id


@pytest.mark.asyncio
async def test_secrets_param_rejects_non_list():
    tool = REGISTRY.get("bash")
    ctx = _make_ctx(None)
    result = await tool.execute(ctx, {"cmd": "echo hi", "secrets": "DEPLOY_TOKEN"})
    assert result == "Error: secrets must be a list of strings"


@pytest.mark.asyncio
async def test_secrets_param_caps_at_twenty():
    tool = REGISTRY.get("bash")
    ctx = _make_ctx(None)
    result = await tool.execute(ctx, {"cmd": "echo hi", "secrets": [f"S{i}" for i in range(21)]})
    assert result == "Error: at most 20 secrets per call"


# ── end-to-end: exec env + masking ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolved_secret_reaches_exec_env_and_is_masked_in_output():
    tool = REGISTRY.get("bash")
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    ws.network_enabled = True
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(ws, room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            return_value=({"DEPLOY_TOKEN": "supersecretvalue"}, {}),
        ),
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
    ):
        mock_exec.return_value = {
            "exit_code": 0,
            "stdout": "token is supersecretvalue",
            "stderr": "",
        }
        result = await tool.execute(
            ctx, {"cmd": "deploy.sh $DEPLOY_TOKEN", "secrets": ["DEPLOY_TOKEN"]}
        )

    assert mock_exec.await_args.kwargs["env"] == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert "supersecretvalue" not in result
    assert "***MASKED:DEPLOY_TOKEN***" in result


@pytest.mark.asyncio
async def test_background_secret_survives_into_bash_output_masking():
    bash_tool = REGISTRY.get("bash")
    bash_output_tool = REGISTRY.get("bash_output")
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    ws.network_enabled = True
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(ws, room)

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_secrets_mixed",
            new_callable=AsyncMock,
            return_value=({"DEPLOY_TOKEN": "supersecretvalue"}, {}),
        ),
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "123 123", "stderr": ""}
        launch_result = await bash_tool.execute(
            ctx, {"cmd": "deploy.sh", "background": True, "secrets": ["DEPLOY_TOKEN"]}
        )

    assert mock_exec.await_args.kwargs["env"]["DEPLOY_TOKEN"] == "supersecretvalue"
    bg_id = next(line for line in launch_result.splitlines() if line.startswith("bg_id:")).split()[1]

    assert room_masking.bg_secrets(bg_id) == {"DEPLOY_TOKEN": "supersecretvalue"}

    proc = MagicMock()
    proc.bg_id = bg_id
    proc.status = "running"
    proc.pid = 123
    proc.log_path = f".legion/bg/{bg_id}.log"
    proc.cmd = "deploy.sh"
    out_ctx = _make_ctx(proc)

    with patch(
        "app.core.workspaced_client.read_workspace_file",
        new_callable=AsyncMock,
        return_value="token is supersecretvalue",
    ):
        out = await bash_output_tool.execute(out_ctx, {"bg_id": bg_id})

    assert "supersecretvalue" not in out
    assert "***MASKED:DEPLOY_TOKEN***" in out

    room_masking.forget_bg_secrets(bg_id)
