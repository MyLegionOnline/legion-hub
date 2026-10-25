"""Wake-on-exit, install timeouts and the language-neutral wake message."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import bg_wake
from app.core.tools import REGISTRY, ToolContext
from app.core.tools import _bash


# ---------------------------------------------------------------------------
# install-aware timeout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "pip install reportlab",
    "pip3 install -r requirements.txt",
    "python -m pip install pandas",
    "cd app && npm install",
    "apt-get update && apt-get install -y fonts-dejavu",
    "sudo apt install ffmpeg",
    "yarn",
    "uv pip install rich",
])
def test_install_commands_get_long_window(cmd):
    assert _bash._is_install_command(cmd)
    assert _bash._effective_timeout(cmd, None) == _bash._INSTALL_MAX_TIMEOUT_SEC
    assert _bash._effective_timeout(cmd, 200) == 200
    assert _bash._effective_timeout(cmd, 10**6) == _bash._INSTALL_MAX_TIMEOUT_SEC


@pytest.mark.parametrize("cmd", ["ls -la", "echo pip install", "python build.py", "git status"])
def test_other_commands_keep_the_120s_cap(cmd):
    assert not _bash._is_install_command(cmd)
    assert _bash._effective_timeout(cmd, None) == 30
    assert _bash._effective_timeout(cmd, 9999) == 120


# ---------------------------------------------------------------------------
# bash(background, wake_on_exit)
# ---------------------------------------------------------------------------

def _bg_ctx(task_id):
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    ws.network_enabled = True
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _result(q):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=ws)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=_result)
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=task_id, db=db), db


async def _start_bg(ctx, **extra):
    with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as ex:
        ex.return_value = {"exit_code": 0, "stdout": "11 11\n", "stderr": ""}
        return await REGISTRY.get("bash").execute(
            ctx, {"cmd": "make render", "background": True, **extra}
        )


@pytest.mark.asyncio
async def test_wake_on_exit_armed_in_direct_chat():
    ctx, db = _bg_ctx(task_id=None)
    out = await _start_bg(ctx, wake_on_exit=True, wake_note="then build the PDF")
    proc = next(c.args[0] for c in db.add.call_args_list if hasattr(c.args[0], "wake_on_exit"))
    assert proc.wake_on_exit is True and proc.wake_note == "then build the PDF"
    assert "wake_on_exit: armed" in out


@pytest.mark.asyncio
async def test_wake_on_exit_ignored_inside_a_task():
    ctx, db = _bg_ctx(task_id=uuid4())
    out = await _start_bg(ctx, wake_on_exit=True)
    proc = next(c.args[0] for c in db.add.call_args_list if hasattr(c.args[0], "wake_on_exit"))
    assert proc.wake_on_exit is False
    assert "bash_wait" in out


@pytest.mark.asyncio
async def test_background_exit_code_wrapper_in_launch_script():
    ctx, _ = _bg_ctx(task_id=None)
    with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as ex:
        ex.return_value = {"exit_code": 0, "stdout": "11 11\n", "stderr": ""}
        await REGISTRY.get("bash").execute(ctx, {"cmd": "x", "background": True})
    script = ex.call_args.args[1][2]
    assert '"$BGLOG.exit"' in script


# ---------------------------------------------------------------------------
# wake message
# ---------------------------------------------------------------------------

def test_wake_message_has_outcome_note_and_log_and_no_cyrillic():
    msg = bg_wake.render_wake_message(
        bg_id="abc", cmd="make", exit_code=2, reason=None,
        note="build the PDF", log_tail="boom",
    )
    assert "exit_code: 2" in msg and "build the PDF" in msg and "boom" in msg
    assert "user's own language" not in msg and "language the user writes in" in msg
    assert not any("\u0400" <= ch <= "\u04ff" for ch in msg)


def test_wake_message_without_exit_code_states_reason():
    msg = bg_wake.render_wake_message(
        bg_id="abc", cmd="make", exit_code=None,
        reason="the workspace stopped before the process finished", note=None, log_tail="",
    )
    assert "outcome: the workspace stopped" in msg and "exit_code" not in msg


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def _row(**kw):
    p = MagicMock()
    p.id = uuid4()
    p.bg_id = "abc"
    p.session_id = str(uuid4())
    p.cmd = "make"
    p.log_path = ".legion/bg/abc.log"
    p.wake_note = "n"
    p.status = "running"
    p.exit_code = None
    p.created_at = None
    for k, v in kw.items():
        setattr(p, k, v)
    ws = MagicMock(room_id=uuid4(), status="running")
    return p, ws


def _factory(rows, rowcount=1):
    db = AsyncMock()
    res = MagicMock()
    res.all = MagicMock(return_value=rows)
    claim = MagicMock(rowcount=rowcount)
    db.execute = AsyncMock(side_effect=[res, claim])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm), db


@pytest.mark.asyncio
async def test_sweep_wakes_when_exit_file_appears():
    factory, db = _factory([_row()])
    with (
        patch.object(bg_wake, "_read_exit_code", AsyncMock(return_value=0)),
        patch.object(bg_wake, "_log_tail", AsyncMock(return_value="ok")),
        patch.object(bg_wake, "_resume", AsyncMock()) as resume,
    ):
        n = await bg_wake.wake_finished_processes(factory)
        await __import__("asyncio").sleep(0)
    assert n == 1
    resume.assert_called_once()
    assert "exit_code: 0" in resume.call_args.args[-1]


@pytest.mark.asyncio
async def test_sweep_leaves_running_process_alone():
    factory, _ = _factory([_row()])
    with (
        patch.object(bg_wake, "_read_exit_code", AsyncMock(return_value=None)),
        patch.object(bg_wake, "_resume", AsyncMock()) as resume,
    ):
        assert await bg_wake.wake_finished_processes(factory) == 0
    resume.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_skips_while_the_agent_turn_is_still_running():
    factory, _ = _factory([_row()])
    with (
        patch.object(bg_wake, "_turn_in_progress", return_value=True),
        patch.object(bg_wake, "_read_exit_code", AsyncMock(return_value=0)),
        patch.object(bg_wake, "_resume", AsyncMock()) as resume,
    ):
        assert await bg_wake.wake_finished_processes(factory) == 0
    resume.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_does_not_resume_when_another_worker_claimed_it():
    factory, _ = _factory([_row()], rowcount=0)
    with (
        patch.object(bg_wake, "_read_exit_code", AsyncMock(return_value=0)),
        patch.object(bg_wake, "_resume", AsyncMock()) as resume,
    ):
        assert await bg_wake.wake_finished_processes(factory) == 0
    resume.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_wakes_on_stopped_workspace_without_exit_code():
    p, ws = _row()
    ws.status = "stopped"
    factory, _ = _factory([(p, ws)])
    with (
        patch.object(bg_wake, "_read_exit_code", AsyncMock(return_value=None)),
        patch.object(bg_wake, "_log_tail", AsyncMock(return_value="")),
        patch.object(bg_wake, "_resume", AsyncMock()) as resume,
    ):
        assert await bg_wake.wake_finished_processes(factory) == 1
    assert "workspace stopped" in resume.call_args.args[-1]
