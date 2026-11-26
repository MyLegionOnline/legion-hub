"""Memory gate of heavy bash commands (core/memory_gate.py, room parallel tasks plan E3)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.core import memory_gate, run_clock


@pytest.mark.parametrize("cmd", [
    "npm run dev",
    "cd web && pnpm build",
    "yarn test",
    "bun run start",
    "npx next build",
    "next dev -p 3001",
    "vite --port 5173",
    "tsc --noEmit",
    "cargo build --release",
    "go test ./...",
    "pytest -q tests/",
    "python3 -m pytest",
    "python scripts/train.py",
    "node server.js",
    "docker build .",
    "make all",
    "FOO=1 && npm run build",
])
def test_heavy_commands_are_gated(cmd):
    assert memory_gate.needs_gate(cmd, background=False, install=False)


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "cat package.json",
    "git status",
    "ps aux",
    "free -m",
    "grep -rn next src/",
    "sed -n 1,20p app/page.tsx",
    "rm -rf .next/cache",
    "echo npm run dev",
])
def test_light_commands_are_not_gated(cmd):
    # "echo npm run dev" contains the words but does not run them at a command start.
    assert not memory_gate.needs_gate(cmd, background=False, install=False)


def test_background_and_install_are_always_gated():
    assert memory_gate.needs_gate("sleep 5", background=True, install=False)
    assert memory_gate.needs_gate("pip install x", background=False, install=True)


@pytest.mark.parametrize("cmd", [
    "kill 412 && npm run build",
    "pkill -f next-server; npm run dev",
    "killall node",
])
def test_killing_commands_are_never_gated(cmd):
    assert not memory_gate.needs_gate(cmd, background=True, install=False)


def test_is_busy_thresholds(monkeypatch):
    monkeypatch.setenv("BASH_MEMORY_GATE_PCT", "80")
    monkeypatch.setenv("BASH_MEMORY_GATE_PSI_FULL", "10")
    assert not memory_gate.is_busy(None)
    assert not memory_gate.is_busy({"pct": None, "psi_full_avg10": None})
    assert not memory_gate.is_busy({"pct": 79.9, "psi_full_avg10": 2.0})
    assert memory_gate.is_busy({"pct": 80.0})
    assert memory_gate.is_busy({"pct": 40.0, "psi_full_avg10": 12.5})


def _ctx(task_id=None, run_id=None):
    return SimpleNamespace(
        db=None, room_id=uuid.uuid4(), session_id=uuid.uuid4(),
        task_id=task_id, run_id=run_id,
    )


def _patch_common(monkeypatch, readings):
    seq = list(readings)

    async def _pressure(room_id):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    async def _sleep(_s):
        return None

    async def _release(ctx, *, reason="tool"):
        return False

    async def _touch(ctx):
        return None

    monkeypatch.setattr(memory_gate, "_pressure", _pressure)
    monkeypatch.setattr(memory_gate.asyncio, "sleep", _sleep)
    monkeypatch.setattr(memory_gate, "_touch_session", _touch)
    monkeypatch.setattr("app.core.tools.registry.release_tool_db", _release)


@pytest.mark.asyncio
async def test_free_memory_passes_without_waiting(monkeypatch):
    _patch_common(monkeypatch, [{"pct": 30.0}])
    res = await memory_gate.wait_for_memory(_ctx(), "r1")
    assert res.ok and res.waited == 0.0


@pytest.mark.asyncio
async def test_unknown_reading_passes(monkeypatch):
    _patch_common(monkeypatch, [None])
    res = await memory_gate.wait_for_memory(_ctx(), "r1")
    assert res.ok


@pytest.mark.asyncio
async def test_waits_until_memory_frees(monkeypatch):
    _patch_common(monkeypatch, [{"pct": 92.0}, {"pct": 91.0}, {"pct": 55.0}])
    res = await memory_gate.wait_for_memory(_ctx(), "r1")
    assert res.ok
    assert res.pressure == {"pct": 55.0}


@pytest.mark.asyncio
async def test_gives_up_after_bound_and_credits_the_deadline(monkeypatch):
    monkeypatch.setenv("BASH_MEMORY_GATE_WAIT_SECONDS", "6")
    _patch_common(monkeypatch, [{"pct": 95.0}])
    clock = {"t": 1000.0}

    def _mono():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(memory_gate.time, "monotonic", _mono)
    task_id = uuid.uuid4()
    run_clock.reset(task_id)
    credited = []
    monkeypatch.setattr(
        run_clock, "add_credit", lambda tid, s: credited.append((str(tid), s)) or s,
    )
    res = await memory_gate.wait_for_memory(_ctx(task_id=task_id), "r1")
    assert not res.ok and not res.stopped
    assert res.waited >= 6
    assert res.pressure == {"pct": 95.0}
    # The wait ran inside run_clock.accruing: it was credited to the task.
    assert credited and all(tid == str(task_id) for tid, _ in credited)


@pytest.mark.asyncio
async def test_stopped_run_ends_the_wait(monkeypatch):
    monkeypatch.setenv("BASH_MEMORY_GATE_WAIT_SECONDS", "600")
    _patch_common(monkeypatch, [{"pct": 95.0}])
    clock = {"t": 0.0}

    def _mono():
        clock["t"] += 4.0
        return clock["t"]

    monkeypatch.setattr(memory_gate.time, "monotonic", _mono)

    async def _not_current(task_id, run_id, db):
        return False

    monkeypatch.setattr("app.core.task_runtime.is_run_current", _not_current)
    res = await memory_gate.wait_for_memory(_ctx(task_id=uuid.uuid4(), run_id="r"), "r1")
    assert not res.ok and res.stopped


def test_held_by_room_compares_agent_share(monkeypatch):
    monkeypatch.delenv("BASH_MEMORY_GATE_AGENT_SHARE", raising=False)
    assert memory_gate.held_by_room({"agent_rss_mb": 200.0, "mem_max_mb": 1024})
    assert not memory_gate.held_by_room({"agent_rss_mb": 400.0, "mem_max_mb": 1024})
    # unknown agent figures (Docker stats fallback): behave as before
    assert not memory_gate.held_by_room({"agent_rss_mb": None, "mem_max_mb": 1024})
    assert not memory_gate.held_by_room(None)


@pytest.mark.asyncio
async def test_memory_held_by_the_room_service_is_not_waited_for(monkeypatch):
    """E4 run 2: a next dev service held 700 MB of 1 GB; agents looped on refusals."""
    reading = {"pct": 92.0, "mem_used_mb": 942.0, "mem_max_mb": 1024, "agent_rss_mb": 120.0}
    _patch_common(monkeypatch, [reading])
    res = await memory_gate.wait_for_memory(_ctx(), "r1")
    assert res.ok and res.held_by_room and res.waited == 0.0


@pytest.mark.asyncio
async def test_wait_ends_once_only_the_room_holds_memory(monkeypatch):
    _patch_common(monkeypatch, [
        {"pct": 95.0, "mem_max_mb": 1024, "agent_rss_mb": 600.0},
        {"pct": 90.0, "mem_max_mb": 1024, "agent_rss_mb": 100.0},
    ])
    res = await memory_gate.wait_for_memory(_ctx(), "r1")
    assert res.ok and res.held_by_room


def test_held_by_room_note_is_english_and_names_tmp():
    note = memory_gate.held_by_room_note(
        {"pct": 91.0, "mem_used_mb": 932.0, "mem_max_mb": 1024,
         "agent_rss_mb": 150.0, "tmp_mb": 200.0},
    )
    assert note.startswith("[workspace memory is 932 of 1024 MB (91%)")
    assert "agent processes hold 150 MB" in note and "/tmp holds 200 MB" in note
    assert "ran anyway" in note
    assert not any("\u0400" <= ch <= "\u04ff" for ch in note)


@pytest.mark.asyncio
async def test_room_memory_notice_is_rate_limited(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    room_id = uuid.uuid4()
    pressure = {"pct": 93.0, "mem_used_mb": 952.0, "mem_max_mb": 1024, "agent_rss_mb": 100.0}

    def _db(recent):
        db = MagicMock()
        res = MagicMock()
        res.first.return_value = recent
        db.execute = AsyncMock(return_value=res)
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        return db

    with patch("app.core.events.emit_event", new=AsyncMock()) as emit:
        db = _db(None)
        assert await memory_gate.notify_room_memory_low(db, room_id, pressure)
        emit.assert_awaited_once()
        kwargs = emit.await_args.kwargs
        assert kwargs["event_type"] == memory_gate.ROOM_MEMORY_LOW
        assert "Workspace tab" in kwargs["content"] and "952 of 1024 MB" in kwargs["content"]
        db.commit.assert_awaited_once()

        emit.reset_mock()
        assert not await memory_gate.notify_room_memory_low(_db((uuid.uuid4(),)), room_id, pressure)
        emit.assert_not_awaited()


def test_busy_message_names_processes_and_own_background():
    msg = memory_gate.busy_message(
        {
            "mem_used_mb": 1890.0, "mem_max_mb": 2048, "pct": 92.3,
            "top": [
                {"pid": 412, "rss_mb": 780.0, "cmd": "next-server (v15)"},
                {"pid": 49, "rss_mb": 12.0, "cmd": "sh -c cd /sys/fs/cgroup 2>/dev/null; echo"},
                {"pid": 7, "rss_mb": 1.4, "cmd": "sleep infinity"},
            ],
        },
        60.0,
        [("ab12cd34ef56", "npm run dev")],
    )
    assert msg.startswith("Error: workspace memory is busy: 1890 of 2048 MB (92%)")
    assert "waited 60s, the command was not started" in msg
    assert "next-server (v15) (pid 412, 780 MB)" in msg
    assert "ab12cd34ef56 (npm run dev)" in msg
    assert "/sys/fs/cgroup" not in msg and "sleep infinity" not in msg
    assert "bash_kill" in msg
    assert not any("\u0400" <= ch <= "\u04ff" for ch in msg)


@pytest.mark.asyncio
async def test_bash_tool_refuses_heavy_command_when_memory_stays_full(monkeypatch):
    from app.core.tools import _bash

    tool = _bash.BashTool() if hasattr(_bash, "BashTool") else None
    if tool is None:
        pytest.skip("bash tool class name changed")

    ws = SimpleNamespace(id=uuid.uuid4())

    async def _ensure(ctx, reason):
        return ws, None

    async def _env(self, ctx, secrets, environment):
        return {}, {}, None, None

    async def _wait(ctx, rid):
        return memory_gate.GateResult(False, pressure={"pct": 93.0, "top": []}, waited=60.0)

    async def _own(self, ctx, ws_, sid):
        return []

    executed = []

    async def _sync(self, *a, **k):
        executed.append(a)
        return "ran"

    monkeypatch.setattr(_bash, "ensure_workspace_running", _ensure)
    monkeypatch.setattr(_bash.BashTool, "_build_exec_env", _env)
    monkeypatch.setattr(_bash.BashTool, "_own_background", _own)
    monkeypatch.setattr(_bash.BashTool, "_run_sync", _sync)
    monkeypatch.setattr(memory_gate, "wait_for_memory", _wait)

    out = await tool.execute(_ctx(), {"cmd": "npm run build"})
    assert out.startswith("Error: workspace memory is busy (93%)")
    assert not executed

    # A light command is not held back by the same full room.
    out = await tool.execute(_ctx(), {"cmd": "ls"})
    assert out == "ran"
