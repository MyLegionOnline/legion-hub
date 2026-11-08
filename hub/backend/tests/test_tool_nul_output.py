"""A NUL byte in a command's output must not end the agent run (2026-11-07).

No database, no network. ``bash`` output with a NUL could not be stored in
PostgreSQL; the failed flush left the session's transaction aborted and the
run's next write crashed the whole task.
"""

import asyncio

from app.core import proxy_session
from app.core.workspaced_client import _without_nul_output

NUL = chr(0)


def test_exec_output_loses_nul_characters_and_keeps_the_text():
    result = _without_nul_output({"stdout": f"PNG{NUL}{NUL}IHDR", "stderr": f"warn{NUL}", "exit_code": 0})
    assert result == {"stdout": "PNGIHDR", "stderr": "warn", "exit_code": 0}


def test_exec_output_without_nul_or_without_text_is_untouched():
    assert _without_nul_output({"stdout": "ok", "exit_code": 0}) == {"stdout": "ok", "exit_code": 0}
    assert _without_nul_output({"stdout": None}) == {"stdout": None}
    assert _without_nul_output(["not", "a", "dict"]) == ["not", "a", "dict"]


class _Tx:
    def __init__(self, active: bool) -> None:
        self.is_active = active


class _Db:
    def __init__(self, tx) -> None:
        self._tx = tx
        self.rolled_back = False

    def get_transaction(self):
        return self._tx

    async def rollback(self) -> None:
        self.rolled_back = True


def test_an_aborted_transaction_is_rolled_back_so_the_run_can_continue():
    db = _Db(_Tx(active=False))
    asyncio.run(proxy_session._rollback_if_aborted(db))
    assert db.rolled_back


def test_a_healthy_or_absent_transaction_is_left_alone():
    for db in (_Db(_Tx(active=True)), _Db(None)):
        asyncio.run(proxy_session._rollback_if_aborted(db))
        assert not db.rolled_back
    asyncio.run(proxy_session._rollback_if_aborted(None))


def test_a_failing_tool_with_an_aborted_session_becomes_a_tool_result(monkeypatch):
    db = _Db(_Tx(active=False))

    class _Ctx:
        session_id = "s"

    ctx = _Ctx()
    ctx.db = db

    class _Tool:
        async def execute(self, _ctx, _input):
            raise RuntimeError("invalid byte sequence for encoding UTF8: 0x00")

    class _Registry:
        def get(self, _name):
            return _Tool()

    import app.core.tools as tools

    monkeypatch.setattr(tools, "REGISTRY", _Registry())
    text = asyncio.run(proxy_session._execute_registry_tool("bash", ctx, {}))
    assert text.startswith("Error:")
    assert db.rolled_back
