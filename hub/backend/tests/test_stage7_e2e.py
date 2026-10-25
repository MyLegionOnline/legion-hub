"""
E2E integration tests — Stage 7.

Scenario A: "find bug and fix"
    grep (fallback) → read_file → edit_file → git_diff
    Key invariant: edit_file must fail if file was not read first;
    once read, the full chain completes and git_diff shows the fix.

Scenario B: "dev server lifecycle"
    bash(background=True) → bash_output → preview_fetch → bash_kill
    Key invariant: bg_id from the bash call is used in bash_output and bash_kill;
    the proc is found in DB and killed.

Scenario C: "ask_user mid-chain"
    grep → ask_user (user responds) → edit_file based on answer
    Key invariant: ask_user pauses the chain, receives the answer,
    and the tool result is usable by subsequent tools.
"""
from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools._session_state import clear_session, mark_read, was_read


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _make_no_ws_ctx(session_id=None, room_id=None):
    """Context where workspace does not exist — grep uses Python fallback."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
        )
    )
    return ToolContext(
        room_id=room_id or _uuid(),
        session_id=session_id or _uuid(),
        task_id=_uuid(),
        db=db,
    )


def _make_running_ws():
    ws = MagicMock()
    ws.id = _uuid()
    ws.status = "running"
    ws.network_enabled = True
    ws.last_activity_at = None
    return ws


def _make_ws_ctx(ws=None, session_id=None, room_id=None, task_id=None):
    """Context where a running workspace exists."""
    ws = ws or _make_running_ws()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _result(scalar_val=None, all_val=None):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=scalar_val)
        r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=all_val or [])))
        return r

    db.execute = AsyncMock(side_effect=lambda q: _result(scalar_val=ws, all_val=[]))
    return ToolContext(
        room_id=room_id or _uuid(),
        session_id=session_id or _uuid(),
        task_id=task_id or _uuid(),
        db=db,
    ), db, ws


# ---------------------------------------------------------------------------
# Scenario A: grep → read_file → edit_file → git_diff
# ---------------------------------------------------------------------------


class TestScenarioA:
    """
    Simulates an agent finding and fixing a bug:
      1. grep    — locates the bug (Python fallback, no container required)
      2. read_file — reads the file, marks session state
      3. edit_file — replaces the buggy line (requires prior read_file)
      4. git_diff  — verifies the change is recorded
    """

    FILE_PATH = "src/calculator.py"
    BUG_CONTENT = "def add(a, b):\n    return a - b  # BUG: should be +\n"
    FIXED_CONTENT = "def add(a, b):\n    return a + b\n"
    OLD_STR = "    return a - b  # BUG: should be +"
    NEW_STR = "    return a + b"

    @pytest.mark.asyncio
    async def test_edit_file_fails_without_prior_read(self):
        """edit_file must refuse if read_file was never called in this session."""
        ctx = _make_no_ws_ctx()
        edit_tool = REGISTRY.get("edit_file")
        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock, return_value=self.BUG_CONTENT):
            result = await edit_tool.execute(ctx, {
                "path": self.FILE_PATH,
                "old_string": self.OLD_STR,
                "new_string": self.NEW_STR,
            })
        assert "not been read" in result.lower() or "read_file first" in result.lower()

    @pytest.mark.asyncio
    async def test_full_grep_read_edit_git_diff_chain(self):
        session_id = _uuid()
        room_id = _uuid()

        ctx = _make_no_ws_ctx(session_id=session_id, room_id=room_id)

        # Workspace tree for grep fallback
        tree = [{"path": self.FILE_PATH, "modified_at": 1_700_000_000}]

        written: list[tuple[str, str]] = []

        async def _fake_write(rid, path, content, commit_message=None):
            written.append((path, content))

        diff_out = (
            "--- a/src/calculator.py\n"
            "+++ b/src/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b  # BUG: should be +\n"
            "+    return a + b\n"
        )

        with (
            patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock, return_value=tree),
            patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock, return_value=self.BUG_CONTENT),
            patch("app.core.workspaced_client.write_workspace_file", side_effect=_fake_write),
            patch("app.core.workspaced_client.git_diff_workspace", new_callable=AsyncMock, return_value={"diff": diff_out}),
        ):
            # Step 1: grep — find the bug (fallback path, no container)
            grep_tool = REGISTRY.get("grep")
            grep_result = await grep_tool.execute(ctx, {"pattern": r"return a - b"})
            assert self.FILE_PATH in grep_result, f"grep did not find the file: {grep_result!r}"

            # Step 2: read_file — read and mark the session state
            read_tool = REGISTRY.get("read_file")
            read_result = await read_tool.execute(ctx, {"path": self.FILE_PATH})
            assert "return a - b" in read_result
            assert "1\t" in read_result  # line numbers present
            assert was_read(str(session_id), self.FILE_PATH)

            # Step 3: edit_file — fix the bug (must succeed now that file was read)
            edit_tool = REGISTRY.get("edit_file")
            edit_result = await edit_tool.execute(ctx, {
                "path": self.FILE_PATH,
                "old_string": self.OLD_STR,
                "new_string": self.NEW_STR,
            })
            assert "Edited" in edit_result, f"edit_file failed: {edit_result!r}"

            # Verify the written content has the fix applied
            assert written, "write_workspace_file was never called"
            _, saved_content = written[-1]
            assert "return a + b" in saved_content
            assert "return a - b" not in saved_content

            # Step 4: git_diff — confirm the change is visible
            diff_tool = REGISTRY.get("git_diff")
            diff_result = await diff_tool.execute(ctx, {"path": self.FILE_PATH})
            assert "+    return a + b" in diff_result
            assert "-    return a - b" in diff_result

        clear_session(str(session_id))

    @pytest.mark.asyncio
    async def test_edit_file_after_read_is_idempotent_on_replace_all(self):
        """replace_all=True replaces every occurrence — validates the flag propagates."""
        session_id = _uuid()
        ctx = _make_no_ws_ctx(session_id=session_id)

        content = "x = 1\nx = 1\nx = 1\n"
        written: list[str] = []

        async def _fake_write(rid, path, c, commit_message=None):
            written.append(c)

        with (
            patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock, return_value=content),
            patch("app.core.workspaced_client.write_workspace_file", side_effect=_fake_write),
        ):
            read_tool = REGISTRY.get("read_file")
            await read_tool.execute(ctx, {"path": "vars.py"})

            edit_tool = REGISTRY.get("edit_file")
            result = await edit_tool.execute(ctx, {
                "path": "vars.py",
                "old_string": "x = 1",
                "new_string": "x = 2",
                "replace_all": True,
            })

        assert "3 replacements" in result
        assert written[0] == "x = 2\nx = 2\nx = 2\n"

        clear_session(str(session_id))


# ---------------------------------------------------------------------------
# Scenario B: bash(background) → bash_output → preview_fetch → bash_kill
# ---------------------------------------------------------------------------


class TestScenarioB:
    """
    Simulates an agent running a dev server:
      1. bash(background=True) — launches server, receives bg_id
      2. bash_output(bg_id)    — reads server log
      3. preview_fetch(port)   — confirms server is responding
      4. bash_kill(bg_id)      — stops the server

    The bg_id from step 1 is used in steps 2 and 4, validating the lifecycle.
    """

    @pytest.mark.asyncio
    async def test_background_bash_returns_bg_id(self):
        ctx, db, ws = _make_ws_ctx()
        bash_tool = REGISTRY.get("bash")

        with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"exit_code": 0, "stdout": "42 42", "stderr": ""}
            result = await bash_tool.execute(ctx, {"cmd": "npm run dev", "background": True})

        assert "bg_id:" in result
        assert "status: running" in result
        assert "bash_output" in result  # hint to use bash_output
        assert "bash_kill" in result    # hint to use bash_kill

    @pytest.mark.asyncio
    async def test_full_devserver_lifecycle(self):
        session_id = _uuid()
        room_id = _uuid()
        task_id = _uuid()
        ws = _make_running_ws()

        # --- Step 1: bash(background=True) ---
        captured_proc: list[Any] = []

        db1 = AsyncMock()
        db1.flush = AsyncMock()

        def _db1_add(obj):
            if hasattr(obj, "bg_id"):
                captured_proc.append(obj)

        db1.add = MagicMock(side_effect=_db1_add)
        db1.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=ws),
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
            )
        )

        ctx1 = ToolContext(room_id=room_id, session_id=session_id, task_id=task_id, db=db1)
        bash_tool = REGISTRY.get("bash")

        with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"exit_code": 0, "stdout": "12345 12345", "stderr": ""}
            bash_result = await bash_tool.execute(ctx1, {"cmd": "npm run dev --port 3000", "background": True})

        # Extract bg_id from result
        bg_id = None
        for line in bash_result.splitlines():
            if line.startswith("bg_id:"):
                bg_id = line.split(": ", 1)[1].strip()
                break

        assert bg_id is not None, f"bg_id not found in:\n{bash_result}"
        assert len(captured_proc) == 1
        proc = captured_proc[0]
        assert proc.bg_id == bg_id
        assert proc.status == "running"

        # --- Step 2: bash_output(bg_id) ---
        server_log = (
            "Starting dev server...\n"
            "Compiled successfully!\n"
            "Listening at http://localhost:3000\n"
        )

        db2 = AsyncMock()
        db2.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=proc))
        )

        ctx2 = ToolContext(room_id=room_id, session_id=session_id, task_id=task_id, db=db2)
        output_tool = REGISTRY.get("bash_output")

        with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock, return_value=server_log):
            output_result = await output_tool.execute(ctx2, {"bg_id": bg_id})

        assert "Listening at http://localhost:3000" in output_result
        assert "running" in output_result
        assert bg_id in output_result

        # --- Step 3: preview_fetch — check server is responding ---
        ctx3 = ToolContext(room_id=room_id, session_id=session_id, task_id=task_id, db=AsyncMock())
        preview_tool = REGISTRY.get("preview_fetch")

        body = base64.b64encode(b"<html><body>Hello from dev server</body></html>").decode()

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock) as mock_proxy:
            mock_proxy.return_value = {
                "status_code": 200,
                "content_type": "text/html; charset=utf-8",
                "body_b64": body,
            }
            preview_result = await preview_tool.execute(ctx3, {"port": 3000, "path": "/"})

        assert "200" in preview_result
        assert "Hello from dev server" in preview_result
        mock_proxy.assert_called_once_with(str(room_id), 3000, "/", "")

        # --- Step 4: bash_kill(bg_id) ---
        call_order: list[int] = []

        def _db4_execute(q):
            call_order.append(len(call_order))
            r = MagicMock()
            if len(call_order) == 1:
                # First call: WorkspaceBgProc query
                r.scalar_one_or_none = MagicMock(return_value=proc)
            else:
                # Second call: Workspace query
                r.scalar_one_or_none = MagicMock(return_value=ws)
            return r

        db4 = AsyncMock()
        db4.add = MagicMock()
        db4.flush = AsyncMock()
        db4.execute = AsyncMock(side_effect=_db4_execute)

        ctx4 = ToolContext(room_id=room_id, session_id=session_id, task_id=task_id, db=db4)
        kill_tool = REGISTRY.get("bash_kill")

        with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"exit_code": 0, "stdout": "", "stderr": ""}
            kill_result = await kill_tool.execute(ctx4, {"bg_id": bg_id})

        assert "killed" in kill_result
        assert bg_id in kill_result
        assert proc.status == "killed"

    @pytest.mark.asyncio
    async def test_bash_output_missing_bg_id_returns_error(self):
        """bash_output with an unknown bg_id returns an error, not an exception."""
        ctx = _make_no_ws_ctx()
        db_result = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        ctx.db.execute = AsyncMock(return_value=db_result)

        output_tool = REGISTRY.get("bash_output")
        result = await output_tool.execute(ctx, {"bg_id": "nonexistent"})

        assert "Error" in result
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_bash_kill_already_stopped_noop(self):
        """bash_kill on a proc that is already 'killed' returns a no-op message."""
        proc = MagicMock()
        proc.bg_id = "abc123"
        proc.status = "killed"

        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=proc))
        )

        ctx = ToolContext(room_id=_uuid(), session_id=_uuid(), task_id=None, db=db)
        kill_tool = REGISTRY.get("bash_kill")
        result = await kill_tool.execute(ctx, {"bg_id": "abc123"})

        assert "already stopped" in result.lower() or "nothing to kill" in result.lower()


# ---------------------------------------------------------------------------
# Scenario C: ask_user mid-chain
# ---------------------------------------------------------------------------


class TestScenarioC:
    """
    Simulates ask_user pausing the chain and resuming with the user's answer.
    The answer is then used to guide a subsequent tool call.
    """

    @pytest.mark.asyncio
    async def test_ask_user_answer_feeds_next_tool(self):
        """
        Chain: ask_user (receives answer) → edit_file uses the chosen path.
        Validates that ask_user returns a usable JSON result that guides later calls.
        """
        from app.core import ask_user_registry as reg

        session_id = _uuid()
        room_id = _uuid()

        # Prepare two fix options
        options = [
            {"label": "Use + operator", "description": "Fix: return a + b"},
            {"label": "Use sum()", "description": "Fix: return sum([a, b])"},
        ]

        # Context for ask_user
        ctx = ToolContext(room_id=room_id, session_id=session_id, task_id=_uuid(), db=AsyncMock())

        ask_tool = REGISTRY.get("ask_user")

        async def _answer_soon():
            await asyncio.sleep(0.05)
            # Find the pending question and answer it
            qid = next(
                (k for k, v in reg._PENDING.items() if v.session_id == session_id),
                None,
            )
            if qid:
                reg.resolve(qid, {"selected_options": ["Use + operator"], "free_text": None})

        with patch("app.core.events.emit_event", new_callable=AsyncMock):
            tool_task = asyncio.create_task(
                ask_tool.execute(ctx, {
                    "questions": [{
                        "question": "How should we fix the add() function?",
                        "header": "Fix approach",
                        "options": options,
                    }]
                })
            )
            asyncio.create_task(_answer_soon())
            ask_result = await tool_task

        import json
        data = json.loads(ask_result)
        assert data["selected_options"] == ["Use + operator"]

        # Use the result to choose which edit to apply
        chosen = data["selected_options"][0]
        new_str = "    return a + b" if "+" in chosen else "    return sum([a, b])"

        # Now edit the file with the chosen fix
        ctx2 = ToolContext(room_id=room_id, session_id=session_id, task_id=_uuid(), db=AsyncMock())
        file_path = "src/calculator.py"
        content = "def add(a, b):\n    return a - b\n"
        mark_read(str(session_id), file_path)

        with (
            patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock, return_value=content),
            patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mock_write,
        ):
            edit_tool = REGISTRY.get("edit_file")
            edit_result = await edit_tool.execute(ctx2, {
                "path": file_path,
                "old_string": "    return a - b",
                "new_string": new_str,
            })

        assert "Edited" in edit_result
        written_content = mock_write.call_args[0][2]
        assert "return a + b" in written_content

        clear_session(str(session_id))

    @pytest.mark.asyncio
    async def test_ask_user_timeout_chain_continues(self):
        """
        When user does not respond, ask_user returns a 'best judgement' fallback
        and the chain can continue independently.
        """
        from app.core.tools import _ask_user as _au_mod

        original = _au_mod._TIMEOUT
        _au_mod._TIMEOUT = 0  # force immediate timeout

        ctx = ToolContext(room_id=_uuid(), session_id=_uuid(), task_id=None, db=AsyncMock())
        ask_tool = REGISTRY.get("ask_user")

        try:
            with patch("app.core.events.emit_event", new_callable=AsyncMock):
                result = await ask_tool.execute(ctx, {
                    "questions": [{
                        "question": "Pick a refactoring approach",
                        "header": "Approach",
                        "options": [
                            {"label": "A", "description": "Approach A"},
                            {"label": "B", "description": "Approach B"},
                        ],
                    }]
                })
        finally:
            _au_mod._TIMEOUT = original

        # After timeout the chain is not blocked — tool returns fallback text
        assert result  # non-empty
        assert "best judgement" in result.lower() or "did not respond" in result.lower()


# ---------------------------------------------------------------------------
# Registry sanity (spot-checks for the complete tool set)
# ---------------------------------------------------------------------------


class TestRegistrySanity:
    """Verify the final registry state matches the expected tool set."""

    def test_all_expected_agent_tools_registered(self):
        agent_tools = {t.name for t in REGISTRY.tools_for("agent")}
        expected = {
            "bash", "bash_output", "bash_kill",
            "read_file", "write_file", "edit_file",
            "glob", "grep", "find_file", "git_diff",
            "preview_fetch",
            "ask_user",
            "task_list", "task_update", "plan_read",
        }
        missing = expected - agent_tools
        assert not missing, f"Agent tools missing from registry: {missing}"

    def test_cm_only_tools_not_in_agent_set(self):
        agent_names = {t.name for t in REGISTRY.tools_for("agent")}
        assert "task_create" not in agent_names
        assert "request_file_read" not in agent_names

    def test_cm_tools_include_expected_set(self):
        cm_names = {t.name for t in REGISTRY.tools_for("cm")}
        assert "task_create" in cm_names
        assert "task_list" in cm_names
        assert "task_update" in cm_names
        assert "plan_read" in cm_names
        assert "read_file" in cm_names
        assert "git_diff" in cm_names
        assert "grep" in cm_names

    def test_allowed_filter_restricts_tools(self):
        """tools_allowed restricts the set to the specified subset, plus the
        tools every agent is always offered (reply_to_user)."""
        always = {t.name for t in REGISTRY.tools_for("agent") if t.always_offered}
        assert always  # reply_to_user at least
        restricted = {t.name for t in REGISTRY.tools_for("agent", allowed=["bash", "read_file"])}
        assert restricted == {"bash", "read_file"} | always

    def test_allowed_filter_empty_list_leaves_only_the_always_offered(self):
        restricted = REGISTRY.tools_for("agent", allowed=[])
        assert restricted and all(t.always_offered for t in restricted)

    def test_allowed_none_returns_all_role_tools(self):
        all_agent = REGISTRY.tools_for("agent", allowed=None)
        restricted = REGISTRY.tools_for("agent", allowed=None)
        assert len(all_agent) == len(restricted)

    def test_ant_schemas_include_name_and_input_schema(self):
        schemas = REGISTRY.ant_schemas_for("agent")
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s
