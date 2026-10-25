"""
Tests for the bash `cat`/`cat -n` duplicate-read short-circuit.

Root cause covered: read_file and grep already avoid re-sending content the
agent already saw this session (record_observation/record_read_interval), but
bash had no such protection at all — an agent that switched from read_file to
`cat -n <file>` for the exact same file paid full token AND call cost every
time, completely bypassing the existing guard rails. This closes that gap for
the dominant, unambiguous case: a plain `cat [-n] <single file>` with no
pipes/redirects/wildcards, cross-checked against the SAME session read ledger
that read_file/grep already write to (so a `cat` after a `read_file` of the
same file — or vice versa — is recognized as a duplicate).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_observations import ObservationResult, fingerprint
from app.core.tools import REGISTRY, ToolContext
from app.core.tools._bash import _classify_single_file_cat


# ---------------------------------------------------------------------------
# Pure classifier — no mocks needed
# ---------------------------------------------------------------------------


class TestClassifySingleFileCat:
    def test_plain_cat(self):
        assert _classify_single_file_cat("cat foo.py") == "foo.py"

    def test_cat_with_numbered_flag(self):
        assert _classify_single_file_cat("cat -n foo.py") == "foo.py"

    def test_cat_nested_path(self):
        assert _classify_single_file_cat("cat blog/templates/blog/base.html") == \
            "blog/templates/blog/base.html"

    def test_two_files_rejected(self):
        assert _classify_single_file_cat("cat foo.py bar.py") is None

    def test_missing_path_rejected(self):
        assert _classify_single_file_cat("cat -n") is None

    def test_unsupported_flag_rejected(self):
        assert _classify_single_file_cat("cat --number foo.py") is None

    def test_not_cat_rejected(self):
        assert _classify_single_file_cat("echo hi") is None

    def test_wildcard_rejected(self):
        assert _classify_single_file_cat("cat *.py") is None

    def test_piped_command_rejected(self):
        assert _classify_single_file_cat("cat foo.py | grep x") is None

    def test_redirect_rejected(self):
        assert _classify_single_file_cat("cat foo.py > out.txt") is None

    def test_chained_command_rejected(self):
        assert _classify_single_file_cat("cat foo.py && echo done") is None

    def test_unbalanced_quotes_returns_none(self):
        assert _classify_single_file_cat("cat 'unterminated") is None


# ---------------------------------------------------------------------------
# Full execute() path
# ---------------------------------------------------------------------------


def _make_ws(status="running"):
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = status
    ws.network_enabled = True
    return ws


def _make_ctx() -> tuple[ToolContext, MagicMock, MagicMock]:
    ws = _make_ws()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _make_result(obj):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=obj)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=lambda q: _make_result(ws))
    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)
    return ctx, db, ws


def _fake_observation(ref: str = "obs-1") -> ObservationResult:
    artifact = MagicMock()
    artifact.id = ref
    artifact.stale = False
    artifact.tool_name = "read_file"
    artifact.token_count = 42
    return ObservationResult(artifact=artifact, duplicate=False)


@pytest.mark.asyncio
async def test_dedup_short_circuits_when_fully_covered():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")
    content = "line1\nline2\nline3\nline4\nline5\n"
    fp = fingerprint(content)

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read,
        patch("app.core.context_observations.load_context_state", new_callable=AsyncMock) as mock_state,
        patch("app.core.context_observations.record_observation", new_callable=AsyncMock) as mock_obs,
        patch("app.core.context_observations.record_read_interval", new_callable=AsyncMock) as mock_interval,
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_read.return_value = content
        mock_state.return_value = {
            "read_ledger": {
                "blog/models.py": {"fingerprint": fp, "ranges": [[1, 5]], "observation_refs": ["obs-1"]}
            }
        }
        mock_obs.return_value = _fake_observation()
        mock_interval.return_value = []
        result = await tool.execute(ctx, {"cmd": "cat -n blog/models.py"})

    assert "Skipped duplicate" in result
    assert "blog/models.py" in result
    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_runs_for_real_when_never_read_before():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")
    content = "line1\nline2\n"

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read,
        patch("app.core.context_observations.load_context_state", new_callable=AsyncMock) as mock_state,
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_read.return_value = content
        mock_state.return_value = {}  # empty ledger — nothing read yet
        mock_exec.return_value = {"exit_code": 0, "stdout": content, "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "cat blog/models.py"})

    assert "exit_code: 0" in result
    mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_runs_for_real_when_fingerprint_changed():
    """File was read before, but its content (and thus fingerprint) changed since —
    stale ranges must not suppress a fresh read of the new content."""
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")
    content = "new content\n"

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read,
        patch("app.core.context_observations.load_context_state", new_callable=AsyncMock) as mock_state,
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_read.return_value = content
        mock_state.return_value = {
            "read_ledger": {
                "blog/models.py": {"fingerprint": "stale-fp", "ranges": [[1, 1]], "observation_refs": []}
            }
        }
        mock_exec.return_value = {"exit_code": 0, "stdout": content, "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "cat blog/models.py"})

    assert "exit_code: 0" in result
    mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_runs_for_real_when_only_partially_covered():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")
    content = "l1\nl2\nl3\nl4\nl5\n"
    fp = fingerprint(content)

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read,
        patch("app.core.context_observations.load_context_state", new_callable=AsyncMock) as mock_state,
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_read.return_value = content
        mock_state.return_value = {
            "read_ledger": {"f.py": {"fingerprint": fp, "ranges": [[1, 2]], "observation_refs": []}}
        }
        mock_exec.return_value = {"exit_code": 0, "stdout": content, "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "cat f.py"})

    assert "exit_code: 0" in result
    mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_piped_command_never_deduped_even_if_fully_covered():
    """Ambiguous commands must always run for real — dedup only ever short-circuits
    the exact single-file 'cat [-n] <path>' shape."""
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")
    content = "line1\n"
    fp = fingerprint(content)

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read,
        patch("app.core.context_observations.load_context_state", new_callable=AsyncMock) as mock_state,
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_read.return_value = content
        mock_state.return_value = {
            "read_ledger": {"f.py": {"fingerprint": fp, "ranges": [[1, 1]], "observation_refs": []}}
        }
        mock_exec.return_value = {"exit_code": 0, "stdout": "line1\n", "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "cat f.py | grep line"})

    assert "exit_code: 0" in result
    mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_dedup_skipped_when_workspace_not_running():
    ctx, db, ws = _make_ctx()
    ws.status = "stopped"
    tool = REGISTRY.get("bash")

    from app.core.workspace_lifecycle import WorkspaceStartError

    # bash tries to wake the workspace; when that fails, dedup must not answer.
    with patch(
        "app.core.workspace_lifecycle.ensure_workspace_started",
        new_callable=AsyncMock,
        side_effect=WorkspaceStartError("quota_exceeded", "limit", status_code=429),
    ):
        result = await tool.execute(ctx, {"cmd": "cat f.py"})
    assert "Error" in result
    assert "quota_exceeded" in result
