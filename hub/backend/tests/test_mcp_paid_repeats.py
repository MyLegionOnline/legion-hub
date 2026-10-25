"""internal design notes Stage 10 — "`loop_guard` sees paid repeats".

Until this stage the MCP branch of the agent tool loop called neither guard:
`ToolErrorGuard` and `RepeatedToolCallGuard` only ever saw workspace tools, so
the sole ceiling on a model re-sending the same billed `vps_read_file` was
`PROXY_AGENT_MAX_TOOL_CALLS` — hundreds of charges away.

These are unit tests of the guard's side of that wiring (the `paid` flag, the
productive-call notion for VPS writes, and the remote read target). The
wiring itself lives in `proxy_session._guard_mcp_call`.
"""

from __future__ import annotations

from app.core.loop_guard import (
    MCP_PRODUCTIVE_TOOLS,
    STALL_DUP_THRESHOLD,
    STALL_PAID_DUP_THRESHOLD,
    RepeatedToolCallGuard,
    mcp_call_productive,
    mcp_read_target,
)

_READ = {"server": "PROD", "path": "/etc/nginx/sites-enabled/site.conf"}


def _observe_n(guard: RepeatedToolCallGuard, n: int, *, paid: bool, args=None):
    """Make the same call `n` times; return the verdict of each."""
    return [
        guard.observe("vps_read_file", dict(args or _READ), paid=paid, result_text="ok")
        for _ in range(n)
    ]


class TestPaidDupThreshold:
    def test_a_paid_repeat_is_warned_one_call_earlier_than_a_free_one(self):
        assert STALL_PAID_DUP_THRESHOLD < STALL_DUP_THRESHOLD

        paid = _observe_n(RepeatedToolCallGuard(), STALL_PAID_DUP_THRESHOLD, paid=True)
        assert paid[-1].warn is not None
        assert not paid[-1].stall

        free = _observe_n(RepeatedToolCallGuard(), STALL_PAID_DUP_THRESHOLD, paid=False)
        assert free[-1].warn is None, "a free call must not be warned this early"

    def test_the_call_after_the_paid_warning_is_a_stall(self):
        guard = RepeatedToolCallGuard()
        verdicts = _observe_n(guard, STALL_PAID_DUP_THRESHOLD + 1, paid=True)
        assert verdicts[-1].stall
        assert verdicts[-1].reason == "dup_call"
        assert verdicts[-1].tool_name == "vps_read_file"

    def test_the_paid_warning_says_the_repeats_were_billed(self):
        guard = RepeatedToolCallGuard()
        warn = _observe_n(guard, STALL_PAID_DUP_THRESHOLD, paid=True)[-1].warn
        assert "billed" in warn
        # And the free wording stays free of a claim that is not true for it.
        free_guard = RepeatedToolCallGuard()
        free_warn = _observe_n(free_guard, STALL_DUP_THRESHOLD, paid=False)[-1].warn
        assert "billed" not in free_warn

    def test_different_paths_on_one_server_are_not_repeats(self):
        guard = RepeatedToolCallGuard()
        for i in range(STALL_PAID_DUP_THRESHOLD + 2):
            verdict = guard.observe(
                "vps_read_file",
                {"server": "PROD", "path": f"/etc/nginx/sites-enabled/s{i}.conf"},
                paid=True,
                result_text="ok",
            )
            assert not verdict.stall

    def test_the_same_path_on_two_servers_is_not_a_repeat(self):
        guard = RepeatedToolCallGuard()
        for server in ("PROD", "STAGING"):
            for _ in range(STALL_PAID_DUP_THRESHOLD):
                verdict = guard.observe(
                    "vps_read_file",
                    {"server": server, "path": "/etc/nginx/nginx.conf"},
                    paid=True,
                    result_text="ok",
                )
                assert not verdict.stall


class TestProductiveMcpCalls:
    def test_a_successful_write_or_service_action_is_progress(self):
        assert MCP_PRODUCTIVE_TOOLS == {"vps_write_file", "vps_upload", "vps_service"}
        for tool in sorted(MCP_PRODUCTIVE_TOOLS):
            assert mcp_call_productive(tool, '{"bytes": 412}') is True

    def test_a_failed_write_is_not_progress(self):
        assert mcp_call_productive(
            "vps_write_file", "Error: [permission_denied] /etc/nginx/site.conf"
        ) is False
        assert mcp_call_productive("vps_write_file", None) is True  # no error text = no failure

    def test_a_read_is_never_progress(self):
        assert mcp_call_productive("vps_read_file", "content") is False
        assert mcp_call_productive("vps_list_dir", "entries") is False

    def test_a_real_deploy_is_not_read_as_thrashing(self):
        # Signal A.2 fires on many low-diversity calls with *zero* productive
        # ones. A deploy that is landing config files must not qualify, however
        # repetitive its surrounding reads look.
        guard = RepeatedToolCallGuard()
        for i in range(20):
            tool, args, result = (
                ("vps_write_file", {"server": "PROD", "path": f"/tmp/s{i}.conf"}, '{"bytes": 1}')
                if i % 4 == 0
                else ("vps_list_dir", {"server": "PROD", "path": f"/etc/nginx/d{i}"}, "entries")
            )
            verdict = guard.observe(
                tool, args, paid=True,
                productive=mcp_call_productive(tool, result),
                result_text=result,
                read_target=mcp_read_target(tool, args),
            )
            assert not verdict.stall, i
        assert guard.productive_calls == 5


class TestRemoteReadTarget:
    def test_the_target_is_scoped_by_server_and_left_unnormalised(self):
        assert mcp_read_target("vps_read_file", _READ) == (
            "vps:PROD:/etc/nginx/sites-enabled/site.conf"
        )
        # An absolute path on someone else's machine must not be put through
        # the workspace path contract, which would strip the leading slash and
        # collide with a workspace file of the same tail.
        assert mcp_read_target("vps_list_dir", {"server": "PROD", "path": "/etc"}) == "vps:PROD:/etc"

    def test_a_non_read_tool_has_no_target(self):
        assert mcp_read_target("vps_write_file", {"server": "PROD", "path": "/etc/x"}) is None
        assert mcp_read_target("vps_exec", {"server": "PROD", "command": ["ls"]}) is None
        assert mcp_read_target("vps_read_file", {"server": "PROD"}) is None

    def test_re_reading_one_remote_file_trips_signal_c(self):
        # Different max_bytes each time, so Signal A's exact-args hash never
        # matches — this is the "Frontend developer" pattern with a billed tool.
        guard = RepeatedToolCallGuard()
        stalled = False
        for i in range(12):
            args = {**_READ, "max_bytes": 1000 + i}
            verdict = guard.observe(
                "vps_read_file", args, paid=True,
                result_text="server { listen 80; }",
                read_target=mcp_read_target("vps_read_file", args),
            )
            if verdict.stall:
                stalled = True
                assert verdict.reason == "stuck_read"
                assert verdict.target == "vps:PROD:/etc/nginx/sites-enabled/site.conf"
                break
        assert stalled, "re-reading one remote file forever should be caught"

    def test_a_write_between_reads_clears_the_read_streak(self):
        guard = RepeatedToolCallGuard()
        for i in range(30):
            if i % 3 == 2:
                guard.observe(
                    "vps_write_file", {"server": "PROD", "path": f"/tmp/x{i}"},
                    paid=True, result_text='{"bytes": 1}',
                    read_target=mcp_read_target("vps_write_file", {}),
                )
                continue
            args = {**_READ, "max_bytes": 1000 + i}
            verdict = guard.observe(
                "vps_read_file", args, paid=True, result_text="server {}",
                read_target=mcp_read_target("vps_read_file", args),
            )
            assert not verdict.stall, i
