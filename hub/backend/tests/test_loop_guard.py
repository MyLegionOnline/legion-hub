"""Unit tests for app/core/loop_guard.py — the agent non-productivity detector."""
from __future__ import annotations

import pytest

from app.core.loop_guard import (
    PRODUCTIVE_TOOLS,
    RepeatedToolCallGuard,
    build_stall_report,
    call_signature,
    looks_like_blocked_read,
    parse_read_span,
    read_loop_target,
)


def _read_result(offset: int, end: int, *, total: int = 500) -> str:
    """Shape of a real read_file result header the guard parses."""
    return (
        f"<file path='blog/static/blog/style.css' lines={offset}-{end} "
        f"total={total} fingerprint=abc>\n{offset}\tbody {{}}\n"
    )


_RECEIPT = (
    "[observation available: 1234; tool=read_file; tokens=90; "
    "use observation_read(ref='1234', offset=..., limit=...) for exact text]"
)
_HARD_BLOCK = (
    "[BLOCKED: 'blog/static/blog/style.css' has been requested 3 times this "
    "session with unchanged content — it will not be served again.]\n" + _RECEIPT
)


class TestCallSignature:
    def test_same_call_same_signature(self):
        a = call_signature("read_file", {"path": "blog/views.py"})
        b = call_signature("read_file", {"path": "blog/views.py"})
        assert a == b

    def test_arg_spelling_is_normalised(self):
        # canonicalize_tool_input strips the /workspace prefix, so these are one call.
        bare = call_signature("read_file", {"path": "blog/views.py"})
        prefixed = call_signature("read_file", {"path": "/workspace/blog/views.py"})
        assert bare == prefixed

    def test_different_tool_different_signature(self):
        assert call_signature("read_file", {"path": "a"}) != call_signature("grep", {"path": "a"})

    def test_different_args_different_signature(self):
        assert call_signature("grep", {"pattern": "foo"}) != call_signature("grep", {"pattern": "bar"})


class TestRepeatedToolCallGuard:
    def test_identical_call_warns_at_threshold_then_stalls(self):
        g = RepeatedToolCallGuard(dup_threshold=3)
        args = {"path": "blog/static/blog/style.css"}

        assert g.observe("read_file", args).warn is None          # 1
        assert g.observe("read_file", args).warn is None          # 2
        v3 = g.observe("read_file", args)                          # 3 → warn
        assert v3.warn is not None and not v3.stall
        assert "2-3" in v3.warn
        v4 = g.observe("read_file", args)                         # 4 → stall
        assert v4.stall and v4.reason == "dup_call"
        assert v4.tool_name == "read_file" and v4.repeat_count == 4

    def test_warning_is_emitted_once_per_signature(self):
        g = RepeatedToolCallGuard(dup_threshold=2)
        a = {"path": "a"}
        b = {"pattern": "b"}
        assert g.observe("read_file", a).warn is None
        assert g.observe("read_file", a).warn is not None          # warn for a
        # b reaches its own threshold and still warns; a already fired → stall
        assert g.observe("grep", b).warn is None
        assert g.observe("grep", b).warn is not None               # warn for b
        assert g.observe("read_file", a).stall                     # a: 3 > 2

    def test_low_diversity_with_no_writes_stalls(self):
        g = RepeatedToolCallGuard(
            dup_threshold=99, min_calls_for_ratio=10, distinct_ratio_floor=0.5
        )
        verdicts = [g.observe("grep", {"pattern": f"p{i % 3}"}) for i in range(12)]
        stalls = [v for v in verdicts if v.stall]
        assert stalls and stalls[0].reason == "low_diversity"

    def test_productive_call_prevents_low_diversity_stall(self):
        g = RepeatedToolCallGuard(
            dup_threshold=99, min_calls_for_ratio=6, distinct_ratio_floor=0.9
        )
        g.observe("write_file", {"path": "out.py", "content": "x"})
        last = None
        for i in range(10):
            last = g.observe("grep", {"pattern": f"p{i % 2}"})
        assert not last.stall
        assert g.productive_calls == 1

    def test_only_fires_once(self):
        g = RepeatedToolCallGuard(dup_threshold=2)
        args = {"path": "a"}
        g.observe("read_file", args)
        g.observe("read_file", args)
        assert g.observe("read_file", args).stall                  # 3 > 2 → stall
        # subsequent observes stay silent — the loop is being torn down
        assert not g.observe("read_file", args).stall
        assert g.observe("read_file", args).warn is None

    def test_productive_tool_membership(self):
        assert "write_file" in PRODUCTIVE_TOOLS
        assert "edit_file" in PRODUCTIVE_TOOLS
        assert "read_file" not in PRODUCTIVE_TOOLS
        assert "grep" not in PRODUCTIVE_TOOLS

    def test_summary_line_reports_top_signature(self):
        g = RepeatedToolCallGuard(dup_threshold=99)
        for _ in range(4):
            g.observe("read_file", {"path": "hot.py"})
        g.observe("grep", {"pattern": "x"})
        line = g.summary_line()
        assert "5 tool calls" in line
        assert "read_file" in line and "x4" in line


class TestBuildStallReport:
    def test_dup_call_report_names_the_tool(self):
        text = build_stall_report(
            {"reason": "dup_call", "tool_name": "grep", "repeat_count": 5, "summary": "s"}
        )
        assert "grep" in text and "5x" in text
        assert "task_update" in text

    def test_low_diversity_report(self):
        text = build_stall_report({"reason": "low_diversity", "summary": "20 tool calls, 4 distinct"})
        assert "20 tool calls, 4 distinct" in text

    def test_restart_streak_report(self):
        text = build_stall_report({"reason": "restart_streak", "restart_streak": 3})
        assert "restarted 3 times" in text

    def test_unknown_reason_falls_back(self):
        text = build_stall_report({"reason": "???"})
        assert "AUTOMATIC STALL REPORT" in text

    def test_stuck_read_report_names_the_file(self):
        text = build_stall_report(
            {
                "reason": "stuck_read",
                "tool_name": "read_file",
                "repeat_count": 6,
                "target": "blog/static/blog/style.css",
                "summary": "s",
            }
        )
        assert "blog/static/blog/style.css" in text and "6" in text

    def test_stuck_read_blocked_report(self):
        text = build_stall_report(
            {"reason": "stuck_read_blocked", "repeat_count": 3, "target": "a.css"}
        )
        assert "a.css" in text and "BLOCKED" in text


class TestReadLoopHelpers:
    def test_parse_read_span_from_header(self):
        assert parse_read_span(_read_result(10, 60)) == (10, 60)
        assert parse_read_span("<observation ref='x' lines=200-260 total=900>") == (200, 260)

    def test_parse_read_span_missing(self):
        assert parse_read_span(None) is None
        assert parse_read_span(_RECEIPT) is None
        assert parse_read_span("No matches for 'foo'") is None

    def test_looks_like_blocked_read(self):
        assert looks_like_blocked_read(_RECEIPT)
        assert looks_like_blocked_read(_HARD_BLOCK)
        assert looks_like_blocked_read("[observation stale: 1; tool=read_file]")
        assert not looks_like_blocked_read(_read_result(1, 100))
        assert not looks_like_blocked_read(None)

    def test_read_loop_target_resolution(self):
        assert read_loop_target("read_file", {"path": "/workspace/blog/a.css"}) == "blog/a.css"
        assert read_loop_target("grep", {"pattern": "x", "path_prefix": "blog/a.css"}) == "blog/a.css"
        assert read_loop_target("grep", {"pattern": "x", "glob": "**/*.css"}) == "glob:**/*.css"
        assert read_loop_target("grep", {"pattern": "x"}) is None
        assert read_loop_target("observation_read", {"ref": "1"}) is None


class TestReadLoopGuard:
    CSS = {"path": "blog/static/blog/style.css"}

    def test_repeated_overlapping_reads_of_one_file_stall(self):
        g = RepeatedToolCallGuard()
        verdict = None
        for i in range(12):
            # same file, drifting offsets, every range overlaps the last
            verdict = g.observe(
                "read_file",
                {**self.CSS, "offset": 1 + i, "limit": 100},
                result_text=_read_result(1 + i, 100 + i),
            )
            if verdict.stall:
                break
        assert verdict.stall and verdict.reason == "stuck_read"
        assert verdict.target == "blog/static/blog/style.css"
        assert verdict.tool_name == "read_file"

    def test_sequential_paging_does_not_stall(self):
        g = RepeatedToolCallGuard()
        last = None
        for page in range(12):
            start = 1 + page * 100
            last = g.observe(
                "read_file",
                {**self.CSS, "offset": start, "limit": 100},
                result_text=_read_result(start, start + 99, total=5000),
            )
        assert not last.stall

    def test_blocked_reads_stall_faster(self):
        g = RepeatedToolCallGuard()
        verdicts = [
            g.observe("read_file", {**self.CSS, "offset": o}, result_text=_RECEIPT)
            for o in (1, 2, 3)
        ]
        assert verdicts[-1].stall
        assert verdicts[-1].reason == "stuck_read_blocked"
        assert verdicts[-1].repeat_count == 3

    def test_hard_block_receipt_counts_as_blocked(self):
        g = RepeatedToolCallGuard()
        v = None
        for o in (1, 2, 3):
            v = g.observe("read_file", {**self.CSS, "offset": o}, result_text=_HARD_BLOCK)
        assert v.stall and v.reason == "stuck_read_blocked"

    def test_other_tool_resets_the_streak(self):
        g = RepeatedToolCallGuard()
        for o in range(4):
            g.observe("read_file", {**self.CSS, "offset": 1 + o},
                      result_text=_read_result(1, 100))
        g.observe("edit_file", {"path": "blog/a.py", "old": "x", "new": "y"})
        last = None
        for o in range(4):
            last = g.observe("read_file", {**self.CSS, "offset": 1 + o},
                             result_text=_read_result(1, 100))
        assert not last.stall

    def test_reading_a_different_file_resets_the_streak(self):
        g = RepeatedToolCallGuard()
        for o in range(4):
            g.observe("read_file", {**self.CSS, "offset": 1 + o},
                      result_text=_read_result(1, 100))
        g.observe("read_file", {"path": "blog/views.py"},
                  result_text="<file path='blog/views.py' lines=1-100 total=100 fingerprint=z>\n")
        last = None
        for o in range(4):
            last = g.observe("read_file", {**self.CSS, "offset": 1 + o},
                             result_text=_read_result(1, 100))
        assert not last.stall

    def test_repeated_grep_of_one_file_stalls(self):
        g = RepeatedToolCallGuard()
        last = None
        for i in range(6):
            last = g.observe(
                "grep",
                {"pattern": f"sel-{i}", "path_prefix": "blog/static/blog/style.css"},
                result_text="blog/static/blog/style.css:12:.foo {}",
            )
        assert last.stall and last.reason == "stuck_read"
        assert last.target == "blog/static/blog/style.css"

    def test_observation_read_shares_target_with_read_file(self):
        g = RepeatedToolCallGuard()
        # 3 real reads that returned receipts, then observation_read of the same
        # underlying file (target passed in the way proxy_session resolves it).
        for o in (1, 2):
            g.observe("read_file", {**self.CSS, "offset": o}, result_text=_RECEIPT)
        v = g.observe(
            "observation_read",
            {"ref": "1234", "offset": 1},
            result_text="[observation stale: 1234; tool=read_file]",
            read_target="blog/static/blog/style.css",
        )
        assert v.stall and v.reason == "stuck_read_blocked"

    def test_whole_workspace_grep_does_not_count(self):
        g = RepeatedToolCallGuard()
        last = None
        for i in range(8):
            last = g.observe("grep", {"pattern": f"p{i}"}, result_text="No matches")
        assert not last.stall

    def test_warns_once_before_the_stall(self):
        g = RepeatedToolCallGuard()
        warns = []
        for i in range(6):
            v = g.observe("read_file", {**self.CSS, "offset": 1 + i},
                          result_text=_read_result(1, 100))
            if v.warn:
                warns.append(v.warn)
        assert len(warns) == 1
        assert "blog/static/blog/style.css" in warns[0]

    def test_non_overlapping_then_overlapping_still_trips(self):
        g = RepeatedToolCallGuard()
        # page forward once (progress), then get stuck re-reading the top with
        # slightly different offsets each time (so Signal A never matches)
        g.observe("read_file", {**self.CSS, "offset": 400},
                  result_text=_read_result(400, 460, total=5000))
        stall = None
        for i in range(8):
            v = g.observe("read_file", {**self.CSS, "offset": 1 + i},
                          result_text=_read_result(1, 60, total=5000))
            if v.stall:
                stall = v
                break
        assert stall is not None and stall.reason == "stuck_read"

    def test_backward_compatible_observe_without_new_kwargs(self):
        g = RepeatedToolCallGuard(dup_threshold=3)
        args = {"path": "blog/static/blog/style.css"}
        assert g.observe("read_file", args).warn is None
        assert g.observe("read_file", args).warn is None
        assert g.observe("read_file", args).warn is not None   # dup warn at 3
        assert g.observe("read_file", args).stall              # dup stall at 4
