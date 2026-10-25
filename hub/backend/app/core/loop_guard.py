"""Non-productivity (stall) detection for the agent tool loop.

The agent side of a proxy pair can fall into a degenerate loop: it re-runs the
same ``read_file`` / ``grep`` / ``bash`` discovery calls over and over, never
writes a file, exhausts its tool-call budget, gets auto-resumed with the
counters reset, and starts the same discovery from scratch. The raw
``PROXY_AGENT_MAX_TOOL_CALLS`` ceiling is a blunt instrument — it stops a
runaway loop only after hundreds of wasted calls and does nothing to break the
pattern.

This module is the smarter circuit breaker. It watches three independent signals:

* **Signal A — repeated identical tool calls.** Every workspace tool call is
  hashed on ``(tool_name, canonical_args)``. The same signature repeating
  ``dup_threshold`` times gets a one-shot in-band warning; repeating past that
  is treated as a stall. A low distinct/total ratio with zero productive calls
  after a warm-up window is the same verdict by a different route (the model
  trying many *different* useless calls).

* **Signal B — restarts without progress.** ``Task.stall_streak`` counts
  consecutive agent sessions that ended on a limit/timeout without producing a
  file write; it is reset to 0 the moment real progress happens. The proxy loop
  reads it at resume time.

* **Signal C — stuck re-reading one file.** ``read_file`` / ``observation_read``
  / ``grep`` aimed at the *same* target path over and over — with different
  ``offset`` / ``limit`` / ``ref`` each time, so Signal A's exact-args hash
  never matches — while never editing anything or reading new ground. This is
  the pattern from the "Frontend developer" session (2026-09-06): ~15 reads of one
  ``style.css`` with the model re-running the same analysis. Non-overlapping
  sequential paging of a big file is explicitly *not* a stall; overlapping /
  repeated ranges and duplicate-read receipts (``[observation …]`` /
  ``[BLOCKED …]`` from ``_session_state``) are.

Since ``internal design notes`` Stage 10 signals A and C also watch **paid** MCP
calls (an SSH deploy tool, a vision call), not just free workspace tools. Two
differences follow from the price: a repeat is warned about one call earlier
than a free one, and the warning says outright that each repeat is billed.
Before this the MCP branch of the tool loop called neither guard, so a model
could re-send the same ``vps_read_file`` thirty times and nothing but the raw
tool-call ceiling noticed.

All three signals converge on the same response: hand the situation to the
ChatMaster as a real mid-task turn so it can give the agent a concrete unstuck
plan (``STALL_INTERVENTION_NOTE`` in ``proxy_session``), instead of silently
re-dispatching the same dead end.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

from app.core.tools.registry import canonicalize_tool_input

# --- Tunables (env-overridable so behaviour can be probed on a live room) ----

def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if 0.0 < value < 1.0 else default


# Same (tool, args) this many times in one tool loop → stage-1 warning; the
# next identical call → stall.
STALL_DUP_THRESHOLD = _int_env("PROXY_STALL_DUP_THRESHOLD", 3)
# The same paid (MCP) call this many times → warning; the next one → stall.
# One lower than the free threshold: the cheapest moment to stop a billed
# loop is before the third charge, not after it.
STALL_PAID_DUP_THRESHOLD = _int_env("PROXY_STALL_PAID_DUP_THRESHOLD", 2)
# Don't judge diversity until the loop has done at least this many calls.
STALL_MIN_CALLS_FOR_RATIO = _int_env("PROXY_STALL_MIN_CALLS", 14)
# distinct signatures / total calls below this, with zero productive calls,
# after the warm-up window → stall.
STALL_DISTINCT_RATIO_FLOOR = _float_env("PROXY_STALL_DIVERSITY_FLOOR", 0.34)
# Consecutive limit/timeout restarts with no file write → Signal B fires at
# the start of the next resumed session.
STALL_RESTART_THRESHOLD = _int_env("PROXY_STALL_RESTART_THRESHOLD", 2)

# Signal C — read tools whose repetition on one target path means "stuck
# re-reading", not progress. Deliberately narrow: the three from the incident
# report, plus the two VPS reads (Stage 10), which are billed per call. Paging
# a large file with these is fine; standing still is not.
READ_LOOP_TOOLS: frozenset[str] = frozenset(
    {"read_file", "observation_read", "grep", "vps_read_file", "vps_list_dir"}
)
# Targets that are already resolved and must not be run through the workspace
# path contract: a `glob:` pattern, and a `vps:` remote path (which is
# absolute on someone else's machine and means nothing as a workspace path).
_OPAQUE_TARGET_PREFIXES = ("glob:", "vps:")
# Consecutive Signal-C reads of the *same* target that made no progress
# (overlapping/repeated range, or a content-less receipt) → stall.
STALL_SAME_TARGET_READS = _int_env("PROXY_STALL_SAME_TARGET_READS", 6)
# Consecutive reads of the same target that came back as a duplicate-read
# receipt / [BLOCKED] notice — a much stronger signal, trips on its own.
STALL_BLOCKED_TARGET_READS = _int_env("PROXY_STALL_BLOCKED_TARGET_READS", 3)

# A call to one of these is real forward progress — it resets both the
# in-loop productive counter and (via proxy_session) Task.stall_streak.
PRODUCTIVE_TOOLS: frozenset[str] = frozenset(
    {"write_file", "edit_file", "run_subtask", "submit_image_prompts"}
)

# The same notion on the MCP side: a write or a service action that *worked*
# changed the world, so a deploy session that is landing config files is not
# standing still even though it never calls `write_file`. A failed one is not
# progress, which is why `mcp_call_productive` needs the result text too.
MCP_PRODUCTIVE_TOOLS: frozenset[str] = frozenset(
    {"vps_write_file", "vps_upload", "vps_service"}
)


def mcp_call_productive(tool_name: str, result_text: str | None) -> bool:
    """Whether this MCP call counts as forward progress for Signal A.2."""
    if tool_name not in MCP_PRODUCTIVE_TOOLS:
        return False
    head = (result_text or "").lstrip().casefold()
    return not head.startswith("error")

# Sentinel: the caller did not resolve a target for this call (Signal C works
# it out from the args). ``None`` means "resolved, but there is no target".
READ_TARGET_UNSET = object()

_READ_SPAN_RE = re.compile(r"lines=(\d+)-(\d+)")


def _norm_target_path(value: object) -> str | None:
    """Normalise a workspace path to the same key ``read_file`` compares on."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        from app.core.path_contract import normalize_workspace_path

        return normalize_workspace_path(text)
    except Exception:
        return text.replace("\\", "/").lstrip("/") or None


def read_loop_target(tool_name: str, tool_input: dict | None) -> str | None:
    """Best-effort target path for a Signal-C read call, from its args alone.

    ``observation_read`` cannot be resolved here (the ref → path lookup needs
    the DB); the proxy loop passes that one in explicitly.
    """
    args = tool_input or {}
    if tool_name == "read_file":
        return _norm_target_path(args.get("path"))
    if tool_name == "grep":
        prefix = _norm_target_path(args.get("path_prefix"))
        if prefix:
            return prefix
        glob = str(args.get("glob") or "").strip()
        return f"glob:{glob}" if glob else None
    return None


def mcp_read_target(tool_name: str, tool_input: dict | None) -> str | None:
    """Signal-C target for a VPS read call: ``vps:<server>:<remote path>``.

    Scoped by server on purpose — reading the same path on `PROD` and on
    `STAGING` is two different questions, not a repeat.
    """
    if tool_name not in READ_LOOP_TOOLS:
        return None
    args = tool_input or {}
    path = str(args.get("path") or "").strip()
    if not path:
        return None
    server = str(args.get("server") or "?").strip() or "?"
    return f"vps:{server}:{path}"


def parse_read_span(result_text: str | None) -> tuple[int, int] | None:
    """Extract the ``lines=A-B`` range a read tool actually returned, if any."""
    if not result_text:
        return None
    match = _READ_SPAN_RE.search(result_text)
    if not match:
        return None
    a, b = int(match.group(1)), int(match.group(2))
    return (a, b) if a <= b else (b, a)


def looks_like_blocked_read(result_text: str | None) -> bool:
    """True when a read came back as a receipt / hard block instead of content.

    Covers ``observation.receipt()`` (``[observation available/stale: …]``) and
    ``_session_state.duplicate_escalation_notice`` (``[BLOCKED: …]``) — both
    mean the workspace declined to re-serve the same bytes.
    """
    if not result_text:
        return False
    head = result_text.lstrip().lower()
    return head.startswith("[observation ") or "[blocked:" in head


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def call_signature(tool_name: str, tool_input: dict | None) -> str:
    """Stable identity for a tool call, independent of arg spelling."""
    args = canonicalize_tool_input(tool_name, tool_input)
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{tool_name}\x00{payload}".encode("utf-8")).hexdigest()[:16]
    return f"{tool_name}:{digest}"


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of observing one tool call."""

    warn: str | None = None          # inject into the tool result, keep looping
    stall: bool = False              # stop the loop, escalate to ChatMaster
    reason: str | None = None        # dup_call | low_diversity | stuck_read | stuck_read_blocked
    tool_name: str | None = None
    repeat_count: int = 0
    target: str | None = None        # Signal C: the path being re-read


@dataclass
class RepeatedToolCallGuard:
    """Bounds unproductive repetition inside a single agent tool loop.

    One instance per ``_call_llm_with_tools`` invocation on the agent side,
    mirroring how ``ToolErrorGuard`` is scoped. Cross-session repetition is
    caught by ``Task.stall_streak`` (Signal B), not here.
    """

    dup_threshold: int = STALL_DUP_THRESHOLD
    paid_dup_threshold: int = STALL_PAID_DUP_THRESHOLD
    min_calls_for_ratio: int = STALL_MIN_CALLS_FOR_RATIO
    distinct_ratio_floor: float = STALL_DISTINCT_RATIO_FLOOR
    same_target_read_limit: int = STALL_SAME_TARGET_READS
    blocked_target_read_limit: int = STALL_BLOCKED_TARGET_READS

    total_calls: int = 0
    productive_calls: int = 0
    _sig_counts: dict[str, int] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set)
    _fired: bool = False

    # Signal C — per-target read-loop state. Reset whenever the target changes
    # or any non-read tool runs.
    _read_target: str | None = None
    _read_streak: int = 0            # consecutive no-progress reads of _read_target
    _read_blocked_streak: int = 0    # …of which came back as a receipt/[BLOCKED]
    _read_calls: int = 0            # every read of _read_target, progress or not
    _read_spans: list[tuple[int, int]] = field(default_factory=list)
    _read_warned: bool = False

    @property
    def distinct_calls(self) -> int:
        return len(self._sig_counts)

    def top_signature(self) -> tuple[str, int] | None:
        if not self._sig_counts:
            return None
        sig, count = max(self._sig_counts.items(), key=lambda kv: kv[1])
        return sig, count

    def observe(
        self,
        tool_name: str,
        tool_input: dict | None,
        *,
        productive: bool | None = None,
        result_text: str | None = None,
        read_target: object = READ_TARGET_UNSET,
        paid: bool = False,
    ) -> GuardVerdict:
        """Record a tool call and return what the loop should do about it.

        ``result_text`` and ``read_target`` feed Signal C: the former lets the
        guard tell a served range from a receipt, the latter is the resolved
        target path for ``observation_read`` (whose ref the guard cannot map on
        its own). ``paid`` marks a billed MCP call, which trips Signal A one
        repeat earlier and says so in the warning. All are optional so
        existing callers keep working.
        """
        if productive is None:
            productive = tool_name in PRODUCTIVE_TOOLS

        self.total_calls += 1
        if productive:
            self.productive_calls += 1

        sig = call_signature(tool_name, tool_input)
        count = self._sig_counts.get(sig, 0) + 1
        self._sig_counts[sig] = count

        if self._fired:
            # Already escalated once — stay quiet, the loop is being torn down.
            return GuardVerdict()

        # Signal A.1 — the exact same call, over and over. A billed call gets
        # one repeat less rope than a free one (Stage 10).
        dup_threshold = self.paid_dup_threshold if paid else self.dup_threshold
        if count > dup_threshold:
            self._fired = True
            return GuardVerdict(
                stall=True, reason="dup_call", tool_name=tool_name, repeat_count=count
            )
        if count == dup_threshold and sig not in self._warned:
            self._warned.add(sig)
            cost = (
                " Every one of those calls was billed to the owner's credit balance."
                if paid else ""
            )
            return GuardVerdict(
                warn=(
                    f"You have now made this exact `{tool_name}` call "
                    f"{count} times in this session with the same arguments."
                    f"{cost} Repeating it will not change the result. Before your next "
                    "tool call, in plain text: (1) state what you are still "
                    "missing, (2) list 2-3 genuinely different ways to get it "
                    "or to move the task forward without it, (3) pick one and "
                    "act on it. Do not run this call again."
                ),
                reason="dup_call",
                tool_name=tool_name,
                repeat_count=count,
            )

        # Signal A.2 — lots of calls, all different, none productive: the agent
        # is thrashing through variations instead of converging.
        if (
            self.total_calls >= self.min_calls_for_ratio
            and self.productive_calls == 0
            and self.distinct_calls / self.total_calls < self.distinct_ratio_floor
        ):
            self._fired = True
            return GuardVerdict(
                stall=True, reason="low_diversity", tool_name=tool_name, repeat_count=count
            )

        # Signal C — stuck re-reading one file (different offset/limit/ref each
        # time, so Signal A never matched).
        return self._observe_read_loop(tool_name, tool_input, result_text, read_target) \
            or GuardVerdict()

    def _reset_read_loop(self) -> None:
        self._read_target = None
        self._read_streak = 0
        self._read_blocked_streak = 0
        self._read_calls = 0
        self._read_spans = []
        self._read_warned = False

    def _observe_read_loop(
        self,
        tool_name: str,
        tool_input: dict | None,
        result_text: str | None,
        read_target: object,
    ) -> GuardVerdict | None:
        if tool_name not in READ_LOOP_TOOLS:
            self._reset_read_loop()
            return None

        if read_target is READ_TARGET_UNSET:
            target = read_loop_target(tool_name, tool_input)
        elif read_target and not str(read_target).startswith(_OPAQUE_TARGET_PREFIXES):
            target = _norm_target_path(read_target)
        else:
            target = read_target or None

        if not target:
            # Whole-workspace grep / unresolvable ref — not a single-file
            # fixation; treat it like any other tool and reset the streak.
            self._reset_read_loop()
            return None

        if target != self._read_target:
            self._reset_read_loop()
            self._read_target = target

        self._read_calls += 1
        blocked = looks_like_blocked_read(result_text)
        span = parse_read_span(result_text)
        progressed = False

        if blocked:
            self._read_blocked_streak += 1
            self._read_streak += 1
        elif span is not None:
            if any(_spans_overlap(span, seen) for seen in self._read_spans):
                self._read_streak += 1
            else:
                progressed = True
                self._read_streak = 0
            # Keep only recent ranges: an overlap loop repeats what it just
            # read, and this bounds work on a genuinely long paging session.
            self._read_spans = [*self._read_spans[-39:], span]
        else:
            # grep (no line range) or a content-less read of the same target.
            self._read_streak += 1

        # Even clean sequential paging of one file has a ceiling — a model that
        # reads the same file 18+ times in a row is not doing real work.
        hard_ceiling = self.same_target_read_limit * 3

        if self._read_blocked_streak >= self.blocked_target_read_limit:
            self._fired = True
            return GuardVerdict(
                stall=True,
                reason="stuck_read_blocked",
                tool_name=tool_name,
                repeat_count=self._read_blocked_streak,
                target=self._read_target,
            )
        if self._read_streak >= self.same_target_read_limit or (
            not progressed and self._read_calls >= hard_ceiling
        ):
            self._fired = True
            return GuardVerdict(
                stall=True,
                reason="stuck_read",
                tool_name=tool_name,
                repeat_count=max(self._read_streak, self._read_calls),
                target=self._read_target,
            )

        # One-shot nudge two no-progress reads before the stall.
        if (
            not self._read_warned
            and self._read_streak >= 2
            and self._read_streak >= self.same_target_read_limit - 2
        ):
            self._read_warned = True
            return GuardVerdict(
                warn=(
                    f"You have re-read `{self._read_target}` {self._read_streak} times "
                    "in a row without acting on it — the same target, just a "
                    "different range or argument. Reading it again will not tell you "
                    "anything new. State in plain text what you still need from it, "
                    "then either search for that exact thing or move to your next "
                    "real action (edit/write/run something)."
                ),
                reason="stuck_read",
                tool_name=tool_name,
                repeat_count=self._read_streak,
                target=self._read_target,
            )
        return None

    def summary_line(self) -> str:
        top = self.top_signature()
        top_txt = f"{top[0]} x{top[1]}" if top else "n/a"
        return (
            f"{self.total_calls} tool calls, {self.distinct_calls} distinct, "
            f"{self.productive_calls} productive (file writes); "
            f"most-repeated: {top_txt}"
        )


# --- Message the stuck agent "sends" to the ChatMaster --------------------

_REASON_LINE = {
    "dup_call": (
        "I keep making the same `{tool_name}` call ({repeat_count}x with identical "
        "arguments) and it is not getting me anywhere."
    ),
    "low_diversity": (
        "I have made many different discovery calls ({summary}) and still have not "
        "written anything or made real progress."
    ),
    "restart_streak": (
        "This task has been restarted {restart_streak} times and every attempt has "
        "ended without a single file change."
    ),
    "stuck_read": (
        "I have re-read `{target}` {repeat_count} times in a row (same file, "
        "different offset/limit/ref) without editing anything or moving on — I "
        "keep re-running the same analysis instead of making progress."
    ),
    "stuck_read_blocked": (
        "I keep re-requesting `{target}` ({repeat_count}x) even though the "
        "workspace is now refusing to serve it again (duplicate-read receipts / "
        "[BLOCKED])."
    ),
}


def build_stall_report(signal: dict) -> str:
    """Plain-text message handed to the ChatMaster when the agent loop stalls.

    System-generated from objective loop stats — it does not speak for the
    model. The ChatMaster answers it as a normal mid-task turn (it may use its
    own tools to look at the workspace first).
    """
    reason = signal.get("reason") or "dup_call"
    line = _REASON_LINE.get(reason, _REASON_LINE["dup_call"]).format(
        tool_name=signal.get("tool_name") or "the same",
        repeat_count=signal.get("repeat_count") or 0,
        summary=signal.get("summary") or "repeated reads/greps",
        restart_streak=signal.get("restart_streak") or 0,
        target=signal.get("target") or "the same file",
    )
    return (
        "[AUTOMATIC STALL REPORT — the agent working on this task is looping "
        "without progress.]\n\n"
        f"{line}\n"
        f"Loop stats: {signal.get('summary') or 'n/a'}.\n\n"
        "The agent has been stopped before it burns more calls on the same "
        "approach. Give it a way forward: a concrete next step it has not "
        "tried, or a choice between 2-3 different approaches, or — if the task "
        "wording itself is the blocker (ambiguous, or it references something "
        "that does not exist in the workspace) — the corrected description via "
        "task_update."
    )
