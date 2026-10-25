"""Lifecycle limits for user-facing sessions.

internal design notes stage 1: a `user_cm_*` / `user_agent_*`
session used to have no upper bound at all — no age, no turn count, no window
share — so one session absorbed every message a room ever received (observed:
8 days open, 185 messages, 3.18M cumulative input tokens).

This module owns the thresholds and the pure decision function, so
``user_sessions`` (rotation before a turn) and ``watchdog`` (idle close) apply
the same numbers and share one kill switch.  Values are read from the
environment on every call: they must be tunable on a live deployment and
overridable per test.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

USER_SESSION_KINDS = ("user_agent_p1", "user_agent_p2", "user_cm_p1", "user_cm_p2")

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

# Rotation reasons, mirrored into the SESSION_CLOSED metadata.
ROTATE_MAX_TURNS = "max_turns"
ROTATE_MAX_AGE = "max_age"
ROTATE_WINDOW_RATIO = "window_ratio"


def _env_number(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (AttributeError, ValueError):
        logger.warning("user_session_limits: invalid %s=%r — using %s", name, raw, default)
        return default


def lifecycle_enforced() -> bool:
    """Master switch for stage 1 (rotation, idle close, orphan cleanup)."""
    raw = os.getenv("USER_SESSION_LIFECYCLE_ENFORCE", "true")
    return str(raw).strip().casefold() in _TRUTHY


def max_turns() -> int:
    """Messages in ``context_snapshot.messages`` before the session rotates.

    Raised 40 → 80 (2026-09-06): a direct ``@agent`` build session routinely
    spends dozens of turns on one feature, and rotating it out mid-work is how
    a reply like "finish" to the agent's own question ended up cold-routed to
    a fresh ChatMaster (see ``waiting_reply_safeguard``).  Age / idle / window
    limits still bound the session.
    """
    return max(2, int(_env_number("USER_SESSION_MAX_TURNS", 80)))


def max_age() -> timedelta:
    return timedelta(hours=max(0.05, _env_number("USER_SESSION_MAX_AGE_HOURS", 24)))


def idle_timeout() -> timedelta:
    return timedelta(minutes=max(1.0, _env_number("USER_SESSION_IDLE_MINUTES", 90)))


def window_soft_ratio() -> float:
    """Share of ``context_window_limit`` a single call may reach before rotating."""
    return min(1.0, max(0.05, _env_number("USER_SESSION_WINDOW_SOFT_RATIO", 0.5)))


def router_continue_enabled() -> bool:
    """Stage 2 switch: may the router answer ``continue_session`` at all?

    ``ROUTER_CONTINUE_SESSION=false`` rolls back to "every message starts a new
    session" — the open session is still closed gracefully, never abandoned.
    """
    raw = os.getenv("ROUTER_CONTINUE_SESSION", "true")
    return str(raw).strip().casefold() in _TRUTHY


def survive_restart() -> bool:
    """May a still-young user session stay open across a backend restart?

    A user session has no coroutine of its own — it lives between HTTP/WS turns,
    its whole state is ``context_snapshot`` in the DB — so a restart orphans
    nothing. Closing every one at startup (the stage 1.4 behaviour) cut the
    conversation the user was in the middle of: the next message found no open
    session, the router had nothing to continue, and a cold ChatMaster started
    from scratch. ``USER_SESSION_SURVIVE_RESTART=false`` restores the old
    close-everything behaviour.
    """
    raw = os.getenv("USER_SESSION_SURVIVE_RESTART", "true")
    return str(raw).strip().casefold() in _TRUTHY


def trivial_reply_chars() -> int:
    """Replies at or below this length skip the router LLM call entirely."""
    return max(0, int(_env_number("ROUTER_TRIVIAL_REPLY_CHARS", 40)))


def rotation_reason(
    *,
    message_count: int,
    started_at: datetime | None,
    last_window_tokens: int = 0,
    window_limit: int = 0,
    now: datetime | None = None,
) -> str | None:
    """Return why this session must rotate before the next turn, else None.

    Pure on purpose: the caller supplies the session's numbers, so the policy
    is testable without a session object, a database or a provider.
    """
    if not lifecycle_enforced():
        return None

    if message_count >= max_turns():
        return ROTATE_MAX_TURNS

    if started_at is not None:
        now = now or datetime.now(timezone.utc)
        started = started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if started < now - max_age():
            return ROTATE_MAX_AGE

    if window_limit > 0 and last_window_tokens > window_soft_ratio() * window_limit:
        return ROTATE_WINDOW_RATIO

    return None


def is_continuable(
    *,
    message_count: int,
    started_at: datetime | None,
    last_activity_at: datetime | None = None,
    last_window_tokens: int = 0,
    window_limit: int = 0,
    now: datetime | None = None,
    has_open_question: bool = False,
) -> bool:
    """May an open session still be offered to the router as ``continue_session``?

    Stage 2 pre-check (§2.2): only a *young* session is a candidate.  Young means
    it would not rotate on the next turn and it has not been silent longer than
    the idle timeout — the same numbers the rotation and the watchdog use, so the
    three never disagree about whether a session is alive.

    ``has_open_question`` (waiting_reply_safeguard S3): the last assistant turn
    put a question to the user, or the session's task is parked waiting for one.
    The user's answer belongs in *this* session, so a size-based limit
    (``max_turns`` / ``window_ratio``) is waived for the one turn that carries
    it — but a genuinely stale session (past ``max_age`` or silent past the idle
    timeout) is still closed: the question is carried forward by
    ``waiting_reply`` instead.
    """
    now = now or datetime.now(timezone.utc)

    reason = rotation_reason(
        message_count=message_count,
        started_at=started_at,
        last_window_tokens=last_window_tokens,
        window_limit=window_limit,
        now=now,
    )
    if reason is not None and not (
        has_open_question and reason in (ROTATE_MAX_TURNS, ROTATE_WINDOW_RATIO)
    ):
        return False

    if lifecycle_enforced() and last_activity_at is not None:
        last = last_activity_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if last < now - idle_timeout():
            return False

    return True
