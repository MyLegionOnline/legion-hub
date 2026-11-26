"""Near-identical agent replies in a row (app/core/repeat_guard.py)."""
from __future__ import annotations

from app.core.repeat_guard import RepeatGuard, near_duplicate, repeat_streak

_LOOP = (
    "environment failure: workspace file tools (`read_file`, `edit_file`, `git_diff`, "
    "`close_session`) not declared for this agent instance — task requires restart on a "
    "fresh instance with the workspace/MCP file tool group attached."
)


def test_small_edits_still_count_as_the_same_reply():
    assert near_duplicate(_LOOP, _LOOP.replace("fresh instance", "new instance"))
    assert near_duplicate("<thinking> x </thinking> Understood.", "UNDERSTOOD.")


def test_different_replies_are_not_duplicates():
    assert not near_duplicate(_LOOP, "Rewrote chapter 3 in episodes/007/007_ru.md, lines 51-104.")
    assert not near_duplicate("", "Done.")


def test_streak_counts_only_the_tail():
    replies = ["Read the file.", _LOOP, _LOOP + " Again.", _LOOP]
    assert repeat_streak(replies) == 3
    assert repeat_streak([]) == 0
    assert repeat_streak([_LOOP, "Something else entirely, chapter rewritten."]) == 1


def test_guard_trips_at_the_limit_and_resets_on_a_new_reply():
    guard = RepeatGuard(limit=3)
    assert guard.observe(_LOOP) == 1
    assert guard.observe(_LOOP) == 2
    assert not guard.tripped
    assert guard.observe(_LOOP) == 3
    assert guard.tripped
    assert guard.observe("A completely different question about the bible files?") == 1
    assert not guard.tripped


def test_zero_limit_disables_the_guard():
    guard = RepeatGuard(limit=0)
    for _ in range(5):
        guard.observe(_LOOP)
    assert not guard.tripped


import pytest


@pytest.mark.asyncio
async def test_direct_session_is_closed_on_a_reply_loop():
    import uuid
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.core import user_sessions

    session = MagicMock()
    session.id = uuid.uuid4()
    session.room_id = uuid.uuid4()
    session.kind = "user_agent_p2"
    session.closed_at = None
    db = AsyncMock()
    db.add = MagicMock()
    with patch.object(user_sessions, "emit_event", new_callable=AsyncMock) as emit:
        await user_sessions._close_on_reply_loop(session, 8, db)

    assert session.closed_at is not None
    assert session.close_reason == "error"
    meta = emit.await_args.kwargs["metadata"]
    assert meta["reason_detail"] == "reply_repeat"
    assert meta["repeat_count"] == 8
    db.commit.assert_awaited_once()
