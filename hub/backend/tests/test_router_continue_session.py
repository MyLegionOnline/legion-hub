"""Stage 2 of internal design notes — the router is back.

Before this stage an open user session short-circuited routing entirely
(`chat.py` returned before `route_user_message`), which is why a room with eight
days of traffic had exactly one row in `router_decision_log` (defect C1). Now
"continue this session" is a route the router picks — or a deterministic
shortcut takes for trivially short replies — and any other route closes the open
session gracefully instead of leaving it to grow.

Covered here:
  * 2.1 no message bypasses the router
  * 2.2 the open session is offered as context; continue_session_id is ours to set
  * 2.3 "ok" continues without paying for an LLM call
  * 2.4 every message leaves exactly one router_decision_log row; fallbacks are loud
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.router import (
    OpenSessionContext,
    RouteDecision,
    route_user_message,
)
from app.core.user_session_limits import is_continuable
from app.models.room import Room
from app.models.router_log import RouterDecisionLog

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _room() -> Room:
    return Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="test-room")


def _db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    return db


def _open_ctx(
    session_id: uuid.UUID | None = None, *, has_open_question: bool = False
) -> OpenSessionContext:
    return OpenSessionContext(
        session_id=session_id or uuid.uuid4(),
        kind="user_cm_p2",
        turns=6,
        summary_so_far="discussing blog SEO",
        last_assistant_excerpt="I suggest starting with sitemap.xml. Agreed?",
        has_open_question=has_open_question,
    )


def _llm(route: str, **extra: Any) -> tuple[str, int, int, str]:
    payload = {
        "route": route,
        "mentioned_agent_ids": [],
        "can_split": None,
        "split_messages": None,
        "matched_task_id": None,
        "matched_goal_id": None,
        "confidence": 0.9,
        "reasoning": "test",
    }
    payload.update(extra)
    return json.dumps(payload), 10, 5, "claude-haiku-4-5-20251001"


def _log_rows(db: AsyncMock) -> list[RouterDecisionLog]:
    return [
        call.args[0]
        for call in db.add.call_args_list
        if isinstance(call.args[0], RouterDecisionLog)
    ]


# ---------------------------------------------------------------------------
# 2.2 — which sessions may be continued at all
# ---------------------------------------------------------------------------


class TestContinuablePolicy:
    def test_a_young_session_is_continuable(self):
        assert is_continuable(message_count=6, started_at=NOW, now=NOW) is True

    def test_a_long_session_is_not(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        assert is_continuable(message_count=40, started_at=NOW, now=NOW) is False

    def test_an_old_session_is_not(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "24")
        assert (
            is_continuable(message_count=2, started_at=NOW - timedelta(hours=25), now=NOW)
            is False
        )

    def test_a_silent_session_is_not(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_IDLE_MINUTES", "90")
        assert (
            is_continuable(
                message_count=2,
                started_at=NOW,
                last_activity_at=NOW - timedelta(minutes=120),
                now=NOW,
            )
            is False
        )

    def test_a_wide_window_is_not(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        assert (
            is_continuable(
                message_count=2,
                started_at=NOW,
                last_window_tokens=70_000,
                window_limit=100_000,
                now=NOW,
            )
            is False
        )


# ---------------------------------------------------------------------------
# 2.3 — the deterministic shortcut
# ---------------------------------------------------------------------------


class TestTrivialShortcut:
    @pytest.mark.asyncio
    async def test_ok_continues_without_an_llm_call(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch("app.core.router._call_router_llm", new=AsyncMock()) as llm,
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(room, "ok", db, open_session=ctx)

        llm.assert_not_awaited()
        assert decision.route == "continue_session"
        assert decision.continue_session_id == ctx.session_id

    @pytest.mark.asyncio
    async def test_shortcut_still_leaves_a_log_row(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch("app.core.router._call_router_llm", new=AsyncMock()),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            await route_user_message(room, "yes, let's go", db, open_session=ctx)

        rows = _log_rows(db)
        assert len(rows) == 1
        assert rows[0].route == "continue_session"
        assert rows[0].outcome == "shortcut"
        assert rows[0].open_session_id == ctx.session_id
        assert rows[0].cost_usd == 0

    @pytest.mark.asyncio
    async def test_a_long_message_is_not_a_shortcut(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("chatmaster_p2")),
            ) as llm,
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room,
                "and now let's move on to a completely different topic — payment integration",
                db,
                open_session=ctx,
            )

        llm.assert_awaited_once()
        assert decision.route == "chatmaster_p2"

    @pytest.mark.asyncio
    async def test_a_mention_is_never_a_shortcut(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("single_agent_p2")),
            ) as llm,
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            await route_user_message(room, "@backend ok", db, open_session=ctx)

        llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_without_an_open_session_there_is_no_shortcut(self):
        room, db = _room(), _db()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("chatmaster_p2")),
            ) as llm,
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(room, "ok", db)

        llm.assert_awaited_once()
        assert decision.route == "chatmaster_p2"


# ---------------------------------------------------------------------------
# 2.2 — continue_session as a routing decision
# ---------------------------------------------------------------------------


class TestContinueRoute:
    @pytest.mark.asyncio
    async def test_the_open_session_reaches_the_prompt(self):
        room, db, ctx = _room(), _db(), _open_ctx()
        captured: dict[str, Any] = {}

        async def _capture(user_msg, cfg, or_session_id=None, **kwargs):
            captured["user_msg"] = user_msg
            captured["system_prompt"] = kwargs.get("system_prompt")
            return _llm("continue_session")

        with (
            patch("app.core.router._call_router_llm", new=_capture),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            await route_user_message(
                room,
                "and what about the second item of the plan that we discussed above?",
                db,
                open_session=ctx,
            )

        assert "continue_session" in captured["system_prompt"]
        assert str(ctx.session_id) in captured["user_msg"]
        assert "discussing blog SEO" in captured["user_msg"]

    @pytest.mark.asyncio
    async def test_the_continued_id_is_ours_not_the_models(self):
        room, db, ctx = _room(), _db(), _open_ctx()
        hallucinated = str(uuid.uuid4())

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(
                    return_value=_llm("continue_session", continue_session_id=hallucinated)
                ),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room, "let's continue what we talked about above, and clarify the details", db, open_session=ctx
            )

        assert decision.continue_session_id == ctx.session_id

    @pytest.mark.asyncio
    async def test_continue_without_an_open_session_is_coerced(self):
        """A route the caller cannot fulfil must never leave the router."""
        room, db = _room(), _db()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("continue_session")),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room, "let's continue what we talked about last time, please", db
            )

        assert decision.route == "chatmaster_p2"
        assert decision.continue_session_id is None

    @pytest.mark.asyncio
    async def test_reply_to_an_open_question_is_coerced_to_continue(self):
        """The assistant asked something, the reply has no @mention and the
        router still returned chatmaster_p2 — a misread answer. Coerce it back."""
        room, db = _room(), _db()
        ctx = _open_ctx(has_open_question=True)

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(
                    return_value=_llm("chatmaster_p2", confidence=0.85)
                ),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room,
                "1) permanently 2) take a ready one from vue-bits 3) a build step in CI, npm",
                db,
                open_session=ctx,
            )

        assert decision.route == "continue_session"
        assert decision.continue_session_id == ctx.session_id

    @pytest.mark.asyncio
    async def test_open_question_does_not_override_an_explicit_topic_drop(self):
        """Only the user saying so outright ("forget it", "let's talk about something else") keeps
        a fresh route in place while a question is open — not the router's
        confidence, which is not trustworthy here."""
        room, db = _room(), _db()
        ctx = _open_ctx(has_open_question=True)

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("chatmaster_p2", confidence=0.95)),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room, "stop, forget it, let's now do a completely different big project", db,
                open_session=ctx,
            )

        assert decision.route == "chatmaster_p2"

    @pytest.mark.asyncio
    async def test_open_question_coerces_regardless_of_confidence(self):
        """project_plan_route_and_decompose_fix (2026-09-08#2): an answer that
        quotes the CM's own question and ends with an imperative ("...describe X,
        draw up the plan") scored 0.95 as a fresh task and cold-rotated a
        user_cm_p2 session mid requirement-gathering. The confidence ceiling is
        gone — with an open question and no @mention and no explicit topic drop,
        the turn stays in the session."""
        room, db = _room(), _db()
        ctx = _open_ctx(has_open_question=True)

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("chatmaster_p2", confidence=0.95)),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room,
                "describe pull the first time with a token and on update without a token. "
                "draw up the plan",
                db,
                open_session=ctx,
            )

        assert decision.route == "continue_session"
        assert decision.continue_session_id == ctx.session_id

    @pytest.mark.asyncio
    async def test_open_question_override_never_fires_with_a_mention(self):
        room, db = _room(), _db()
        ctx = _open_ctx(has_open_question=True)

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("single_agent_p2", confidence=0.7)),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room, "@backend yes, let's go with this option", db, open_session=ctx
            )

        assert decision.route == "single_agent_p2"

    @pytest.mark.asyncio
    async def test_open_question_flag_reaches_the_prompt(self):
        room, db = _room(), _db()
        ctx = _open_ctx(has_open_question=True)
        captured: dict[str, Any] = {}

        async def _capture(user_msg, cfg, or_session_id=None, **kwargs):
            captured["user_msg"] = user_msg
            return _llm("continue_session")

        with (
            patch("app.core.router._call_router_llm", new=_capture),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            await route_user_message(
                room, "a fairly long answer to three questions of the assistant at once", db,
                open_session=ctx,
            )

        assert "asked the user a QUESTION" in captured["user_msg"]

    @pytest.mark.asyncio
    async def test_a_normal_route_never_carries_a_session_id(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(
                    return_value=_llm(
                        "chatmaster_p2", continue_session_id=str(ctx.session_id)
                    )
                ),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            decision = await route_user_message(
                room, "let's discuss a completely new big project topic", db, open_session=ctx
            )

        assert decision.continue_session_id is None


class TestOpenQuestionFromFeed:
    """incident 2026-09-08#3: the CM asked its clarifying questions through the
    force_text retry after a premature close_session on turn 1. That reply
    reached the room feed but not the session's persisted history when the next
    turn's router read it — so `has_open_question` came back False and the answer
    cold-routed to a fresh ChatMaster. open_session_context now falls back to the
    last agent message in the room feed."""

    def _session(self) -> MagicMock:
        s = MagicMock()
        s.id = uuid.uuid4()
        s.room_id = uuid.uuid4()
        s.kind = "user_cm_p2"
        s.is_open = True
        s.last_context_window_tokens = 0
        _now = datetime.now(timezone.utc)
        s.started_at = _now - timedelta(minutes=3)
        s.last_activity_at = _now - timedelta(seconds=20)
        s.chat_master_agent_id = uuid.uuid4()
        s.agent_id = None
        s.goal_id = None
        s.task_id = None
        s.summary = ""
        # only the user's opening turn is persisted — the CM's question is not
        s.context_snapshot = {"messages": [{"role": "user", "content": "site deploy"}]}
        return s

    @pytest.mark.asyncio
    async def test_feed_question_lifts_has_open_question(self):
        from app.core.user_sessions import open_session_context

        session = self._session()
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = (
            "Rewrite the README entirely for the deploy, or add a new section?"
        )
        db.execute = AsyncMock(return_value=result)

        ctx = await open_session_context(session, db)

        assert ctx is not None
        assert ctx.has_open_question is True

    @pytest.mark.asyncio
    async def test_feed_without_a_question_does_not(self):
        from app.core.user_sessions import open_session_context

        session = self._session()
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = "Done, README updated."
        db.execute = AsyncMock(return_value=result)

        ctx = await open_session_context(session, db)

        assert ctx is not None
        assert ctx.has_open_question is False


# ---------------------------------------------------------------------------
# 2.4 — observability
# ---------------------------------------------------------------------------


class TestRouterObservability:
    @pytest.mark.asyncio
    async def test_a_failed_router_keeps_the_conversation_and_says_so(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(side_effect=RuntimeError("provider down")),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()) as emit,
            patch("app.core.metrics.record_route_fallback") as metric,
        ):
            decision = await route_user_message(
                room, "a fairly long message about the project state", db, open_session=ctx
            )

        assert decision.route == "continue_session"
        assert decision.continue_session_id == ctx.session_id
        assert "ROUTE_LLM_FAILURE" in [
            c.kwargs["event_type"] for c in emit.call_args_list
        ]
        metric.assert_called_once()
        assert _log_rows(db)[0].outcome == "fallback"

    @pytest.mark.asyncio
    async def test_a_failed_router_without_an_open_session_still_falls_back_to_p2(self):
        room, db = _room(), _db()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(side_effect=RuntimeError("provider down")),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
            patch("app.core.metrics.record_route_fallback"),
        ):
            decision = await route_user_message(
                room, "a fairly long message about the project state", db
            )

        assert decision.route == "chatmaster_p2"

    @pytest.mark.asyncio
    async def test_one_log_row_per_message(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch(
                "app.core.router._call_router_llm",
                new=AsyncMock(return_value=_llm("chatmaster_p2")),
            ),
            patch("app.core.router.emit_event", new=AsyncMock()),
        ):
            await route_user_message(room, "ok", db, open_session=ctx)          # shortcut
            await route_user_message(
                room, "let's discuss a completely new big project topic", db, open_session=ctx
            )
            await route_user_message(
                room, "and one more completely separate topic with no open session", db
            )

        rows = _log_rows(db)
        assert len(rows) == 3
        assert [r.outcome for r in rows] == ["shortcut", "llm", "llm"]

    @pytest.mark.asyncio
    async def test_route_decided_metadata_names_the_open_session(self):
        room, db, ctx = _room(), _db(), _open_ctx()

        with (
            patch("app.core.router._call_router_llm", new=AsyncMock()),
            patch("app.core.router.emit_event", new=AsyncMock()) as emit,
        ):
            await route_user_message(room, "ok", db, open_session=ctx)

        decided = next(
            c for c in emit.call_args_list if c.kwargs["event_type"] == "ROUTE_DECIDED"
        )
        assert decided.kwargs["metadata"]["open_session_id"] == str(ctx.session_id)
        assert decided.kwargs["metadata"]["shortcut"] is True


# ---------------------------------------------------------------------------
# 2.1 — the group-room dispatcher no longer bypasses the router
# ---------------------------------------------------------------------------


def _group_room() -> Room:
    return Room(
        id=uuid.uuid4(),
        creator_id=uuid.uuid4(),
        name="group",
        coordinator_config={"agent_id": str(uuid.uuid4())},
    )


def _open_session(session_id: uuid.UUID | None = None) -> MagicMock:
    session = MagicMock()
    session.id = session_id or uuid.uuid4()
    session.kind = "user_cm_p2"
    session.is_open = True
    return session


class _DispatchEnv:
    """Patch every collaborator of _dispatch_group_route; expose the mocks."""

    def __init__(self, *, existing: MagicMock | None, open_ctx: OpenSessionContext | None,
                 decision: RouteDecision) -> None:
        repo = MagicMock()
        repo.find_open_user_session = AsyncMock(return_value=existing)
        self.repo = repo
        self.open_ctx = open_ctx
        self.decision = decision
        self._patches: list[Any] = []

    def __enter__(self):
        targets = {
            "app.repositories.session.SessionRepository": MagicMock(return_value=self.repo),
            "app.core.user_sessions.open_session_context": AsyncMock(return_value=self.open_ctx),
            "app.core.user_sessions.close_user_session_gracefully": AsyncMock(return_value="summary"),
            "app.api.v1.chat.route_user_message": AsyncMock(return_value=self.decision),
            "app.api.v1.chat.process_user_turn": AsyncMock(return_value=True),
            "app.api.v1.chat.start_user_session": AsyncMock(return_value=_open_session()),
        }
        self.mocks: dict[str, Any] = {}
        for target, replacement in targets.items():
            patcher = patch(target, new=replacement)
            patcher.start()
            self._patches.append(patcher)
            self.mocks[target.rsplit(".", 1)[1]] = replacement
        return self

    def __exit__(self, *exc: Any) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()


class TestGroupDispatch:
    @pytest.mark.asyncio
    async def test_every_message_goes_through_the_router(self):
        """Defect C1: an open session used to return before the router ran."""
        from app.api.v1.chat import _dispatch_group_route

        room, user, db = _group_room(), MagicMock(id=uuid.uuid4()), _db()
        existing = _open_session()
        ctx = _open_ctx(existing.id)
        decision = RouteDecision(
            route="continue_session",
            continue_session_id=existing.id,
            confidence=1.0,
            reasoning="same topic",
        )

        with _DispatchEnv(existing=existing, open_ctx=ctx, decision=decision) as env:
            await _dispatch_group_route(room, user, "ok", db)

        env.mocks["route_user_message"].assert_awaited_once()
        assert env.mocks["route_user_message"].await_args.kwargs["open_session"] is ctx
        env.mocks["process_user_turn"].assert_awaited_once()
        assert env.mocks["process_user_turn"].await_args.args[0] is existing
        env.mocks["close_user_session_gracefully"].assert_not_awaited()
        env.mocks["start_user_session"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_topic_change_closes_the_old_session_and_opens_a_new_one(self):
        from app.api.v1.chat import _dispatch_group_route

        room, user, db = _group_room(), MagicMock(id=uuid.uuid4()), _db()
        existing = _open_session()
        decision = RouteDecision(route="chatmaster_p2", confidence=0.9, reasoning="new topic")

        with _DispatchEnv(
            existing=existing, open_ctx=_open_ctx(existing.id), decision=decision
        ) as env:
            await _dispatch_group_route(room, user, "and now let's talk about billing", db)

        close = env.mocks["close_user_session_gracefully"]
        close.assert_awaited_once()
        assert close.await_args.args[0] is existing
        assert close.await_args.kwargs["rotation_reason"] == "topic_change"
        env.mocks["start_user_session"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_over_limit_session_is_closed_and_not_offered_to_the_router(self):
        from app.api.v1.chat import _dispatch_group_route

        room, user, db = _group_room(), MagicMock(id=uuid.uuid4()), _db()
        existing = _open_session()
        decision = RouteDecision(route="chatmaster_p2", confidence=0.9, reasoning="fresh")

        with _DispatchEnv(existing=existing, open_ctx=None, decision=decision) as env:
            await _dispatch_group_route(room, user, "ok", db)

        assert env.mocks["route_user_message"].await_args.kwargs["open_session"] is None
        close = env.mocks["close_user_session_gracefully"]
        close.assert_awaited_once()
        assert close.await_args.kwargs["rotation_reason"] == "not_continuable"

    @pytest.mark.asyncio
    async def test_a_stale_continue_decision_never_resurrects_another_session(self):
        """continue_session for an id we did not offer must not be honoured."""
        from app.api.v1.chat import _dispatch_group_route

        room, user, db = _group_room(), MagicMock(id=uuid.uuid4()), _db()
        existing = _open_session()
        decision = RouteDecision(
            route="continue_session",
            continue_session_id=uuid.uuid4(),
            confidence=1.0,
            reasoning="stale",
        )

        with _DispatchEnv(
            existing=existing, open_ctx=_open_ctx(existing.id), decision=decision
        ) as env:
            await _dispatch_group_route(room, user, "ok", db)

        # Not answered in the wrong session — but answered: the turn falls back
        # to a fresh ChatMaster session rather than disappearing.
        env.mocks["close_user_session_gracefully"].assert_awaited_once()
        env.mocks["start_user_session"].assert_awaited_once()
        turn_session = env.mocks["process_user_turn"].await_args.args[0]
        assert turn_session is not existing

    @pytest.mark.asyncio
    async def test_without_an_open_session_nothing_is_closed(self):
        from app.api.v1.chat import _dispatch_group_route

        room, user, db = _group_room(), MagicMock(id=uuid.uuid4()), _db()
        decision = RouteDecision(route="chatmaster_p2", confidence=0.9, reasoning="new")

        with _DispatchEnv(existing=None, open_ctx=None, decision=decision) as env:
            await _dispatch_group_route(room, user, "hi, let's start a project", db)

        env.mocks["close_user_session_gracefully"].assert_not_awaited()
        env.mocks["start_user_session"].assert_awaited_once()
        env.mocks["process_user_turn"].assert_awaited_once()
