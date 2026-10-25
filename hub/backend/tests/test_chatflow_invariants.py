"""
Stage 6.1 of internal design notes — regression guards
for the chain as a whole.

Stages 1-5 each proved their own fix in isolation (rotation, the router,
the Board snapshot, the memory flush, the Sessions tab). This file asserts the
five load-bearing guarantees the whole plan promised (§6 "Target contract"),
so a future change to any one collaborator fails a test here before it quietly
reopens the immortal-session bug the plan started from:

  1. no user-facing session outlives USER_SESSION_MAX_AGE
  2. every user message leaves exactly one router_decision_log row
  3. a user_cm_p2 system prompt carries the Board snapshot, never a previous
     session's raw transcript
  4. closing a session — any close_reason — leaves non-empty Agent memory
  5. app startup never registers two routes for the same (method, path)

No database — every test builds its own scripted AsyncMock, matching the rest
of this stage's suite (test_user_session_lifecycle.py,
test_router_continue_session.py, test_session_close_memory_flush.py).
"""
from __future__ import annotations

import json
import types
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.memory_layer import build_memory_layer, build_system_prompt
from app.core.router import OpenSessionContext, route_user_message
from app.core.session_close import _KIND_ALLOWED, flush_session_memory
from app.core.user_session_limits import max_age
from app.core.user_sessions import _LLMTurnResult, process_user_turn
from app.models.room import Room
from app.models.router_log import RouterDecisionLog
from app.models.session.base import CLOSE_REASONS

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1 — no user session outlives USER_SESSION_MAX_AGE
# ---------------------------------------------------------------------------
# Harness mirrors tests/test_user_session_lifecycle.py's TestRotation — each
# test file in this stage builds its own scaffolding, there is no shared
# conftest fixture for the mocked session/db pair.


def _make_session(
    kind: str = "user_cm_p2", *, messages: int = 0, age_hours: float = 0.0,
    last_window: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.side = None
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = last_window
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.started_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    s.last_activity_at = None
    s.user_id = uuid.uuid4()
    s.agent_id = uuid.uuid4() if "agent" in kind else None
    s.chat_master_agent_id = uuid.uuid4() if "cm" in kind else None
    s.goal_id = None
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.context_snapshot = {
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(messages)
        ]
    }
    return s


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.get = AsyncMock(return_value=None)
    return db


def _text_result(text: str = "reply", tokens_in: int = 900) -> _LLMTurnResult:
    return _LLMTurnResult(
        text=text, tool_input=None, tokens_in=tokens_in, tokens_out=100,
        model="claude-haiku-4-5-20251001",
        max_window_in=tokens_in, last_window_in=tokens_in,
    )


def _close_result(summary: str = "dialogue summary") -> _LLMTurnResult:
    return _LLMTurnResult(
        text=None,
        tool_input={"agent_memory_patch": {"content": "state"}, "summary": summary},
        tokens_in=500, tokens_out=50, model="claude-haiku-4-5-20251001",
        max_window_in=500, last_window_in=500,
    )


@contextmanager
def _turn_env(llm_results: list[_LLMTurnResult], successor: MagicMock):
    calls = iter(llm_results)

    async def _fake_call(*_args, **_kwargs) -> _LLMTurnResult:
        return next(calls)

    repo_instance = AsyncMock()
    repo_instance.create = AsyncMock(return_value=successor)

    with ExitStack() as stack:
        for target, kwargs in (
            ("build_memory_layer", {"new_callable": AsyncMock, "return_value": {}}),
            ("build_system_prompt", {"return_value": "SYS"}),
            ("_get_agent_llm_config", {"new_callable": AsyncMock, "return_value": {}}),
            ("_call_llm_with_tools", {"side_effect": _fake_call}),
            ("apply_close_actions", {"new_callable": AsyncMock}),
            ("emit_event", {"new_callable": AsyncMock}),
            ("_context_window_limit", {"new_callable": AsyncMock, "return_value": 100_000}),
            ("SessionRepository", {"return_value": repo_instance}),
        ):
            stack.enter_context(patch(f"app.core.user_sessions.{target}", **kwargs))
        manager = stack.enter_context(patch("app.core.user_sessions.manager"))
        manager.broadcast = AsyncMock()
        yield repo_instance


@pytest.fixture(autouse=True)
def _no_workspace():
    with patch(
        "app.core.workspace_tools.workspace_tools_available",
        new_callable=AsyncMock, return_value=False,
    ):
        yield


class TestSessionNeverOutlivesMaxAge:
    """§6 contract #1 / §6.1 bullet 1."""

    @pytest.mark.asyncio
    async def test_an_overage_session_always_rotates_before_answering(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "24")
        session = _make_session("user_cm_p2", messages=4, age_hours=30)
        successor = _make_session("user_cm_p2")
        db = _make_db()

        with _turn_env([_close_result(), _text_result()], successor):
            still_open = await process_user_turn(session, "we continue in a week", db)

        # The turn is still answered — the user never sees a dead end.
        assert still_open is True
        assert session.closed_at is not None
        assert session.close_reason == "rotated"
        # The overage session never got to answer this turn itself — it was
        # retired instead. (Its wall-clock age *before* rotation may exceed
        # max_age — that's the silence the user left it in; what matters is it
        # can never process another turn once it does.)
        assert datetime.now(timezone.utc) - session.started_at > max_age()
        # Its successor starts a fresh clock; it must not inherit the overage.
        assert datetime.now(timezone.utc) - successor.started_at < max_age()

    @pytest.mark.asyncio
    async def test_a_chain_of_late_replies_rotates_independently_every_time(self, monkeypatch):
        """Several long silences in a row — each one must rotate on its own."""
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "6")
        db = _make_db()
        current = _make_session("user_cm_p2")

        for _ in range(5):
            current.started_at = datetime.now(timezone.utc) - timedelta(hours=30)
            successor = _make_session("user_cm_p2")
            with _turn_env([_close_result(), _text_result()], successor):
                await process_user_turn(current, "one more day", db)
            assert current.closed_at is not None
            assert current.close_reason == "rotated"
            current = successor
            assert datetime.now(timezone.utc) - current.started_at < max_age()

        assert datetime.now(timezone.utc) - current.started_at < max_age()


# ---------------------------------------------------------------------------
# 2 — every user message leaves exactly one router_decision_log row
# ---------------------------------------------------------------------------


def _room() -> Room:
    return Room(id=uuid.uuid4(), creator_id=uuid.uuid4(), name="test-room")


def _router_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    return db


def _open_ctx(session_id: uuid.UUID | None = None) -> OpenSessionContext:
    return OpenSessionContext(
        session_id=session_id or uuid.uuid4(),
        kind="user_cm_p2",
        turns=6,
        summary_so_far="discussing blog SEO",
        last_assistant_excerpt="I suggest starting with sitemap.xml. Agreed?",
    )


def _log_rows(db: AsyncMock) -> list[RouterDecisionLog]:
    return [
        call.args[0]
        for call in db.add.call_args_list
        if isinstance(call.args[0], RouterDecisionLog)
    ]


def _llm_payload(route: str, **extra) -> tuple[str, int, int, str]:
    payload = {
        "route": route, "mentioned_agent_ids": [], "can_split": None,
        "split_messages": None, "matched_task_id": None, "matched_goal_id": None,
        "confidence": 0.9, "reasoning": "test",
    }
    payload.update(extra)
    return json.dumps(payload), 10, 5, "claude-haiku-4-5-20251001"


class TestEveryMessageGetsExactlyOneRouterLogRow:
    """§6 contract #2 / §6.1 bullet 2 — a mixed run: shortcut, LLM routes, a
    provider failure, an @mention and a message with no open session at all.
    """

    @pytest.mark.asyncio
    async def test_a_mixed_run_leaves_one_row_per_message_with_the_right_outcome(self):
        room, db, ctx = _room(), _router_db(), _open_ctx()
        outcomes = iter([
            _llm_payload("chatmaster_p2"),
            RuntimeError("provider down"),
            _llm_payload("single_agent_p2"),
            _llm_payload("chatmaster_p2"),
        ])

        async def _fake_llm(*_args, **_kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        messages = [
            ("ok", ctx),
            ("let's discuss a completely new big project topic", ctx),
            ("a fairly long message about billing and payment", ctx),
            ("@backend a long message mentioning the agent directly", ctx),
            ("hi, this message came in without an open session today", None),
        ]

        with (
            patch("app.core.router._call_router_llm", new=_fake_llm),
            patch("app.core.router.emit_event", new=AsyncMock()),
            patch("app.core.metrics.record_route_fallback"),
        ):
            for text, open_session in messages:
                await route_user_message(room, text, db, open_session=open_session)

        rows = _log_rows(db)
        assert len(rows) == len(messages)
        assert [r.outcome for r in rows] == ["shortcut", "llm", "fallback", "llm", "llm"]


# ---------------------------------------------------------------------------
# 3 — user_cm_p2 gets the Board, never a previous session's raw transcript
# ---------------------------------------------------------------------------


class TestBoardSnapshotReplacesRawTranscript:
    """§6 contract #3 / §6.1 bullet 3 (the fix for defect M1)."""

    @pytest.mark.asyncio
    async def test_prompt_has_the_board_and_never_the_old_transcript(self):
        session = types.SimpleNamespace(
            kind="user_cm_p2",
            room_id=uuid.uuid4(),
            chat_master_agent_id=uuid.uuid4(),
            # Nothing in build_memory_layer/build_system_prompt should ever read
            # this — the assertion below is that raw transcript text can never
            # leak into the rendered prompt no matter what a session carries.
            context_snapshot={
                "messages": [
                    {"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"raw-transcript-line-{i}"}
                    for i in range(50)
                ]
            },
        )
        board = {
            "goals": [{
                "id": str(uuid.uuid4()), "title": "Blog SEO", "status": "active",
                "total": 2, "plan": None, "columns": {}, "collapsed": {},
            }],
            "without_goal": {"columns": {}, "collapsed": {}, "total": 0},
            "blocked_summary": [],
            "counters": {"total": 2, "active": 1, "blocked": 0, "completed": 1},
            "scan_truncated": False,
        }

        with patch.multiple(
            "app.core.memory_layer",
            _room_memory=AsyncMock(return_value=""),
            _repo_snapshot=AsyncMock(return_value=[]),
            _agents_list_short=AsyncMock(return_value=[]),
            _agent_memory=AsyncMock(return_value=""),
            _board_snapshot=AsyncMock(return_value=board),
            _list_of_goals=AsyncMock(return_value=[]),
        ):
            layer = await build_memory_layer(session, AsyncMock())
        prompt = build_system_prompt(layer, "user_cm_p2")

        assert "board_snapshot" in layer
        assert "messages" not in layer
        assert "## Task Board" in prompt
        for i in range(50):
            assert f"raw-transcript-line-{i}" not in prompt


# ---------------------------------------------------------------------------
# 4 — closing a session, any close_reason, leaves non-empty Agent memory
# ---------------------------------------------------------------------------


def _flush_session(kind: str, **overrides) -> object:
    common = dict(
        id=uuid.uuid4(), room_id=uuid.uuid4(), kind=kind, agent_id=None,
        chat_master_agent_id=None, task_id=None, context_snapshot=None,
        summary=None, side=None,
    )
    common.update(overrides)
    return types.SimpleNamespace(**common)


def _flush_result(first_row=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.first.return_value = first_row
    return m


def _flush_db(*results) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(side_effect=list(results))
    db.get = AsyncMock(return_value=None)
    return db


# Every kind whose §4 close-actions matrix allows agent_memory_patch —
# cm_self_plan is deliberately excluded, it never gets its own agent memory.
_KINDS_WITH_AGENT_MEMORY = [
    ("user_agent_p1", {"agent_id": uuid.uuid4()}),
    ("user_agent_p2", {"agent_id": uuid.uuid4()}),
    ("user_cm_p1", {"chat_master_agent_id": uuid.uuid4()}),
    ("user_cm_p2", {"chat_master_agent_id": uuid.uuid4()}),
    ("cm_agent_proxy", {"side": "agent", "agent_id": uuid.uuid4()}),
    ("cm_agent_proxy", {"side": "chat_master", "agent_id": uuid.uuid4()}),
    ("agent_subtask", {"agent_id": uuid.uuid4()}),
]


class TestEveryCloseReasonLeavesNonEmptyAgentMemory:
    """§6 contract #4 / §6.1 bullet 4 (the fix for defect M3)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reason", CLOSE_REASONS)
    @pytest.mark.parametrize("kind,attrs", _KINDS_WITH_AGENT_MEMORY)
    async def test_flush_never_leaves_agent_memory_empty(self, kind, attrs, reason):
        session = _flush_session(kind, summary=f"Summary for reason {reason}", **attrs)
        db = _flush_db(_flush_result(None), _flush_result(None))

        with patch("app.core.events.emit_event", new_callable=AsyncMock):
            result = await flush_session_memory(session, db, reason=reason)

        assert result is True
        added = [c.args[0] for c in db.add.call_args_list]
        assert added, f"{kind}/{reason} must write an AgentMemory row"
        content = added[-1].content
        assert content.strip()
        assert reason in content

    @pytest.mark.asyncio
    async def test_cm_self_plan_is_the_documented_exception(self):
        """cm_self_plan has no personal-memory slot in its close-actions matrix (§4)."""
        assert "agent_memory_patch" not in _KIND_ALLOWED["cm_self_plan"]
        session = _flush_session("cm_self_plan", chat_master_agent_id=uuid.uuid4())
        db = _flush_db()

        result = await flush_session_memory(session, db, reason="task_done")

        assert result is False
        db.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5 — no two routes claim the same (method, path)
# ---------------------------------------------------------------------------


def _registered_pairs(router, prefix: str = "") -> list:
    """Every (method, path) the app actually serves, flattened.

    FastAPI >= 0.141 no longer expands include_router() into app.routes: each
    call leaves a single lazy fastapi.routing._IncludedRouter there, so
    len(app.routes) collapsed from ~200 to 31 while the routes themselves stayed
    registered. Walk through those wrappers (and through plain sub-routers, the
    pre-0.141 layout) so the count keeps meaning the same thing on either
    version.
    """
    pairs = []
    for route in getattr(router, "routes", ()) or ():
        included = getattr(route, "original_router", None)
        if included is not None:  # fastapi >= 0.141 lazy include wrapper
            context = getattr(route, "include_context", None)
            pairs += _registered_pairs(
                included, prefix + (getattr(context, "prefix", "") or "")
            )
            continue
        path = prefix + getattr(route, "path", "")
        if getattr(route, "routes", None) and not hasattr(route, "endpoint"):
            pairs += _registered_pairs(route, path)  # Mount / sub-router
            continue
        # Websocket routes carry no methods.
        for method in getattr(route, "methods", None) or {"WEBSOCKET"}:
            pairs.append((method, path))
    return pairs


class TestNoDuplicateRoutes:
    """§6 contract #5 / §6.1 bullet 5 (the fix for defect O1).

    main.py's _include() already refuses a duplicate (method, path) at import
    time (stage 5.1) — this turns "the container fails to start" into a test
    failure a change can catch before it ships.
    """

    def test_app_import_registers_no_duplicate_method_path_pairs(self):
        # app.main pulls in the skills/market stack, which needs PyYAML — not
        # installed in every local dev venv (same gap this stage's other
        # releases already worked around by excluding a handful of test
        # modules). Skip gracefully here instead of failing on an unrelated
        # missing dependency; CI and the container image both have it.
        pytest.importorskip("yaml")
        from app.main import _ROUTE_OWNERS, app  # RuntimeError here if a dup exists

        # _include()'s own ledger: the pairs it accepted, prefixes applied.
        # Counting app.routes instead would measure include_router() calls on
        # fastapi >= 0.141, not endpoints.
        assert len(_ROUTE_OWNERS) > 50

        # ...and the same pairs really reached the app, not just the ledger.
        pairs = _registered_pairs(app)
        assert len(pairs) > 50
        assert not {pair for pair in pairs if pairs.count(pair) > 1}
        assert set(_ROUTE_OWNERS) <= set(pairs)
