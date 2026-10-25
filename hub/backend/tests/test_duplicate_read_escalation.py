"""
Tests for the hard-refusal escalation on repeated duplicate reads.

Root cause covered: read_file/grep/bash-cat already avoided re-sending
content the agent had already seen this session, but the receipt was a
bland, easy-to-ignore note — nothing stopped the agent from requesting the
exact same content over and over (each repeat still burning a full tool
call). After DUPLICATE_READ_HARD_LIMIT repeats of the same (tool-class,
target, fingerprint), the receipt now leads with an explicit "[BLOCKED: ...]"
directive instead of a soft note. read_file and bash `cat` share one counter
per (path, fingerprint) since both represent "you already saw this file's
content" regardless of which tool asked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_observations import ObservationResult
from app.core.tools import REGISTRY, ToolContext
from app.core.tools._session_state import (
    DUPLICATE_READ_HARD_LIMIT,
    clear_session,
    duplicate_escalation_notice,
    note_duplicate_read,
)


# ---------------------------------------------------------------------------
# Pure counter/notice unit tests
# ---------------------------------------------------------------------------


class TestNoteDuplicateReadAndNotice:
    def test_counter_increments_per_session_and_key(self):
        sid = str(uuid4())
        assert note_duplicate_read(sid, "read:a.py:fp1") == 1
        assert note_duplicate_read(sid, "read:a.py:fp1") == 2
        assert note_duplicate_read(sid, "read:b.py:fp2") == 1  # different key, separate count
        clear_session(sid)

    def test_counter_isolated_per_session(self):
        sid1, sid2 = str(uuid4()), str(uuid4())
        note_duplicate_read(sid1, "read:a.py:fp1")
        note_duplicate_read(sid1, "read:a.py:fp1")
        assert note_duplicate_read(sid2, "read:a.py:fp1") == 1
        clear_session(sid1)
        clear_session(sid2)

    def test_notice_empty_below_limit(self):
        assert duplicate_escalation_notice("'a.py'", DUPLICATE_READ_HARD_LIMIT - 1) == ""

    def test_notice_present_at_limit(self):
        text = duplicate_escalation_notice("'a.py'", DUPLICATE_READ_HARD_LIMIT)
        assert "BLOCKED" in text
        assert "'a.py'" in text

    def test_notice_present_above_limit(self):
        text = duplicate_escalation_notice("'a.py'", DUPLICATE_READ_HARD_LIMIT + 5)
        assert "BLOCKED" in text

    def test_clear_session_resets_counter(self):
        sid = str(uuid4())
        note_duplicate_read(sid, "read:a.py:fp1")
        note_duplicate_read(sid, "read:a.py:fp1")
        clear_session(sid)
        assert note_duplicate_read(sid, "read:a.py:fp1") == 1
        clear_session(sid)


# ---------------------------------------------------------------------------
# read_file integration
# ---------------------------------------------------------------------------


def _make_ctx(session_id=None) -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    execute_mock = AsyncMock()
    execute_mock.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    db.execute = execute_mock
    return ToolContext(room_id=uuid4(), session_id=session_id or uuid4(), task_id=uuid4(), db=db)


class _FakeLedger:
    """In-memory stand-in for the DB-backed read ledger.

    record_read_interval/load_context_state are normally backed by Postgres
    JSONB; a bare AsyncMock db never actually persists state between calls,
    so tests using one would never see a real "already covered" range and
    the duplicate branch would never trigger. This fake reproduces the same
    merge/subtract contract as context_observations.record_read_interval so
    the escalation counter gets exercised the way it would in production.
    """

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}

    async def load_context_state(self, db, session_id):
        return self.state.get(str(session_id), {})

    async def record_read_interval(
        self, db, *, session_id, path, source_fingerprint, requested_start, requested_end, observation_ref
    ):
        from app.core.context_observations import merge_intervals, subtract_interval

        st = self.state.setdefault(str(session_id), {})
        ledger = st.setdefault("read_ledger", {})
        entry = ledger.get(path) or {}
        if entry.get("fingerprint") != source_fingerprint:
            entry = {"fingerprint": source_fingerprint, "ranges": [], "observation_refs": []}
        covered = entry.get("ranges") or []
        unseen = subtract_interval(requested_start, requested_end, covered)
        entry["ranges"] = merge_intervals([*covered, [requested_start, requested_end]])
        refs = list(entry.get("observation_refs") or [])
        if observation_ref not in refs:
            refs.append(observation_ref)
        entry["observation_refs"] = refs
        ledger[path] = entry
        st["read_ledger"] = ledger
        return unseen


async def _fake_record_observation(db, **kwargs):
    return _fake_observation()


@pytest.mark.asyncio
async def test_read_file_escalates_after_hard_limit_repeats():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    clear_session(str(sid))
    tool = REGISTRY.get("read_file")
    content = "line one\nline two\n"
    fake = _FakeLedger()

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m,
        patch("app.core.context_observations.record_observation", new=_fake_record_observation),
        patch("app.core.context_observations.record_read_interval", new=fake.record_read_interval),
    ):
        m.return_value = content
        results = []
        for _ in range(DUPLICATE_READ_HARD_LIMIT + 1):
            results.append(await tool.execute(ctx, {"path": "foo.py"}))

    # First real read is not a duplicate (nothing to compare against yet).
    assert "BLOCKED" not in results[0]
    # By the time we've hit the hard limit, later repeats must be blocked.
    assert "BLOCKED" in results[-1]
    clear_session(str(sid))


@pytest.mark.asyncio
async def test_read_file_first_couple_duplicates_stay_soft():
    sid = uuid4()
    ctx = _make_ctx(session_id=sid)
    clear_session(str(sid))
    tool = REGISTRY.get("read_file")
    content = "only line\n"

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = content
        first = await tool.execute(ctx, {"path": "foo.py"})
        second = await tool.execute(ctx, {"path": "foo.py"})

    assert "BLOCKED" not in first
    assert "BLOCKED" not in second
    clear_session(str(sid))


# ---------------------------------------------------------------------------
# bash `cat` shares the counter with read_file
# ---------------------------------------------------------------------------


def _make_bash_ctx(session_id) -> tuple[ToolContext, MagicMock]:
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    ws.network_enabled = True
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _make_result(obj):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=obj)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=lambda q: _make_result(ws))
    ctx = ToolContext(room_id=uuid4(), session_id=session_id, task_id=uuid4(), db=db)
    return ctx, db


def _fake_observation(ref: str = "obs-1") -> ObservationResult:
    artifact = MagicMock()
    artifact.id = ref
    artifact.stale = False
    artifact.tool_name = "read_file"
    artifact.token_count = 42
    return ObservationResult(artifact=artifact, duplicate=False)


@pytest.mark.asyncio
async def test_cat_and_read_file_share_escalation_counter():
    sid = uuid4()
    clear_session(str(sid))
    read_ctx = _make_ctx(session_id=sid)
    bash_ctx, _ = _make_bash_ctx(sid)
    read_tool = REGISTRY.get("read_file")
    bash_tool = REGISTRY.get("bash")
    content = "l1\nl2\nl3\n"
    fake = _FakeLedger()

    with (
        patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m,
        patch("app.core.context_observations.load_context_state", new=fake.load_context_state),
        patch("app.core.context_observations.record_observation", new=_fake_record_observation),
        patch("app.core.context_observations.record_read_interval", new=fake.record_read_interval),
    ):
        m.return_value = content

        # 1st: real read via read_file.
        r1 = await read_tool.execute(read_ctx, {"path": "shared.py"})
        assert "BLOCKED" not in r1

        # 2nd + 3rd: duplicates via bash `cat` of the SAME file — must count
        # toward the same limit as read_file's duplicates.
        r2 = await bash_tool.execute(bash_ctx, {"cmd": "cat shared.py"})
        r3 = await bash_tool.execute(bash_ctx, {"cmd": "cat -n shared.py"})

        # 4th: another duplicate via read_file itself — this is repeat #4
        # (1 real + 2 cats + this one), past DUPLICATE_READ_HARD_LIMIT=3.
        r4 = await read_tool.execute(read_ctx, {"path": "shared.py"})

    assert "BLOCKED" not in r2
    assert "BLOCKED" not in r3
    assert "BLOCKED" in r4
    clear_session(str(sid))


# ---------------------------------------------------------------------------
# grep escalation
# ---------------------------------------------------------------------------


class _FakeObservationStore:
    """Reproduces record_observation's exact-repeat dedup (by args+fingerprint)
    without touching the DB, so grep's `observation.duplicate` branch — the
    only thing it checks — behaves as it would in production."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    async def record_observation(self, db, *, canonical_args, source_fingerprint, **kwargs):
        import json as _json

        key = _json.dumps(canonical_args, sort_keys=True, default=str) + "|" + source_fingerprint
        is_dup = key in self.seen
        self.seen.add(key)
        return ObservationResult(artifact=_fake_observation().artifact, duplicate=is_dup)


@pytest.mark.asyncio
async def test_grep_duplicate_escalates_after_hard_limit():
    sid = uuid4()
    clear_session(str(sid))
    ctx = _make_ctx(session_id=sid)
    tool = REGISTRY.get("grep")
    store = _FakeObservationStore()

    with (
        patch("app.core.tools._grep._grep_fallback", new_callable=AsyncMock) as m,
        # _grep.py imports record_observation by name at module scope (unlike
        # read_file/bash, which import it locally inside execute()), so the
        # patch target is the name bound in _grep's own namespace.
        patch("app.core.tools._grep.record_observation", new=store.record_observation),
    ):
        m.return_value = "foo.py:1:match\n"
        results = []
        for _ in range(DUPLICATE_READ_HARD_LIMIT + 2):
            results.append(await tool.execute(ctx, {"pattern": "match"}))

    assert "BLOCKED" not in results[0]
    assert "BLOCKED" in results[-1]
    clear_session(str(sid))
