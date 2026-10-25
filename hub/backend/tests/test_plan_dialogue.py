"""internal design notes Stage 1 — L0 journal.

No Redis, no PostgreSQL: the real client is swapped for an in-memory fake, and
the ``context_snapshot`` fallback is exercised directly.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from app.core import plan_dialogue
from app.core.plan_dialogue import (
    _is_synthetic,
    journal_append,
    journal_read,
)


class _FakeRedis:
    """Minimal RPUSH / LRANGE / EXPIRE / DEL over a dict of lists."""

    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}
        self.expires: dict[str, int] = {}
        self.kv: dict[str, str] = {}

    async def rpush(self, key: str, value: str) -> None:
        self.store.setdefault(key, []).append(value)

    async def lpush(self, key: str, value: str) -> None:
        self.store.setdefault(key, []).insert(0, value)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        items = self.store.get(key, [])
        self.store[key] = items[start : (end + 1 if end != -1 else None)]

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set(self, key: str, value: str) -> None:
        self.kv[key] = value

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.store.get(key, [])
        if end == -1:
            return list(items[start:])
        return list(items[start : end + 1])

    async def expire(self, key: str, ttl: int) -> None:
        self.expires[key] = ttl

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.expires.pop(key, None)


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: r)
    return r


def _session(messages):
    return SimpleNamespace(id=uuid.uuid4(), context_snapshot={"messages": messages})


# --- filter ---------------------------------------------------------------

@pytest.mark.parametrize(
    "role,content,expected",
    [
        ("user", "install nginx", False),
        ("assistant", "Please clarify the upstreams", False),
        ("assistant", "[tool call] read_file", True),
        ("assistant", "[tool results]\n\n...", True),
        ("assistant", "[session closed: token limit]", True),
        ("assistant", "[Requesting permission to read the file: `x`]", True),
        ("tool", "whatever", True),
        ("user", "   ", True),
        ("user", None, True),
    ],
)
def test_is_synthetic(role, content, expected):
    assert _is_synthetic(role, content) is expected


# --- append never raises -------------------------------------------------

@pytest.mark.asyncio
async def test_append_no_redis_is_silent(monkeypatch):
    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: None)
    await journal_append(uuid.uuid4(), "user", "hello")  # must not raise


@pytest.mark.asyncio
async def test_append_broken_redis_is_silent(monkeypatch):
    class _Broken:
        async def rpush(self, *a):
            raise RuntimeError("connection refused")

        async def expire(self, *a):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: _Broken())
    await journal_append(uuid.uuid4(), "user", "hello")  # must not raise


# --- roundtrip ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_journal_roundtrip_filters_synthetic(fake_redis):
    sid = uuid.uuid4()
    await journal_append(sid, "user", "install nginx")
    await journal_append(sid, "assistant", "[tool call] grep")  # dropped
    await journal_append(sid, "assistant", "How many upstreams?")

    key = f"plan_dialogue:{sid}"
    assert len(fake_redis.store[key]) == 2
    assert fake_redis.expires[key] == plan_dialogue.JOURNAL_TTL_SEC

    session = SimpleNamespace(id=sid, context_snapshot={})
    turns = await journal_read(session)
    assert [(t["turn"], t["role"]) for t in turns] == [(1, "user"), (2, "assistant")]
    assert turns[0]["content"] == "install nginx"


@pytest.mark.asyncio
async def test_journal_read_falls_back_to_snapshot_when_key_gone(fake_redis):
    # Nothing written to Redis — key is "expired".
    session = _session(
        [
            {"role": "user", "content": "install nginx"},
            {"role": "assistant", "content": "[tool call] grep"},
            {"role": "assistant", "content": "Done, here is the plan"},
        ]
    )
    turns = await journal_read(session)
    assert [t["content"] for t in turns] == ["install nginx", "Done, here is the plan"]
    assert [t["turn"] for t in turns] == [1, 2]


@pytest.mark.asyncio
async def test_expired_key_and_snapshot_agree(fake_redis):
    sid = uuid.uuid4()
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    for m in msgs:
        await journal_append(sid, m["role"], m["content"])
    live = await journal_read(SimpleNamespace(id=sid, context_snapshot={}))

    await plan_dialogue.journal_delete(sid)
    fell_back = await journal_read(SimpleNamespace(id=sid, context_snapshot={"messages": msgs}))

    assert [(t["role"], t["content"]) for t in live] == [
        (t["role"], t["content"]) for t in fell_back
    ]


# --- L2: archive_for_goal ------------------------------------------------

@pytest.fixture
def fake_workspace(monkeypatch):
    """Capture every write_workspace_file call as {path: content}."""
    written: dict[str, str] = {}

    class _WSD:
        @staticmethod
        async def write_workspace_file(room_id, path, content, commit_message=None):
            written[path] = content
            return {"ok": True}

    import app.core.workspaced_client as real_wsd

    monkeypatch.setattr(real_wsd, "write_workspace_file", _WSD.write_workspace_file)
    return written


@pytest.fixture(autouse=True)
def _no_digest(monkeypatch):
    async def _empty(turns, llm_config, **_kw):
        return ""

    monkeypatch.setattr(plan_dialogue, "_generate_digest", _empty)


NGINX = "\n".join(f"    location /p{i} {{ proxy_pass http://up; }}" for i in range(20))


@pytest.mark.asyncio
async def test_archive_for_goal_writes_transcript_and_attachment(fake_redis, fake_workspace):
    sid = uuid.uuid4()
    await journal_append(sid, "user", f"install nginx, here is the config:\n```nginx\n{NGINX}\n```")
    await journal_append(sid, "assistant", "[tool call] read_file")  # synthetic
    await journal_append(sid, "assistant", "Done, the plan is: ...")

    session = SimpleNamespace(id=sid, context_snapshot={})
    goal_id = uuid.uuid4()
    path = await plan_dialogue.archive_for_goal(
        uuid.uuid4(), goal_id, "Configure NGINX", session, None,
        pending_plan={"plan_summary": "Two upstreams", "plan_steps": [{"title": "Config"}]},
    )
    assert path == f".legion/plans/goal-{goal_id}.md"

    md = fake_workspace[path]
    assert "# Spec: Configure NGINX" in md
    assert "[tool call]" not in md          # synthetic filtered
    assert "## Attachments" in md
    assert "## Agreed plan" in md
    assert "Two upstreams" in md
    # code block lifted out of the transcript into a real file
    assert "proxy_pass" not in md
    att_path = f".legion/plans/goal-{goal_id}-attachments/01-nginx.conf"
    assert "proxy_pass" in fake_workspace[att_path]


@pytest.mark.asyncio
async def test_archive_redacts_secrets(fake_redis, fake_workspace):
    sid = uuid.uuid4()
    await journal_append(sid, "user", "config with a password: password=supersecret123 in prod")
    session = SimpleNamespace(id=sid, context_snapshot={})
    gid = uuid.uuid4()
    await plan_dialogue.archive_for_goal(uuid.uuid4(), gid, "T", session, None)
    md = fake_workspace[f".legion/plans/goal-{gid}.md"]
    assert "supersecret123" not in md


@pytest.mark.asyncio
async def test_archive_middle_truncation_keeps_attachments(fake_redis, fake_workspace, monkeypatch):
    monkeypatch.setattr(plan_dialogue, "ARCHIVE_MAX_CHARS", 2000)
    sid = uuid.uuid4()
    for i in range(40):
        await journal_append(sid, "user", f"turn number {i} " + "blah " * 50)
    session = SimpleNamespace(id=sid, context_snapshot={})
    gid = uuid.uuid4()
    await plan_dialogue.archive_for_goal(uuid.uuid4(), gid, "T", session, None)
    md = fake_workspace[f".legion/plans/goal-{gid}.md"]
    assert "the middle of the dialogue is omitted" in md


@pytest.mark.asyncio
async def test_archive_no_turns_returns_none(fake_redis, fake_workspace):
    session = SimpleNamespace(id=uuid.uuid4(), context_snapshot={})
    out = await plan_dialogue.archive_for_goal(uuid.uuid4(), uuid.uuid4(), "T", session, None)
    assert out is None
    assert fake_workspace == {}


@pytest.mark.asyncio
async def test_archive_write_failure_is_swallowed(fake_redis, monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("workspaced down")

    import app.core.workspaced_client as real_wsd
    monkeypatch.setattr(real_wsd, "write_workspace_file", _boom)

    sid = uuid.uuid4()
    await journal_append(sid, "user", "install nginx")
    session = SimpleNamespace(id=sid, context_snapshot={})
    out = await plan_dialogue.archive_for_goal(uuid.uuid4(), uuid.uuid4(), "T", session, None)
    assert out is None  # no raise


# --- L3: plan_dialogue read access (Stage 3) ----------------------------------


class _FakeDB:
    """``.get(Model, id)`` over a preloaded {(model_name, id): row} map."""

    def __init__(self, rows: dict) -> None:
        self._rows = rows

    async def get(self, model, ident):
        return self._rows.get((model.__name__, ident))


@pytest.fixture
def fake_ws_files(monkeypatch):
    """Serve a {path: content} map through read_workspace_file."""
    files: dict[str, str] = {}

    async def _read(room_id, path):
        return files.get(path)

    import app.core.workspaced_client as real_wsd

    monkeypatch.setattr(real_wsd, "read_workspace_file", _read)
    return files


def _ctx(db, *, room_id=None, session_id=None, task_id=None):
    return SimpleNamespace(
        db=db,
        room_id=room_id or uuid.uuid4(),
        session_id=session_id or uuid.uuid4(),
        task_id=task_id,
    )


async def _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace, extra_turn=""):
    await journal_append(sid, "user", "install nginx, rate limit 100 r/s per IP")
    await journal_append(sid, "assistant", "Understood: two upstreams, round-robin." + extra_turn)
    session = SimpleNamespace(id=sid, context_snapshot={})
    await plan_dialogue.archive_for_goal(room_id, gid, "Configure NGINX", session, None)
    return fake_workspace[f".legion/plans/goal-{gid}.md"]


@pytest.mark.asyncio
async def test_plan_dialogue_search_returns_matching_turn(
    fake_redis, fake_workspace, fake_ws_files
):
    from app.models.session.base import Session

    room_id, gid, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = md

    db = _FakeDB({("Session", sid): SimpleNamespace(id=sid, task_id=None, goal_id=gid)})
    ctx = _ctx(db, room_id=room_id, session_id=sid)

    out = await plan_dialogue.read_dialogue_for_session(ctx, mode="search", query="rate limit")
    assert "[#1]" in out
    assert "rate limit" in out.lower()


@pytest.mark.asyncio
async def test_plan_dialogue_digest_returns_header_only(
    fake_redis, fake_workspace, fake_ws_files
):
    room_id, gid, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = md

    db = _FakeDB({("Session", sid): SimpleNamespace(id=sid, task_id=None, goal_id=gid)})
    out = await plan_dialogue.read_dialogue_for_session(
        _ctx(db, room_id=room_id, session_id=sid), mode="digest"
    )
    assert "# Spec: Configure NGINX" in out
    assert "## [#1]" not in out  # transcript excluded


@pytest.mark.asyncio
async def test_plan_dialogue_turns_mode(fake_redis, fake_workspace, fake_ws_files):
    room_id, gid, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = md

    db = _FakeDB({("Session", sid): SimpleNamespace(id=sid, task_id=None, goal_id=gid)})
    out = await plan_dialogue.read_dialogue_for_session(
        _ctx(db, room_id=room_id, session_id=sid), mode="turns", turn_ids=[2]
    )
    assert "[#2]" in out
    assert "round-robin" in out


@pytest.mark.asyncio
async def test_plan_dialogue_scope_isolation(fake_redis, fake_workspace, fake_ws_files):
    """An agent on goal A cannot reach goal B's archive by any argument."""
    room_id = uuid.uuid4()
    gid_a, sid_a = uuid.uuid4(), uuid.uuid4()
    gid_b, sid_b = uuid.uuid4(), uuid.uuid4()
    md_a = await _seed_goal_archive(gid_a, sid_a, room_id, fake_redis, fake_workspace)
    md_b = await _seed_goal_archive(
        gid_b, sid_b, room_id, fake_redis, fake_workspace, extra_turn=" SECRET-B"
    )
    fake_ws_files[f".legion/plans/goal-{gid_a}.md"] = md_a
    fake_ws_files[f".legion/plans/goal-{gid_b}.md"] = md_b

    db = _FakeDB({("Session", sid_a): SimpleNamespace(id=sid_a, task_id=None, goal_id=gid_a)})
    out = await plan_dialogue.read_dialogue_for_session(
        _ctx(db, room_id=room_id, session_id=sid_a),
        mode="search",
        query="round-robin",
    )
    assert f"goal-{gid_a}" not in out  # only A's label, never B
    assert "SECRET-B" not in out
    # digest mode likewise sees only archive A
    out2 = await plan_dialogue.read_dialogue_for_session(
        _ctx(db, room_id=room_id, session_id=sid_a), mode="digest"
    )
    assert "SECRET-B" not in out2


@pytest.mark.asyncio
async def test_plan_dialogue_ref_only_existing(fake_redis, fake_workspace, fake_ws_files):
    room_id, gid = uuid.uuid4(), uuid.uuid4()
    sid = uuid.uuid4()
    md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = md

    ref = await plan_dialogue.plan_dialogue_ref(room_id, uuid.uuid4(), gid)
    assert ref == {"goal": f".legion/plans/goal-{gid}.md"}  # task archive absent


@pytest.mark.asyncio
async def test_plan_dialogue_no_archive_is_graceful(fake_ws_files):
    db = _FakeDB({("Session", "s"): SimpleNamespace(id="s", task_id=None, goal_id=uuid.uuid4())})
    ctx = SimpleNamespace(db=db, room_id=uuid.uuid4(), session_id="s", task_id=None)
    out = await plan_dialogue.read_dialogue_for_session(ctx, mode="digest")
    assert "not been created" in out or "no dialogue archive" in out


@pytest.mark.asyncio
async def test_digest_for_decomposition_returns_header_only(
    fake_redis, fake_workspace, fake_ws_files
):
    room_id, gid, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = md

    out = await plan_dialogue.digest_for_decomposition(room_id, goal_id=gid)
    assert out is not None
    assert out.startswith("# Spec:")
    assert "## [#1]" not in out  # transcript turns excluded


@pytest.mark.asyncio
async def test_digest_for_decomposition_none_without_archive(fake_ws_files):
    out = await plan_dialogue.digest_for_decomposition(uuid.uuid4(), goal_id=uuid.uuid4())
    assert out is None


@pytest.mark.asyncio
async def test_digest_for_decomposition_prefers_task_archive(
    fake_redis, fake_workspace, fake_ws_files
):
    room_id, gid, tid, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    goal_md = await _seed_goal_archive(gid, sid, room_id, fake_redis, fake_workspace)
    fake_ws_files[f".legion/plans/goal-{gid}.md"] = goal_md
    fake_ws_files[f".legion/plans/task-{tid}.md"] = "# Spec: task-scoped\n\n> **Task:** x\n"

    out = await plan_dialogue.digest_for_decomposition(room_id, goal_id=gid, task_id=tid)
    assert "task-scoped" in out


@pytest.mark.asyncio
async def test_plan_dialogue_tool_registered_for_agent():
    from app.core.tools import REGISTRY

    tool = REGISTRY.get("plan_dialogue")
    assert tool is not None
    assert "agent" in tool.roles and "cm" in tool.roles
    assert tool.requires_container is False


# --- L1: drafts (Stage 5) ----------------------------------------------------


def _draft_session(sid, room_id, user_id):
    return SimpleNamespace(
        id=sid, room_id=room_id, user_id=user_id, kind="user_cm_p2",
        context_snapshot={},
    )


@pytest.mark.asyncio
async def test_save_draft_writes_file_and_head_pointer(fake_redis, fake_workspace):
    room_id, user_id, sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await journal_append(sid, "user", "install nginx, here is a reference https://ex.com/a")
    await journal_append(sid, "assistant", "[tool call] grep")  # synthetic
    await journal_append(sid, "assistant", "How many upstreams?")

    path = await plan_dialogue.save_draft(
        _draft_session(sid, room_id, user_id), None, close_reason="topic_change"
    )
    assert path == f".legion/drafts/{sid}.md"
    md = fake_workspace[path]
    assert "[tool call]" not in md
    assert "topic_change" in md
    assert "https://ex.com/a" in md

    cands = await plan_dialogue.list_draft_candidates(room_id, user_id)
    assert len(cands) == 1
    assert cands[0]["session_id"] == str(sid)
    assert cands[0]["close_reason"] == "topic_change"
    assert "nginx" in cands[0]["preview"]


@pytest.mark.asyncio
async def test_save_draft_no_turns_returns_none(fake_redis, fake_workspace):
    out = await plan_dialogue.save_draft(
        _draft_session(uuid.uuid4(), uuid.uuid4(), uuid.uuid4()), None
    )
    assert out is None
    assert fake_workspace == {}


@pytest.mark.asyncio
async def test_save_draft_no_redis_still_writes_file(monkeypatch, fake_workspace):
    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: None)
    sid = uuid.uuid4()
    session = SimpleNamespace(
        id=sid, room_id=uuid.uuid4(), user_id=uuid.uuid4(), kind="user_cm_p2",
        context_snapshot={"messages": [{"role": "user", "content": "hi"}]},
    )
    path = await plan_dialogue.save_draft(session, None)
    assert path == f".legion/drafts/{sid}.md"


@pytest.mark.asyncio
async def test_list_draft_candidates_skips_merged_and_old(fake_redis, fake_workspace):
    import json as _json
    from datetime import datetime, timedelta, timezone

    room_id, user_id = uuid.uuid4(), uuid.uuid4()
    fresh, merged, stale = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = datetime.now(timezone.utc)
    k = plan_dialogue._head_key(room_id, user_id)
    for s, at in [
        (fresh, now),
        (merged, now),
        (stale, now - timedelta(hours=plan_dialogue.DRAFT_WINDOW_HOURS + 5)),
    ]:
        await fake_redis.lpush(k, _json.dumps(
            {"session_id": str(s), "closed_at": at.isoformat(),
             "close_reason": "rotated", "topic": "t", "preview": "p"}
        ))
    await plan_dialogue._mark_draft_merged(merged)

    cands = await plan_dialogue.list_draft_candidates(room_id, user_id)
    ids = {c["session_id"] for c in cands}
    assert ids == {str(fresh)}


@pytest.mark.asyncio
async def test_merge_draft_prepends_and_is_idempotent(fake_redis):
    draft_sid, cur_sid = uuid.uuid4(), uuid.uuid4()
    await journal_append(draft_sid, "user", "first dialogue: install nginx")
    await journal_append(draft_sid, "assistant", "what rate limit?")

    draft_row = SimpleNamespace(id=draft_sid, context_snapshot={})
    cur = SimpleNamespace(
        id=cur_sid, kind="user_cm_p2",
        context_snapshot={"messages": [{"role": "user", "content": "yes, 100 r/s"}]},
    )
    db = _FakeDB({("Session", draft_sid): draft_row})

    ok = await plan_dialogue.merge_draft_into_session(cur, draft_sid, db)
    assert ok is True
    msgs = cur.context_snapshot["messages"]
    assert msgs[0]["content"].startswith("[unfinished dialogue merged")
    assert "install nginx" in msgs[1]["content"]
    assert msgs[-1]["content"] == "yes, 100 r/s"

    # journal of the current session now starts with the carried turns
    carried = await journal_read(SimpleNamespace(id=cur_sid, context_snapshot={}))
    assert "install nginx" in carried[0]["content"]

    before = list(cur.context_snapshot["messages"])
    assert await plan_dialogue.merge_draft_into_session(cur, draft_sid, db) is True
    assert cur.context_snapshot["messages"] == before  # idempotent

    assert await plan_dialogue.list_draft_candidates(uuid.uuid4(), uuid.uuid4()) == []


@pytest.mark.asyncio
async def test_merge_draft_missing_or_empty_returns_false(fake_redis):
    db = _FakeDB({})
    cur = SimpleNamespace(id=uuid.uuid4(), kind="user_cm_p2", context_snapshot={})
    assert await plan_dialogue.merge_draft_into_session(cur, uuid.uuid4(), db) is False


@pytest.mark.asyncio
async def test_clear_draft_heads(fake_redis):
    import json as _json

    room_id, user_id = uuid.uuid4(), uuid.uuid4()
    k = plan_dialogue._head_key(room_id, user_id)
    await fake_redis.lpush(k, _json.dumps({"session_id": str(uuid.uuid4()),
                                           "closed_at": "", "topic": "", "preview": ""}))
    await plan_dialogue.clear_draft_heads(room_id, user_id)
    assert await plan_dialogue.list_draft_candidates(room_id, user_id) == []


@pytest.mark.asyncio
async def test_merge_draft_tool_registered_and_excluded_from_proxy():
    from app.core.tools import REGISTRY
    from app.core.tools.registry import strip_dialogue_only

    tool = REGISTRY.get("merge_draft")
    assert tool is not None and tool.requires_container is False

    ant = REGISTRY.ant_schemas_for("cm")
    assert any(s["name"] == "merge_draft" for s in ant)
    assert all(s["name"] != "merge_draft" for s in strip_dialogue_only(ant))


# --- Stage 6: archive_for_task + append_to_task_archive ----------------------


@pytest.fixture
def fake_ws_rw(monkeypatch):
    """A single {path: content} map behind both read and write."""
    files: dict[str, str] = {}

    async def _read(room_id, path):
        return files.get(path)

    async def _write(room_id, path, content, commit_message=None):
        files[path] = content
        return {"ok": True}

    import app.core.workspaced_client as real_wsd

    monkeypatch.setattr(real_wsd, "read_workspace_file", _read)
    monkeypatch.setattr(real_wsd, "write_workspace_file", _write)
    return files


@pytest.mark.asyncio
async def test_archive_for_task_writes_task_archive(fake_redis, fake_ws_rw):
    sid = uuid.uuid4()
    await journal_append(sid, "user", "rework the proxy, here is the new config")
    await journal_append(sid, "assistant", "Got it, I will clarify the details")
    session = SimpleNamespace(id=sid, context_snapshot={})
    tid = uuid.uuid4()
    path = await plan_dialogue.archive_for_task(
        uuid.uuid4(), tid, session, None, task_title="Rework the proxy"
    )
    assert path == f".legion/plans/task-{tid}.md"
    assert "# Spec: Rework the proxy" in fake_ws_rw[path]
    assert "**Task:**" in fake_ws_rw[path]


@pytest.mark.asyncio
async def test_append_to_task_archive_appends_and_is_idempotent(fake_redis, fake_ws_rw):
    tid = uuid.uuid4()
    path = f".legion/plans/task-{tid}.md"
    fake_ws_rw[path] = "# Spec: X\n\n## [#1] User\n\nbody\n"

    turns = [
        {"role": "agent", "content": "which upstream — from the config or the current one?"},
        {"role": "cm", "content": "from the config that was sent"},
    ]
    out = await plan_dialogue.append_to_task_archive(tid, uuid.uuid4(), turns, None)
    assert out == path
    body = fake_ws_rw[path]
    assert "## Clarifications during execution" in body
    assert "from the config that was sent" in body

    # second identical call does not double-append
    await plan_dialogue.append_to_task_archive(tid, uuid.uuid4(), turns, None)
    assert fake_ws_rw[path].count("from the config that was sent") == 1


@pytest.mark.asyncio
async def test_append_to_task_archive_falls_back_to_goal_archive(fake_redis, fake_ws_rw):
    from app.models.task import Task

    tid = uuid.uuid4()
    gid = uuid.uuid4()
    goal_path = f".legion/plans/goal-{gid}.md"
    fake_ws_rw[goal_path] = "# Spec: G\n\n## [#1] User\n\nbody\n"

    db = _FakeDB({("Task", tid): SimpleNamespace(goal_id=gid)})
    out = await plan_dialogue.append_to_task_archive(
        tid, uuid.uuid4(),
        [{"role": "cm", "content": "detail"}],
        db,
    )
    assert out == goal_path
    assert "detail" in fake_ws_rw[goal_path]


@pytest.mark.asyncio
async def test_append_to_task_archive_no_archive_returns_none(fake_redis, fake_ws_rw):
    out = await plan_dialogue.append_to_task_archive(
        uuid.uuid4(), uuid.uuid4(),
        [{"role": "cm", "content": "detail"}],
        None,
    )
    assert out is None


# --- Stage 7: archive_meta + unlink_archive (UI chip) -----------------------


@pytest.mark.asyncio
async def test_archive_meta_counts_turns_and_attachments(fake_redis, fake_ws_rw):
    sid, gid, room_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await journal_append(sid, "user", "install nginx, here is the config:\n```nginx\n" + NGINX + "\n```")
    await journal_append(sid, "assistant", "Done, the plan is this.")
    session = SimpleNamespace(id=sid, context_snapshot={})
    await plan_dialogue.archive_for_goal(room_id, gid, "Configure NGINX", session, None)

    meta = await plan_dialogue.archive_meta(room_id, goal_id=gid)
    assert meta["bound"] is True
    assert meta["kind"] == "goal"
    assert meta["message_count"] == 2
    assert meta["attachment_count"] == 1
    assert meta["path"] == f".legion/plans/goal-{gid}.md"


@pytest.mark.asyncio
async def test_archive_meta_prefers_task_over_goal(fake_redis, fake_ws_rw):
    room_id, gid, tid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    fake_ws_rw[f".legion/plans/goal-{gid}.md"] = "# Spec: g\n\n## [#1] User\n\nx\n"
    fake_ws_rw[f".legion/plans/task-{tid}.md"] = (
        "# Spec: t\n\n## [#1] User\n\na\n\n## [#2] ChatMaster\n\nb\n"
    )
    meta = await plan_dialogue.archive_meta(room_id, goal_id=gid, task_id=tid)
    assert meta["kind"] == "task"
    assert meta["message_count"] == 2


@pytest.mark.asyncio
async def test_archive_meta_unbound_when_no_file(fake_ws_rw):
    meta = await plan_dialogue.archive_meta(uuid.uuid4(), goal_id=uuid.uuid4())
    assert meta["bound"] is False
    assert meta["message_count"] == 0


@pytest.mark.asyncio
async def test_unlink_archive_deletes_file(fake_redis, monkeypatch):
    deleted: list[str] = []

    async def _del(room_id, path, commit_message=None):
        deleted.append(path)
        return True

    import app.core.workspaced_client as real_wsd
    monkeypatch.setattr(real_wsd, "delete_workspace_file", _del)

    tid = uuid.uuid4()
    ok = await plan_dialogue.unlink_archive(uuid.uuid4(), task_id=tid)
    assert ok is True
    assert deleted == [f".legion/plans/task-{tid}.md"]


@pytest.mark.asyncio
async def test_unlink_archive_swallows_workspace_error(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("workspaced down")

    import app.core.workspaced_client as real_wsd
    monkeypatch.setattr(real_wsd, "delete_workspace_file", _boom)

    ok = await plan_dialogue.unlink_archive(uuid.uuid4(), goal_id=uuid.uuid4())
    assert ok is False  # no raise


def test_draft_candidates_prompt_block_renders():
    from app.core.memory_layer import build_system_prompt

    prompt = build_system_prompt(
        {
            "draft_candidates": [
                {"session_id": "abc", "closed_at": "2026-09-10T11:30:00+00:00",
                 "close_reason": "topic_change", "topic": "nginx proxy",
                 "preview": "install nginx"},
            ]
        },
        "user_cm_p2",
    )
    assert "Unfinished dialogue" in prompt
    assert "merge_draft" in prompt
    assert "`abc`" in prompt
