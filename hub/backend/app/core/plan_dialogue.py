"""User↔ChatMaster task-formation dialogue: journal, drafts, archives.

internal design notes.

L0  journal_append / journal_read    — Redis, best-effort, with a DB fallback
L1  save_draft / list_draft_candidates / merge_draft_into_session   (Stage 5)
L2  archive_for_goal / archive_for_task / append_to_task_archive     (Stage 2/6)

Stage 1 delivers L0 only: every real turn of a task-formation dialogue is
appended to a Redis list the moment it happens, so the archive that Stage 2
writes at close time never has to trust ``session.context_snapshot`` as its
sole source (that column has a history of silently dropping a turn —
incident 2026-09-08#3, fix 17c08f9). Redis is short-lived, so ``journal_read``
falls back to reconstructing the same turns from ``context_snapshot`` with the
identical filter when the key is gone.

Everything here is best-effort: a missing or unreachable Redis must never make
a dialogue turn fail (same contract as ``RedisLogBuffer`` in
``core/log_storage.py``).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- L0 ---------------------------------------------------------------

JOURNAL_TTL_SEC = 7 * 24 * 3600
DRAFT_WINDOW_HOURS = 24
ARCHIVE_MAX_CHARS = 400_000          # middle truncation, never the tail
ATTACHMENT_MIN_LINES = 10

_KEY_PREFIX = "plan_dialogue:"

# Content prefixes that mark a synthetic history entry — a tool-loop note, a
# close marker, a file-read permission stub. These carry no dialogue meaning
# and must never reach the journal or the archive.
_SYNTHETIC_PREFIXES = (
    "[tool call",
    "[tools executed",
    "[tool results",
    "[session closed",
    "[Requesting permission to read the file",
)

_redis: Any = None
_redis_disabled = False


def _get_redis() -> Any:
    """Lazily build a shared async Redis client, or return None.

    Short connect/socket timeouts so a turn never blocks on an unreachable
    Redis; a hard import/config failure disables the client for the process.
    """
    global _redis, _redis_disabled
    if _redis_disabled:
        return None
    if _redis is None:
        try:
            import redis.asyncio as aioredis

            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            _redis = aioredis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        except Exception as exc:  # pragma: no cover - import/config failure
            logger.warning("plan_dialogue: Redis unavailable, journal disabled: %s", exc)
            _redis_disabled = True
            return None
    return _redis


def _key(session_id: Any) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _is_synthetic(role: Any, content: Any) -> bool:
    """True for entries that must not be journaled or archived."""
    if role not in ("user", "assistant"):
        return True
    text = "" if content is None else str(content)
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.startswith(_SYNTHETIC_PREFIXES)


async def journal_append(
    session_id: Any,
    role: str,
    content: Any,
    *,
    turn_no: Optional[int] = None,
) -> None:
    """RPUSH one real dialogue turn onto ``plan_dialogue:{session_id}``.

    Synthetic entries (see ``_is_synthetic``) are dropped at the door. Never
    raises: a Redis outage degrades to the ``context_snapshot`` fallback in
    ``journal_read``.
    """
    if _is_synthetic(role, content):
        return
    r = _get_redis()
    if r is None:
        return
    entry = {
        "turn": turn_no,
        "role": role,
        "content": str(content),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        k = _key(session_id)
        await r.rpush(k, json.dumps(entry, ensure_ascii=False))
        await r.expire(k, JOURNAL_TTL_SEC)
    except Exception as exc:
        logger.warning("plan_dialogue.journal_append failed (session %s): %s", session_id, exc)


def _reconstruct_from_snapshot(session: Any) -> list[dict[str, Any]]:
    snap = getattr(session, "context_snapshot", None) or {}
    messages = snap.get("messages", []) if isinstance(snap, dict) else []
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if _is_synthetic(role, content):
            continue
        out.append({"turn": None, "role": role, "content": str(content), "at": None})
    return out


async def journal_read(session: Any, db: Any = None) -> list[dict[str, Any]]:
    """Return this session's real dialogue turns as ``[{turn, role, content, at}]``.

    Redis first; on an empty/expired key, reconstruct from
    ``session.context_snapshot["messages"]`` with the same filter. Turn numbers
    are assigned sequentially on read so ``[#N]`` markers stay stable regardless
    of source.
    """
    entries: list[dict[str, Any]] = []
    r = _get_redis()
    if r is not None:
        try:
            raw = await r.lrange(_key(session.id), 0, -1)
            for item in raw:
                try:
                    e = json.loads(item)
                except Exception:
                    continue
                if _is_synthetic(e.get("role"), e.get("content")):
                    continue
                entries.append(
                    {
                        "turn": None,
                        "role": e.get("role"),
                        "content": str(e.get("content") or ""),
                        "at": e.get("at"),
                    }
                )
        except Exception as exc:
            logger.warning(
                "plan_dialogue.journal_read: Redis read failed (session %s): %s",
                getattr(session, "id", "?"), exc,
            )

    if not entries:
        entries = _reconstruct_from_snapshot(session)

    for i, e in enumerate(entries, start=1):
        e["turn"] = i
    return entries


async def journal_delete(session_id: Any) -> None:
    """Best-effort DEL of a session's journal key (used by retention / tests)."""
    r = _get_redis()
    if r is None:
        return
    try:
        await r.delete(_key(session_id))
    except Exception as exc:  # pragma: no cover
        logger.warning("plan_dialogue.journal_delete failed (session %s): %s", session_id, exc)


# --- L2: archive at Goal/Task creation ------------------------------------
#
# When a task-formation dialogue produces a Goal (or Task), the filtered
# transcript is materialized to ``.legion/plans/goal-{id}.md`` so the agent that
# later executes the work reads the user's own words — the reference links, the
# pasted 50-line nginx.conf — instead of ChatMaster's lossy paraphrase in
# ``task_data``. Long verbatim code blocks are lifted out into real files under
# a per-archive ``…-attachments/`` directory and referenced from an index, so a
# precise path lands in the task description. Everything is rendered through
# ``redact_secrets_in_text`` first: users paste configs with passwords in them
# and this file is read by every agent and sub-agent and rides along into volume
# backups.

_ROLE_LABEL = {"user": "User", "assistant": "ChatMaster"}

# ``` fence, optional info string, body, closing fence. Non-greedy body.
_FENCE_RE = re.compile(
    r"(?P<indent>[ \t]*)```(?P<info>[^\n`]*)\n(?P<body>.*?)(?P=indent)```",
    re.DOTALL,
)

_INFO_EXT = {
    "nginx": "conf", "conf": "conf", "ini": "ini", "toml": "toml",
    "yaml": "yaml", "yml": "yaml", "json": "json", "dockerfile": "dockerfile",
    "docker": "dockerfile", "bash": "sh", "sh": "sh", "shell": "sh",
    "python": "py", "py": "py", "javascript": "js", "js": "js", "ts": "ts",
    "typescript": "ts", "sql": "sql", "html": "html", "css": "css", "xml": "xml",
    "go": "go", "rust": "rs", "rs": "rs", "java": "java", "c": "c", "cpp": "cpp",
    "make": "mk", "makefile": "mk", "env": "env", "text": "txt", "": "txt",
}


def _redact(text: str) -> str:
    try:
        from app.core.secrets.redaction import redact_secrets_in_text

        return redact_secrets_in_text(text)
    except Exception:  # pragma: no cover - defensive
        return text


def _attachment_name(info: str, index: int) -> str:
    token = (info or "").strip().split()[0].lower() if (info or "").strip() else ""
    ext = _INFO_EXT.get(token, "txt")
    stem = token or "snippet"
    return f"{index:02d}-{stem}.{ext}"


def _extract_attachments(
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lift code blocks >= ``ATTACHMENT_MIN_LINES`` lines out of the transcript.

    Returns ``(attachments, rewritten_turns)``. Each attachment is
    ``{name, content, turn, lines}``; the matching span in the turn text is
    replaced with a one-line pointer so the transcript stays readable and small.
    """
    attachments: list[dict[str, Any]] = []
    rewritten: list[dict[str, Any]] = []
    counter = 1
    for t in turns:
        content = str(t.get("content") or "")

        def _sub(m: "Any") -> str:
            nonlocal counter
            body = m.group("body")
            line_count = body.count("\n") + (0 if body.endswith("\n") else 1)
            if line_count < ATTACHMENT_MIN_LINES:
                return m.group(0)
            name = _attachment_name(m.group("info"), counter)
            counter += 1
            attachments.append(
                {
                    "name": name,
                    "content": body if body.endswith("\n") else body + "\n",
                    "turn": t.get("turn"),
                    "lines": line_count,
                }
            )
            return (
                f"[attachment: `{name}` — {line_count} lines, "
                f"see the 'Attachments' section]"
            )

        rewritten.append({**t, "content": _FENCE_RE.sub(_sub, content)})
    return attachments, rewritten


def _truncate_middle(transcript: str, budget: int) -> str:
    if len(transcript) <= budget or budget <= 0:
        return transcript
    head = budget // 2
    tail = budget - head
    marker = "\n\n[… the middle of the dialogue is omitted, turns are available through plan_dialogue(mode=\"turns\") …]\n\n"
    return transcript[:head] + marker + transcript[-tail:]


async def _generate_digest(
    turns: list[dict[str, Any]],
    llm_config: Optional[dict[str, Any]],
    *,
    session: Any = None,
) -> str:
    """One cheap LLM call summarising the agreed requirements, or ``""``.

    Best-effort: any failure (no config, network, bad response) yields an empty
    string and the archive is written without the digest section — the turns are
    always there regardless.
    """
    if not llm_config:
        return ""
    convo = "\n\n".join(
        f"[#{t['turn']}] {_ROLE_LABEL.get(t['role'], t['role'])}: {t['content']}"
        for t in turns
    )[:60_000]
    prompt = (
        "Below is a task-setting dialogue between the user and the coordinator. "
        "List the agreed requirements — one per line, as a bulleted "
        "list, each with the number of its source turn in square brackets, "
        "for example '- Rate limit 100 r/s [#3]'. Requirements only, without "
        "an introduction or conclusion.\n\n" + convo
    )
    try:
        from app.core.proxy_session import _call_compact_summary

        # §8.1: the digest is the ChatMaster's smart model's job — do NOT let
        # _call_compact_summary swap in Haiku for Anthropic here.
        text = await _call_compact_summary(
            dict(llm_config),
            prompt,
            downgrade_model=False,
            purpose="plan_digest",
            log_session=session,
        )
        return (text or "").strip()
    except Exception as exc:
        logger.info("plan_dialogue: digest generation skipped: %s", exc)
        return ""


def _render_archive(
    *,
    title: str,
    header_id_label: str,
    header_id: Any,
    session: Any,
    turns: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
    digest: str,
    pending_plan: Optional[dict[str, Any]],
    attachments_dir: str,
    requirements_digest: str = "",
) -> str:
    session_id = getattr(session, "id", "?")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# Spec: {title}".rstrip(),
        "",
        f"> **{header_id_label}:** {header_id}  ·  **Session:** {session_id}  ·  **{now}**",
        "",
    ]

    if digest:
        lines += ["## Digest of agreed requirements", "", digest, ""]

    if requirements_digest:
        lines += [requirements_digest, ""]

    if attachments:
        lines += ["## Attachments", "", "| File | Turn | Lines |", "|------|-----|-------|"]
        for att in attachments:
            turn = f"[#{att['turn']}]" if att.get("turn") else "—"
            lines.append(f"| `{attachments_dir}/{att['name']}` | {turn} | {att['lines']} |")
        lines.append("")

    links = _extract_links(turns)
    if links:
        lines += ["## Links", ""]
        lines += [f"- {url}" for url in links]
        lines.append("")

    if pending_plan:
        plan_text = _render_pending_plan(pending_plan)
        if plan_text:
            lines += ["## Agreed plan", "", plan_text, ""]

    lines += ["---", ""]
    transcript_parts: list[str] = []
    for t in turns:
        label = _ROLE_LABEL.get(t["role"], t["role"])
        at = t.get("at") or ""
        at_short = at[11:16] if isinstance(at, str) and len(at) >= 16 else ""
        head = f"## [#{t['turn']}] {label}"
        if at_short:
            head += f" · {at_short}"
        transcript_parts.append(f"{head}\n\n{t['content']}")
    transcript = "\n\n".join(transcript_parts)

    body = "\n".join(lines)
    budget = ARCHIVE_MAX_CHARS - len(body)
    transcript = _truncate_middle(transcript, budget)
    return _redact(body + transcript + "\n")


_URL_RE = re.compile(r"https?://[^\s<>()\"'\]]+")


def _extract_links(turns: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for t in turns:
        for m in _URL_RE.finditer(str(t.get("content") or "")):
            url = m.group(0).rstrip(".,;")
            if url not in seen:
                seen.append(url)
    return seen


def _render_pending_plan(pending_plan: dict[str, Any]) -> str:
    parts: list[str] = []
    summary = str(pending_plan.get("plan_summary") or "").strip()
    if summary:
        parts.append(summary)
    steps = pending_plan.get("plan_steps") or pending_plan.get("steps") or []
    for i, st in enumerate(steps, start=1):
        if isinstance(st, dict):
            title = str(st.get("title") or "").strip()
            desc = str(st.get("description") or "").strip()
            line = f"{i}. {title}" if title else f"{i}."
            if desc:
                line += f" — {desc}"
        else:
            line = f"{i}. {st}"
        parts.append(line)
    return "\n".join(parts).strip()


async def _write_archive(
    room_id: Any,
    plan_path: str,
    attachments_dir: str,
    body: str,
    attachments: list[dict[str, Any]],
) -> Optional[str]:
    from app.core import workspaced_client as wsd

    try:
        await wsd.write_workspace_file(str(room_id), plan_path, body)
        for att in attachments:
            await wsd.write_workspace_file(
                str(room_id),
                f"{attachments_dir}/{att['name']}",
                _redact(att["content"]),
            )
    except Exception as exc:
        logger.warning(
            "plan_dialogue: archive write failed (%s): %s", plan_path, exc
        )
        return None
    return plan_path


async def archive_for_goal(
    room_id: Any,
    goal_id: Any,
    goal_title: str,
    session: Any,
    db: Any = None,
    *,
    pending_plan: Optional[dict[str, Any]] = None,
    llm_config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Materialize ``.legion/plans/goal-{goal_id}.md`` from the dialogue journal.

    Returns the archive path, or ``None`` when there is nothing to archive or
    the workspace write failed (never raises — a missing archive is degraded,
    not broken).
    """
    turns = await journal_read(session, db)
    if not turns:
        return None
    attachments, rewritten = _extract_attachments(turns)
    attachments_dir = f".legion/plans/goal-{goal_id}-attachments"
    # Stage 1.2.2: the digest is a model call; do not wait it out in a transaction.
    from app.db.release import release_db

    await release_db(db, reason="plan_dialogue/digest")
    digest = await _generate_digest(turns, llm_config, session=session)
    requirements_digest = await _load_requirements_digest(goal_id, db)
    body = _render_archive(
        title=goal_title or "(untitled)",
        header_id_label="Goal",
        header_id=goal_id,
        session=session,
        turns=rewritten,
        attachments=attachments,
        digest=digest,
        pending_plan=pending_plan,
        attachments_dir=attachments_dir,
        requirements_digest=requirements_digest,
    )
    return await _write_archive(
        room_id, f".legion/plans/goal-{goal_id}.md", attachments_dir, body, attachments
    )


async def archive_for_task(
    room_id: Any,
    task_id: Any,
    session: Any,
    db: Any = None,
    *,
    goal_id: Any = None,
    task_title: str = "",
    pending_plan: Optional[dict[str, Any]] = None,
    llm_config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Materialize ``.legion/plans/task-{task_id}.md`` — user_cm_p1 / direct sessions.

    Stage 6 wires the call sites; the renderer is shared with ``archive_for_goal``.
    """
    turns = await journal_read(session, db)
    if not turns:
        return None
    attachments, rewritten = _extract_attachments(turns)
    attachments_dir = f".legion/plans/task-{task_id}-attachments"
    # Stage 1.2.2: the digest is a model call; do not wait it out in a transaction.
    from app.db.release import release_db

    await release_db(db, reason="plan_dialogue/digest")
    digest = await _generate_digest(turns, llm_config, session=session)
    requirements_digest = await _load_requirements_digest(goal_id, db)
    body = _render_archive(
        title=task_title or "(untitled)",
        header_id_label="Task",
        header_id=task_id,
        session=session,
        turns=rewritten,
        attachments=attachments,
        digest=digest,
        pending_plan=pending_plan,
        attachments_dir=attachments_dir,
        requirements_digest=requirements_digest,
    )
    return await _write_archive(
        room_id, f".legion/plans/task-{task_id}.md", attachments_dir, body, attachments
    )


async def _load_requirements_digest(goal_id: Any, db: Any) -> str:
    """Stage 9 — the archive's "## Requirements" section.

    Best-effort: no *goal_id* (a task with no goal), no *db* (tests exercise
    the renderer standalone), or the query failing all degrade to "" rather
    than breaking the archive write — a plan archive without a requirements
    table is still a usable archive.
    """
    if goal_id is None or db is None:
        return ""
    try:
        from app.core.requirements import load_goal_requirements, render_requirements_digest

        rows = await load_goal_requirements(goal_id, db)
        return render_requirements_digest(rows, for_prompt=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue: requirements digest failed for goal %s: %s", goal_id, exc)
        return ""


async def append_to_task_archive(
    task_id: Any,
    room_id: Any,
    turns: list[dict[str, Any]],
    db: Any = None,
    *,
    goal_id: Any = None,
) -> Optional[str]:
    """Append a "Clarifications during execution" section — mid-task proxy dialogue.

    Stage 6: while an agent executes a task it may ask the ChatMaster a
    clarifying question ("which upstream? the pasted config or the current
    one?"). That Q&A carries requirement detail that is not in ``task_data`` and
    would otherwise be lost. Each round trip is appended to the task's archive
    (or, when the task only has a goal archive, to ``goal-{id}.md``) so a later
    rework or a sibling task can read it through ``plan_dialogue``.

    ``turns`` is ``[{role, content}]`` (``role`` ∈ ``agent`` | ``cm``). Best
    effort, idempotent: an identical trailing block is not appended twice.
    """
    clean = [
        t for t in turns
        if str(t.get("content") or "").strip()
        and not _is_synthetic(
            "user" if t.get("role") in ("agent", "user") else "assistant",
            t.get("content"),
        )
    ]
    if not clean:
        return None

    if goal_id is None and db is not None and task_id is not None:
        try:
            from app.models.task import Task

            task = await db.get(Task, task_id)
            goal_id = getattr(task, "goal_id", None) if task else None
        except Exception:  # pragma: no cover - defensive
            goal_id = None

    path = f".legion/plans/task-{task_id}.md"
    raw = await read_archive(room_id, path)
    if raw is None and goal_id is not None:
        alt = f".legion/plans/goal-{goal_id}.md"
        if await read_archive(room_id, alt) is not None:
            path, raw = alt, await read_archive(room_id, alt)
    if raw is None:
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    label = {"agent": "Executor", "cm": "ChatMaster", "user": "User"}
    block_lines = [f"### {now}", ""]
    for t in clean:
        who = label.get(t.get("role"), str(t.get("role") or "?"))
        block_lines.append(f"**{who}:** {str(t['content']).strip()}")
        block_lines.append("")
    block = "\n".join(block_lines).rstrip() + "\n"

    if block.strip() and block.strip() in raw:
        return path  # already appended

    heading = "## Clarifications during execution"
    if heading in raw:
        new_body = raw.rstrip() + "\n\n" + block
    else:
        new_body = raw.rstrip() + "\n\n" + heading + "\n\n" + block

    from app.core import workspaced_client as wsd

    try:
        await wsd.write_workspace_file(str(room_id), path, _redact(new_body))
    except Exception as exc:
        logger.warning(
            "plan_dialogue.append_to_task_archive: write failed (%s): %s", path, exc
        )
        return None
    return path


# --- L1: drafts for a dialogue that closed without a Goal/Task -----------------
#
# The router sometimes fails to recognise the user's next message as a
# continuation of an open session and cold-closes it. No Goal, no Plan, no
# archive — everything said before the abort is gone. Stage 5 fixes that: ANY
# close of a user↔CM / user↔agent session that did not produce a Goal or Task
# materializes ``.legion/drafts/{session_id}.md`` and pushes a head pointer onto
# a per-(room, user) Redis list. The next session's memory layer lists the
# candidates (id + topic + preview); ChatMaster calls ``merge_draft`` when the
# new message continues that same dialogue, and the carried turns are prepended
# to the live session (and to its journal, so a later archive is complete).

_DRAFT_HEAD_PREFIX = "plan_draft_head:"
_DRAFT_MERGED_PREFIX = "plan_draft_merged:"
_DRAFT_HEAD_TTL_SEC = DRAFT_WINDOW_HOURS * 3600 + 3600
_DRAFT_HEAD_MAX = 10
_DRAFT_CANDIDATE_MAX = 3
_DRAFT_PREVIEW_CHARS = 300
_MERGE_MARKER_PREFIX = "[unfinished dialogue merged: "
PENDING_MERGED_PREFIX_KEY = "pending_merged_prefix"


def _head_key(room_id: Any, user_id: Any) -> str:
    return f"{_DRAFT_HEAD_PREFIX}{room_id}:{user_id}"


def _merged_key(draft_session_id: Any) -> str:
    return f"{_DRAFT_MERGED_PREFIX}{draft_session_id}"


def _draft_path(session_id: Any) -> str:
    return f".legion/drafts/{session_id}.md"


def _topic_and_preview(turns: list[dict[str, Any]]) -> tuple[str, str]:
    """Deterministic topic / preview from the first real user turn."""
    first = next((t for t in turns if t.get("role") == "user"), None)
    if first is None:
        first = turns[0] if turns else {"content": ""}
    text = " ".join(str(first.get("content") or "").split())
    topic = text[:80] + ("…" if len(text) > 80 else "")
    preview = text[:_DRAFT_PREVIEW_CHARS] + ("…" if len(text) > _DRAFT_PREVIEW_CHARS else "")
    return topic or "(no topic)", preview


def _render_draft(session: Any, turns: list[dict[str, Any]], close_reason: str) -> str:
    session_id = getattr(session, "id", "?")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    topic, _ = _topic_and_preview(turns)
    lines = [
        f"# Dialogue draft: {topic}".rstrip(),
        "",
        f"> **Session:** {session_id}  ·  **Closed:** {now}  ·  "
        f"**Reason:** {close_reason}",
        "",
        "> The task-setting dialogue closed without producing a Goal or task. "
        "ChatMaster can merge it into the next message through `merge_draft`."
        "",
        "---",
        "",
    ]
    links = _extract_links(turns)
    if links:
        lines += ["## Links", ""] + [f"- {u}" for u in links] + ["", "---", ""]
    parts: list[str] = []
    for t in turns:
        label = _ROLE_LABEL.get(t["role"], t["role"])
        parts.append(f"## [#{t['turn']}] {label}\n\n{t['content']}")
    body = "\n".join(lines) + "\n\n".join(parts) + "\n"
    return _redact(_truncate_middle(body, ARCHIVE_MAX_CHARS))


async def save_draft(session: Any, db: Any = None, *, close_reason: str = "rotated") -> Optional[str]:
    """Materialize ``.legion/drafts/{session_id}.md`` and push a head pointer.

    Called on ANY close of a user-formation session that produced no Goal/Task.
    Best-effort — a workspace or Redis failure yields ``None`` and no pointer,
    never an exception (the session is already closing).
    """
    try:
        turns = await journal_read(session, db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue.save_draft: journal_read failed: %s", exc)
        return None
    if not turns:
        return None

    room_id = getattr(session, "room_id", None)
    user_id = getattr(session, "user_id", None)
    path = _draft_path(getattr(session, "id", "?"))
    body = _render_draft(session, turns, close_reason)

    from app.core import workspaced_client as wsd

    try:
        await wsd.write_workspace_file(str(room_id), path, body)
    except Exception as exc:
        logger.warning("plan_dialogue.save_draft: workspace write failed (%s): %s", path, exc)
        return None

    if room_id is not None and user_id is not None:
        topic, preview = _topic_and_preview(turns)
        entry = {
            "session_id": str(getattr(session, "id", "")),
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "close_reason": close_reason,
            "topic": topic,
            "preview": preview,
        }
        r = _get_redis()
        if r is not None:
            try:
                k = _head_key(room_id, user_id)
                await r.lpush(k, json.dumps(entry, ensure_ascii=False))
                await r.ltrim(k, 0, _DRAFT_HEAD_MAX - 1)
                await r.expire(k, _DRAFT_HEAD_TTL_SEC)
            except Exception as exc:
                logger.warning("plan_dialogue.save_draft: head pointer push failed: %s", exc)
    return path


async def list_draft_candidates(room_id: Any, user_id: Any, db: Any = None) -> list[dict[str, Any]]:
    """Recent un-merged drafts for this (room, user), newest first, capped.

    Returns ``[{session_id, closed_at, close_reason, topic, preview}]``. Empty
    when Redis is unavailable — the fallback here is simply "no suggestions".
    """
    r = _get_redis()
    if r is None:
        return []
    try:
        raw = await r.lrange(_head_key(room_id, user_id), 0, _DRAFT_HEAD_MAX - 1)
    except Exception as exc:
        logger.warning("plan_dialogue.list_draft_candidates: Redis read failed: %s", exc)
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - DRAFT_WINDOW_HOURS * 3600
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        try:
            e = json.loads(item)
        except Exception:
            continue
        sid = str(e.get("session_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        at = str(e.get("closed_at") or "")
        try:
            ts = datetime.fromisoformat(at).timestamp()
        except Exception:
            ts = None
        if ts is not None and ts < cutoff:
            continue
        try:
            if await r.get(_merged_key(sid)):
                continue
        except Exception:
            pass
        out.append(
            {
                "session_id": sid,
                "closed_at": at,
                "close_reason": e.get("close_reason"),
                "topic": e.get("topic") or "",
                "preview": e.get("preview") or "",
            }
        )
        if len(out) >= _DRAFT_CANDIDATE_MAX:
            break
    return out


async def _mark_draft_merged(draft_session_id: Any) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        await r.set(_merged_key(draft_session_id), "1")
        await r.expire(_merged_key(draft_session_id), JOURNAL_TTL_SEC)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue._mark_draft_merged failed: %s", exc)


async def clear_draft_heads(room_id: Any, user_id: Any) -> None:
    """Drop every head pointer for this (room, user) — a Goal/Task just landed."""
    if room_id is None or user_id is None:
        return
    r = _get_redis()
    if r is None:
        return
    try:
        await r.delete(_head_key(room_id, user_id))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue.clear_draft_heads failed: %s", exc)


async def _journal_prepend_many(session_id: Any, turns: list[dict[str, Any]]) -> None:
    """LPUSH carried turns to the FRONT of a session journal, order preserved."""
    r = _get_redis()
    if r is None:
        return
    try:
        for t in reversed(turns):
            entry = {
                "turn": None,
                "role": t.get("role"),
                "content": str(t.get("content") or ""),
                "at": t.get("at"),
            }
            await r.lpush(_key(session_id), json.dumps(entry, ensure_ascii=False))
        await r.expire(_key(session_id), JOURNAL_TTL_SEC)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue._journal_prepend_many failed: %s", exc)


async def merge_draft_into_session(
    session: Any, draft_session_id: Any, db: Any
) -> bool:
    """Prepend a closed draft's turns to *session* (history + journal).

    Idempotent: a second call with the same ``draft_session_id`` is a no-op.
    The draft is marked merged (not deleted) so a rollback after this point
    cannot lose it. Markdown is never parsed — the turns come from the journal.
    """
    from app.models.session.base import Session

    try:
        draft = await db.get(Session, draft_session_id)
    except Exception:
        draft = None
    if draft is None:
        return False

    draft_turns = await journal_read(draft, db)
    if not draft_turns:
        return False

    snap = dict(getattr(session, "context_snapshot", None) or {})
    msgs = list(snap.get("messages", []))
    marker = f"{_MERGE_MARKER_PREFIX}{draft_session_id}]"
    if any(
        isinstance(m, dict) and str(m.get("content") or "").startswith(marker)
        for m in msgs
    ):
        return True  # already merged

    prepend: list[dict[str, Any]] = [{"role": "assistant", "content": marker}]
    for t in draft_turns:
        prepend.append({"role": t["role"], "content": t["content"]})
    snap["messages"] = prepend + msgs
    # The turn that called merge_draft holds its own copy of the history and
    # saves it back when it ends, which would drop the prepend above. It picks
    # this up instead (user_sessions._adopt_merged_draft_prefix). Newest merge
    # first, matching the order in "messages".
    snap[PENDING_MERGED_PREFIX_KEY] = prepend + list(snap.get(PENDING_MERGED_PREFIX_KEY) or [])
    session.context_snapshot = snap
    try:
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(session, "context_snapshot")
    except Exception:
        pass

    await _journal_prepend_many(session.id, draft_turns)
    await _mark_draft_merged(draft_session_id)
    return True


# --- L3: read access for the executing agent ----------------------------------
#
# The dialogue archive is deliberately invisible to the ordinary file tools:
# ``.legion`` is stripped from ``repo_snapshot`` / ``list_files`` and the grep
# fallback never sees it without a live container. ``plan_dialogue`` is the
# supported way in — the scope (which goal / which task) is resolved from the
# session on the backend, so an agent cannot reach into another goal's archive
# by passing a path or an id.

_TURN_HEAD_RE = re.compile(r"^## \[#(\d+)\]\s*(.*)$", re.MULTILINE)
_DIGEST_MAX_CHARS = 6_000


async def read_archive(room_id: Any, path: str) -> Optional[str]:
    """Return the raw markdown of an archive file, or ``None``."""
    from app.core import workspaced_client as wsd

    try:
        return await wsd.read_workspace_file(str(room_id), path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("plan_dialogue.read_archive failed (%s): %s", path, exc)
        return None


def split_header_and_turns(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split a rendered archive into its digest header and the transcript turns.

    Returns ``(header, [{turn, label, body}])``. When the text has no ``[#N]``
    turn markers (a truncated or malformed archive), the whole text is the
    header and the turn list is empty.
    """
    matches = list(_TURN_HEAD_RE.finditer(text))
    if not matches:
        return text.strip(), []
    header = text[: matches[0].start()].rstrip()
    turns: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append(
            {
                "turn": int(m.group(1)),
                "label": (m.group(2) or "").strip(),
                "body": text[m.end():end].strip(),
            }
        )
    return header, turns


def _digest_section(header: str) -> str:
    """The header, trimmed to the digest / attachments / links / plan block."""
    cut = header.find("\n---")
    if cut != -1:
        header = header[:cut]
    return header.strip()[:_DIGEST_MAX_CHARS]


def _search_turns(
    turns: list[dict[str, Any]], query: str, budget: int
) -> list[str]:
    try:
        pat = re.compile(query, re.IGNORECASE)
    except re.error:
        pat = re.compile(re.escape(query), re.IGNORECASE)
    out: list[str] = []
    used = 0
    for t in turns:
        body = t["body"]
        m = pat.search(body)
        if not m:
            continue
        lo = max(0, m.start() - 300)
        hi = min(len(body), m.end() + 300)
        snippet = body[lo:hi].strip()
        if lo > 0:
            snippet = "… " + snippet
        if hi < len(body):
            snippet = snippet + " …"
        block = f"## [#{t['turn']}] {t['label']}\n\n{snippet}"
        if used + len(block) > budget and out:
            out.append("[… more matches omitted, narrow the query …]")
            break
        out.append(block)
        used += len(block)
    return out


async def read_dialogue_for_session(
    ctx: Any,
    *,
    mode: str = "search",
    query: str = "",
    turn_ids: Optional[list[int]] = None,
    max_chars: int = 6000,
) -> str:
    """Backend of the ``plan_dialogue`` tool — see ``core/tools/_plan_dialogue.py``.

    Resolves the scope (task archive, then goal archive) from the session row,
    never from a caller-supplied argument.
    """
    from app.models.session.base import Session
    from app.models.task import Task

    task_id = getattr(ctx, "task_id", None)
    goal_id = None
    session = None
    try:
        session = await ctx.db.get(Session, ctx.session_id)
    except Exception:  # pragma: no cover - defensive
        session = None
    if session is not None:
        task_id = task_id or getattr(session, "task_id", None)
        goal_id = getattr(session, "goal_id", None)
    if goal_id is None and task_id is not None:
        try:
            task = await ctx.db.get(Task, task_id)
            goal_id = getattr(task, "goal_id", None) if task else None
        except Exception:  # pragma: no cover - defensive
            goal_id = None

    sources: list[tuple[str, str]] = []
    if task_id is not None:
        sources.append(("task", f".legion/plans/task-{task_id}.md"))
    if goal_id is not None:
        sources.append(("goal", f".legion/plans/goal-{goal_id}.md"))
    if not sources:
        return (
            "No task or goal is linked to this session — there is no dialogue archive."
        )

    try:
        budget = max(500, min(int(max_chars), ARCHIVE_MAX_CHARS))
    except (TypeError, ValueError):
        budget = 6000

    loaded: list[tuple[str, str, list[dict[str, Any]]]] = []
    for label, path in sources:
        raw = await read_archive(ctx.room_id, path)
        if raw:
            header, turns = split_header_and_turns(raw)
            loaded.append((label, header, turns))

    if not loaded:
        return (
            "The agreement dialogue archive has not been created for this task/goal yet "
            "(the dialogue may not have produced a separate archive)."
        )

    parts: list[str] = []
    if mode == "digest":
        for label, header, _turns in loaded:
            parts.append(f"# Archive: {label}\n\n{_digest_section(header)}")
        return "\n\n---\n\n".join(parts)

    if mode == "turns":
        wanted = {int(x) for x in (turn_ids or []) if str(x).lstrip("-").isdigit()}
        if not wanted:
            return "Provide turn_ids — a list of turn numbers, e.g. [3, 7]."
        for label, _header, turns in loaded:
            picked = [t for t in turns if t["turn"] in wanted]
            if picked:
                body = "\n\n".join(
                    f"## [#{t['turn']}] {t['label']}\n\n{t['body']}" for t in picked
                )
                parts.append(f"# Archive: {label}\n\n{body}")
        text = "\n\n---\n\n".join(parts) if parts else "No turns with such numbers were found."
        return text[:budget]

    # mode == "search" (default)
    if not query.strip():
        return 'Provide query — what to search for in the dialogue (mode="digest" returns the whole header).'
    for label, _header, turns in loaded:
        hits = _search_turns(turns, query, budget // max(1, len(loaded)))
        if hits:
            parts.append(f"# Archive: {label}\n\n" + "\n\n".join(hits))
    if not parts:
        return f"Nothing found in the dialogue for the query '{query}'. Try mode=\"digest\"."
    return "\n\n---\n\n".join(parts)[:budget]


async def digest_for_decomposition(
    room_id: Any, *, goal_id: Any = None, task_id: Any = None
) -> Optional[str]:
    """Archive header (digest, attachments, links, agreed plan) for injection
    into the ``cm_self_plan`` decomposition turn — Stage 4.

    The task archive wins over the goal archive when both exist. Returns
    ``None`` when no archive has been written yet; decomposition then runs on
    the plan text alone.
    """
    candidates: list[str] = []
    if task_id is not None:
        candidates.append(f".legion/plans/task-{task_id}.md")
    if goal_id is not None:
        candidates.append(f".legion/plans/goal-{goal_id}.md")
    for path in candidates:
        raw = await read_archive(room_id, path)
        if not raw:
            continue
        header, _turns = split_header_and_turns(raw)
        section = _digest_section(header)
        if section:
            return section
    return None


# --- Stage 7: UI chip on the Goal / Task card --------------------------------
#
# "Discussion context: N messages, M attachments" — a one-line, at-a-glance sign
# that this Goal/Task carries a task-formation dialogue archive, with a link to
# read it and a way to detach it (``unlink_archive`` deletes the file; the API
# layer also clears ``origin_session_id``). All best-effort: a workspace that is
# down yields ``bound=False`` / ``ok=False``, never an exception.

_ATTACHMENT_ROW_RE = re.compile(r"^\|\s*`[^`]+`\s*\|\s*\[?#", re.MULTILINE)


async def archive_meta(
    room_id: Any, *, goal_id: Any = None, task_id: Any = None
) -> dict[str, Any]:
    """Summary of the dialogue archive bound to a goal/task, for the UI chip.

    The task archive wins over the goal archive when both exist. Returns
    ``{bound, kind, path, message_count, attachment_count, link_count,
    digest}``; ``bound`` is ``False`` when no archive file is present.
    """
    empty = {
        "bound": False,
        "kind": None,
        "path": None,
        "message_count": 0,
        "attachment_count": 0,
        "link_count": 0,
        "digest": None,
    }
    candidates: list[tuple[str, str]] = []
    if task_id is not None:
        candidates.append(("task", f".legion/plans/task-{task_id}.md"))
    if goal_id is not None:
        candidates.append(("goal", f".legion/plans/goal-{goal_id}.md"))
    for kind, path in candidates:
        raw = await read_archive(room_id, path)
        if not raw:
            continue
        header, turns = split_header_and_turns(raw)
        digest = _digest_section(header)
        link_count = sum(
            1 for ln in header.splitlines() if ln.strip().startswith("- http")
        )
        return {
            "bound": True,
            "kind": kind,
            "path": path,
            "message_count": len(turns),
            "attachment_count": len(_ATTACHMENT_ROW_RE.findall(header)),
            "link_count": link_count,
            "digest": digest or None,
        }
    return empty


async def unlink_archive(
    room_id: Any, *, goal_id: Any = None, task_id: Any = None
) -> bool:
    """Delete the dialogue archive file bound to a goal/task.

    Best-effort and idempotent: a missing file or a workspace error yields
    ``False``. The attachments directory is left for the retention sweep
    (Stage 8) — ``delete_workspace_file`` only removes one path at a time and a
    stale attachment folder is harmless.
    """
    from app.core import workspaced_client as wsd

    done = False
    targets: list[str] = []
    if task_id is not None:
        targets.append(f".legion/plans/task-{task_id}.md")
    if goal_id is not None:
        targets.append(f".legion/plans/goal-{goal_id}.md")
    for path in targets:
        try:
            if await wsd.delete_workspace_file(str(room_id), path):
                done = True
        except Exception as exc:
            logger.warning(
                "plan_dialogue.unlink_archive: delete failed (%s): %s", path, exc
            )
    return done


async def plan_dialogue_ref(room_id: Any, task_id: Any, goal_id: Any) -> Optional[dict[str, str]]:
    """``{"task": path, "goal": path}`` for the archives that actually exist."""
    ref: dict[str, str] = {}
    if task_id is not None:
        path = f".legion/plans/task-{task_id}.md"
        if await read_archive(room_id, path) is not None:
            ref["task"] = path
    if goal_id is not None:
        path = f".legion/plans/goal-{goal_id}.md"
        if await read_archive(room_id, path) is not None:
            ref["goal"] = path
    return ref or None
