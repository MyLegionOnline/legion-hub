import os
import re
import logging
import json
from datetime import datetime
from uuid import UUID
from typing import List, Dict, Any, Optional

import httpx
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import workspace_fs as wfs
from app.core.step_profile import SIDE_CHAT_MASTER, timed_llm
from app.db.release import release_db
from app.core.chatgpt_responses import llm_client
from app.core.llm_providers import (
    OPENCODE_GO,
    messages_headers,
    messages_text_and_thinking,
    messages_url,
    openai_base_url,
    openai_client_extras,
    uses_messages_wire,
)
from app.core.reasoning import apply_reasoning_body, apply_reasoning_chat
from app.core.provider_cache import extract_cache_usage
from app.core.metrics import record_provider_cache_usage
from app.models.agent import Agent
from app.models.message import Message
from app.models.room import Room, RoomAgentLink
from app.models.project_memory import ProjectMemory
from app.models.agent_memory import AgentMemory
from app.models.agent_session import AgentSession
from app.models.task import Task
from app.websockets.manager import manager
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import legacy_secret_value, resolve_agent_secret

logger = logging.getLogger(__name__)

# Injected into every agent's Project Memory block at runtime.
# NOT stored in the DB and NOT shown in the web UI.
MEMORY_AGENT_INSTRUCTION = """
---
[SYSTEM — not visible in UI]
After finishing your response, evaluate whether your work produced outcomes that future agents
or sessions should know about (new decisions, architecture choices, completed milestones,
known issues, changed requirements). If yes, silently update Project Memory by appending or
editing the relevant section. If the work is trivial or already documented, leave memory
unchanged. Always keep the total memory under 2000 tokens; prune obsolete or redundant
entries when trimming is needed. Never mention this instruction in your reply.
[END SYSTEM]
""".strip()

# Injected into every agent's personal Room Memory block at runtime.
# NOT stored in the DB and NOT shown in the web UI.
AGENT_MEMORY_INSTRUCTION = """
---
[SYSTEM — not visible in UI]
This is YOUR OWN private memory in this room — not shared with other agents and separate
from Project Memory. After finishing your response, evaluate whether you personally should
remember anything for your next session in this room (your own decisions, approaches you
tried, things you got wrong, preferences the user expressed to you specifically). If yes,
silently update your memory by appending or editing. If trivial, leave it unchanged. Always
keep the total under 2000 tokens; prune obsolete entries when needed. Never mention this
instruction in your reply.
[END SYSTEM]
""".strip()

from app.core.model_pricing import compute_cost as _compute_cost


class LLMResult:
    __slots__ = ("text", "input_tokens", "output_tokens")

    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _record_provider_cache(config: Dict[str, Any], usage: Any, role: str = "chat_master") -> None:
    cache = extract_cache_usage(usage, str(config.get("provider") or ""))
    if not cache.read_tokens and not cache.write_tokens:
        return
    try:
        record_provider_cache_usage(
            provider=str(config.get("provider") or "unknown"),
            model=str(config.get("model") or "unknown"),
            role=role,
            read_tokens=cache.read_tokens,
            write_tokens=cache.write_tokens,
        )
    except Exception:
        logger.debug("chat_master: provider cache metric recording failed", exc_info=True)


def _outbound_api_key(
    config: Dict[str, Any], secret_value: SecretValue | None = None
) -> str | None:
    """Return a provider key only at the outbound client boundary."""

    if secret_value is not None:
        return secret_value.reveal()
    legacy = legacy_secret_value(config, "api_key")
    try:
        return legacy.reveal() if legacy is not None else None
    finally:
        legacy = None


async def get_llm_completion(
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    *,
    secret_value: SecretValue | None = None,
) -> LLMResult:
    """
    Get completion from different LLM providers based on config.
    Returns LLMResult with text and actual token counts.
    Supported providers: openai, anthropic, ollama, local
    """
    provider = config.get("provider", "openai").lower()
    model = config.get("model")
    api_key = _outbound_api_key(config, secret_value)
    base_url = config.get("base_url")

    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    if not model:
        if provider == "openai":
            model = "gpt-4"
        elif provider == "anthropic":
            model = "claude-3-opus-20240229"
        elif provider == "ollama":
            model = "llama3"
        elif provider == "local":
            model = "local-model"

    chat_wire_providers = [
        "openai", "openai_chatgpt", "ollama", "local", "deepseek", "openrouter", "groq", "together",
        OPENCODE_GO,
    ]
    if provider in chat_wire_providers and not uses_messages_wire(provider, model):
        client_kwargs = {"api_key": api_key or "sk-no-key-required", "timeout": 60.0}
        client_kwargs["base_url"] = openai_base_url(provider, base_url)
        client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

        client = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
        create_extra: dict[str, Any] = {}
        apply_reasoning_chat(create_extra, config, provider, model)
        import asyncio as _asyncio
        for attempt in range(3):
            try:
                response = await timed_llm(
                    client.chat.completions.create(
                        model=model, messages=messages, **create_extra
                    ),
                    provider=provider,
                    side=SIDE_CHAT_MASTER,
                )
                text = response.choices[0].message.content
                usage = response.usage
                _record_provider_cache(config, usage)
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok)
            except Exception as e:
                if attempt < 2 and ("429" in str(e) or "rate" in str(e).lower()):
                    await _asyncio.sleep(2 ** attempt)
                    continue
                raise

    elif uses_messages_wire(provider, model):
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        other_messages = [m for m in messages if m["role"] != "system"]

        async with httpx.AsyncClient() as client:
            headers = messages_headers(provider, api_key)
            payload = {
                "model": model,
                "system": system_msg,
                "messages": other_messages,
                "max_tokens": 4096,
            }
            apply_reasoning_body(payload, config, provider, model)
            response = await timed_llm(
                client.post(
                    messages_url(provider, base_url),
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                ),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
            response.raise_for_status()
            data = response.json()
            text, _thinking = messages_text_and_thinking(data)
            usage = data.get("usage", {})
            _record_provider_cache(config, usage)
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok)

    # Generic fallback
    if base_url:
        client = llm_client(provider, chat_client_factory=AsyncOpenAI,
            api_key=api_key or "sk-no-key-required",
            base_url=base_url,
            **openai_client_extras(base_url),
        )
        response = await timed_llm(
            client.chat.completions.create(
                model=model or "default",
                messages=messages,
            ),
            provider=provider,
            side=SIDE_CHAT_MASTER,
        )
        text = response.choices[0].message.content
        usage = response.usage
        _record_provider_cache(config, usage)
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok)

    raise ValueError(f"Unsupported LLM provider: {provider}")


async def get_llm_json_completion(
    config: Dict[str, Any],
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    *,
    secret_value: SecretValue | None = None,
) -> LLMResult:
    """
    Like get_llm_completion but guarantees a valid JSON response matching `schema`.
    - OpenAI-compatible providers: uses response_format=json_object + json_schema
    - Anthropic: uses tool_use to force structured output
    Falls back to plain completion + manual parse if the provider doesn't support it.
    """
    provider = config.get("provider", "openai").lower()
    model = config.get("model")
    api_key = _outbound_api_key(config, secret_value)
    base_url = config.get("base_url")

    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    if not model:
        if provider == "openai":
            model = "gpt-4o"
        elif provider == "anthropic":
            model = "claude-sonnet-4-6"
        elif provider in ("ollama", "local"):
            model = "llama3"

    if uses_messages_wire(provider, model):
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        other_messages = [m for m in messages if m["role"] != "system"]
        tool_def = {
            "name": "structured_output",
            "description": "Output the structured plan as required.",
            "input_schema": schema,
        }
        async with httpx.AsyncClient() as client:
            headers = messages_headers(provider, api_key)
            payload = {
                "model": model,
                "system": system_msg,
                "messages": other_messages,
                "max_tokens": 4096,
                "tools": [tool_def],
                "tool_choice": {"type": "tool", "name": "structured_output"},
            }
            response = await timed_llm(
                client.post(
                    messages_url(provider, base_url),
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                ),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
            response.raise_for_status()
            data = response.json()
            tool_block = next(
                (b for b in data.get("content", []) if b.get("type") == "tool_use"),
                None,
            )
            if tool_block:
                text = json.dumps(tool_block["input"])
            else:
                text, _thinking = messages_text_and_thinking(data)
            usage = data.get("usage", {})
            _record_provider_cache(config, usage)
            return LLMResult(
                text=text,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

    # OpenAI-compatible providers
    client_kwargs: Dict[str, Any] = {"api_key": api_key or "sk-no-key-required", "timeout": 60.0}
    client_kwargs["base_url"] = openai_base_url(provider, base_url)
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    client = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)

    # Class B of the reasoning plan (§5.3.2): the Hub's own structured call, so
    # no reasoning fields are attached even when the agent has reasoning on.
    # The user paid for the agent to think about the *work*, not about the JSON
    # wrapper, and `response_format` together with reasoning is a combination
    # nothing in this plan measured.  Do not add apply_reasoning_* below.
    # Try structured output with json_schema first; fall back to json_object
    import asyncio as _asyncio
    for attempt in range(3):
        try:
            response = await timed_llm(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "plan", "strict": True, "schema": schema},
                    },
                ),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
            text = response.choices[0].message.content
            usage = response.usage
            _record_provider_cache(config, usage)
            return LLMResult(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            err = str(e)
            if "json_schema" in err or "response_format" in err or "unsupported" in err.lower():
                # Provider doesn't support json_schema — try json_object mode
                break
            if attempt < 2 and ("429" in err or "rate" in err.lower()):
                await _asyncio.sleep(2 ** attempt)
                continue
            raise

    # Fallback: json_object mode (supported by almost all OpenAI-compatible APIs)
    for attempt in range(3):
        try:
            response = await timed_llm(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                ),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
            text = response.choices[0].message.content
            usage = response.usage
            _record_provider_cache(config, usage)
            return LLMResult(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            err = str(e)
            if "json_object" in err or "response_format" in err or "unsupported" in err.lower():
                # Provider doesn't support response_format at all — fall through to plain completion
                break
            if attempt < 2 and ("429" in err or "rate" in err.lower()):
                await _asyncio.sleep(2 ** attempt)
                continue
            raise

    # Last resort: plain completion (same as get_llm_completion)
    return await get_llm_completion(config, messages, secret_value=secret_value)


async def get_project_memory(room_id: UUID, db: AsyncSession) -> Optional[ProjectMemory]:
    result = await db.execute(select(ProjectMemory).where(ProjectMemory.room_id == room_id))
    return result.scalars().first()


async def get_agent_memory(room_id: UUID, agent_id: UUID, db: AsyncSession) -> Optional[AgentMemory]:
    result = await db.execute(
        select(AgentMemory).where(AgentMemory.room_id == room_id, AgentMemory.agent_id == agent_id)
    )
    return result.scalars().first()


async def get_repo_snapshot(room_id: UUID, db: AsyncSession) -> List[Dict[str, str]]:
    """Return a bounded path manifest — never file contents or an unbounded tree."""
    files = await wfs.fs_list_files(room_id, db)
    # The complete tree remains available to explicit glob/repo_map calls.  A
    # prompt only needs enough orientation to choose its next on-demand read.
    return files[:200]


async def _read_file_chunk(
    room_id: UUID, path: str, from_line: int, to_line: int, db: AsyncSession
) -> str:
    """Return lines [from_line..to_line] (1-based, inclusive) of a repo file."""
    content = await wfs.fs_read_file(room_id, path, db)
    if content is None:
        return f"[ERROR: file {path} not found or empty]"
    if content.startswith("data:"):
        return f"[{path}] (binary file — content omitted)"
    lines = content.splitlines()
    total = len(lines)
    start = max(0, from_line - 1)
    end = min(total, to_line)
    chunk = "\n".join(lines[start:end])
    header = f"[{path} lines {start+1}–{end} of {total}]"
    return f"{header}\n{chunk}"


import re as _re

_READ_FILE_RE = _re.compile(
    r'<read_file\s+path=["\']([^"\']+)["\']\s+from_line=["\']?(\d+)["\']?\s+to_line=["\']?(\d+)["\']?\s*/>',
    _re.IGNORECASE,
)

def _parse_read_requests(text: str) -> List[Dict[str, Any]]:
    """Extract <read_file path="..." from_line="N" to_line="M"/> tags from agent reply."""
    requests = []
    for m in _READ_FILE_RE.finditer(text):
        requests.append({
            "path": m.group(1),
            "from_line": int(m.group(2)),
            "to_line": int(m.group(3)),
        })
    return requests


_MAX_AGENT_TURNS = 8  # max read-cycles per session to prevent runaway loops


async def run_agent_loop(
    agent: Agent,
    task_id: UUID,
    room_id: UUID,
    initial_user_message: str,
    db: AsyncSession,
    session_logger=None,
) -> tuple[str, int, int]:
    """
    Agentic loop: agent sees snapshot → requests file chunks → reads → continues.
    Returns (final_reply, total_input_tokens, total_output_tokens).
    """
    from app.core.skills import resolve_agent_skills_text

    context = await build_agent_context(task_id, db, agent_id=agent.id)
    skills_text = await resolve_agent_skills_text(agent, db)
    system_prompt = build_isolated_system_prompt(agent, context, skills_text=skills_text)
    agent_config = dict(agent.llm_config or {})

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message},
    ]

    total_in = 0
    total_out = 0
    final_reply = ""

    for turn in range(_MAX_AGENT_TURNS):
        if not agent_config.get("provider"):
            raise ValueError("Agent has no LLM provider configured")

        secret_value = await resolve_agent_secret(
            db=db,
            owner_id=agent.owner_id,
            agent_id=agent.id,
            config=agent_config,
            request_id=task_id,
        )
        try:
            await release_db(db, reason="chat_master/run_agent_loop")
            llm_result = await get_llm_completion(
                agent_config, messages, secret_value=secret_value
            )
        finally:
            secret_value = None
        reply = llm_result.text or ""
        total_in += llm_result.input_tokens
        total_out += llm_result.output_tokens

        read_requests = _parse_read_requests(reply)

        if session_logger:
            await session_logger.log_turn(
                turn=turn,
                messages_sent=messages,
                response=reply,
                tool_calls=read_requests,
                tool_results=[],
                usage={"input": llm_result.input_tokens, "output": llm_result.output_tokens},
                model=agent_config.get("model", ""),
            )

        if not read_requests:
            # No more file reads — this is the final answer
            final_reply = reply
            break

        logger.info(
            "Agent @%s turn %d: %d read_file request(s)", agent.nickname, turn + 1, len(read_requests)
        )

        # Execute reads and build tool-result message
        tool_results: List[str] = []
        for req in read_requests:
            chunk = await _read_file_chunk(room_id, req["path"], req["from_line"], req["to_line"], db)
            tool_results.append(chunk)

        tool_result_text = "\n\n".join(tool_results)

        # Append assistant turn and tool results as next user message
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": tool_result_text})
    else:
        # Exhausted turns — use last reply as-is
        final_reply = reply  # noqa: F821 (assigned in loop body)
        logger.warning("Agent @%s hit max turns (%d), using last reply", agent.nickname, _MAX_AGENT_TURNS)

    return final_reply, total_in, total_out


async def resolve_file_mentions(text: str, room_id: UUID, db: AsyncSession) -> str:
    """
    Replace @path/to/file.ext mentions with actual file content blocks.
    Supports both @files/foo.md and @/files/foo.md forms.
    """
    pattern = _re.compile(r"@(/?[\w./\-]+\.\w+)")
    replacements: List[tuple[str, str]] = []

    for match in pattern.finditer(text):
        raw_path = match.group(1)
        content = await wfs.fs_read_file(room_id, raw_path, db)

        if content:
            norm_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
            if content.startswith("data:"):
                block = f"[File: {norm_path}] (binary file — content omitted)"
            else:
                block = f"[File: {norm_path}]\n```\n{content}\n```"
            replacements.append((match.group(0), block))
            logger.info("Resolved file mention %s in room %s", norm_path, room_id)
        else:
            logger.warning("File mention %s not found in room %s", raw_path, room_id)

    for original, replacement in replacements:
        text = text.replace(original, replacement, 1)
    return text


async def _fetch_referenced_files(description: str, room_id: UUID, db: AsyncSession) -> List[Dict[str, str]]:
    """
    Extract file paths mentioned in task description and return their content.
    Detects patterns like 'files/foo.md', '/files/foo.md', or inside [File: ...] blocks.
    """
    patterns = [
        _re.compile(r"\[File:\s*([\w./\-]+)\]"),
        _re.compile(r"(?<!\w)(/?[\w./\-]+\.\w+)"),
    ]
    found_paths: List[str] = []
    for p in patterns:
        for m in p.finditer(description):
            path = m.group(1)
            if path not in found_paths:
                found_paths.append(path)

    files: List[Dict[str, str]] = []
    for raw_path in found_paths:
        content = await wfs.fs_read_file(room_id, raw_path, db)
        if content:
            norm_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
            if content.startswith("data:"):
                content = f"(binary file — content omitted)"
            files.append({"path": norm_path, "content": content})
    return files


async def get_last_session_summary(task_id: UUID, db: AsyncSession) -> Optional[str]:
    result = await db.execute(
        select(AgentSession)
        .where(AgentSession.task_id == task_id, AgentSession.summary.isnot(None))
        .order_by(AgentSession.started_at.desc())
        .limit(1)
    )
    session = result.scalars().first()
    return session.summary if session else None


async def build_agent_context(task_id: UUID, db: AsyncSession, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
    """
    Build an isolated context dict for an agent working on a task.
    Contains task details, project memory, repo snapshot, and previous session summary.
    This replaces the full room message history as the agent's context.
    """
    task = await db.get(Task, task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    memory = await get_project_memory(task.room_id, db)
    agent_memory = await get_agent_memory(task.room_id, agent_id, db) if agent_id else None
    repo_snapshot = await get_repo_snapshot(task.room_id, db)
    previous_summary = await get_last_session_summary(task_id, db)
    referenced_files = await _fetch_referenced_files(task.description or "", task.room_id, db)

    return {
        "task": {
            "id": str(task.id),
            "title": task.title,
            "description": task.description,
            "state": task.state,
        },
        "project_memory": memory.content if memory else "",
        "agent_memory": agent_memory.content if agent_memory else "",
        "repository_snapshot": repo_snapshot,
        "previous_session_summary": previous_summary,
        "referenced_files": referenced_files,
    }


def build_isolated_system_prompt(
    agent: Agent, context: Dict[str, Any], skills_text: Optional[str] = None
) -> str:
    """Compose a system prompt from isolated context rather than room history.

    skills_text is the pre-resolved skills section (see core.skills).
    """
    base_prompt = (agent.llm_config or {}).get("system_prompt", "")
    skills_text = skills_text or ""

    prev_summary = context["previous_session_summary"] or "This is the first session for this task."

    parts = [
        f"You are {agent.nickname}.",
        "",
        "## Your Task",
        f"**{context['task']['title']}**",
        context["task"]["description"],
        "",
    ]

    # Inject explicitly referenced file contents (from @-mentions in task description)
    ref_files = context.get("referenced_files") or []
    if ref_files:
        parts += ["## Referenced Files", ""]
        for f in ref_files:
            parts += [f"### {f['path']}", "```", f["content"], "```", ""]

    # Repo snapshot — bounded orientation only; use repo_map → get_outline →
    # read_file for task-relevant code.
    snapshot = context["repository_snapshot"]
    if snapshot:
        parts += [
            "## Repository Snapshot (bounded orientation — use repo_map, get_outline, then read_file)",
            "",
        ]
        for entry in snapshot[:40]:
            lc = entry.get("line_count", "?")
            parts += [f"- {entry['path']}  ({lc} lines, version {entry.get('version', 1)})"]
        if len(snapshot) > 40:
            parts += ["- *(snapshot truncated; use repo_map for ranked navigation)*"]
        parts += [""]

    memory_content = context["project_memory"] or "(empty)"
    agent_memory_content = context.get("agent_memory") or "(empty)"
    parts += [
        "## Project Memory",
        memory_content,
        "",
        MEMORY_AGENT_INSTRUCTION,
        "",
        "## Your Personal Memory (this room only)",
        agent_memory_content,
        "",
        AGENT_MEMORY_INSTRUCTION,
        "",
        "## Previous Session Summary",
        prev_summary,
        "",
        "## Reading Files",
        "You can read any file from the repository in chunks of up to 30 lines.",
        "To request a chunk, output ONE OR MORE of these tags anywhere in your reply:",
        "",
        '  <read_file path="/files/botinki.md" from_line="1" to_line="30"/>',
        "",
        "The system will return the requested lines and you can continue working.",
        "Read as many chunks as you need before writing your final output.",
        "Do NOT include read_file tags in your final answer — only in intermediate steps.",
        "",
        "## Saving Files",
        "When your response includes content that should be saved as a file, use this format:",
        "",
        "```<ext> /path/to/filename.ext",
        "file content here",
        "```",
        "",
        "Example:",
        "```html /index.html",
        "<h1>Hello</h1>",
        "```",
        "",
        "Use meaningful paths. Files are saved to the room repository and optionally synced to git.",
    ]

    if base_prompt:
        parts = [base_prompt, ""] + parts
    if skills_text:
        parts += ["", "## Skills", skills_text]

    return "\n".join(parts)

def _extract_files_from_reply(reply: str) -> List[Dict[str, str]]:
    """
    Parse agent reply for file blocks in the format:
      ```<ext> /path/to/file.ext
      content
      ```
    Returns list of {"path": str, "content": str}.
    """
    files: List[Dict[str, str]] = []
    # Match: ```<optional-lang> /absolute/path\ncontent\n```
    pattern = _re.compile(
        r"```(?:\w+)?\s+(/[^\n`]+\.\w+)\n(.*?)```",
        _re.DOTALL,
    )
    for m in pattern.finditer(reply):
        path = m.group(1).strip()
        content = m.group(2)
        # Strip trailing newline that's part of the fence, not the file
        if content.endswith("\n"):
            content = content[:-1]
        files.append({"path": path, "content": content})
    return files


async def _save_agent_files(
    room_id: UUID,
    agent_id: UUID,
    files: List[Dict[str, str]],
    db: AsyncSession,
) -> List[str]:
    """Save extracted files to the room repository. Returns list of saved paths."""
    saved: List[str] = []
    for f in files:
        try:
            await wfs.fs_write_file(room_id, f["path"], f["content"], db, agent_id=agent_id)
            saved.append(f["path"])
            logger.info("Agent saved file %s in room %s", f["path"], room_id)
        except Exception as exc:
            logger.warning("Failed to save agent file %s: %s", f["path"], exc)
    return saved


async def invoke_agent_with_context(
    agent: Agent,
    task_id: UUID,
    user_message: str,
    db: AsyncSession,
) -> str:
    """
    Invoke an internal agent using an isolated task context instead of full room history.
    Uses the agentic file-read loop so the agent can request file chunks as needed.
    """
    task = await db.get(Task, task_id)
    room_id = task.room_id if task else agent.owner_id  # fallback

    context = await build_agent_context(task_id, db, agent_id=agent.id)
    reply, in_tok, out_tok = await run_agent_loop(
        agent=agent,
        task_id=task_id,
        room_id=room_id,
        initial_user_message=user_message,
        db=db,
    )
    model_id = dict(agent.llm_config or {}).get("model", "")

    session = AgentSession(
        task_id=task_id,
        agent_id=agent.id,
        context_snapshot=context,
        summary=reply,
        ended_at=datetime.utcnow(),
        session_title=context.get("task", {}).get("title"),
        initiator_type="user",
        model_id=model_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        token_count=in_tok + out_tok,
        cost_usd=_compute_cost(model_id, in_tok, out_tok, agent.llm_config),
    )
    db.add(session)
    await db.commit()

    return reply


async def _resolve_master_config(room: Room, db: AsyncSession) -> tuple[Dict[str, Any], Optional[Agent]]:
    """Return (llm_config, master_agent) for the room coordinator."""
    coord_cfg = room.coordinator_config or {}
    master_agent_id = (
        coord_cfg.get("agent_id")
        or coord_cfg.get("chat_master_agent_id")
        or coord_cfg.get("master_agent_id")
    )
    if master_agent_id:
        agent_res = await db.execute(select(Agent).where(Agent.id == master_agent_id))
        master_agent = agent_res.scalars().first()
        if master_agent:
            return (master_agent.llm_config or {}), master_agent
    return coord_cfg, None


async def _broadcast_master_message(
    room_id: UUID,
    content: str,
    db: AsyncSession,
    master_agent: Optional[Agent] = None,
) -> Message:
    """Save and broadcast a Chat-Master MESSAGE event, attributed to the master agent if known."""
    from app.core.events import emit_event
    nickname = master_agent.nickname if master_agent else "Chat-Master"
    return await emit_event(
        room_id=room_id,
        event_type="MESSAGE",
        content=content,
        db=db,
        sender_agent_id=master_agent.id if master_agent else None,
        sender_type="master",
        metadata={"sender_type": "master", "nickname": nickname},
    )


_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "needs_tasks": {"type": "boolean"},
        "reply": {"type": ["string", "null"]},
        "confirmation_message": {"type": ["string", "null"]},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "agent_nickname": {"type": "string"},
                },
                "required": ["title", "description", "agent_nickname"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["needs_tasks", "reply", "confirmation_message", "tasks"],
    "additionalProperties": False,
}


async def plan_tasks(
    room: Room,
    user_request: str,
    room_agents: list[Agent],
    config: Dict[str, Any],
    db: AsyncSession,
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to produce a structured plan using JSON-mode / tool-use so that
    the response is always valid JSON — no manual parsing needed.
    Returns parsed plan dict or None on failure.
    """
    agents_info = "\n".join([
        f"- nickname: {a.nickname}" + (f" — {a.short_description}" if a.short_description else "")
        for a in room_agents
    ])
    memory = await get_project_memory(room.id, db)

    repo_snapshot = await get_repo_snapshot(room.id, db)
    repo_files_list = "\n".join(
        f"  - {e['path']} ({e.get('line_count', '?')} lines)" for e in repo_snapshot[:40]
    ) if repo_snapshot else "  (empty)"
    if len(repo_snapshot) > 40:
        repo_files_list += "\n  - *(manifest truncated; use repo_map/glob for navigation)*"

    system_prompt = f"""You are Chat-Master, an orchestrator that routes user requests to agents.

Available agents:
{agents_info or "(none yet)"}

Project memory:
{memory.content if memory else "(empty)"}

Repository files (agents can read these on demand):
{repo_files_list}

RULES:
- SIMPLE QUESTIONS / GREETINGS / STATUS → needs_tasks: false, set reply="your answer", confirmation_message=null, tasks=[].
- ANYTHING that requires building, writing, designing, or creating → needs_tasks: true, set reply=null, fill confirmation_message and tasks.
- agent_nickname MUST exactly match one of the nicknames listed above.
- When the task involves files from the repository, include their exact paths in the task description (e.g. "read /files/botinki.md and /files/valentki.md") — the agent will fetch the content itself.
- If the user's request contains [File: path] blocks, copy them verbatim into the task description.
- When the task requires writing or updating a file, explicitly state in the task description which file path must be saved (e.g. "save the result to /html/index.html").
- Always output valid JSON. All four fields (needs_tasks, reply, confirmation_message, tasks) must be present."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

    known_nicknames = {a.nickname for a in room_agents}
    try:
        master_id = (
            (room.coordinator_config or {}).get("agent_id")
            or (room.coordinator_config or {}).get("master_agent_id")
        )
        master_agent = await db.get(Agent, UUID(str(master_id))) if master_id else None
        secret_value = (
            await resolve_agent_secret(
                db=db,
                owner_id=master_agent.owner_id,
                agent_id=master_agent.id,
                config=config,
                request_id=room.id,
            )
            if master_agent
            else None
        )
        try:
            await release_db(db, reason="chat_master/plan_tasks")
            result = await get_llm_json_completion(
                config, messages, _PLAN_SCHEMA, secret_value=secret_value
            )
        finally:
            secret_value = None
        if not result.text or not result.text.strip():
            raise ValueError("LLM returned an empty response — model may not support JSON mode or hit a quota limit")
        plan = json.loads(result.text)

        # Validate plan structure — reject garbage LLM output early
        if plan.get("needs_tasks"):
            tasks = plan.get("tasks") or []
            if not tasks:
                raise ValueError("needs_tasks=true but tasks list is empty")
            for t in tasks:
                title = (t.get("title") or "").strip()
                nickname = (t.get("agent_nickname") or "").strip()
                if not title or title in ("...", "task_title", "string"):
                    raise ValueError(f"LLM returned placeholder task title: {title!r}")
                if nickname not in known_nicknames:
                    raise ValueError(
                        f"LLM returned unknown agent_nickname {nickname!r}. "
                        f"Known: {list(known_nicknames)}"
                    )
    except Exception as e:
        logger.error(f"plan_tasks: structured completion failed or produced invalid plan: {e}")
        plan = None
        result = None

    await _save_master_session(room, config, result, db)
    return plan


async def _save_master_session(
    room: Room,
    config: Dict[str, Any],
    result,  # LLMResult or None
    db: AsyncSession,
) -> None:
    """Persist a Chat-Master LLM call as a room-level AgentSession (no task)."""
    from app.models.agent_session import AgentSession as _AS

    # Resolve master agent id if available
    coord_cfg = room.coordinator_config or {}
    master_agent_id = (
        coord_cfg.get("agent_id")
        or coord_cfg.get("chat_master_agent_id")
        or coord_cfg.get("master_agent_id")
    )
    if not master_agent_id:
        return  # nothing to attach it to

    model_id = config.get("model", "")
    in_tok = result.input_tokens if result else 0
    out_tok = result.output_tokens if result else 0

    session = _AS(
        room_id=room.id,
        task_id=None,
        agent_id=master_agent_id,
        context_snapshot={},
        session_title="Chat-Master planning",
        initiator_type="user",
        model_id=model_id,
        ended_at=datetime.utcnow(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        token_count=in_tok + out_tok,
        cost_usd=_compute_cost(model_id, in_tok, out_tok, config),
    )
    db.add(session)
    await db.commit()


async def approve_and_start(
    room: Room,
    plan: Dict[str, Any],
    room_agents: list[Agent],
    config: Dict[str, Any],
    db: AsyncSession,
) -> None:
    """
    Create Tasks from a confirmed plan, assign agents, start sessions, update Room state.
    After each agent session, save summary and update project memory.
    """
    from app.core.events import emit_event
    from app.models.room import ROOM_TRANSITIONS, RoomState
    from app.models.task import TASK_TRANSITIONS, TaskState

    nickname_map = {a.nickname: a for a in room_agents}

    # Transition room to EXECUTION
    if RoomState.EXECUTION in ROOM_TRANSITIONS.get(room.state, []):
        room.state = RoomState.EXECUTION
        db.add(room)
        await db.commit()
        await emit_event(
            room_id=room.id,
            event_type="ROOM_STATE_CHANGED",
            content="Execution started.",
            db=db,
            metadata={"new_state": "execution"},
        )

    for task_spec in plan.get("tasks", []):
        requested_nickname = task_spec.get("agent_nickname", "")
        agent = nickname_map.get(requested_nickname)
        if not agent:
            # Try case-insensitive fallback
            lower_map = {k.lower(): v for k, v in nickname_map.items()}
            agent = lower_map.get(requested_nickname.lower())
        if not agent:
            logger.warning(
                "plan: unknown agent_nickname %r, available: %s",
                requested_nickname,
                list(nickname_map.keys()),
            )

        task = Task(
            room_id=room.id,
            title=task_spec["title"],
            description=task_spec.get("description", ""),
            state=TaskState.ASSIGNED,
            assigned_agent_id=agent.id if agent else None,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)

        await emit_event(
            room_id=room.id,
            event_type="TASK_CREATED",
            content=f"Task created: {task.title}",
            db=db,
            task_id=task.id,
            sender_type="master",
            metadata={
                "task_title": task.title,
                "task_state": task.state,
                "task_description": task.description or "",
                "agent_nickname": agent.nickname if agent else requested_nickname,
                "sender_type": "master",
            },
        )

        if not agent:
            await _broadcast_master_message(
                room.id,
                f"⚠️ No agent found for nickname '{requested_nickname}'. Task '{task_spec['title']}' left unassigned.",
                db,
            )
            continue

        if not (agent.llm_config or {}).get("provider"):
            await _broadcast_master_message(
                room.id,
                f"⚠️ Agent @{agent.nickname} has no LLM provider configured. Configure it in Settings → Agents.",
                db,
            )
            continue

        # Start session
        task.state = TaskState.IN_PROGRESS
        task.updated_at = datetime.utcnow()
        db.add(task)
        await db.commit()

        await emit_event(
            room_id=room.id,
            event_type="SESSION_STARTED",
            content=f"Session started for task: {task.title} (agent: @{agent.nickname})",
            db=db,
            task_id=task.id,
            sender_type="master",
            metadata={"agent_nickname": agent.nickname, "sender_type": "master"},
        )

        try:
            from app.core.log_storage import get_log_buffer, get_log_archive
            from app.core.session_logger import SessionLogger

            agent_config_pre = dict(agent.llm_config or {})
            model_id_pre = agent_config_pre.get("model", "")

            # Create session upfront so failures are visible in the list
            session = AgentSession(
                task_id=task.id,
                agent_id=agent.id,
                context_snapshot={},
                session_title=task.title,
                initiator_type="master",
                model_id=model_id_pre,
                log_status="recording",
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)

            # internal design notes Stage 1: develop on `task/<id>` (off `dev` or
            # a goal branch), never straight on `dev`/`main`. Non-blocking - a
            # workspace without a repo just keeps writing on the current branch.
            try:
                from app.core.room_git_flow import ensure_task_branch
                await ensure_task_branch(str(room.id), task, db)
                await db.commit()
            except Exception as _exc:
                logger.warning("chat_master: could not prepare task branch for task %s: %s", task.id, _exc)

            session_logger = SessionLogger(
                session_id=session.id,
                user_id=room.creator_id,
                buffer=get_log_buffer(),
            )

            context = await build_agent_context(task.id, db, agent_id=agent.id)
            task_description = task_spec.get("description", task_spec["title"])
            # If the task involves writing/updating files, remind the agent of the save format.
            # This ensures _extract_files_from_reply can detect and persist the output.
            _write_keywords = ("update", "write", "add", "change",
                               "edit",
                               "save", "create", "modify")
            if any(kw in task_description.lower() for kw in _write_keywords):
                task_description = (
                    task_description
                    + "\n\n**Important:** for every file you change or create, "
                    "use the following save format (the path right after the language on the same line):\n"
                    "```html /html/index.html\n"
                    "<file contents>\n"
                    "```\n"
                    "Without this format the file will not be saved to the repository."
                )
            reply, in_tok, out_tok = await run_agent_loop(
                agent=agent,
                task_id=task.id,
                room_id=room.id,
                initial_user_message=task_description,
                db=db,
                session_logger=session_logger,
            )
            agent_config = dict(agent.llm_config or {})

            if not reply:
                raise ValueError("Agent returned an empty response — the LLM may have refused or hit a quota limit")

            # Save any files the agent produced into the room repository
            extracted_files = _extract_files_from_reply(reply)
            if extracted_files:
                saved_paths = await _save_agent_files(room.id, agent.id, extracted_files, db)
                for path in saved_paths:
                    await emit_event(
                        room_id=room.id,
                        event_type="FILE_CHANGED",
                        content=f"Agent @{agent.nickname} saved file: {path}",
                        db=db,
                        task_id=task.id,
                        metadata={"path": path, "agent_nickname": agent.nickname, "source": "agent"},
                    )
                # Git push is manual-only (Push to Git button) — no auto-push here.

            # Flush session log to S3
            log_key, log_bytes = await session_logger.flush_to_s3(get_log_archive())

            # Update the pre-created session with final data
            model_id = agent_config.get("model", "")
            session.context_snapshot = context
            session.summary = reply
            session.ended_at = datetime.utcnow()
            session.model_id = model_id
            session.input_tokens = in_tok
            session.output_tokens = out_tok
            session.token_count = in_tok + out_tok
            session.cost_usd = _compute_cost(model_id, in_tok, out_tok, agent_config)
            session.log_key = log_key or None
            session.log_size_bytes = log_bytes
            session.log_status = "flushed" if log_key else "recording"
            db.add(session)

            db.add(task)
            await db.commit()

            await emit_event(
                room_id=room.id,
                event_type="SESSION_COMPLETED",
                content=f"@{agent.nickname} completed: {task.title}",
                db=db,
                task_id=task.id,
                sender_type="master",
                metadata={"agent_nickname": agent.nickname, "summary_preview": reply[:200], "sender_type": "master"},
            )

            # Merge session summary into project memory and this agent's personal room memory
            await update_project_memory(room.id, reply, config, db)
            await update_agent_memory(room.id, agent.id, reply, agent_config, db, agent=agent)

            # internal design notes Stages 2–4 — Gate 1 in one call: quick checks
            # on the task branch, deterministic risk class, then one of three
            # outcomes — BLOCKED (red), auto-merged into `dev` (`trivial`), or
            # WAITING_REVIEW for a human. All the events (CI_*, TASK_*) and the
            # Gate 2 `promote_policy` hook come from there. Shared with the
            # dialogue path (internal design notes) so the merge policy
            # exists exactly once.
            from app.core.room_git_flow import finalize_task_gate1

            await finalize_task_gate1(room, task, db)

            # Broadcast agent reply as a message
            await emit_event(
                room_id=room.id,
                event_type="MESSAGE",
                content=f"@{agent.nickname}: {reply}",
                db=db,
                sender_agent_id=agent.id,
                task_id=task.id,
                metadata={"nickname": agent.nickname},
            )

        except Exception as e:
            import traceback as _tb
            logger.error(f"Agent session failed for task {task.id}: {e}")
            # Log the error turn and flush what we have
            try:
                await session_logger.log_error(str(e), _tb.format_exc())
                log_key, log_bytes = await session_logger.flush_to_s3(get_log_archive())
                session.log_key = log_key or None
                session.log_size_bytes = log_bytes
                session.log_status = "flush_failed"
                session.ended_at = datetime.utcnow()
                db.add(session)
                await db.commit()
            except Exception:
                pass
            # Reset task so it doesn't stay stuck in IN_PROGRESS
            task.state = TaskState.ASSIGNED
            task.updated_at = datetime.utcnow()
            db.add(task)
            await db.commit()
            await emit_event(
                room_id=room.id,
                event_type="TASK_STATE_CHANGED",
                content=f"Task reset to assigned after error.",
                db=db,
                task_id=task.id,
                sender_type="master",
                metadata={"task_title": task.title, "task_state": "assigned", "sender_type": "master"},
            )
            await _broadcast_master_message(
                room.id,
                f"⚠️ Agent @{agent.nickname} failed on task '{task.title}': {e}",
                db,
            )

    await monitor_completion(room, db)


async def monitor_completion(room: Room, db: AsyncSession) -> None:
    """Check if all tasks are COMPLETED and transition room to COMPLETED if so."""
    from app.models.room import ROOM_TRANSITIONS, RoomState
    from app.models.task import TaskState
    from app.core.events import emit_event

    result = await db.execute(select(Task).where(Task.room_id == room.id))
    tasks = result.scalars().all()

    if not tasks:
        return

    terminal = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.REJECTED}
    all_done = all(t.state in terminal for t in tasks)
    any_completed = any(t.state == TaskState.COMPLETED for t in tasks)
    if all_done and any_completed and RoomState.COMPLETED in ROOM_TRANSITIONS.get(room.state, []):
        room.state = RoomState.COMPLETED
        db.add(room)
        await db.commit()
        await emit_event(
            room_id=room.id,
            event_type="ROOM_STATE_CHANGED",
            content="All tasks completed.",
            db=db,
            metadata={"new_state": "completed"},
        )
        await emit_event(
            room_id=room.id,
            event_type="MESSAGE",
            content="All tasks completed. Project is done.",
            db=db,
            metadata={"sender_type": "master", "nickname": "Chat-Master"},
        )


async def update_project_memory(
    room_id: UUID,
    session_summary: str,
    config: Dict[str, Any],
    db: AsyncSession,
    agent: Optional[Agent] = None,
) -> None:
    """Merge session_summary into project memory via LLM (or append-and-trim fallback)."""
    from app.core.events import emit_event

    if agent is None:
        room = await db.get(Room, room_id)
        master_id = (room.coordinator_config or {}).get("agent_id") if room else None
        master_id = master_id or ((room.coordinator_config or {}).get("master_agent_id") if room else None)
        if master_id:
            try:
                agent = await db.get(Agent, UUID(str(master_id)))
            except (TypeError, ValueError):
                agent = None

    memory = await get_project_memory(room_id, db)
    if memory is None:
        from app.models.project_memory import ProjectMemory
        memory = ProjectMemory(room_id=room_id, content="# Project Summary\n\n")
        db.add(memory)
        await db.commit()

    existing = memory.content

    TOKEN_LIMIT = 2000
    CHAR_LIMIT = TOKEN_LIMIT * 4  # ~4 chars per token

    if config.get("provider"):
        merge_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a project memory curator. Your job:\n"
                    "1. Read the session summary and decide whether it contains information "
                    "that future agents or sessions need to know (decisions, milestones, "
                    "architecture changes, known issues, completed features).\n"
                    "2. If YES — update the relevant section(s) of Project Memory with a "
                    "concise addition. Remove outdated or redundant entries to stay under "
                    f"{CHAR_LIMIT} characters (~{TOKEN_LIMIT} tokens).\n"
                    "3. If NO (trivial work, already documented, or no future impact) — "
                    "return the existing memory UNCHANGED.\n"
                    "Return ONLY the final memory document text, no commentary, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Current Project Memory\n{existing}\n\n"
                    f"## Session Summary\n{session_summary}"
                ),
            },
        ]
        try:
            secret_value = None
            if agent is not None:
                secret_value = await resolve_agent_secret(
                    db=db,
                    owner_id=agent.owner_id,
                    agent_id=agent.id,
                    config=config,
                    request_id=agent.id,
                )
            try:
                await release_db(db, reason="chat_master/project_memory")
                updated = (
                    await get_llm_completion(
                        config, merge_prompt, secret_value=secret_value
                    )
                ).text
            finally:
                secret_value = None
        except Exception as e:
            logger.warning(f"Memory merge LLM failed: {e}, falling back to append")
            updated = existing + f"\n\n---\n{session_summary}"
    else:
        updated = existing + f"\n\n---\n{session_summary}"

    # Hard cap at CHAR_LIMIT — trim from the top (oldest content) if exceeded
    if len(updated) > CHAR_LIMIT:
        updated = updated[-CHAR_LIMIT:]

    memory.content = updated
    memory.token_count = len(updated) // 4
    memory.updated_at = datetime.utcnow()
    db.add(memory)
    await db.commit()

    await emit_event(
        room_id=room_id,
        event_type="MEMORY_UPDATED",
        content="Project memory updated after session.",
        db=db,
        metadata={"token_count": memory.token_count},
    )


async def _merge_agent_memory_content(
    current: str,
    session_summary: str,
    llm_config: Optional[Dict[str, Any]],
    agent: Optional[Agent],
    db: AsyncSession,
) -> str:
    """Merge a session summary into an agent's personal room memory via LLM (or append-and-trim)."""
    TOKEN_LIMIT = 2000
    CHAR_LIMIT = TOKEN_LIMIT * 4  # ~4 chars per token
    nickname = agent.nickname if agent else "the agent"

    if llm_config and llm_config.get("provider"):
        merge_prompt = [
            {
                "role": "system",
                "content": (
                    f"You are curating the private memory of the agent {nickname} for this room. "
                    "This memory is personal to this agent and not shared with other agents.\n"
                    "1. Read the session summary and decide whether it contains something this "
                    "agent should personally remember for its next session in this room (its own "
                    "decisions, approaches it tried, mistakes to avoid, room-specific preferences).\n"
                    "2. If YES — update the memory with a concise addition. Remove outdated or "
                    f"redundant entries to stay under {CHAR_LIMIT} characters (~{TOKEN_LIMIT} tokens).\n"
                    "3. If NO (trivial work, already documented, or no future relevance) — return "
                    "the existing memory UNCHANGED.\n"
                    "Return ONLY the final memory document text, no commentary, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Current Memory\n{current}\n\n"
                    f"## Session Summary\n{session_summary}"
                ),
            },
        ]
        try:
            secret_value = None
            if agent is not None:
                secret_value = await resolve_agent_secret(
                    db=db,
                    owner_id=agent.owner_id,
                    agent_id=agent.id,
                    config=llm_config,
                    request_id=agent.id,
                )
            try:
                await release_db(db, reason="chat_master/agent_memory")
                updated = (
                    await get_llm_completion(
                        llm_config, merge_prompt, secret_value=secret_value
                    )
                ).text
            finally:
                secret_value = None
        except Exception as e:
            logger.warning(f"Agent memory merge LLM failed: {e}, falling back to append")
            updated = current + f"\n\n---\n{session_summary}"
    else:
        updated = current + f"\n\n---\n{session_summary}"

    if len(updated) > CHAR_LIMIT:
        updated = updated[-CHAR_LIMIT:]
    return updated


async def update_agent_memory(
    room_id: UUID,
    agent_id: UUID,
    session_summary: str,
    llm_config: Optional[Dict[str, Any]],
    db: AsyncSession,
    agent: Optional[Agent] = None,
) -> None:
    """Merge session_summary into this agent's personal room memory. No-op if the agent
    has no memory row (i.e. it isn't currently a member of the room)."""
    from app.core.events import emit_event

    memory = await get_agent_memory(room_id, agent_id, db)
    if memory is None:
        return

    updated = await _merge_agent_memory_content(
        memory.content, session_summary, llm_config, agent, db
    )

    memory.content = updated
    memory.token_count = len(updated) // 4
    memory.updated_at = datetime.utcnow()
    db.add(memory)
    await db.commit()

    await emit_event(
        room_id=room_id,
        event_type="AGENT_MEMORY_UPDATED",
        content="Agent memory updated after session.",
        db=db,
        metadata={"agent_id": str(agent_id), "token_count": memory.token_count},
    )


async def _classify_confirmation(
    user_message: str,
    pending_plan: Dict[str, Any],
    config: Dict[str, Any],
    *,
    db: AsyncSession,
    agent: Optional[Agent],
) -> str:
    """Ask the LLM whether the user confirmed, rejected, or ignored the pending plan.
    Returns: 'confirmed' | 'rejected' | 'other'
    """
    plan_summary = pending_plan.get("confirmation_message", "")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. The assistant proposed a plan to the user and is waiting for their response.\n"
                "Classify the user's reply as one of:\n"
                "- confirmed: user agrees to start the plan, in any language (e.g. 'yes', 'ok', 'do it', 'go ahead', 'sure', 'start')\n"
                "- rejected: user cancels or declines, in any language (e.g. 'no', 'cancel', 'stop', 'never mind')\n"
                "- other: user is asking a question, changing the request, or saying something unrelated\n"
                "Respond with exactly one word: confirmed, rejected, or other."
            ),
        },
        {
            "role": "user",
            "content": f"Proposed plan: {plan_summary}\n\nUser reply: {user_message}",
        },
    ]
    try:
        secret_value = (
            await resolve_agent_secret(
                db=db,
                owner_id=agent.owner_id,
                agent_id=agent.id,
                config=config,
                request_id=agent.id,
            )
            if agent is not None
            else None
        )
        try:
            await release_db(db, reason="chat_master/classify_confirmation")
            result = (
                await get_llm_completion(config, messages, secret_value=secret_value)
            ).text
        finally:
            secret_value = None
        result = result.strip().lower().split()[0]
        if result in ("confirmed", "rejected", "other"):
            return result
    except Exception as e:
        logger.warning(f"_classify_confirmation failed: {e}")
    return "other"


async def chat_master_process(room_id: UUID, content: str, db: AsyncSession):
    result = await db.execute(select(Room).where(Room.id == room_id))
    room = result.scalars().first()

    if not room or not room.has_coordinator:
        return

    logger.info(f"Chat-Master triggered in room {room.name} ({room_id})")

    config, master_agent = await _resolve_master_config(room, db)
    if not config.get("provider"):
        logger.warning(f"Chat-Master: no LLM provider configured for room {room_id}")
        return

    coord_cfg = dict(room.coordinator_config or {})

    # ── Auto-recover room stuck in EXECUTION with no active tasks ────────────
    from app.models.room import RoomState, ROOM_TRANSITIONS
    from app.models.task import TaskState as _TS
    if room.state == RoomState.EXECUTION:
        tasks_res = await db.execute(select(Task).where(Task.room_id == room_id))
        all_tasks = tasks_res.scalars().all()
        terminal_states = {_TS.COMPLETED, _TS.CANCELLED, _TS.REJECTED}
        if all_tasks and all(t.state in terminal_states for t in all_tasks):
            room.state = RoomState.PLANNING
            db.add(room)
            await db.commit()
            from app.core.events import emit_event
            await emit_event(
                room_id=room_id,
                event_type="ROOM_STATE_CHANGED",
                content="All tasks are completed. The room is back in the planning state.",
                db=db,
                metadata={"new_state": "planning"},
            )
            logger.info("Auto-recovered room %s from stuck EXECUTION to PLANNING", room_id)

    # ── If a pending plan is waiting for user confirmation ───────────────────
    pending_plan = coord_cfg.get("pending_plan")
    if pending_plan:
        intent = await _classify_confirmation(
            content, pending_plan, config, db=db, agent=master_agent
        )
        confirmed = intent == "confirmed"
        rejected = intent == "rejected"

        if confirmed:
            coord_cfg.pop("pending_plan", None)
            room.coordinator_config = coord_cfg
            db.add(room)
            await db.commit()

            result2 = await db.execute(
                select(Agent).join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
                .where(RoomAgentLink.room_id == room_id)
            )
            room_agents = result2.scalars().all()

            await _broadcast_master_message(room_id, "Plan confirmed. Starting execution...", db, master_agent)
            await approve_and_start(room, pending_plan, room_agents, config, db)
            return

        elif rejected:
            coord_cfg.pop("pending_plan", None)
            room.coordinator_config = coord_cfg
            from app.models.room import RoomState, ROOM_TRANSITIONS
            from app.core.events import emit_event
            if RoomState.PLANNING in ROOM_TRANSITIONS.get(room.state, []):
                room.state = RoomState.PLANNING
            db.add(room)
            await db.commit()
            await emit_event(room_id=room_id, event_type="ROOM_STATE_CHANGED", content="Plan rejected.", db=db, metadata={"new_state": "planning"})
            await _broadcast_master_message(room_id, "Plan cancelled. Let me know if you want to start over.", db, master_agent)
            return

    # ── Normal flow: ask LLM to plan or reply ───────────────────────────────
    result2 = await db.execute(
        select(Agent).join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
        .where(RoomAgentLink.room_id == room_id)
    )
    room_agents = result2.scalars().all()

    # Resolve @file mentions before planning so the LLM sees actual file contents
    resolved_content = await resolve_file_mentions(content, room_id, db)

    try:
        plan = await plan_tasks(room, resolved_content, room_agents, config, db)
    except Exception as e:
        logger.error(f"Chat-Master plan_tasks error: {e}")
        await _broadcast_master_message(room_id, f"⚠️ LLM error: {e}", db, master_agent)
        return

    if plan is None:
        # Reset state so the room doesn't stay stuck in WAITING_USER
        coord_cfg.pop("pending_plan", None)
        room.coordinator_config = coord_cfg
        from app.models.room import RoomState, ROOM_TRANSITIONS
        from app.core.events import emit_event
        if RoomState.PLANNING in ROOM_TRANSITIONS.get(room.state, []):
            room.state = RoomState.PLANNING
        db.add(room)
        await db.commit()
        await emit_event(room_id=room_id, event_type="ROOM_STATE_CHANGED", content="Plan parse failed.", db=db, metadata={"new_state": "planning"})
        await _broadcast_master_message(room_id, "Sorry, I couldn't parse the plan. Please try again.", db, master_agent)
        return

    if not plan.get("needs_tasks"):
        await _broadcast_master_message(room_id, plan.get("reply", "(no reply)"), db, master_agent)
        return

    # Plan requires tasks — ask user for confirmation
    confirmation_msg = plan.get("confirmation_message", "Here is the plan. Reply 'yes' to confirm or 'no' to cancel.")
    coord_cfg["pending_plan"] = plan
    room.coordinator_config = coord_cfg
    from app.models.room import RoomState, ROOM_TRANSITIONS
    from app.core.events import emit_event
    if RoomState.WAITING_USER in ROOM_TRANSITIONS.get(room.state, []):
        room.state = RoomState.WAITING_USER
    db.add(room)
    await db.commit()

    await emit_event(
        room_id=room_id,
        event_type="ROOM_STATE_CHANGED",
        content="Room is now waiting for your confirmation.",
        db=db,
        metadata={
            "new_state": "waiting_user",
            "confirmation_message": confirmation_msg,
            "tasks": plan.get("tasks", []),
        },
    )
