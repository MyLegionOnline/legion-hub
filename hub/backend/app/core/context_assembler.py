"""Pairing-safe active context compaction and request preflight."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context_policy import (
    ContextPolicy,
    context_limit_mode,
    uses_cumulative_limit,
)
from app.core.token_counter import TokenCounter, compact_json

_MUTATING_TOOLS = frozenset({"edit_file", "write_file", "bash", "git_apply", "git_commit"})
_READ_TOOLS = frozenset({"read_file", "observation_read"})

# Mirrors the default page size of the read_file tool, so a result evicted
# without an explicit limit is attributed to the range it actually served.
_DEFAULT_READ_LIMIT = 100

# Tools that keep their full payload externally stamp the ref into their output.
_OBSERVATION_REF = re.compile(
    r"observation_ref=['\"]?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

# Bulky arguments of file-writing calls.  Once such a call has run, its payload
# lives in the file itself, so a past call can keep its path and a receipt
# instead of the whole file body — and must give way before fresh results do.
_ARG_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "write_file": ("content",),
    "edit_file": ("old_string", "new_string"),
}

# Keep enough successful assemblies for the operational 24-hour prompt-size
# report without turning the session snapshot into an unbounded event log.
_SUCCESSFUL_ASSEMBLY_HISTORY_MAX = 64

logger = logging.getLogger(__name__)


@dataclass
class ContextAssembly:
    messages: list[dict[str, Any]]
    estimated_input_tokens: int
    active_tool_tokens: int
    active_tool_tokens_by_name: dict[str, int]
    evicted_tokens: int
    evicted_results: int
    dropped_messages: int
    dropped_tokens: int
    checkpoint: dict[str, Any]
    rejected_reason: str | None = None
    eviction_mode: str = "on_demand"
    #: Reads whose text this assembly removed from the active context, as
    #: ``{"path", "start", "end"}`` (``None`` bounds mean the whole file).
    evicted_reads: list[dict[str, Any]] = field(default_factory=list)
    #: Observation refs whose payload this assembly removed from the context.
    evicted_observations: list[str] = field(default_factory=list)
    #: Arguments of past write_file/edit_file calls replaced by receipts.
    compacted_args: int = 0
    compacted_arg_tokens: int = 0

    @property
    def allowed(self) -> bool:
        return self.rejected_reason is None


@dataclass
class _ToolResult:
    message_index: int
    block_index: int | None
    call_id: str
    tool_name: str
    args: dict[str, Any]
    tokens: int
    content: Any
    is_error: bool = False


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return compact_json(value)


def _tool_calls(messages: list[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                result[str(block.get("id") or "")] = (
                    str(block.get("name") or "unknown"),
                    block.get("input") if isinstance(block.get("input"), dict) else {},
                )
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (TypeError, ValueError, json.JSONDecodeError):
                args = {}
            result[str(call.get("id") or "")] = (
                str(function.get("name") or "unknown"),
                args,
            )
    return result


def _tool_results(
    messages: list[dict[str, Any]],
    calls: dict[str, tuple[str, dict[str, Any]]],
    counter: TokenCounter,
) -> list[_ToolResult]:
    results: list[_ToolResult] = []
    for message_index, message in enumerate(messages):
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name, args = calls.get(call_id, ("unknown", {}))
            results.append(
                _ToolResult(
                    message_index=message_index,
                    block_index=None,
                    call_id=call_id,
                    tool_name=name,
                    args=args,
                    tokens=counter.count(message),
                    content=message.get("content"),
                    is_error=bool(message.get("is_error")),
                )
            )
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = str(block.get("tool_use_id") or "")
            name, args = calls.get(call_id, ("unknown", {}))
            results.append(
                _ToolResult(
                    message_index=message_index,
                    block_index=block_index,
                    call_id=call_id,
                    tool_name=name,
                    args=args,
                    tokens=counter.count(block),
                    content=block.get("content"),
                    is_error=bool(block.get("is_error")),
                )
            )
    return results


def _target(args: dict[str, Any]) -> str:
    value = (
        args.get("path")
        or args.get("ref")
        or args.get("pattern")
        or args.get("cmd")
        or args.get("command")
    )
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")[:160]


def _result_status(result: _ToolResult) -> str:
    if result.is_error:
        return "error"
    text = _text(result.content).casefold()
    match = re.search(r"\bexit_code\s*[:=]\s*(-?\d+)", text)
    if match:
        return "ok" if int(match.group(1)) == 0 else f"exit_{match.group(1)}"
    if any(word in text for word in ("traceback", "error:", "failed")):
        return "error"
    return "ok"


def _receipt(result: _ToolResult) -> str:
    target = _target(result.args)
    target_part = f"; target={target}" if target else ""
    range_part = ""
    if result.tool_name in _READ_TOOLS:
        offset = result.args.get("offset", 1)
        limit = result.args.get("limit", 100)
        range_part = f"; range={offset}+{limit}"
    digest = hashlib.sha256(_text(result.content).encode("utf-8")).hexdigest()[:12]
    return (
        f"[context receipt: tool={result.tool_name}{target_part}{range_part}; "
        f"status={_result_status(result)}; result_sha256={digest}; "
        f"evicted_tokens={result.tokens}. Exact output is no longer active; "
        "rerun a precise tool call only if it is still needed.]"
    )


def _evicted_reads(
    results: list[_ToolResult], live_call_ids: set[str]
) -> list[dict[str, Any]]:
    """Describe the file ranges this assembly removed from the active context.

    The read ledger refuses to resend text the agent has already been given,
    so it has to be told when compaction takes that text away again —
    otherwise a re-read of an evicted file comes back as a bare receipt for
    content that is nowhere in the prompt.
    """
    evicted: list[dict[str, Any]] = []
    for result in results:
        if result.call_id in live_call_ids or _is_receipt(result):
            continue
        if result.tool_name == "read_file":
            path = result.args.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                offset = max(1, int(result.args.get("offset") or 1))
                limit = max(1, int(result.args.get("limit") or _DEFAULT_READ_LIMIT))
            except (TypeError, ValueError):
                offset, limit = 1, _DEFAULT_READ_LIMIT
            evicted.append(
                {"path": path, "start": offset, "end": offset + limit - 1}
            )
        elif result.tool_name == "read_files":
            for path in result.args.get("paths") or []:
                if isinstance(path, str) and path:
                    # read_files decides its own range per file, so the whole
                    # path is what left the context.
                    evicted.append({"path": path, "start": None, "end": None})
    return evicted


def _evicted_observations(
    results: list[_ToolResult], live_call_ids: set[str]
) -> list[str]:
    """Collect observation refs whose payload leaves the active context."""
    refs: list[str] = []
    for result in results:
        if result.call_id in live_call_ids or _is_receipt(result):
            continue
        refs.extend(_OBSERVATION_REF.findall(_text(result.content)))
    return list(dict.fromkeys(refs))


def _is_receipt(result: _ToolResult) -> bool:
    return isinstance(result.content, str) and result.content.startswith(
        "[context receipt:"
    )


def _arg_receipt(tool_name: str, field_name: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return (
        f"[context receipt: argument {field_name} of {tool_name}; chars={len(value)}; "
        f"sha256={digest}. The call already ran; read_file shows the current "
        "file if its text is still needed.]"
    )


def _compacted_args(
    tool_name: str, args: dict[str, Any], counter: TokenCounter
) -> tuple[dict[str, Any], int] | None:
    """Return ``args`` with bulky payload fields as receipts and the tokens saved."""
    saved = 0
    compacted = dict(args)
    for field_name in _ARG_PAYLOAD_FIELDS.get(tool_name, ()):
        value = args.get(field_name)
        if not isinstance(value, str) or value.startswith("[context receipt:"):
            continue
        receipt = _arg_receipt(tool_name, field_name, value)
        gain = counter.count(value) - counter.count(receipt)
        if gain <= 0:
            continue
        compacted[field_name] = receipt
        saved += gain
    return (compacted, saved) if saved else None


def _normal_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _checkpoint(
    messages: list[dict[str, Any]],
    calls: dict[str, tuple[str, dict[str, Any]]],
    results: list[_ToolResult],
    counter: TokenCounter,
    cap: int,
) -> dict[str, Any]:
    user_texts = [
        _normal_text(message).strip()
        for message in messages
        if message.get("role") == "user" and _normal_text(message).strip()
    ]
    assistant_texts = [
        _normal_text(message).strip()
        for message in messages
        if message.get("role") == "assistant" and _normal_text(message).strip()
    ]
    reads: list[str] = []
    changed: list[str] = []
    commands: list[str] = []
    for name, args in calls.values():
        target = _target(args)
        if name in _READ_TOOLS and target:
            reads.append(f"{target}:{args.get('offset', 1)}+{args.get('limit', 100)}")
        if name in {"edit_file", "write_file"} and target:
            changed.append(target)
        if name == "bash" and target:
            commands.append(target)
    checks = [
        f"{result.tool_name}:{_result_status(result)}"
        for result in results
        if result.tool_name == "bash" or result.is_error
    ]
    checkpoint = {
        "objective": user_texts[0][:500] if user_texts else "",
        "reads": list(dict.fromkeys(reads))[-20:],
        "changed_files": list(dict.fromkeys(changed))[-20:],
        "commands": list(dict.fromkeys(commands))[-10:],
        "checks": checks[-10:],
        "latest_decision": assistant_texts[-1][:500] if assistant_texts else "",
        "latest_user": user_texts[-1][:500] if user_texts else "",
    }
    rendered = compact_json(checkpoint)
    if counter.count(rendered) <= cap:
        return checkpoint
    # Preserve structure and the most actionable facts before truncating prose.
    checkpoint["objective"] = counter.truncate(checkpoint["objective"], max(32, cap // 4))
    checkpoint["latest_decision"] = counter.truncate(
        checkpoint["latest_decision"], max(32, cap // 4)
    )
    checkpoint["latest_user"] = counter.truncate(
        checkpoint["latest_user"], max(32, cap // 4)
    )
    while counter.count(compact_json(checkpoint)) > cap:
        removable = next(
            (
                key
                for key in ("reads", "commands", "checks", "changed_files")
                if checkpoint[key]
            ),
            None,
        )
        if removable is None:
            break
        checkpoint[removable].pop(0)
    return checkpoint


def _groups(messages: list[dict[str, Any]]) -> list[list[int]]:
    """Group assistant tool calls with every immediately following result."""
    groups: list[list[int]] = []
    index = 0
    while index < len(messages):
        group = [index]
        message = messages[index]
        has_calls = bool(message.get("tool_calls"))
        content = message.get("content")
        has_calls = has_calls or (
            isinstance(content, list)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in content
            )
        )
        if message.get("role") == "assistant" and has_calls:
            cursor = index + 1
            while cursor < len(messages):
                candidate = messages[cursor]
                is_oai_result = candidate.get("role") == "tool"
                candidate_content = candidate.get("content")
                is_ant_result = (
                    candidate.get("role") == "user"
                    and isinstance(candidate_content, list)
                    and any(
                        isinstance(block, dict) and block.get("type") == "tool_result"
                        for block in candidate_content
                    )
                )
                if not (is_oai_result or is_ant_result):
                    break
                group.append(cursor)
                cursor += 1
            index = cursor
        else:
            index += 1
        groups.append(group)
    return groups


class ContextAssembler:
    def __init__(self, policy: ContextPolicy, counter: TokenCounter | None = None):
        self.policy = policy
        self.counter = counter or TokenCounter()

    def build(
        self,
        *,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        session_remaining_tokens: int | None = None,
    ) -> ContextAssembly:
        working = copy.deepcopy(messages)
        calls = _tool_calls(working)
        results = _tool_results(working, calls, self.counter)
        checkpoint = _checkpoint(
            working,
            calls,
            results,
            self.counter,
            self.policy.checkpoint_cap,
        )

        if not self.policy.enabled:
            estimated = self._total(system_prompt, tool_schemas, working)
            return ContextAssembly(
                messages=working,
                estimated_input_tokens=estimated,
                active_tool_tokens=sum(result.tokens for result in results),
                active_tool_tokens_by_name=self._active_by_tool(results, set(range(len(results)))),
                evicted_tokens=0,
                evicted_results=0,
                dropped_messages=0,
                dropped_tokens=0,
                checkpoint=checkpoint,
                eviction_mode=self.policy.eviction_mode,
            )

        # Compaction must leave space for the response as well as for the
        # assembled input.  Otherwise a replaceable history can be left just
        # below ``input_cap`` and then incorrectly fail the final window check.
        compaction_input_limit = min(
            self.policy.input_cap,
            self.policy.model_context_limit - self.policy.output_reserve,
            self.policy.context_window_limit - self.policy.output_reserve,
        )
        # Past file bodies go first: they are already on disk, while a tool
        # result the agent has just asked for is the one thing it needs next.
        compacted_args, compacted_arg_tokens = self._compact_call_args(
            working,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            budget=compaction_input_limit,
        )
        keep = self._keep_raw(
            results,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            messages=working,
            budget=compaction_input_limit,
        )

        evicted_tokens = 0
        evicted_results = 0
        for result_index, result in enumerate(results):
            if result_index in keep:
                continue
            evicted_results += 1
            receipt = _receipt(result)
            message = working[result.message_index]
            if result.block_index is None:
                message["content"] = receipt
                receipt_tokens = self.counter.count(message)
            else:
                message["content"][result.block_index]["content"] = receipt
                receipt_tokens = self.counter.count(
                    message["content"][result.block_index]
                )
            evicted_tokens += max(0, result.tokens - receipt_tokens)

        dropped_messages = 0
        total = self._total(system_prompt, tool_schemas, working)
        before_conversation_compaction = total
        if total > compaction_input_limit:
            grouped = _groups(working)
            protected: set[int] = {
                index for group in grouped[-2:] for index in group
            }
            first_user = next(
                (index for index, message in enumerate(working) if message.get("role") == "user"),
                None,
            )
            if first_user is not None:
                protected.update(
                    next((group for group in grouped if first_user in group), [first_user])
                )
            dropped: set[int] = set()
            for group in grouped:
                if any(index in protected for index in group):
                    continue
                dropped.update(group)
                candidate = [
                    message for index, message in enumerate(working) if index not in dropped
                ]
                total = self._total(system_prompt, tool_schemas, candidate)
                if total <= compaction_input_limit:
                    working = candidate
                    dropped_messages = len(dropped)
                    break
            else:
                working = [
                    message for index, message in enumerate(working) if index not in dropped
                ]
                dropped_messages = len(dropped)
                total = self._total(system_prompt, tool_schemas, working)

            if dropped_messages:
                checkpoint_message = {
                    "role": "user",
                    "content": "[deterministic context checkpoint]\n" + compact_json(checkpoint),
                }
                insert_at = 1 if working else 0
                working.insert(insert_at, checkpoint_message)
                total = self._total(system_prompt, tool_schemas, working)
                # The checkpoint is useful but never allowed to cause a cap violation.
                if total > compaction_input_limit:
                    working.pop(insert_at)
                    total = self._total(system_prompt, tool_schemas, working)

        dropped_tokens = max(0, before_conversation_compaction - total)
        final_calls = _tool_calls(working)
        final_results = _tool_results(working, final_calls, self.counter)
        final_raw = {
            index for index, result in enumerate(final_results) if not _is_receipt(result)
        }
        live_call_ids = {final_results[index].call_id for index in final_raw}
        evicted_reads = _evicted_reads(results, live_call_ids)
        evicted_observations = _evicted_observations(results, live_call_ids)
        rejected_reason: str | None = None
        if (
            uses_cumulative_limit()
            and session_remaining_tokens is not None
            and session_remaining_tokens <= self.policy.output_reserve
        ):
            rejected_reason = "session_budget"
        else:
            if total > self.policy.input_cap:
                rejected_reason = "prompt_cap"
            elif total + self.policy.output_reserve > self.policy.model_context_limit:
                rejected_reason = "model_context_limit"
            elif total + self.policy.output_reserve > self.policy.context_window_limit:
                rejected_reason = "context_window_limit"

        return ContextAssembly(
            messages=working,
            estimated_input_tokens=total,
            active_tool_tokens=sum(final_results[index].tokens for index in final_raw),
            active_tool_tokens_by_name=self._active_by_tool(final_results, final_raw),
            evicted_tokens=evicted_tokens,
            evicted_results=evicted_results,
            dropped_messages=dropped_messages,
            dropped_tokens=dropped_tokens,
            checkpoint=checkpoint,
            rejected_reason=rejected_reason,
            eviction_mode=self.policy.eviction_mode,
            evicted_reads=evicted_reads,
            evicted_observations=evicted_observations,
            compacted_args=compacted_args,
            compacted_arg_tokens=compacted_arg_tokens,
        )

    def _compact_call_args(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        budget: int,
    ) -> tuple[int, int]:
        """Replace payloads of past write_file/edit_file calls, oldest first.

        Runs only while the prompt is over ``budget`` and stops as soon as it
        fits.  The newest call group is left intact: its result is what the
        agent is reacting to right now.  Mutates ``messages`` (the assembler's
        own copy); returns how many calls were compacted and tokens saved.
        """
        overflow = self._total(system_prompt, tool_schemas, messages) - budget
        if overflow <= 0:
            return 0, 0
        grouped = _groups(messages)
        protected = set(grouped[-1]) if grouped else set()
        compacted = 0
        saved_total = 0
        for message_index, message in enumerate(messages):
            if overflow <= 0:
                break
            if message_index in protected or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block_index, block in enumerate(content):
                    if overflow <= 0:
                        break
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    args = block.get("input")
                    if not isinstance(args, dict):
                        continue
                    outcome = _compacted_args(str(block.get("name") or ""), args, self.counter)
                    if outcome is None:
                        continue
                    content[block_index] = {**block, "input": outcome[0]}
                    compacted += 1
                    saved_total += outcome[1]
                    overflow -= outcome[1]
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call_index, call in enumerate(calls):
                if overflow <= 0:
                    break
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                raw_args = function.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (TypeError, ValueError):
                    continue
                if not isinstance(args, dict):
                    continue
                outcome = _compacted_args(str(function.get("name") or ""), args, self.counter)
                if outcome is None:
                    continue
                encoded: Any = (
                    json.dumps(outcome[0], ensure_ascii=False)
                    if isinstance(raw_args, str)
                    else outcome[0]
                )
                calls[call_index] = {**call, "function": {**function, "arguments": encoded}}
                compacted += 1
                saved_total += outcome[1]
                overflow -= outcome[1]
        return compacted, saved_total

    def _keep_raw(
        self,
        results: list[_ToolResult],
        *,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        budget: int,
    ) -> set[int]:
        """Choose which tool results stay verbatim in the active window.

        ``on_demand`` keeps every result as the agent saw it while the whole
        assembled prompt still fits ``budget``, and only then demotes the
        oldest ones to receipts, just far enough to fit.  An agent reading a
        handful of source files therefore keeps all of them in view instead of
        losing a file it read three tool calls ago.

        ``eager`` is the previous behaviour, kept for rollback: a fixed
        ``keep_raw_results``/``active_tool_result_cap`` window applied on every
        request no matter how much of the context window is actually in use.
        """
        if self.policy.eviction_mode == "eager":
            keep: set[int] = set()
            active_tokens = 0
            for index in range(len(results) - 1, -1, -1):
                if len(keep) >= self.policy.keep_raw_results:
                    continue
                if active_tokens + results[index].tokens > self.policy.active_tool_result_cap:
                    continue
                keep.add(index)
                active_tokens += results[index].tokens
            return keep

        keep = set(range(len(results)))
        overflow = self._total(system_prompt, tool_schemas, messages) - budget
        for index, result in enumerate(results):
            if overflow <= 0:
                break
            if _is_receipt(result):
                # Already a receipt from an earlier turn — evicting it again
                # frees nothing and would only shrink the raw history further.
                continue
            keep.discard(index)
            overflow -= max(0, result.tokens - self.counter.count(_receipt(result)))
        return keep

    def _total(
        self,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> int:
        return (
            self.counter.count(system_prompt)
            + self.counter.count(tool_schemas)
            + self.counter.count(messages)
        )

    @staticmethod
    def _active_by_tool(
        results: list[_ToolResult],
        keep: set[int],
    ) -> dict[str, int]:
        active: dict[str, int] = {}
        for index in keep:
            result = results[index]
            active[result.tool_name] = active.get(result.tool_name, 0) + result.tokens
        return active


async def persist_evicted_reads(
    assembly: ContextAssembly, *, session_id: UUID, db: AsyncSession
) -> None:
    """Tell the dedupe layer what this assembly compacted out of the context."""
    if not assembly.evicted_reads and not assembly.evicted_observations:
        return
    try:
        from app.core.context_observations import (
            mark_observations_evicted,
            mark_reads_evicted,
        )

        if assembly.evicted_reads:
            await mark_reads_evicted(
                db, session_id=session_id, reads=assembly.evicted_reads
            )
        if assembly.evicted_observations:
            await mark_observations_evicted(
                db, session_id=session_id, refs=assembly.evicted_observations
            )
    except Exception:
        # Worst case the agent is refused a re-read it cannot see — noisy, but
        # never a reason to fail the provider request.
        logger.exception(
            "context: failed to mark evicted reads for session %s", session_id
        )


async def persist_context_assembly(
    assembly: ContextAssembly,
    *,
    session_id: UUID,
    db: AsyncSession,
    role: str = "unknown",
    request_kind: str = "unknown",
) -> None:
    """Persist the deterministic checkpoint and prompt-size observations.

    The bounded ``successful_assemblies`` history is intentionally kept in the
    existing snapshot namespace: it makes an operational top-prompts report
    possible without adding a high-write observability table to the request
    path.
    """
    await persist_evicted_reads(assembly, session_id=session_id, db=db)
    try:
        from app.models.session.base import Session

        row = await db.get(Session, session_id)
        if row is None:
            return
        current = getattr(row, "context_snapshot", None)
        if current is not None and not isinstance(current, dict):
            return
        snapshot = dict(current or {})
        previous = snapshot.get("context_gateway")
        previous = previous if isinstance(previous, dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        gateway = dict(previous)
        gateway.update({
            "version": 2,
            "checkpoint": assembly.checkpoint,
            "last_assembled_input_tokens": assembly.estimated_input_tokens,
            "last_active_tool_tokens": assembly.active_tool_tokens,
            "last_rejected_reason": assembly.rejected_reason,
            "eviction_mode": assembly.eviction_mode,
            "tokens_evicted_total": int(previous.get("tokens_evicted_total") or 0)
            + assembly.evicted_tokens
            + assembly.dropped_tokens
            + assembly.compacted_arg_tokens,
            "results_evicted_total": int(previous.get("results_evicted_total") or 0)
            + assembly.evicted_results,
            "args_compacted_total": int(previous.get("args_compacted_total") or 0)
            + assembly.compacted_args,
            "compactions_total": int(previous.get("compactions_total") or 0)
            + int(
                bool(
                    assembly.evicted_results
                    or assembly.dropped_messages
                    or assembly.compacted_args
                )
            ),
            "provider_cache_read_tokens": int(
                previous.get("provider_cache_read_tokens") or 0
            ),
            "provider_cache_write_tokens": int(
                previous.get("provider_cache_write_tokens") or 0
            ),
            "updated_at": now,
        })
        if assembly.allowed:
            history = previous.get("successful_assemblies")
            history = history if isinstance(history, list) else []
            history = [entry for entry in history if isinstance(entry, dict)]
            history.append({
                "at": now,
                "tokens": max(0, int(assembly.estimated_input_tokens)),
                "role": str(role or "unknown"),
                "kind": str(request_kind or "unknown"),
            })
            gateway["successful_assemblies"] = history[-_SUCCESSFUL_ASSEMBLY_HISTORY_MAX:]
            gateway["last_successful_assembled_input_tokens"] = max(
                0, int(assembly.estimated_input_tokens)
            )
            gateway["last_successful_assembled_at"] = now
            old_max = max(0, int(previous.get("max_successful_assembled_input_tokens") or 0))
            if assembly.estimated_input_tokens >= old_max:
                gateway["max_successful_assembled_input_tokens"] = max(
                    0, int(assembly.estimated_input_tokens)
                )
                gateway["max_successful_assembled_at"] = now
            else:
                gateway["max_successful_assembled_input_tokens"] = old_max
                gateway["max_successful_assembled_at"] = previous.get(
                    "max_successful_assembled_at"
                )
        snapshot["context_gateway"] = gateway
        row.context_snapshot = snapshot
        db.add(row)
    except Exception:
        # Context observability must never break the provider request path.
        return


def preflight_rejection_details(
    assembly: ContextAssembly,
    *,
    session_id: UUID,
    role: str,
    request_kind: str,
    policy: ContextPolicy,
    session_remaining_tokens: int | None = None,
    cumulative_tokens: int | None = None,
) -> dict[str, Any]:
    """Return a stable, aggregation-friendly record for a rejected assembly."""
    reason = str(assembly.rejected_reason or "unknown")
    input_cap = max(0, int(policy.input_cap))
    model_cap = max(0, int(policy.model_context_limit))
    if reason == "prompt_cap":
        # Report the effective input ceiling: an administrative input cap can
        # never enlarge the provider's physical context capacity.
        cap = min(input_cap, model_cap)
        cap_source = "input_cap" if input_cap <= model_cap else "model_context_limit"
    elif reason == "model_context_limit":
        cap = model_cap
        cap_source = "model_context_limit"
    elif reason == "session_budget":
        cap = max(0, int(session_remaining_tokens or 0))
        cap_source = "session_remaining_tokens"
    else:
        cap = max(0, int(policy.context_window_limit))
        cap_source = "context_window_limit"
    return {
        "event": "context_preflight_rejected",
        "session_id": str(session_id),
        "role": str(role or "unknown"),
        "kind": str(request_kind or "unknown"),
        "reason": reason,
        "tokens": max(0, int(assembly.estimated_input_tokens)),
        "cap": cap,
        "cap_source": cap_source,
        "mode": context_limit_mode(),
        "assembled_input_tokens": max(0, int(assembly.estimated_input_tokens)),
        "output_reserve": max(0, int(policy.output_reserve)),
        "app_cap": max(0, int(policy.context_window_limit)),
        "model_cap": model_cap,
        "cumulative_tokens": max(0, int(cumulative_tokens or 0)),
    }


def preflight_observation_details(
    assembly: ContextAssembly,
    *,
    session_id: UUID,
    role: str,
    request_kind: str,
    policy: ContextPolicy,
    session_remaining_tokens: int | None = None,
    cumulative_tokens: int | None = None,
) -> dict[str, Any]:
    """Return the structured record emitted for every preflight attempt.

    Successful admissions are logged as ``reason=allowed``.  Rejections retain
    their specific reason and cap fields, so one record explains both the
    decision and the limits that were considered.
    """
    details = preflight_rejection_details(
        assembly,
        session_id=session_id,
        role=role,
        request_kind=request_kind,
        policy=policy,
        session_remaining_tokens=session_remaining_tokens,
        cumulative_tokens=cumulative_tokens,
    )
    details["event"] = (
        "context_preflight_rejected"
        if assembly.rejected_reason
        else "context_preflight"
    )
    details["reason"] = str(assembly.rejected_reason or "allowed")
    details["allowed"] = assembly.rejected_reason is None
    return details


def log_preflight_observation(
    assembly: ContextAssembly,
    *,
    session_id: UUID,
    role: str,
    request_kind: str,
    policy: ContextPolicy,
    session_remaining_tokens: int | None = None,
    cumulative_tokens: int | None = None,
) -> None:
    """Emit one structured observability record for every preflight attempt."""
    details = preflight_observation_details(
        assembly,
        session_id=session_id,
        role=role,
        request_kind=request_kind,
        policy=policy,
        session_remaining_tokens=session_remaining_tokens,
        cumulative_tokens=cumulative_tokens,
    )
    log = logger.warning if not details["allowed"] else logger.info
    log(
        "context_preflight %s",
        json.dumps(
            details,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def log_preflight_rejection(
    assembly: ContextAssembly,
    *,
    session_id: UUID,
    role: str,
    request_kind: str,
    policy: ContextPolicy,
    session_remaining_tokens: int | None = None,
    cumulative_tokens: int | None = None,
) -> None:
    """Emit one JSON log line for every rejected preflight request."""
    logger.warning(
        "context_preflight_rejected %s",
        json.dumps(
            preflight_rejection_details(
                assembly,
                session_id=session_id,
                role=role,
                request_kind=request_kind,
                policy=policy,
                session_remaining_tokens=session_remaining_tokens,
                cumulative_tokens=cumulative_tokens,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


async def persist_provider_cache_usage(
    *,
    session_id: UUID,
    db: AsyncSession,
    read_tokens: int = 0,
    write_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Accumulate provider cache usage in the session's compact gateway state.

    ``cost_usd`` is the cache price correction for these tokens
    (``model_pricing.cache_cost_adjustment``).  It is deferred and lands on
    the session cost only when ``track_llm_usage`` charges the turn these
    requests belong to — never before the tokens it corrects.
    """
    if not read_tokens and not write_tokens:
        return
    if cost_usd:
        from app.core.token_budget import defer_cache_cost

        defer_cache_cost(session_id, cost_usd)
    try:
        from app.models.session.base import Session

        row = await db.get(Session, session_id)
        if row is None:
            return
        current = getattr(row, "context_snapshot", None)
        if current is not None and not isinstance(current, dict):
            return
        snapshot = dict(current or {})
        gateway = snapshot.get("context_gateway")
        gateway = dict(gateway) if isinstance(gateway, dict) else {"version": 1}
        gateway["provider_cache_read_tokens"] = int(
            gateway.get("provider_cache_read_tokens") or 0
        ) + max(0, int(read_tokens))
        gateway["provider_cache_write_tokens"] = int(
            gateway.get("provider_cache_write_tokens") or 0
        ) + max(0, int(write_tokens))
        gateway["updated_at"] = datetime.now(timezone.utc).isoformat()
        snapshot["context_gateway"] = gateway
        row.context_snapshot = snapshot
        db.add(row)
    except Exception:
        return
