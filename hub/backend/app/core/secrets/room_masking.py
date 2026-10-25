"""Room-secret output masking — second line of defence (Stage 3).

``internal design notes`` §3.1 / §4. Opt-in injection (Stage 4+)
keeps secret values out of the model's context in the common case, but any
command can still ``echo $DEPLOY_TOKEN`` or print a failing deploy's raw
output. This module is the catch for that: it replaces exact secret values
(and a few encoded forms of them) with ``***MASKED:NAME***`` wherever tool
output, service logs, MCP results, or session-log turns are about to be
shown to a model or persisted.

Values to mask are not looked up here — a call site either passes them
explicitly (``secrets={"DEPLOY_TOKEN": value}``) or, more commonly, wraps the
resolve-then-exec span in :class:`track_room_secrets`, and every
:func:`mask_room_secrets` call downstream picks them up automatically. The
mapping lives only on a ``contextvar`` for the duration of that ``with``
block — no cache, no persistence, gone the moment it exits (successfully or
not).
"""

from __future__ import annotations

import base64
import urllib.parse
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, Iterable, Optional

MASK_TEMPLATE = "***MASKED:{name}***"

# Mirrors room_secrets.MIN_SECRET_VALUE_LEN: values shorter than this are
# rejected on the way into the store, but guard here too so a stray short
# value (e.g. passed directly by a test) can't turn into mask-everything noise.
MIN_MASKABLE_LEN = 8

_active_secrets: ContextVar[dict[str, str]] = ContextVar(
    "room_masking_active_secrets", default={}
)


class track_room_secrets:
    """Context manager: make ``secrets`` maskable for the duration of the block.

    Usage (Stage 4+ call sites)::

        with track_room_secrets(resolved):
            output = await run_exec(...)
        output = mask_room_secrets(output)

    Nested / concurrent uses merge onto whatever the current async context
    already has and restore it on exit, so an outer scope's names stay
    maskable inside an inner one.
    """

    __slots__ = ("_secrets", "_token")

    def __init__(self, secrets: Optional[Mapping[str, str]]) -> None:
        self._secrets: dict[str, str] = {
            name: value for name, value in (secrets or {}).items() if value
        }
        self._token: Optional[Token] = None

    def __enter__(self) -> "track_room_secrets":
        if self._secrets:
            merged = {**_active_secrets.get(), **self._secrets}
            self._token = _active_secrets.set(merged)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if self._token is not None:
            _active_secrets.reset(self._token)
        return False


def _candidate_forms(name: str, value: str) -> list[tuple[str, str]]:
    """Exact value plus a few encoded forms it commonly shows up in."""

    replacement = MASK_TEMPLATE.format(name=name)
    seen: set[str] = set()
    forms: list[tuple[str, str]] = []

    def add(literal: str) -> None:
        if literal and len(literal) >= MIN_MASKABLE_LEN and literal not in seen:
            seen.add(literal)
            forms.append((literal, replacement))

    add(value)
    stripped = value.strip()
    if stripped != value:
        add(stripped)
    try:
        add(base64.b64encode(value.encode("utf-8")).decode("ascii"))
    except Exception:
        pass
    try:
        add(urllib.parse.quote(value, safe=""))
    except Exception:
        pass
    return forms


def _resolve_secrets(secrets: Optional[Mapping[str, str]]) -> dict[str, str]:
    combined: dict[str, str] = dict(_active_secrets.get())
    if secrets:
        combined.update({k: v for k, v in secrets.items() if v})
    return combined


def mask_room_secrets(
    text: Optional[str], secrets: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Replace any exact (or lightly-encoded) secret value found in ``text``.

    ``secrets`` supplements whatever :class:`track_room_secrets` currently has
    active; either source alone is enough. Returns ``text`` unchanged
    (same object) when there is nothing to mask, which keeps this a strict
    no-op before Stage 4 starts populating the active set.
    """

    if not text:
        return text
    combined = _resolve_secrets(secrets)
    if not combined:
        return text

    forms: list[tuple[str, str]] = []
    for name, value in combined.items():
        if isinstance(value, str):
            forms.extend(_candidate_forms(name, value))
    if not forms:
        return text

    # Longest literal first: a short value that happens to be a substring of
    # a longer one must not chew a bite out of the longer match first.
    forms.sort(key=lambda pair: len(pair[0]), reverse=True)

    masked = text
    for literal, replacement in forms:
        if literal in masked:
            masked = masked.replace(literal, replacement)
    return masked


# A background process (bash(background=true)) resolves its secrets in one
# request and is then read back by bash_output in later, unrelated requests —
# ``track_room_secrets``'s contextvar does not survive across that gap. This
# tiny process-wide registry bridges it: values live only in memory, keyed by
# bg_id, and are dropped once the caller forgets them (bash_kill, or the
# process finishing) or the process restarts.
_bg_secrets: dict[str, dict[str, str]] = {}


def remember_bg_secrets(key: str, secrets: Optional[Mapping[str, str]]) -> None:
    """Keep ``secrets`` maskable for later ``bash_output`` reads of ``key`` (bg_id)."""
    if secrets:
        _bg_secrets[key] = {name: value for name, value in secrets.items() if value}


def forget_bg_secrets(key: str) -> None:
    _bg_secrets.pop(key, None)


def bg_secrets(key: str) -> dict[str, str]:
    return dict(_bg_secrets.get(key) or {})


def mask_room_secrets_deep(value: Any, secrets: Optional[Mapping[str, str]] = None) -> Any:
    """Recursively apply :func:`mask_room_secrets` to every string leaf.

    For structured payloads (tool-result blocks, session-log turns, MCP
    results) where the secret-bearing text could be nested at any depth.
    """

    combined = _resolve_secrets(secrets)
    if not combined:
        return value
    if isinstance(value, str):
        return mask_room_secrets(value, combined)
    if isinstance(value, Mapping):
        return {k: mask_room_secrets_deep(v, combined) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_room_secrets_deep(v, combined) for v in value]
    if isinstance(value, tuple):
        return tuple(mask_room_secrets_deep(v, combined) for v in value)
    return value


__all__ = [
    "MASK_TEMPLATE",
    "MIN_MASKABLE_LEN",
    "track_room_secrets",
    "mask_room_secrets",
    "mask_room_secrets_deep",
    "remember_bg_secrets",
    "forget_bg_secrets",
    "bg_secrets",
]
