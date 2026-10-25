"""Validation helpers for room-scoped workspace identities.

Room ids normally are UUIDs.  The workspaced test harness and a few older
callers use opaque ids, so the contract intentionally allows a conservative
identifier alphabet while rejecting path separators and traversal segments.
"""

from __future__ import annotations

import re


_SAFE_ROOM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def is_safe_room_id(room_id: str) -> bool:
    value = str(room_id)
    return value not in {".", ".."} and _SAFE_ROOM_ID.fullmatch(value) is not None


def require_safe_room_id(room_id: str) -> str:
    value = str(room_id)
    if not is_safe_room_id(value):
        raise ValueError("Invalid room id")
    return value
