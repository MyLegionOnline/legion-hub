"""Counts an agent's near-identical text replies in a row.

A model that has convinced itself it cannot work (2026-11-20: a provider that
lost the tool declarations, so the agent kept reporting "tools are not
declared") sends the same reply over and over.  Each one is a paid request on
both sides of a proxy pair, and a direct session keeps the reply in its
history, which talks every later turn into the same answer.  The comparison
is purely textual, so it works in any language.
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

#: Near-identical replies in a row that stop the exchange.
REPEAT_LIMIT = int(os.getenv("AGENT_REPLY_REPEAT_LIMIT", "8"))
#: Similarity ratio (0..1) at which two replies count as the same.
SIMILARITY = float(os.getenv("AGENT_REPLY_REPEAT_SIMILARITY", "0.8"))
#: Only this many leading characters are compared; long replies stay cheap.
_COMPARE_CHARS = 2_000

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Reply text as compared: no reasoning block, case and spacing folded."""
    value = _THINKING_RE.sub(" ", str(text or ""))
    value = value.replace("`", "").replace("*", "")
    return _SPACE_RE.sub(" ", value).strip().lower()[:_COMPARE_CHARS]


def near_duplicate(a: str | None, b: str | None, threshold: float = SIMILARITY) -> bool:
    left, right = normalize(a), normalize(b)
    if left == right:
        return True
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold


def repeat_streak(replies: list[str], threshold: float = SIMILARITY) -> int:
    """Length of the run of near-identical replies at the end of *replies*."""
    if not replies:
        return 0
    streak = 1
    for index in range(len(replies) - 1, 0, -1):
        if not near_duplicate(replies[index], replies[index - 1], threshold):
            break
        streak += 1
    return streak


class RepeatGuard:
    """Streak of near-identical replies, fed one reply at a time."""

    def __init__(self, limit: int = REPEAT_LIMIT, threshold: float = SIMILARITY) -> None:
        self.limit = limit
        self.threshold = threshold
        self.count = 0
        self._last: str | None = None

    def observe(self, reply: str | None) -> int:
        if self._last is not None and near_duplicate(self._last, reply, self.threshold):
            self.count += 1
        else:
            self.count = 1
        self._last = reply or ""
        return self.count

    @property
    def tripped(self) -> bool:
        return self.limit > 0 and self.count >= self.limit
