"""Provider-neutral token counting used by the active context gateway.

The production request usage returned by a provider remains authoritative.
Preflight cannot wait for that response, so it uses a conservative UTF-8
estimate with a safety margin and caches immutable blocks by digest.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Any


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class TokenCounter:
    """Small cached counter suitable for request preflight.

    Four UTF-8 bytes per token is deliberately followed by a 20% margin.  It is
    less precise than a provider tokenizer, but errs on the safe side for
    Cyrillic and works for Anthropic, OpenAI-compatible and local providers
    without adding a network request to the hot loop.
    """

    def __init__(self, *, safety_margin: float = 0.20, cache_size: int = 4096):
        self.safety_margin = max(0.0, float(safety_margin))
        self.cache_size = max(32, int(cache_size))
        self._cache: OrderedDict[str, int] = OrderedDict()

    def count(self, value: Any) -> int:
        text = value if isinstance(value, str) else compact_json(value)
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(digest)
        if cached is not None:
            self._cache.move_to_end(digest)
            return cached
        tokens = max(
            1,
            math.ceil((len(text.encode("utf-8")) / 4) * (1 + self.safety_margin)),
        )
        self._cache[digest] = tokens
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tokens

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return a UTF-8-safe prefix whose conservative estimate fits."""
        max_tokens = max(0, int(max_tokens))
        if self.count(text) <= max_tokens:
            return text
        if max_tokens == 0:
            return ""
        raw = text.encode("utf-8")
        low, high = 0, len(raw)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = raw[:middle].decode("utf-8", errors="ignore")
            if self.count(candidate) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return raw[:low].decode("utf-8", errors="ignore")
