"""Keep credentials out of uvicorn's request lines.

A browser cannot set headers on a WebSocket, so the user-events and connector
sockets carry their bearer token in the query string (``?token=...``), and
uvicorn logs the whole path of every request and socket. Without this filter
a log line is enough to act as the user, or as their connector, until the
token expires. One-time codes of the sign-in flows travel the same way.
"""

from __future__ import annotations

import logging
import re

SECRET_QUERY_PARAMS = ("token", "access_token", "refresh_token", "code", "state", "ticket")

_SECRET_QUERY = re.compile(
    r"([?&](?:" + "|".join(SECRET_QUERY_PARAMS) + r")=)[^&\s\"']+",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    return _SECRET_QUERY.sub(r"\1[redacted]", text)


class RedactQuerySecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


def install() -> None:
    """uvicorn.access writes HTTP request lines, uvicorn.error the WebSocket ones."""
    for name in ("uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactQuerySecrets) for f in logger.filters):
            logger.addFilter(RedactQuerySecrets())
