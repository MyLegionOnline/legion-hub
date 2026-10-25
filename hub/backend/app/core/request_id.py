"""Correlation ID helpers that never inspect request bodies."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import Request


REQUEST_ID_HEADER = "X-Request-ID"


def request_id_from_request(request: Request) -> UUID:
    existing = getattr(request.state, "request_id", None)
    if isinstance(existing, UUID):
        return existing
    raw = request.headers.get(REQUEST_ID_HEADER)
    try:
        request_id = UUID(raw) if raw else uuid4()
    except (ValueError, AttributeError):
        request_id = uuid4()
    request.state.request_id = request_id
    return request_id
