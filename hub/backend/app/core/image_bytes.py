"""Shared raster-image primitives: data-URI parsing and magic-byte sniffing.

Split out of ``mcp_gateway`` (internal design notes, stage 1) so
the media reference resolver can validate the same PNG/JPEG/WebP/GIF bytes
the mcp-vision path already validates, without a second diverging
implementation. Callers keep their own size limits and exception types —
this module only recognizes bytes, it never decides whether they are
acceptable.
"""
from __future__ import annotations

import re

IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

_DATA_URI_RE = re.compile(r"data:(image/[a-z0-9.+-]+);base64,([A-Za-z0-9+/]*={0,2})", re.IGNORECASE)


def split_image_data_uri(value: str) -> tuple[str, str] | None:
    """Split ``data:image/...;base64,...`` into ``(mime_type, base64_payload)``."""

    match = _DATA_URI_RE.fullmatch(value)
    if match is None:
        return None
    return match.group(1).lower(), match.group(2)


def sniff_image_mime(decoded: bytes) -> str | None:
    """Infer the image MIME type from magic bytes, or ``None`` if unrecognized."""

    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if decoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(decoded) >= 12 and decoded[:4] == b"RIFF" and decoded[8:12] == b"WEBP":
        return "image/webp"
    if decoded.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None
