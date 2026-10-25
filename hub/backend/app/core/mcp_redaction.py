"""Pattern redaction for MCP results — internal design notes §11 (Stage 7).

``mask_room_secrets_deep`` masks values the Hub *knows* are secret: they were
resolved out of the Room's own store moments earlier. Once an agent can read
an arbitrary server, that stops being enough — ``/etc/shadow``, a deploy key
in ``/root/.ssh``, another tenant's ``.env`` are all secrets the Hub has
never seen and cannot enumerate. This module is the complementary half: it
recognizes secrets *by shape* rather than by value, and runs on the result of
every MCP call (``_apply_mcp_result_safety_net``), not only the VPS ones.

Two deliberate properties:

* **Structure is preserved.** A redacted config still reads as a config —
  ``password=hunter2`` becomes ``password=[REDACTED]``, not a dropped line —
  so an agent can still reason about, and rewrite, the file it just read.
* **Base64 payloads are looked inside.** ``vps_read_file`` returns
  ``content_base64``; redacting only plain text would mean the one tool most
  likely to hand back ``/etc/shadow`` is the one tool that escapes redaction.
  A base64 string that decodes to UTF-8 text is redacted and re-encoded; one
  that decodes to binary (an image from ``mcp-vision``) is left untouched.

Like the policy blacklist (``core/mcp_policy.py``), this is a net for the
common shapes, not a proof: a secret with no recognizable framing — a bare
40-character API key on a line of its own — passes through, because a rule
broad enough to catch it would redact half of every legitimate result.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

#: What replaces a matched secret. Deliberately obvious in a chat transcript:
#: a human reading the agent's reasoning should be able to tell "the Hub hid
#: this" from "the file really said that".
REDACTED = "[REDACTED]"
REDACTED_PRIVATE_KEY = "[REDACTED_PRIVATE_KEY]"
REDACTED_PASSWORD_HASH = "[REDACTED_PASSWORD_HASH]"

#: Do not try to base64-decode-and-rescan anything larger than this. The
#: safety net's own char cap applies *after* redaction, so an 8 MiB result
#: does reach this module.
_MAX_B64_SCAN_CHARS = 1_400_000
_MIN_B64_SCAN_CHARS = 24

# ── patterns ───────────────────────────────────────────────────────────────

#: A whole PEM block, header to footer. Non-greedy so several keys in one
#: file each collapse separately rather than swallowing what sits between.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
#: A header with no matching footer (a truncated read, a log excerpt) still
#: leaks key material on the lines that follow — redact to end of block.
_PEM_HEADER_ONLY = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*")

#: AWS long- and short-lived key ids. The id alone is not the credential, but
#: it is the reliable marker that the secret key sits next to it, and it is
#: what §11 names explicitly.
_AWS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA)[0-9A-Z]{16}\b")

#: crypt(3) hashes — ``$6$salt$hash`` in ``/etc/shadow``, ``$2b$`` bcrypt in
#: an application config. The ``!``/``*``/empty fields of a locked account
#: are not secrets and are deliberately not matched.
_CRYPT_HASH = re.compile(r"\$(?:1|2[abxy]?|5|6|7|y|gy|md5|sha1)\$[A-Za-z0-9./=+,$-]{12,}")

#: ``key=value`` / ``key: value`` in a config, an env file, a connection
#: string or a URL query. The key must *end* in a sensitive word so that
#: ``tokens:`` (mcp-vision's design-system tokens) and ``token_count`` do not
#: match — only ``token``, ``api_key``, ``db_password`` and friends do.
_SENSITIVE_KEY_WORD = r"(?:password|passwd|passphrase|secret|token|api[_.-]?key|access[_.-]?key|private[_.-]?key|credentials?|auth[_.-]?key)"
_KEY_VALUE = re.compile(
    r"(?i)(?P<key>[A-Za-z0-9_.\-\[\]]*" + _SENSITIVE_KEY_WORD + r")"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<quote>[\"']?)(?P<value>[^\s\"'&,;<>]{4,})(?P=quote)"
)

#: Dict keys whose scalar value is redacted outright, whatever it looks like.
#: Same suffix rule as ``_KEY_VALUE`` so the two agree on what "sensitive
#: name" means.
_SENSITIVE_DICT_KEY = re.compile(r"(?i)^[A-Za-z0-9_.\-]*" + _SENSITIVE_KEY_WORD + r"$")

#: A plausible base64 blob. Line breaks are tolerated (wrapped base64 is
#: common) but spaces are not: without that an ordinary English paragraph
#: matches, and every MCP result in the system pays for a decode attempt.
_B64_SHAPED = re.compile(r"^[A-Za-z0-9+/]+(?:[\r\n]+[A-Za-z0-9+/]+)*={0,2}$")


def _redact_key_value(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{match.group('key')}{match.group('sep')}{quote}{REDACTED}{quote}"


def redact_text(text: str) -> str:
    """Redact the known secret shapes in one string, preserving its structure."""

    if not text:
        return text
    redacted = _PEM_BLOCK.sub(REDACTED_PRIVATE_KEY, text)
    if "PRIVATE KEY-----" in redacted:
        redacted = _PEM_HEADER_ONLY.sub(REDACTED_PRIVATE_KEY, redacted)
    redacted = _AWS_KEY_ID.sub(REDACTED, redacted)
    redacted = _CRYPT_HASH.sub(REDACTED_PASSWORD_HASH, redacted)
    redacted = _KEY_VALUE.sub(_redact_key_value, redacted)
    return redacted


def _redact_base64(value: str) -> str | None:
    """Redact inside a base64 payload, or return ``None`` to leave it alone.

    ``None`` covers both "not base64" and "base64 of something that is not
    text" — an image, a tarball — where re-encoding would be pure damage and
    there is nothing a text rule could have matched anyway.
    """

    if not (_MIN_B64_SCAN_CHARS <= len(value) <= _MAX_B64_SCAN_CHARS):
        return None
    if not _B64_SHAPED.match(value):
        return None
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    redacted = redact_text(text)
    if redacted == text:
        return None
    return base64.b64encode(redacted.encode("utf-8")).decode("ascii")


def redact_sensitive_deep(value: Any) -> Any:
    """Redact secret-shaped content anywhere in an MCP result.

    A no-op for values with nothing to redact — the same object comes back,
    so the common case (a search result, a vision description) costs one
    regex sweep and no allocation.
    """

    if isinstance(value, str):
        decoded = _redact_base64(value)
        if decoded is not None:
            return decoded
        return redact_text(value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and _SENSITIVE_DICT_KEY.match(key) and isinstance(child, (str, int, float)) and not isinstance(child, bool):
                out[key] = REDACTED
            else:
                out[key] = redact_sensitive_deep(child)
        return out
    if isinstance(value, list):
        return [redact_sensitive_deep(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_deep(item) for item in value)
    return value
