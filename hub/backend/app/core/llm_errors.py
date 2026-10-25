"""One place that turns an LLM/gateway exception into a user-facing category.

Reasoning and router plan, stage 9 (§10.3): the OpenCode Go gateway refuses
some models with 403 ``RegionError`` / ``DataPolicyError`` — the key is valid,
the model is locked behind a browser-side workspace toggle and the error body
carries the working opt-in URL (measured §0.8: no programmatic opt-in exists,
five header variants tried).  These two failures must reach the user as
"open this link and flip the switch", not as a generic LLM failure that looks
like the agent is broken.

`classify_gateway_error` reads the status code and the JSON error body from
whatever shape the exception arrived in — the OpenAI SDK
(``PermissionDeniedError.body``), a hand-rolled httpx call
(``HTTPStatusError.response``) — and returns one of:

    ('region_opt_in' | 'data_policy_opt_in' | 'auth' | 'credits' | 'timeout'
     | 'other', opt_in_url)

`OPT_IN_REASONS` are the two reasons the Hub can do something about; every
other category stays on the caller's existing generic failure path.
"""

from __future__ import annotations

import asyncio
import re
from httpx import TimeoutException as HttpxTimeoutException
from typing import Any

# 403 error.type values that mean "locked behind a workspace opt-in toggle".
# Both carry the working link inside the human message — it embeds the user's
# own ``wrk_…`` workspace id, so it must be extracted, never invented.
_OPT_IN_ERROR_TYPES: dict[str, str] = {
    "RegionError": "region_opt_in",
    "DataPolicyError": "data_policy_opt_in",
}

OPT_IN_REASONS = frozenset(_OPT_IN_ERROR_TYPES.values())

_URL_RE = re.compile(r"https://\S+")
_URL_TRAILING = ".,);:>»\"'"


def extract_opt_in_url(message: str | None) -> str | None:
    """The first URL inside a gateway error message, if any.

    The RegionError/DataPolicyError body embeds the user's workspace settings
    link; any ``https://…`` found there is the link to show.  Trailing
    punctuation that sneaks into a regex ``\\S+`` match is trimmed.
    """
    match = _URL_RE.search(message or "")
    if match is None:
        return None
    return match.group(0).rstrip(_URL_TRAILING) or None


def model_opt_in_content(model: str | None, reason: str) -> str:
    """User-visible text of the MODEL_OPT_IN_REQUIRED event (plan §10.4)."""
    if reason == "data_policy_opt_in":
        return (
            f"Model {model or '?'} trains on your prompts and requires "
            "explicit consent in the OpenCode workspace settings"
        )
    return (
        f"Model {model or '?'} requires enabling models hosted "
        "in China in the OpenCode workspace settings"
    )


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _error_body(exc: Exception) -> dict[str, Any]:
    """The gateway's JSON error payload, from whatever attribute it landed in."""
    for source in (getattr(exc, "body", None),):
        if isinstance(source, dict):
            return source
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    return {}


def _error_type(body: dict[str, Any]) -> str | None:
    error = body.get("error")
    if isinstance(error, dict):
        value = error.get("type")
        return value if isinstance(value, str) else None
    value = body.get("type")
    return value if isinstance(value, str) else None


def _error_message(body: dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return body.get("message", "") if isinstance(body.get("message"), str) else ""


def classify_gateway_error(exc: Exception) -> tuple[str, str | None]:
    """('region_opt_in' | 'data_policy_opt_in' | 'auth' | 'credits' | 'timeout'
    | 'other', opt_in_url).

    Tolerant by design: an exception nobody taught this function about is
    ``other`` with no URL, never a crash — the caller keeps its existing
    failure handling for that category.
    """
    if isinstance(exc, (HttpxTimeoutException, asyncio.TimeoutError)) or (
        type(exc).__name__ == "APITimeoutError"
    ):
        return "timeout", None

    status = _status_code(exc)
    body = _error_body(exc)
    error_type = _error_type(body)
    message = _error_message(body)

    # The SDK stringifies the body into the message; a body-less exception
    # whose text names the opt-in error still classifies (defensive — measured
    # SDK behaviour keeps `.body`, so this is a fallback, not the main path).
    if error_type is None and status == 403:
        text = str(exc)
        for sdk_type, reason in _OPT_IN_ERROR_TYPES.items():
            if sdk_type in text:
                error_type = sdk_type
                if not message:
                    message = text
                break

    if status == 403 or error_type in _OPT_IN_ERROR_TYPES:
        if error_type in _OPT_IN_ERROR_TYPES:
            return _OPT_IN_ERROR_TYPES[error_type], extract_opt_in_url(message)
        if error_type == "CreditsError":
            return "credits", None
        return "other", None
    if status == 401:
        return "auth", None
    return "other", None
