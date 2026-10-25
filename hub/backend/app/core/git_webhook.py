"""Provider-specific Git webhook authentication."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value).strip()
    return ""


def verify_github_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub's ``sha256=<hex digest>`` header."""

    supplied = str(signature or "").strip()
    if not supplied.startswith("sha256=") or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(supplied, expected)


def verify_gitlab_token(token: str, secret: str) -> bool:
    """Verify GitLab's exact ``X-Gitlab-Token`` value."""

    if not token or not secret:
        return False
    return hmac.compare_digest(str(token).strip(), secret)


def verify_git_webhook_signature(
    provider: str,
    raw_body: bytes,
    headers: Mapping[str, str],
    secret: str,
) -> bool:
    provider = str(provider or "github").strip().lower()
    if provider == "github":
        return verify_github_signature(
            raw_body,
            _header(headers, "X-Hub-Signature-256"),
            secret,
        )
    if provider == "gitlab":
        return verify_gitlab_token(_header(headers, "X-Gitlab-Token"), secret)
    return False


__all__ = [
    "verify_github_signature",
    "verify_gitlab_token",
    "verify_git_webhook_signature",
]
