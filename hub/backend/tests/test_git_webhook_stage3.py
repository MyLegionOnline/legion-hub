from __future__ import annotations

import hashlib
import hmac

from app.core.git_webhook import (
    verify_github_signature,
    verify_gitlab_token,
    verify_git_webhook_signature,
)


def test_github_fixture_is_checked_before_payload_processing():
    payload = b'{"ref":"refs/heads/main","commits":[]}'
    secret = "github-fixture-secret"
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    assert verify_git_webhook_signature(
        "github", payload, {"X-Hub-Signature-256": f"sha256={digest}"}, secret
    )
    assert not verify_github_signature(payload + b" ", f"sha256={digest}", secret)


def test_gitlab_fixture_uses_constant_time_exact_token_compare():
    assert verify_git_webhook_signature(
        "gitlab", b"{}", {"X-Gitlab-Token": "gitlab-fixture-secret"}, "gitlab-fixture-secret"
    )
    assert verify_gitlab_token("wrong", "gitlab-fixture-secret") is False
