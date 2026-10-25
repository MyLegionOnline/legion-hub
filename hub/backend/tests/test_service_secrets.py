"""The Hub ships no built-in secrets (internal design notes stage 4.2).

A missing token switches the feature off; it never falls back to a password
that the public source code spells out.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.v1 import secrets_internal
from app.core import security, vps_inventory


def test_service_tokens_have_no_built_in_values():
    """Import the modules in a clean process: the values are read at import."""
    code = (
        "from app.core import legion_auth, mcp_gateway, vps_inventory, workspaced_client;"
        "print(repr((workspaced_client.WORKSPACED_TOKEN, mcp_gateway.MCP_GATEWAY_TOKEN,"
        " vps_inventory._EGRESS_TICKET_SECRET, legion_auth.LEGION_CLIENT_SECRET)))"
    )
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("WORKSPACED_TOKEN", "MCP_GATEWAY_TOKEN", "VPS_EGRESS_TICKET_SECRET",
                                "LEGION_CLIENT_SECRET", "SECRETS_STRICT_CONFIG"))}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent.parent, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "('', '', None, None)"


def test_without_a_signing_key_a_random_one_is_made(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY_FILE", raising=False)
    monkeypatch.setenv("SECRETS_STRICT_CONFIG", "false")
    first, second = security._signing_key(), security._signing_key()
    assert first and second and first != second
    assert "development" not in first


def test_signing_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "from-the-operator")
    assert security._signing_key() == "from-the-operator"


def test_no_egress_ticket_without_a_secret(monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(vps_inventory, "_EGRESS_TICKET_SECRET", None)
    with pytest.raises(vps_inventory.EgressNotConfigured) as exc:
        vps_inventory.issue_egress_ticket(room_id=uuid4(), server="WEB", host="203.0.113.1", port=22)
    assert exc.value.http_status == 503


def test_mcp_host_callback_is_off_without_a_token(monkeypatch):
    monkeypatch.delenv("MCP_HOST_TOKEN", raising=False)
    monkeypatch.delenv("MCP_HOST_TOKEN_FILE", raising=False)
    monkeypatch.setenv("SECRETS_STRICT_CONFIG", "false")
    # "Bearer " is what an unset token would expect; it must not open the endpoint.
    for sent in (None, "", "Bearer ", "Bearer guess"):
        with pytest.raises(HTTPException) as exc:
            secrets_internal._require_mcp_host(sent)
        assert exc.value.status_code == 503


def test_mcp_host_callback_accepts_only_the_configured_token(monkeypatch):
    monkeypatch.setenv("MCP_HOST_TOKEN", "operator-token")
    secrets_internal._require_mcp_host("Bearer operator-token")
    with pytest.raises(HTTPException) as exc:
        secrets_internal._require_mcp_host("Bearer other")
    assert exc.value.status_code == 401
