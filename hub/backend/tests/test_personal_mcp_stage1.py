from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.core.personal_mcp import (
    PersonalMcpError,
    launch_spec_from_candidate,
    parse_import_payload,
    transition_runtime,
)
from app.models.personal_mcp import PersonalMcpRuntime


def test_depwire_config_is_normalized_without_shell_or_secrets():
    parsed = parse_import_payload(
        {"mcpServers": {"depwire": {"command": "npx", "args": ["-y", "depwire-cli", "mcp"]}}}
    )

    assert len(parsed.candidates) == 1
    candidate = parsed.candidates[0]
    assert candidate["runtime_profile"] == "node_npx_v1"
    assert candidate["package"] == {
        "requested": "depwire-cli",
        "registry_name": "depwire-cli",
        "requested_version": None,
    }
    assert candidate["command_args"] == ["mcp"]
    assert candidate["normalized_argv"] == ["npx", "--yes", "depwire-cli", "mcp"]
    assert candidate["shell"] is False
    assert "env_names" in candidate and "secret" not in str(candidate)


def test_multi_server_preview_keeps_valid_candidates_and_safe_rejections():
    parsed = parse_import_payload(
        {
            "mcpServers": {
                "good": {"command": "npx", "args": ["--yes", "@acme/calendar-mcp@^2.4.0", "--stdio"]},
                "bad": {"command": "npm", "args": ["install", "unsafe-package"]},
            }
        }
    )

    assert [candidate["source_name"] for candidate in parsed.candidates] == ["good"]
    assert parsed.rejected[0]["source_name"] == "bad"
    assert parsed.rejected[0]["error"]["code"] == "unsupported_npm_subcommand"
    assert "unsafe-package" not in str(parsed.rejected[0]["error"])


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"command": "npx", "args": ["depwire-cli"]}, "interactive_npx_not_allowed"),
        ({"command": "npx", "args": ["-y", "https://example.test/mcp"]}, "package_source_not_allowed"),
        ({"command": "npx", "args": ["-y", "depwire-cli", "$(whoami)"]}, "shell_syntax_not_allowed"),
        ({"command": "npx", "args": ["-y", "depwire-cli"], "env": {"TOKEN": "plain-secret"}}, "secret_value_not_allowed"),
    ],
)
def test_unsupported_or_secret_input_is_rejected_without_echo(payload, code):
    with pytest.raises(PersonalMcpError) as raised:
        parse_import_payload(payload)
    assert raised.value.code == code
    assert "plain-secret" not in str(raised.value.as_error())
    assert "whoami" not in str(raised.value.as_error())


def test_npm_exec_is_canonicalized_to_argv_profile():
    parsed = parse_import_payload(
        {"command": "npm", "args": ["exec", "-y", "--package", "depwire-cli@1.4.2", "--", "mcp"]}
    )
    candidate = parsed.candidates[0]
    assert candidate["package"]["requested_version"] == "1.4.2"
    assert candidate["command_args"] == ["mcp"]
    assert candidate["normalized_argv"] == ["npm", "exec", "--yes", "--package", "depwire-cli@1.4.2", "--", "mcp"]


def test_launch_spec_contains_only_normalized_non_secret_fields():
    candidate = parse_import_payload(
        {"command": "npx", "args": ["-y", "depwire-cli", "mcp"], "env": {"API_KEY": ""}}
    ).candidates[0]
    launch_spec = launch_spec_from_candidate(candidate)
    assert launch_spec["env_names"] == ["API_KEY"]
    assert "secret_id" not in str(launch_spec)
    assert launch_spec["resolved_artifact"] == {"version": None, "integrity": None, "resolved_at": None}


def test_runtime_state_machine_allows_idempotent_transition_and_rejects_invalid():
    runtime = PersonalMcpRuntime(
        installed_item_id=uuid4(),
        owner_id=uuid4(),
        preview_id=uuid4(),
        candidate_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    transition_runtime(runtime, "starting")
    transition_runtime(runtime, "starting")
    assert runtime.actual_status == "starting"
    with pytest.raises(PersonalMcpError) as raised:
        transition_runtime(runtime, "verified")
    assert raised.value.code == "invalid_state_transition"
