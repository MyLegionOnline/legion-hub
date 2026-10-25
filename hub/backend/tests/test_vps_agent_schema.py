"""The agent-facing shape of a VPS tool schema (internal design notes §3.3/Stage 16).

No database, no network: this is about what the model is shown. The rule it
pins down is one sentence long — an argument the model must not supply must
not be advertised to it. It was violated for `_legion_conn` (shown, and in
`required`, on every tool) and for `vps_upload.content_base64`, whose own
description said the Hub fills it in. A model does what the schema says: one
live run put the literal string "inject" on a production server and then
spent a dozen turns trying to squeeze a 530 KB bundle through the field.
"""

from app.core.mcp_gateway import _vps_agent_schema

UPLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "server": {"type": "string"},
        "workspace_path": {"type": "string"},
        "remote_path": {"type": "string"},
        "content_base64": {"type": "string"},
        "_legion_conn": {"type": "object"},
    },
    "required": ["server", "workspace_path", "remote_path", "content_base64", "_legion_conn"],
}


def test_hub_injected_arguments_are_hidden_from_the_model():
    schema = _vps_agent_schema(UPLOAD_SCHEMA)

    assert "content_base64" not in schema["properties"]
    assert "_legion_conn" not in schema["properties"]
    assert schema["required"] == ["server", "workspace_path", "remote_path"]
    # What the model *does* fill in is untouched.
    assert schema["properties"]["workspace_path"] == {"type": "string"}


def test_environment_is_still_advertised_and_the_input_is_not_mutated():
    schema = _vps_agent_schema(UPLOAD_SCHEMA)

    assert "environment" in schema["properties"]
    # The catalog snapshot this was derived from must survive intact —
    # `room_mcp_tools` hands the same dict to other consumers.
    assert "content_base64" in UPLOAD_SCHEMA["properties"]
    assert "content_base64" in UPLOAD_SCHEMA["required"]


def test_list_servers_loses_its_injected_array_too():
    schema = _vps_agent_schema({
        "type": "object",
        "properties": {"_legion_servers": {"type": "array"}},
        "required": ["_legion_servers"],
    })

    assert schema["properties"] == {"environment": schema["properties"]["environment"]}
    assert schema["required"] == []


def test_repo_status_asks_the_model_for_the_server_and_nothing_else():
    schema = _vps_agent_schema({
        "type": "object",
        "properties": {
            "server": {"type": "string"},
            "_legion_conn": {"type": "object"},
            "_legion_repo": {"type": "object"},
        },
        "required": ["server", "_legion_conn", "_legion_repo"],
    })

    assert schema["required"] == ["server"]
    assert "_legion_repo" not in schema["properties"]
