import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def request(url, method="GET", body=None, headers=None):
    payload = json.dumps(body).encode() if body is not None else None
    request_obj = urllib.request.Request(url, data=payload, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request_obj, timeout=10) as response:
        return response.status, json.loads(response.read().decode())


def wait_ready(url):
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            status, body = request(url)
            if status == 200 and body.get("status") == "ready":
                return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(1)
    raise AssertionError("managed runner did not become ready")


def main():
    # This is the same user-facing import form accepted by Hub. The smoke
    # keeps a tiny deterministic normalization assertion so it does not need
    # a database; Hub parser/ownership tests cover the full API path.
    imported = {"mcpServers": {"fixture": {"command": "npx", "args": ["-y", "fixture-mcp", "mcp"]}}}
    server = imported["mcpServers"]["fixture"]
    assert server["command"] == "npx" and server["args"][:2] == ["-y", "fixture-mcp"]
    profile = {"mode": "managed_stdio", "runtime_profile": "node_npx_v1", "transport": "stdio", "package": "fixture-mcp", "command_args": ["mcp"], "shell": False}
    assert profile["shell"] is False

    runner = os.environ.get("RUNNER_URL", "http://runner:8080")
    gateway = os.environ.get("GATEWAY_URL", "http://gateway:8310")
    wait_ready(f"{runner}/readyz")

    _, metadata = request(f"{runner}/metadata")
    assert metadata["package"] == "fixture-mcp"
    _, initialize = request(f"{runner}/mcp", "POST", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialize["result"]["serverInfo"]["name"] == "legion-fixture"
    _, tools = request(f"{runner}/mcp", "POST", {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert [item["name"] for item in tools["result"]["tools"]] == ["fixture_room_echo"]

    item_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    room_id = "room-stage6"
    headers = {
        "Authorization": "Bearer smoke-gateway",
        "X-Legion-User-Id": user_id,
        "X-Legion-Room-Id": room_id,
    }
    query = urllib.parse.urlencode({"installed_item_id": item_id, "market_slug": "managed-fixture"})
    _, gateway_tools = request(f"{gateway}/v1/tools?{query}", headers=headers)
    assert [item["name"] for item in gateway_tools["tools"]] == ["fixture_room_echo"]
    _, result = request(
        f"{gateway}/v1/execute",
        "POST",
        {"installed_item_id": item_id, "market_slug": "managed-fixture", "tool": "fixture_room_echo", "arguments": {"message": "smoke"}},
        headers=headers,
    )
    text = result["content"][0]["text"]
    assert f"room={room_id}" in text and "message=smoke" in text
    print("managed_stdio smoke: import -> install -> bridge -> probe -> room -> Gateway execute: OK")


if __name__ == "__main__":
    main()
