"""The pairing dialog shows one install command; it must carry the Hub's address.

Without the address the connector dials its own default (localhost), so the
command is built from HUB_PUBLIC_URL, the browser's Origin, or the request host.
"""

from starlette.requests import Request

from app.api.v1 import connectors


def _request(headers: dict[str, str] | None = None, scheme: str = "http", host: str = "backend:8000") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if "host" not in (headers or {}):
        raw.append((b"host", host.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": scheme,
            "path": "/api/v1/connectors",
            "query_string": b"",
            "headers": raw,
            "server": (host.split(":")[0], 8000),
        }
    )


def test_command_uses_the_browser_origin_and_a_wss_url_for_https(monkeypatch):
    monkeypatch.delenv("HUB_PUBLIC_URL", raising=False)
    monkeypatch.delenv("LEGION_CONNECTOR_REPO_URL", raising=False)
    cmd = connectors._install_command("CODE-123_x", _request({"origin": "https://hub.example.com"}))
    assert cmd == (
        "git clone https://github.com/MyLegionOnline/legion-connector.git legion-connector"
        " && cd legion-connector"
        " && ./install.sh CODE-123_x wss://hub.example.com/api/v1/connector/ws"
    )


def test_plain_http_origin_gives_a_ws_url(monkeypatch):
    monkeypatch.delenv("HUB_PUBLIC_URL", raising=False)
    cmd = connectors._install_command("abc", _request({"origin": "http://localhost:3000"}))
    assert cmd.endswith("./install.sh abc ws://localhost:3000/api/v1/connector/ws")


def test_hub_public_url_wins_over_the_origin(monkeypatch):
    monkeypatch.setenv("HUB_PUBLIC_URL", "https://hub.example.org/")
    cmd = connectors._install_command("abc", _request({"origin": "http://internal:8000"}))
    assert cmd.endswith("./install.sh abc wss://hub.example.org/api/v1/connector/ws")


def test_without_origin_the_host_and_forwarded_scheme_are_used(monkeypatch):
    monkeypatch.delenv("HUB_PUBLIC_URL", raising=False)
    req = _request({"host": "hub.example.net", "x-forwarded-proto": "https"})
    assert connectors._connector_ws_url(req) == "wss://hub.example.net/api/v1/connector/ws"


def test_a_hostile_origin_cannot_inject_shell(monkeypatch):
    monkeypatch.delenv("HUB_PUBLIC_URL", raising=False)
    req = _request({"origin": "https://x.example/;touch /tmp/pwned"})
    cmd = connectors._install_command("abc", req)
    assert "touch" not in cmd.split("&&")[-1].split("wss://")[0]
    assert cmd.endswith("./install.sh abc wss://x.example/api/v1/connector/ws")


def test_repo_url_override_only_accepts_http_urls(monkeypatch):
    monkeypatch.setenv("LEGION_CONNECTOR_REPO_URL", "https://git.example.com/fork/legion-connector.git")
    assert connectors._connector_repo_url() == "https://git.example.com/fork/legion-connector.git"
    monkeypatch.setenv("LEGION_CONNECTOR_REPO_URL", "$(rm -rf /)")
    assert connectors._connector_repo_url() == connectors.DEFAULT_CONNECTOR_REPO_URL
