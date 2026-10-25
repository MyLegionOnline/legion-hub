"""[[personal_mcp]] in legion.toml (internal design notes stage 4.6): parsing, no database."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import personal_mcp_config as pmc
from app.core import registration

EXAMPLE = Path(__file__).resolve().parents[2] / "legion.example.toml"


def _load(tmp_path, body: str):
    path = tmp_path / "legion.toml"
    path.write_text(body, encoding="utf-8")
    return pmc.load_declared(str(path), {})


def test_no_file_means_nothing_declared():
    assert pmc.load_declared(None, {}) == []


def test_a_link_and_an_image(tmp_path):
    entries = _load(tmp_path, '''
[[personal_mcp]]
name = "docs"
url = "http://docs-mcp:8080/mcp"

[[personal_mcp]]
name = "notes"
image = "ghcr.io/owner/notes-mcp:1.2"
port = 9000

[[personal_mcp]]
name = "default-port"
image = "ghcr.io/owner/x:1"
''')
    assert entries == [
        {"name": "docs", "custom_config": {"url": "http://docs-mcp:8080/mcp"}},
        {"name": "notes", "custom_config": {"mode": "hosted", "image": "ghcr.io/owner/notes-mcp:1.2", "port": 9000}},
        {"name": "default-port", "custom_config": {"mode": "hosted", "image": "ghcr.io/owner/x:1", "port": 8000}},
    ]


@pytest.mark.parametrize("body, message", [
    ('[[personal_mcp]]\nurl = "http://x/mcp"\n', "name is required"),
    ('[[personal_mcp]]\nname = "a"\n', "either url"),
    ('[[personal_mcp]]\nname = "a"\nurl = "http://x/mcp"\nimage = "i"\n', "either url"),
    ('[[personal_mcp]]\nname = "a"\nurl = "ftp://x/mcp"\n', "absolute http"),
    ('[[personal_mcp]]\nname = "a"\nurl = "not a url"\n', "absolute http"),
    ('[[personal_mcp]]\nname = "a"\nurl = "http://u:p@x/mcp"\n', "credentials"),
    ('[[personal_mcp]]\nname = "a"\nurl = "http://x/mcp"\nport = 80\n', "port only goes with image"),
    ('[[personal_mcp]]\nname = "a"\nimage = "i"\nport = 70000\n', "TCP port"),
    ('[[personal_mcp]]\nname = "a"\nimage = "i"\nport = true\n', "TCP port"),
    ('[[personal_mcp]]\nname = "a"\nurl = "http://x/mcp"\ntoken = "secret"\n', "unknown key 'token'"),
    ('[[personal_mcp]]\nname = "a"\nurl = "http://x/mcp"\nbearer = "s"\n', "Secret"),
    ('[[personal_mcp]]\nname = "A"\nurl = "http://x/mcp"\n[[personal_mcp]]\nname = "a"\nurl = "http://y/mcp"\n', "used twice"),
    ('personal_mcp = "docs"\n', "[[personal_mcp]]"),
    ('[personal_mcp]\nname = "a"\n', "[[personal_mcp]]"),
])
def test_mistakes_stop_the_startup(tmp_path, body, message):
    with pytest.raises(pmc.PersonalMcpConfigError, match=message.replace("[", r"\[")):
        _load(tmp_path, body)


def test_a_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(pmc.PersonalMcpConfigError, match="not a file"):
        pmc.load_declared(None, {"LEGION_CONFIG": str(tmp_path / "nope.toml")})


def test_the_example_file_works_with_every_example_enabled(tmp_path):
    lines = []
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        lines.append(line[2:] if line.startswith("# ") and (" = " in line or line.startswith("# [[")) else line)
    path = tmp_path / "legion.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    entries = pmc.load_declared(str(path), {})
    assert [e["name"] for e in entries] == ["docs", "notes"]
    assert registration.load_mode(str(path), {}) == "closed"
