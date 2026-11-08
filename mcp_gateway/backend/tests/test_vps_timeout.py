"""The upstream request ceiling for a legion-vps-mcp command follows the command's own deadline.

A flat 30 s cut abandoned every longer build or restart while it was still
running on the server (2026-11-07: 57 cut-offs in one evening).
"""

from __future__ import annotations

from app import config
from app import main as gateway_main


def test_vps_exec_waits_for_its_own_timeout_plus_margin(monkeypatch):
    monkeypatch.setattr(config, "VPS_TIMEOUT_MARGIN_SEC", 30.0)
    monkeypatch.setattr(config, "REQUEST_TIMEOUT_SEC", 30.0)
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_exec", {"timeout_sec": 300}) == 330.0
    # The server's own defaults when the model names no timeout.
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_exec", {}) == 90.0
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_service", {}) == 150.0


def test_vps_exec_timeout_is_capped_like_the_server_caps_it(monkeypatch):
    monkeypatch.setattr(config, "VPS_TIMEOUT_MARGIN_SEC", 30.0)
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_exec", {"timeout_sec": 99999}) == 630.0
    # Garbage is the server's default, not a crash or an unbounded wait.
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_exec", {"timeout_sec": "long"}) == 90.0
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_exec", {"timeout_sec": True}) == 90.0


def test_other_tools_keep_the_default_ceiling():
    assert gateway_main._upstream_timeout_sec("legion-vps-mcp", "vps_read_file", {"timeout_sec": 500}) is None
    assert gateway_main._upstream_timeout_sec("mcp-searxng", "vps_exec", {"timeout_sec": 500}) is None
    assert gateway_main._upstream_timeout_sec("mcp-vision", "describe_image", {}) == config.VISION_REQUEST_TIMEOUT_SEC
