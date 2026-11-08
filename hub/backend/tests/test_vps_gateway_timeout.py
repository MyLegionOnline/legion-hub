"""The Hub waits for a legion-vps-mcp command as long as the command may run.

No database, no network. The Hub used to give up on every Gateway call after
a flat MCP_GATEWAY_TIMEOUT_SEC (45 s on production) while vps_exec allows
60 s by default and 600 s at most, so long builds came back to the agent as a
gateway timeout while still running on the server.
"""

from app.core import mcp_gateway as gw


def test_vps_command_gets_its_deadline_plus_a_margin_above_the_gateways(monkeypatch):
    monkeypatch.setattr(gw, "MCP_GATEWAY_TIMEOUT", 45.0)
    monkeypatch.setattr(gw, "MCP_VPS_GATEWAY_TIMEOUT_MARGIN", 45.0)
    assert gw._gateway_timeout_for("legion-vps-mcp", "vps_exec", {"timeout_sec": 300}) == 345.0
    assert gw._gateway_timeout_for("legion-vps-mcp", "vps_exec", {}) == 105.0
    assert gw._gateway_timeout_for("legion-vps-mcp", "vps_service", {}) == 165.0
    assert gw._gateway_timeout_for("legion-vps-mcp", "vps_exec", {"timeout_sec": 10_000}) == 645.0


def test_non_command_and_non_vps_calls_keep_their_ceiling(monkeypatch):
    monkeypatch.setattr(gw, "MCP_GATEWAY_TIMEOUT", 45.0)
    assert gw._gateway_timeout_for("legion-vps-mcp", "vps_read_file", {}) == 45.0
    assert gw._gateway_timeout_for("mcp-searxng", "vps_exec", {"timeout_sec": 500}) == 45.0
    assert gw._gateway_timeout_for("mcp-vision", "describe_image", {}) == gw.MCP_VISION_GATEWAY_TIMEOUT
