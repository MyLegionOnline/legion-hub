"""The gateway has no built-in service tokens (internal design notes stage 4): the
tests get their own, read by ``app.config`` when it is first imported."""

import os

os.environ.setdefault("MCP_GATEWAY_TOKEN", "test-gateway-token")
os.environ.setdefault("MCP_GATEWAY_ADMIN_TOKEN", "test-gateway-admin-token")
os.environ.setdefault("MARKET_API_TOKEN", "test-market-token")
