# Legion MCP Gateway

The Gateway is a deliberately stateless FastAPI reverse proxy between Hub and
HTTP-transport MCP servers. It owns no user or catalogue database. Hub sends
an `Authorization: Bearer <MCP_GATEWAY_TOKEN>` service credential plus
`X-Legion-User-Id`/`X-Legion-Room-Id`; the Gateway resolves the Market snapshot,
meters the call, and proxies MCP JSON-RPC.

## Local run

```powershell
cd mcp_gateway/backend
py -m pip install -r requirements.txt
py -m uvicorn app.main:app --reload --port 8310
```

`deployment/searxng/` builds the SearXNG image with the Mojeek API engine; the
Gateway tests read that engine from there. Self-hosters run the Gateway as part
of `deploy/selfhost/`; the hosted shared SearXNG and Vision deployment is
operator configuration and is not shipped with the Hub.

Endpoints:

- `POST /v1/tools/discover` — verified HTTP link MCP handshake. Hub supplies
  the service token and, for bearer links, the separate resolved upstream
  auth header; the request also carries an opaque correlation ID.
- `GET /v1/tools?installed_item_id=...&market_item_id=...` — proxy MCP `tools/list`.
- `POST /v1/execute` — proxy MCP `tools/call`; the body includes the installed
  item, tool name, and arguments. For `tenant_aware` catalog items, the
  Gateway adds the trusted `_meta.user_id` and optional `_meta.room_id` to the
  MCP request; callers cannot override the catalog isolation mode.
- `GET / PUT /v1/admin/vision-config` — separate Admin Center-authenticated
  operator channel. It forwards provider settings to `mcp-vision`; provider
  keys are never stored by Gateway or returned in responses.
- `POST /v1/admin/vision-models` — validates a submitted or encrypted stored
  provider key and returns the provider's filtered image-capable model catalog.

`mcp-vision` (a Market tool, not part of this repository) uses separate MCP,
service-admin, Gateway-admin, and encryption tokens.

Custom HTTP links are an SSRF boundary: production requires HTTPS, private
and link-local destinations are rejected after DNS resolution, and redirects
are not followed. Configure `MCP_GATEWAY_REQUIRE_HTTPS` and
`MCP_GATEWAY_DNS_REBIND_PROTECTION` in deployment rather than in request data.

The server-side contract for stateful MCP implementations is documented in
[`docs/MCP_TENANT_CONTRACT.md`](../docs/MCP_TENANT_CONTRACT.md).
