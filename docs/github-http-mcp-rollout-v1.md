# HTTP MCP discovery v1: rollout and operations

This runbook completes stage 6 of `internal design notes`.
It covers the verified HTTP link path only. Marketplace/SearXNG and managed
stdio keep their existing compatibility paths.

## Safety gates

The production Hub compose file sets
`HTTP_MCP_VERIFIED_DISCOVERY_V1=off`. A link MCP is not agent-available until
the flag is enabled for its owner and a non-expired verified snapshot exists.
Rollback is fail-closed: it hides link tools but does not delete bindings or
snapshots.

Allowed values are:

- `off` — no link MCP verification or tool injection;
- `internal` — only `HTTP_MCP_ROLLOUT_USER_IDS` may use the path;
- `canary` — only the explicit canary allowlist may use the path;
- `on` — enabled for all owners, unless the optional allowlist is non-empty.

Move through `internal → canary → on` only after the previous gate is signed
off. The flag is read at request time, so a rollback does not require a deploy.

The Gateway independently enforces the custom-link perimeter: HTTPS is
required in production, userinfo/query/fragment are rejected, redirects are
not followed, and DNS results resolving to loopback, private, link-local,
reserved, multicast, unspecified or metadata destinations are refused. Market
catalog endpoints and verified managed-stdio runtime endpoints remain on their
trusted compatibility paths.

## CI and release gate

The workflow `.github/workflows/github-http-mcp-stage6.yml` runs the bounded
HTTP MCP contract suite and the Hub discovery/runtime regression tests. The
fixture requires `Bearer fixture-upstream-token`, returns a deterministic
`tools/list`, and asserts that the bearer is present on both handshake frames,
absent from safe responses/log assertions, and reused for `tools/call`.

Before enabling a cohort, require:

1. CI green, including marketplace/SearXNG, managed stdio and HTTP MCP tests;
2. SSRF/redirect/TLS checks green;
3. dashboards showing no unexpected auth or transport spike;
4. one successful verified snapshot and one successful room/proxy/call smoke;
5. owner sign-off recorded with the opaque correlation ID only.

## Metrics

All labels are bounded `origin`, `outcome`, and `code`; no URL, item name,
user, room or tool name is a label.

- `mcp_discovery_total{origin,outcome,code}` — bounded discovery outcomes;
- `mcp_discovery_duration_seconds{origin,outcome,code}` — discovery latency;
- `mcp_tools_injected_total{origin,outcome,code}` — validated tools injected;
- `mcp_auth_failure_total{origin,outcome,code}` — `mcp_auth_failed` and
  `mcp_secret_unavailable` failures.

Useful dashboard panels:

```promql
sum by (origin, outcome, code) (rate(mcp_discovery_total[15m]))
histogram_quantile(0.95, sum by (le, origin) (rate(mcp_discovery_duration_seconds_bucket[15m])))
sum by (origin) (rate(mcp_tools_injected_total{outcome="injected"}[15m]))
sum by (origin, code) (rate(mcp_auth_failure_total[15m]))
```

The alert rules live in `hub/deployment/alerting_rules.yml` and intentionally
contain no credential-bearing diagnostic fields.

## Correlation and incident handling

Hub creates an opaque `mcpd_hub_*` correlation ID for verification, sends it in
`X-Legion-Correlation-ID` alongside (but separately from) the resolved upstream
header, and stores it in the safe snapshot/error DTO. Gateway validates and
returns the same ID. If a proxy task is blocked, the room `TASK_STALLED`
metadata contains only the safe error code, remedy and correlation ID.

For `mcp_auth_failed`:

1. Find the correlation ID in the Hub snapshot/room task event.
2. Confirm the item owner and active `MCP_AUTH_HEADER` binding without opening
   or copying its value.
3. Rotate/replace the Secret if required; binding mutation invalidates the
   snapshot.
4. Use the explicit UI retry once the binding is active.

For `mcp_unreachable` or `mcp_http_error`, check endpoint reachability, DNS,
TLS certificate and the Gateway SSRF policy. Do not inspect or paste upstream
response bodies into a room message.

## Rollback

Set `HTTP_MCP_VERIFIED_DISCOVERY_V1=off` and restart/reload the Hub deployment
if the environment does not support live environment refresh. New link tools
are hidden immediately at the request boundary; existing bindings and
`mcp_connection_snapshot` rows remain available for later recovery. Restore
the previous cohort only after the relevant alert has cleared and a manual
verified retry succeeds.
