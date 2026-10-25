# HTTP MCP discovery and upstream-auth contract v1

Status: Stage-0 contract/security baseline; it allows implementation of
stages 1–2 of the HTTP MCP discovery and upstream-auth work. Production rollout
is not approved by this document.

Decision date: 2026-08-12.

## 1. Purpose and boundaries

The contract describes how a connected HTTP(S) MCP is verified before its tools
are passed to the model, and how a bearer secret is safely passed from the Hub to
the Gateway. It applies to link MCP with a `SecretBinding` of purpose
`MCP_AUTH_HEADER` and to public/market MCP without upstream credentials.

In scope:

- `initialize → tools/list` discovery and a safe tool snapshot;
- the same upstream-auth path for `tools/call`;
- stable statuses and error codes for the Hub, the UI and task preflight;
- size limits, TTL, rate limit, redaction and audit;
- prompt/capability context rules so that a Secret never reaches the LLM.

Out of scope: OAuth/refresh tokens, the GitHub API, changing GitHub App
permissions and bypassing the SSRF/network policy. OAuth requires a separate
contract version.

The normative words MUST, MUST NOT, SHOULD and MAY have their usual RFC 2119 meaning.

## 2. Stage-0 decision record

| Decision | Stage-0 status | Gate before production |
| --- | --- | --- |
| Discovery runs through the versioned `POST /v1/tools/discover` | accepted | Gateway contract tests |
| Hub service auth and the upstream bearer are different credentials/headers | accepted | negative auth tests |
| The Secret is resolved just-in-time from an active item-scoped binding | accepted | owner/resource/purpose isolation tests |
| A verified snapshot has a 60-second TTL | accepted | stale/expiry tests |
| An unverified or expired link MCP is not visible to the agent as a capability | accepted | room/preflight E2E |
| The tool snapshot is bounded by count, fields and total size | accepted | size-limit fixtures |
| Gateway/Hub logs, events, metrics and LLM history contain no bearer | accepted | redaction suite |
| The feature flag rollout stays off until stage 6 | accepted | canary sign-off |

This is the security/Secrets decision baseline for the implementation. It does
not replace production approval, TLS/mTLS review, SSRF review or rollout sign-off.

### 2.1 Trust boundaries and assets

1. Browser → Hub: the URL, room assignment and binding ID are untrusted input.
2. Hub → Gateway: a service-authenticated internal request; only the Hub may
   pass the resolved upstream credential.
3. Gateway → HTTP MCP: an external network boundary; the upstream receives only
   the required bearer header and MCP JSON-RPC frames.
4. Gateway/Hub → model/agent: snapshot metadata is untrusted tool metadata, not
   an instruction channel.

The Secret value is a separate asset in the SecretStore. The endpoint URL, tool
description and `inputSchema` are treated as external data and go through
SSRF/size/schema checks.

## 3. Internal Gateway API

### 3.1 Service authentication

Every Hub request to the Gateway MUST contain:

```http
Authorization: Bearer <gateway-service-token>
```

This header authenticates Hub → Gateway and is not passed to the upstream MCP.

For `auth.mode = bearer` the request additionally contains:

```http
X-Legion-MCP-Upstream-Authorization: Bearer <resolved-upstream-value>
```

`X-Legion-MCP-Upstream-Authorization` MUST be read only after the service
Authorization has been verified successfully. The Gateway MUST NOT use this
header as a replacement for its service token, store it, or include its value in
logs, exceptions, metrics, traces, response headers or cache keys.

The upstream header value must be a single line without CR/LF, in the format
`Bearer` + one space + an opaque value of at most 4096 bytes. The Hub obtains the
value only through `resolve_for_use()` of an active binding. A label, a
`secret_id` and user input cannot be substituted for the value.

### 3.2 Discovery request

Endpoint: `POST /v1/tools/discover`.

The request body contains only non-secret data, created by the Hub after
checking owner/room/item:

```json
{
  "installed_item_id": "00000000-0000-0000-0000-000000000042",
  "market_item_id": null,
  "market_slug": null,
  "custom_config": {
    "url": "https://api.githubcopilot.com/mcp/readonly",
    "auth": {"mode": "bearer"}
  }
}
```

Fields:

| Field | Requirement |
| --- | --- |
| `installed_item_id` | required UUID; the item, binding and room are checked by the Hub |
| `market_item_id` | optional UUID; `null` for a custom link |
| `market_slug` | optional string, max 100 chars |
| `custom_config` | an object for link MCP, or `null` for a catalog item |
| `custom_config.url` | required for a link; max 2048 chars; no userinfo |
| `custom_config.auth.mode` | only `bearer` or `none` |

`custom_config` MUST reject, or strip before the network call, the following
fields: `token`, `api_key`, `secret`, `secret_id`, `headers`, `authorization`,
`runtime_auth`, `runtime`, environment values and any unknown fields. In
particular, `runtime_auth` is not an alternative authorization channel.
Secret-bearing data MUST NOT be in the query string, URL fragment, JSON body or
`installed_item_id`/slug.

`url` in this internal contract is Hub-provided server configuration, not an
unverified browser URL. The Gateway MUST still enforce the URL policy:
production HTTPS, a ban on localhost/private/link-local/metadata ranges,
DNS-rebind protection, bounded redirects and a connect/read timeout. This check
is part of the SSRF perimeter and cannot be disabled by a request field.

If `auth.mode = none`, the upstream-auth header MUST be absent. If
`auth.mode = bearer`, the header is required; its absence must not lead to an
anonymous upstream request.

### 3.3 Discovery response

A successful discovery returns HTTP 200 and a safe DTO:

```json
{
  "installed_item_id": "00000000-0000-0000-0000-000000000042",
  "status": "verified",
  "tools": [
    {
      "name": "get_commit",
      "description": "Read a commit by repository and SHA.",
      "inputSchema": {
        "type": "object",
        "properties": {"repository": {"type": "string"}},
        "required": ["repository"]
      }
    }
  ],
  "fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "checked_at": "2026-08-12T12:00:00Z",
  "expires_at": "2026-08-12T12:01:00Z",
  "ttl_seconds": 60,
  "correlation_id": "mcpd_01HXXXXXXXXXXXXXXXXXXXX"
}
```

`checked_at` and `expires_at` are UTC RFC 3339 timestamps. `fingerprint` is a
lowercase SHA-256 with the prefix `sha256:`. `correlation_id` is an opaque ID
without a URL, item label or Secret data; it is needed only for safe
cross-service diagnostics.

On an error the Gateway/Hub returns the same shape with `status = "error"`, an
empty `tools`, `fingerprint = null` and a safe error object:

```json
{
  "installed_item_id": "00000000-0000-0000-0000-000000000042",
  "status": "error",
  "tools": [],
  "fingerprint": null,
  "checked_at": "2026-08-12T12:00:00Z",
  "expires_at": null,
  "ttl_seconds": 60,
  "error": {
    "code": "mcp_auth_failed",
    "message": "MCP server authentication failed",
    "retryable": false,
    "retry_after_seconds": null
  },
  "correlation_id": "mcpd_01HXXXXXXXXXXXXXXXXXXXX"
}
```

The body must not include the upstream response body, URL, request headers,
Secret label/ID/value, a stack trace, or tool metadata received before the
snapshot was validated successfully.

HTTP status boundary:

- `200` — `status = verified`;
- `401/403` — the Gateway service Authorization is missing or wrong; this is not
  an upstream `mcp_auth_failed`;
- `422` — request shape/field validation;
- `502` — a safe discovery error from the table below;
- `429` — a local cooldown/rate limit, with a bounded `Retry-After`, without a
  request to the upstream.

### 3.4 Execute compatibility

`POST /v1/execute` uses the same service/upstream headers and resolution policy.
Its JSON body MUST NOT contain `runtime_auth`, `token` or a plaintext
credential. Gateway applies one resolved bearer to session `initialize` and
`tools/call`; Hub uses the same binding helper as discovery.

The legacy `GET /v1/tools` may remain until a controlled migration, but it cannot
be the new source of truth for link MCP and must not accept secret-bearing
query/config fields.

## 4. MCP handshake and validation

For discovery the Gateway performs exactly one bounded handshake per
single-flight:

1. `POST <endpoint>` JSON-RPC `initialize` with protocol version
   `2025-03-26`, `clientInfo.name = legion-mcp-gateway` and empty capabilities.
2. For Streamable HTTP it passes `MCP-Protocol-Version` and, after a successful
   initialize, sends `notifications/initialized` in the same MCP session.
   For protocol `2026-07-28` and newer the lifecycle notification is not sent.
3. It stores only the opaque `Mcp-Session-Id`, if one is returned, and performs
   `tools/list` in the same MCP session.
4. All handshake requests get the same upstream `Authorization: Bearer ...`; the
   session id and protocol version are repeated on subsequent requests.
5. On success the tools go through limits/schema validation and only then become
   the verified snapshot.

Minimal transport headers:

```http
Content-Type: application/json
Accept: application/json, text/event-stream
Authorization: Bearer <resolved-upstream-value>
MCP-Protocol-Version: 2025-03-26
```

The MCP response body is not copied into diagnostics. A JSON-RPC error,
malformed JSON/SSE, an invalid tools shape and duplicate tool names lead to a
safe code from the table below.

### 4.1 Snapshot limits

| Limit | v1 value | When exceeded |
| --- | ---: | --- |
| Request body (without headers) | 16 KiB | `422` |
| Endpoint URL | 2048 UTF-8 bytes | `422` |
| MCP response body | 1 MiB | `mcp_bad_response` |
| Tools count | 128 | `mcp_bad_response` |
| Tool name | 255 UTF-8 bytes | `mcp_bad_response` |
| Description | 4096 UTF-8 bytes | `mcp_bad_response` |
| One `inputSchema` | 32 KiB | `mcp_bad_response` |
| JSON nesting depth | 20 | `mcp_bad_response` |
| Complete normalized snapshot | 256 KiB | `mcp_bad_response` |
| Successful snapshot TTL | 60 seconds | expired; tools hidden |
| Per-operation timeout | 10 seconds | `mcp_unreachable` |

Tool name MUST match `^[A-Za-z0-9_.:-]{1,255}$`. `inputSchema` MUST be a JSON
object. Descriptions are untrusted text: strip NUL/unsafe control characters,
keep newlines/tabs, and UI-escape before display. Tool metadata MUST NOT be
interpreted as system/developer instructions.

The snapshot fingerprint is calculated over canonical UTF-8 JSON of the list
sorted by tool name. Each item has exactly `name`, `description` and
`inputSchema`; object keys are recursively sorted, array order is preserved,
and JSON has no insignificant whitespace. Hash input is never logged.

An empty tool list is not agent-available and results in `mcp_protocol_error`.
Duplicate names, invalid names/schema, or a changed fingerprint during a
refresh invalidate the previous snapshot; old tools MUST NOT remain injected.

## 5. Error taxonomy

Only these upstream/domain codes are stable v1 contract values:

| Code | Meaning | Retryable | Safe message |
| --- | --- | --- | --- |
| `mcp_auth_failed` | upstream returned 401/403 during handshake/call | no; manual retry after credential change | `MCP server authentication failed` |
| `mcp_unreachable` | DNS/connect/read timeout or transport failure | yes, bounded | `MCP server is unavailable` |
| `mcp_protocol_error` | initialize/tools/list protocol or semantic contract failure | no until endpoint changes | `MCP server protocol validation failed` |
| `mcp_bad_response` | invalid JSON/SSE, oversize body, invalid/oversize tools | no until endpoint changes | `MCP server returned an invalid response` |
| `mcp_http_error` | upstream HTTP error other than 401/403 | bounded; honor safe Retry-After | `MCP server returned an HTTP error` |
| `mcp_secret_unavailable` | no active item-scoped binding or `resolve_for_use()` failed | no network retry | `MCP authorization is not configured` |

401/403 MUST never be generalized to `mcp_unreachable`. Upstream response text,
WWW-Authenticate contents and `Retry-After` are not copied to user-visible
messages; only a bounded integer retry delay MAY be retained.

## 6. Snapshot lifecycle and triggers

### 6.1 Availability rule

"Available to agents" is true only when all conditions hold:

- item active and explicitly assigned to the room;
- owner/room access check passed;
- exactly one active binding exists when `auth.mode = bearer`;
- snapshot status is `verified`;
- current time is before `expires_at`;
- snapshot revision matches endpoint and active binding revision;
- snapshot contains at least one validated tool.

`unknown`, `checking`, `error` and an expired `verified` snapshot are not
available. Expiry is a derived effective state; the stored enum remains
`unknown | checking | verified | error`.

### 6.2 Discovery triggers

| Trigger | Action | Automatic network request |
| --- | --- | --- |
| item/link + binding committed | set `unknown`, then async `checking` | one bounded discovery after commit |
| explicit UI "Check again" | set `checking` if cooldown allows | one bounded discovery |
| session preflight with fresh verified snapshot | use snapshot | none |
| session preflight with expired/unknown snapshot | set `checking` | one single-flight discovery |
| room sidebar/render | read safe DTO | none |
| missing/disabled/revoked binding | invalidate to `unknown` | none |

Discovery never auto-runs merely because a component rendered. A failed
discovery does not auto-start the task and does not cause raw HTTP/curl/token
fallback.

### 6.3 Binding, endpoint and fingerprint changes

The binding helper MUST select by owner, installed item and purpose
`MCP_AUTH_HEADER`, never by label. Create, rotate, replace, disable or delete
of that binding invalidates the snapshot atomically. URL/market revision
changes do the same. The next verification uses the new binding through a
fresh `resolve_for_use()` call; it never reuses a previous header.

If refresh returns a different tool fingerprint, the old tools are removed and
the item stays unavailable until the new snapshot is explicitly accepted by
the verification flow.

### 6.4 Rate and retry policy

- At most one in-flight discovery exists per `installed_item_id` (single-flight).
- A successful snapshot is not refreshed before its 60-second TTL.
- Manual retry cooldown is 10 seconds per item; no more than 6 attempts per
  item in any rolling 10-minute window and 30 attempts per owner in that
  window.
- One trigger performs one `initialize` and one `tools/list`; no retry loop is
  allowed inside a handshake. Transport failures use the next eligible
  trigger, not a tight loop.
- Repeated transport/HTTP failures use bounded backoff up to 300 seconds.
  401/403, missing Secret and validation errors do not receive automatic
  retries; changing the binding/endpoint or an explicit retry is required.
- If upstream returns `Retry-After`, retain only a bounded integer in
  `[1, 300]` seconds and never expose the raw header.

These limits protect the remote MCP rate limit and prevent a UI render or LLM
turn from generating repeated `tools/list` calls.

## 7. Secret, logging and audit policy

The following values MUST NOT occur in database snapshots, room events,
messages, task descriptions, LLM history, exception text, access logs, metrics
labels, traces or response bodies:

- bearer value and Gateway service token;
- Secret value, label, ID, binding label and environment variable name;
- full endpoint URL, query, fragment, authorization header and upstream body.

Safe diagnostics MAY contain only:

```json
{
  "server_kind": "http_link",
  "origin_hash": "sha256:<64 lowercase hex>",
  "installed_item_id": "<opaque internal id>",
  "outcome": "error",
  "code": "mcp_auth_failed",
  "duration_ms": 812,
  "correlation_id": "mcpd_<opaque>"
}
```

The `installed_item_id` above is for operator-scoped diagnostics only and MUST
not be sent to the LLM or used as a metrics label. Hashes must be computed
after URL sanitization and are not reversible substitutes for logging the URL.
Secret usage is audited as metadata: owner, item, purpose, binding revision,
timestamp, outcome and correlation ID. The value is never audited.

## 8. Agent/task context rule

Before `verified`, task context MAY say only:

> GitHub (read-only) MCP is unavailable: MCP authorization failed. Do not use
> raw HTTP, curl/requests or a Secret; a Retry is required after the binding is
> fixed.

After `verified`, task context MAY expose only the capability label and
validated `mcp__<server>__<tool>` schemas. It MUST NOT expose endpoint URL,
Secret label/ID/value, binding ID, raw gateway error or instructions to guess
credentials. The task generator MUST use capability status, not a Secret label,
as the source of truth.

If a requested resource requires an unavailable MCP, preflight returns the safe
code and blocks/asks for retry. It does not grant shell/network fallback.

## 9. Contract fixtures

Fixtures are versioned under
`mcp_gateway/backend/tests/fixtures/http_mcp_discovery/` and use placeholders
such as `<gateway-token>` and `<fixture-upstream-token>`. These placeholders
are deliberately not credentials and MUST be replaced in a test process, never
committed with real values.

| Fixture | Covers | Expected result |
| --- | --- | --- |
| `protected_success.json` | bearer header on initialize and tools/list | `verified`, one tool snapshot |
| `public_success.json` | public MCP without upstream auth | `verified`, no Authorization upstream |
| `auth_failed.json` | 401 on initialize | `mcp_auth_failed`, no tools/list |
| `unreachable.json` | connect/read timeout | `mcp_unreachable` |
| `protocol_error.json` | invalid initialize/tools/list protocol | `mcp_protocol_error` |
| `bad_response.json` | malformed/oversize/invalid tools payload | `mcp_bad_response` |
| `http_error.json` | upstream 429/5xx without raw body leakage | `mcp_http_error` |
| `secret_unavailable.json` | no active binding; no Gateway call | `mcp_secret_unavailable` |
| `execute_protected_success.json` | same bearer path for tools/call | successful call |

Each fixture states request body, expected upstream frames/headers, safe
response and forbidden leakage assertions. Contract tests MUST assert header
presence/absence rather than compare against a committed secret value.

## 10. Stage-0 acceptance checklist

- [x] DTO, status enum, error codes, response mapping and fingerprint format
  are defined.
- [x] Snapshot size limits, 60-second TTL, timeouts, single-flight and retry
  cooldown are defined.
- [x] Separate service Authorization and upstream-auth header are defined.
- [x] Secret resolution scope, logging/redaction/audit and prompt prohibition
  are defined.
- [x] Discovery triggers, binding invalidation and remote rate-limit behavior
  are defined.
- [x] Contract fixtures cover success, public, auth, transport, protocol,
  response, HTTP-error, missing-secret and execute paths.
- [ ] Stage 1 implementation and Gateway tests — next stage.
- [x] Stage 6 implementation: SSRF/redirect perimeter, metrics, correlation,
  feature flag, CI workflow and rollback runbook.
- [ ] Production Security/TLS and canary sign-off — operator gate described in
  `docs/github-http-mcp-rollout-v1.md`.
