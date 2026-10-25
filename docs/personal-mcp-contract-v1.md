# Personal MCP contract v1

Status: fixed for stage 0. Contract version: personal-mcp.v1.

This document is the domain contract for managed personal MCP. The canonical
OpenAPI schema is in
[personal-mcp-openapi-v1.yaml](personal-mcp-openapi-v1.yaml), and the full
Depwire example is in [personal-mcp-depwire-v1.json](personal-mcp-depwire-v1.json).

## 1. Contract boundaries

The basic scenario supports only the node_npx_v1 profile:

- the transport on the runner's input is stdio; only an internal Streamable HTTP
  bridge is exposed outward;
- the user chooses the npm package and the MCP arguments, but not the image, host
  port or shell command;
- the runner resolves the package to a concrete version and integrity before the
  first launch;
- a floating latest is never launched: an update is a separate operation with a
  new revision and a repeated check;
- environment variable values are not accepted by this API. The configuration may
  carry only names and placeholders, after which the user picks existing Legion
  Secrets.

The existing Marketplace, HTTP(S)-link and personal_image modes do not change
their semantics. personal_image remains the advanced Docker path.

## 2. Allowed command forms

Every mcpServers server requires a string command and an array of string args.
Values are passed as argv; a shell is never launched.

### 2.1 npx

Exactly the following forms are supported:

~~~text
npx -y <package-spec> [server-arg...]
npx --yes <package-spec> [server-arg...]
~~~

-y and --yes are mandatory: interactive installation is incompatible with a cloud
runner. <package-spec> is an npm package name with an optional semver
version/range.

~~~json
{"command":"npx","args":["-y","depwire-cli","mcp"]}
{"command":"npx","args":["--yes","@acme/calendar-mcp@^2.4.0","--stdio"]}
~~~

In the internal launch profile the first argument is normalized to command_args:
["mcp"] or ["--stdio"]; npx, -y/--yes and the package are stored separately.

### 2.2 npm exec

For command: "npm" only the non-interactive form with a single package is
supported:

~~~text
npm exec --yes --package=<package-spec> -- <binary-or-server-arg...>
npm exec -y --package <package-spec> -- <binary-or-server-arg...>
~~~

In the first form there must be no second package after --package=. In the second
form --package and its value are an adjacent pair of arguments. The -- marker is
mandatory. The part after -- is stored as command_args; if the binary matches the
package bin, this is reflected in the preview, but it is not executed through a
shell.

### 2.3 General restrictions

The following are rejected:

- npm install, npm install -g, npm run, npm config, npm init and any other npm
  subcommands;
- npx without -y/--yes, several package specs, --package for npx and arbitrary npm
  flags;
- URL, file:, git:, git+*, github:, workspace:, local paths and package aliases;
- shell separators/redirection (;, |, &, <, >), backticks, $()/${...}, CR/LF and
  environment expansion in any argument;
- extra keys in the server object other than command, args and env;
- more than 20 servers per preview, more than 128 arguments per server, or an
  argument longer than 2048 UTF-8 bytes.

The package name is checked against the npm name grammar. The name may be
unscoped or scoped (@scope/name); the registry URL and the package source are not
chosen by the user. Unsupported input never degrades to shell execution.

### 2.4 Environment and secrets

env is an optional object of variable names. Only the following values are
allowed:

- "" — the value will be chosen in Legion Secrets;
- "${ENV_NAME}" or "$ENV_NAME" — a placeholder, which is not stored;
- null — the name is declared, no value is passed.

Any other non-empty string returns secret_value_not_allowed. The Hub does not echo
such input, does not put it into custom_config and does not write it to a log.

After the preview the API accepts only references of the form:

~~~json
{
  "env_name": "DEPWIRE_API_KEY",
  "secret_id": "00000000-0000-0000-0000-000000000042"
}
~~~

Responses contain env_name, secret_id, a safe secret label and the binding state,
but never contain a secret value.

## 3. Normalized launch profile

For the Depwire input:

~~~json
{
  "command": "npx",
  "args": ["-y", "depwire-cli", "mcp"]
}
~~~

the following profile is created:

~~~json
{
  "mode": "managed_stdio",
  "runtime_profile": "node_npx_v1",
  "transport": "stdio",
  "package": "depwire-cli",
  "command_args": ["mcp"],
  "env_names": [],
  "shell": false,
  "resolved_artifact": {
    "version": null,
    "integrity": null,
    "resolved_at": null
  }
}
~~~

resolved_artifact appears at pinning, before start/verify. The bridge endpoint is
not a user field: it is stored and passed only over the internal Hub → Gateway
contract.

The preview returns the following for each candidate:

| Field | Meaning |
| --- | --- |
| candidate_id | short-lived opaque id for create |
| source_name | the key from mcpServers |
| display_name | a safe initial name, editable by the user |
| mode | always managed_stdio |
| runtime_profile | always node_npx_v1 |
| package | the original spec, registry name and requested version/range |
| command_args | MCP process arguments after the package |
| env_names | environment variable names only |
| normalized_argv | a safe argv preview; shell: false |
| pinning | pending until resolution, then version/integrity |
| warnings | safe warnings, without input secrets |

## 4. State machine and room availability

The public state is split into desired_status and actual_status.

| State | Meaning | New room binding | Tools for agents |
| --- | --- | --- | --- |
| preparing | the item is created, the preview is accepted | no | no |
| starting | quota admission and runner start | no | no |
| running | the process is alive, the handshake is not finished yet | no | no |
| verifying | initialize and tools/list are in progress | no | no |
| verified | the revision and the tool fingerprint are verified | yes | yes |
| stopping | the runner is being stopped | no | no |
| stopped | the process is stopped, data and bindings are kept | no | no |
| updating | a new revision is being verified | no | no for the candidate |
| error | the last operation ended with a safe error | no | no |
| deleting | confirmed deletion of the item and data | no | no |
| deleted | terminal state, no longer returned in the list | no | no |

Allowed transitions:

~~~text
create       -> preparing
preparing    -> starting | error
stopped      -> starting | deleting
error        -> starting | stopped | deleting
starting     -> running | error
running      -> verifying | stopping | error
verifying    -> verified | error
verified     -> stopping | updating | error | deleting
stopping     -> stopped | error
updating     -> verifying | verified (rollback) | error
~~~

desired_status is stopped, running or deleted. POST /verify and POST /start set
running; POST /stop sets stopped. Every operation is idempotent for an
already-reached result: a repeated start does not create a second runner or
reservation.

Only verified is admitted to a new room binding. Stopping does not delete an
existing binding, but its available becomes false until the MCP passes verify
again. Assigning to a room never starts the MCP automatically.

## 5. API surface

The canonical HTTP surface is described in OpenAPI; the operation semantics:

| Method | Path | Semantics |
| --- | --- | --- |
| POST | /api/v1/personal-mcps/import-preview | parse/normalize, starts nothing |
| GET | /api/v1/personal-mcps | list of the owner's personal MCPs |
| POST | /api/v1/personal-mcps | create a candidate in preparing |
| GET | /api/v1/personal-mcps/usage | limits and allocated/used resources |
| GET | /api/v1/personal-mcps/{id} | state without endpoint and secret values |
| GET | /api/v1/personal-mcps/{id}/events | safe progress polling; SSE optional |
| POST | /api/v1/personal-mcps/{id}/verify | start + initialize + tools/list |
| POST | /api/v1/personal-mcps/{id}/start | start + verify |
| POST | /api/v1/personal-mcps/{id}/stop | stop, storage is kept |
| POST | /api/v1/personal-mcps/{id}/restart | stop/start with a new check |
| POST | /api/v1/personal-mcps/{id}/update | candidate → new revision, verify, atomic switch |
| DELETE | /api/v1/personal-mcps/{id} | explicit deletion of the process, data and bindings |
| GET/PUT | /api/v1/personal-mcps/{id}/secret-bindings | list/replace env bindings |
| GET/POST | /api/v1/personal-mcps/{id}/room-bindings | list/assign to a room |
| DELETE | /api/v1/personal-mcps/{id}/room-bindings/{room_id} | detach from a room |

GET .../events returns a JSON array with a cursor by default; an implementation
may add SSE for Accept: text/event-stream, keeping the same event types and
fields.

### 5.1 Create request

~~~json
{
  "contract_version": "personal-mcp.v1",
  "preview_id": "00000000-0000-0000-0000-000000000101",
  "candidate_id": "00000000-0000-0000-0000-000000000102",
  "display_name": "Depwire",
  "memory_reservation_mb": 1024,
  "storage_reservation_mb": 1024,
  "secret_bindings": []
}
~~~

Create only registers the item and performs a quota check against the
installed/storage limits. After a successful create, the UI button "Connect and
verify" calls POST .../{id}/verify; this lets the UI show progress and repeat
verify without re-importing.

### 5.2 Verify response

A successful verify returns actual_status: verified, the pinned revision, the
fingerprint and a safe tool snapshot:

~~~json
{
  "id": "00000000-0000-0000-0000-000000000103",
  "actual_status": "verified",
  "ready_for_room": true,
  "revision": 1,
  "resolved_artifact": {
    "package": "depwire-cli",
    "version": "1.4.2",
    "integrity": "sha512-<base64-integrity>",
    "resolved_at": "2026-08-05T18:30:00Z"
  },
  "tools": {
    "count": 2,
    "names": ["search", "fetch"],
    "fingerprint": "sha256:<hex>"
  },
  "correlation_id": "01JXXXXXVERIFY"
}
~~~

### 5.3 Usage response

Usage distinguishes a reservation from actual consumption:

~~~json
{
  "limits": {
    "mcp_personal_max_installed": 5,
    "mcp_personal_max_running": 1,
    "mcp_personal_max_total_memory_mb": 1024,
    "mcp_personal_max_memory_per_server_mb": 1024,
    "mcp_personal_storage_quota_mb": 2048,
    "mcp_personal_max_storage_per_server_mb": 1024,
    "mcp_personal_max_room_bindings_per_server": 3,
    "mcp_personal_monthly_hours": 40
  },
  "installed_count": 1,
  "running_count": 1,
  "memory_reserved_mb": 1024,
  "memory_used_mb": 80,
  "storage_allocated_mb": 1024,
  "storage_used_mb": 420
}
~~~

Stopping releases running_count and memory_reserved_mb, but not
storage_allocated_mb; deleting releases storage only after the volume has been
cleaned up successfully.

Within the legacy plan, the key mcp_personal_max_rooms is allowed only as a
temporary alias for mcp_personal_max_room_bindings_per_server. It does not mean a
total limit on the user's rooms and must not appear in a new usage payload.

## 6. Progress events

An event payload always contains event_id, type, step, occurred_at and
correlation_id. message is safe user-facing text. Technical logs are not part of
this API and are issued only through a sanitized operator diagnostics channel.

Allowed steps:

~~~text
config_accepted
package_resolving
package_pinned
runner_starting
protocol_initializing
tools_discovered
ready
failed
~~~

tools_discovered contains only names, count and fingerprint. It has no
descriptions, which could carry secret values; the snapshot limit and schema are
defined in OpenAPI.

## 7. Stable error codes

Error envelope:

~~~json
{
  "error": {
    "code": "personal_mcp_memory_limit",
    "message": "Not enough reserved RAM to start the MCP.",
    "remedy": "Stop another MCP or choose a smaller memory profile.",
    "retryable": false,
    "details": {"requested_mb": 1024, "available_mb": 512},
    "correlation_id": "01JXXXXXERROR"
  }
}
~~~

The code and fields are stable; message and remedy may be localized. details must
not contain command secrets, bearer tokens, full args, an npm response body or an
internal endpoint.

| Code | HTTP | When used | Recommended action |
| --- | ---: | --- | --- |
| invalid_json | 400 | the input is not JSON | fix the JSON |
| invalid_payload | 422 | the general schema is violated | fix the field |
| mcp_servers_required | 422 | the mcpServers object is missing | paste an MCP config |
| too_many_servers | 422 | more than 20 candidates | import in parts |
| duplicate_server_name | 422 | a duplicate server key | rename the key |
| server_name_invalid | 422 | an unsafe name | use a name from a-z0-9_- |
| unsupported_command | 422 | the command is not npx/npm | use a supported form or Docker |
| unsupported_runtime_profile | 422 | the profile is not node_npx_v1 | choose a supported runtime |
| interactive_npx_not_allowed | 422 | no -y/--yes | add -y |
| args_invalid | 422 | args is not an array of strings / limit exceeded | fix args |
| package_spec_invalid | 422 | an invalid npm name/spec | specify a registry package |
| package_source_not_allowed | 422 | a URL, git, file or path spec | use a registry package |
| unsupported_npm_subcommand | 422 | an install/run/config attempt | use the npx/npm exec form |
| shell_syntax_not_allowed | 422 | a shell delimiter/expansion | pass a literal argv |
| unsupported_config_key | 422 | an unknown key in the server object | remove the key |
| secret_value_not_allowed | 422 | the input contains a plaintext secret | choose a Legion Secret |
| env_name_invalid | 422 | an unsafe env name | use ^[A-Z_][A-Z0-9_]*$ |
| preview_not_found | 404 | the preview expired / does not belong to the user | repeat the preview |
| candidate_not_found | 404 | the candidate is missing | repeat the preview |
| personal_mcp_not_found | 404 | the item is missing / belongs to someone else | check the list |
| personal_mcp_feature_disabled | 403 | the feature flag is off | contact the administrator |
| personal_mcp_count_limit | 409 | the installed limit is exceeded | delete an MCP or change the plan |
| personal_mcp_running_limit | 409 | the running limit is exceeded | stop another MCP |
| personal_mcp_memory_limit | 409 | not enough RAM reservation | stop an MCP / reduce the preset |
| personal_mcp_storage_limit | 409 | not enough storage | delete data / reduce the preset |
| room_binding_limit | 409 | the binding limit is exceeded | detach another room |
| invalid_state_transition | 409 | the action is not allowed in this state | wait for the operation / choose another action |
| revision_conflict | 409 | the revision changed concurrently | refresh the card and retry |
| not_verified | 409 | attach before a successful probe | start and verify the MCP |
| secret_binding_invalid | 422 | the secret does not belong to the user / is inactive | choose an active Secret |
| secret_required | 409 | the runner needs an unbound env | add a binding and retry |
| package_not_found | 422 | the registry did not find the package | check the package name |
| package_registry_unavailable | 503 | the registry is unavailable | retry later |
| package_integrity_mismatch | 502 | the artifact did not match the pin | do not start, retry/update |
| runner_unavailable | 503 | the host/runner is unavailable | retry later |
| startup_timeout | 504 | the runner did not become ready in time | check logs/correlation id |
| protocol_handshake_timeout | 504 | no response to initialize/tools | check the server args |
| not_mcp_server | 422 | the process does not answer MCP JSON-RPC | choose another package |
| invalid_jsonrpc | 422 | a malformed protocol response | update/replace the package |
| invalid_tools_response | 422 | tools/list failed the schema | update/replace the package |
| tool_name_not_allowed | 422 | a tool name violates the allowlist | update the package |
| runtime_crash_loop | 503 | the runner was stopped after a crash loop | fix the config and retry |
| room_not_allowed | 403 | the room does not belong to the user | choose your own room |
| delete_confirmation_required | 428 | data deletion is not confirmed | repeat with confirm |
| delete_cleanup_failed | 502 | the volume was not cleaned up | do not treat storage as released |

## 8. Contract invariants for later stages

Later stages must not silently change the following invariants:

1. managed_stdio does not accept a user-selected image/port and does not launch a
   shell.
2. An unverified runtime gets no room binding and no Gateway endpoint.
3. Assignment does not start the process automatically; start/verify is a
   separate confirmed action.
4. Stop releases the running/RAM reservation but keeps storage/data.
5. Delete requires explicit confirmation and releases storage only after a
   confirmed cleanup.
6. Secrets are represented only as (env_name, secret_id) and a runtime lease;
   plaintext never appears in the DB, API responses, audit or logs.
7. Any change to the launch profile creates a new revision with pinning,
   re-verify and the ability to roll back.
