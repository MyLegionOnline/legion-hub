# Legion Connector Protocol v2

This document specifies the WebSocket protocol between Legion Hub and the
[Legion Connector](https://github.com/MyLegionOnline/legion-connector). The
connector runs an OpenCode agent on a machine controlled by its owner.

## Transport and authentication

The connector opens an outbound WebSocket to:

```text
wss://<hub>/api/v1/connector/ws?token=<access-jwt>
```

The Hub creates a one-time pairing code with `POST /connectors`; the connector
exchanges it for a refresh token at `POST /connectors/pair`, then renews its
access JWT at `POST /connectors/token`. Pairing codes expire after 15 minutes.
The connection is multiplexed: frames carry `room_id`, `agent_id`, `run_id`,
and `session_id` where they apply.

Frames are JSON objects. The Hub sends `ping` every 30 seconds and the
connector responds with `pong`. A missing `pong` marks a connector offline;
dispatch is bounded by `CONNECTOR_DISPATCH_TIMEOUT_MINUTES`.

## Common fields

- `type` identifies the frame.
- `seq` is a monotonically increasing sequence number for Hub-to-connector
  frames. The connector acknowledges it with `ack`.
- `run_id` fences an execution. A receiver must discard a frame for a stale
  run.
- `session_id` identifies the session affected by an execution frame.

## Hub-to-connector frames

### `dispatch`

Starts work for an agent. Its payload includes task content, the selected
`mode`, an optional `cwd_hint`, non-secret room variables, and execution
metadata. Room secrets are never included. The only currently supported mode
is `operator`.

### `answer`, `ask_cm`, and `stop`

`answer` supplies the response to a pending human question. `ask_cm` is a
ChatMaster response. `stop` requests that the connector stop the run. The
connector must stop its local session within five seconds.

### `ping`

`{"type":"ping"}` has no run or session identifier.

## Connector-to-Hub frames

### `hello`

The first frame announces the connector version, supported modes, and the
required `protocol_version`. Optional `cli_version` and `opencode_api` fields
describe the detected local OpenCode installation.

### `status` and `reasoning`

`status` reports lifecycle events such as `LLM_REQUEST_SENT`,
`LLM_HEARTBEAT`, `TOOL_CALL_STARTED`, and `TOOL_CALL_FINISHED`. `reasoning`
is streamed reasoning content associated with a current run.

### `ask_user`, `ask_cm`, and `result`

`ask_user` creates a question for the human, `ask_cm` asks the ChatMaster, and
`result` completes the execution. A result may include optional usage data;
usage is informational and is not a billing source.

### `error`

The connector reports failures using a stable `payload.code`. Relevant codes
include `mode_not_supported`, `unknown_frame`, `cwd_not_allowed`,
`malformed_dispatch`, `driver_start_failed`, `opencode_version_unsupported`,
`opencode_api_mismatch`, and `confirm_ops_not_applied`.

For an unknown frame, send `error` only when both `run_id` and `session_id`
are present, then acknowledge the frame. A connection-level unknown frame is
logged locally because the Hub cannot associate an error with a run.

### `ack` and `resume`

`ack` confirms a Hub sequence number. After reconnecting, `resume` includes
the last acknowledged sequence number; the Hub replays later frames from its
durable outbox.

## Compatibility policy

The connector announces `protocol_version` in `hello`. Hub keeps an inclusive
range, `MIN_CONNECTOR_PROTOCOL` through `CONNECTOR_PROTOCOL`; the current
release uses version 2 for both bounds. A connector outside the range, or an
older connector that omits the field and is therefore treated as version 1,
is marked `incompatible`. Its WebSocket remains open so it can update, but it
receives no `dispatch`, replay, or run-specific response frames.

The Hub completes a rejected task immediately with
`connector_incompatible`; it does not wait for the dispatch timeout. A mode
absent from `capabilities.modes` is rejected with
`connector_mode_unsupported`.

Compatible changes include new optional fields and new optional error codes.
New mandatory frames or fields, and changed semantics of existing fields,
require a new protocol version. Hub supports the preceding version for at
least one minor release: first it expands its range, then a later release may
raise the minimum.
