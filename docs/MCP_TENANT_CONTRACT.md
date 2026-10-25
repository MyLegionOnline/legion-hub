# Tenant-aware MCP contract

This contract applies to a Market item whose `type` is `mcp` and whose
`isolation` is `tenant_aware`. It is the boundary between Legion's MCP
Gateway and a stateful MCP server. The server author, not Legion, owns the
actual data isolation implementation.

## Request context

For every `tools/call` request, the Gateway adds a trusted context object to
the JSON-RPC `params` object:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "save_note",
    "arguments": {"title": "Roadmap", "body": "..."},
    "_meta": {
      "user_id": "<Legion user UUID>",
      "room_id": "<Legion room UUID>"
    }
  }
}
```

`user_id` always identifies the authenticated Legion user. `room_id` is
included when the call belongs to a room; it is omitted for a user-level call.
The values come from the authenticated Hub-to-Gateway headers and are not
accepted from tool arguments or from a caller-supplied `_meta` object. A
server MUST treat this context as authoritative for the tenant and MUST NOT
trust a duplicate value in `arguments`.

The context is attached to `tools/call` only. `tools/list` is a catalogue
operation and does not receive a tenant context.

## Server responsibilities

An implementation published as `tenant_aware` MUST:

1. Require `user_id` and reject requests that do not contain it.
2. Scope every persistent read, write, cache key, and background job to
   `user_id` (and to `room_id` where room-level separation is required).
3. Enforce isolation in the server's own storage layer, for example with
   PostgreSQL row-level security or a per-user schema/namespace. The Gateway
   only transports identity; it cannot provide database isolation.
4. Document any per-user credentials or data required in the Market item's
   `config_schema`. OAuth/OIDC exchange and secret storage are future work;
   this contract does not grant a server access to Legion credentials.

Servers MUST NOT log complete authorization headers, user secrets, or full
tool arguments when those may contain sensitive tenant data.

## Stateless items and custom MCPs

`stateless` items do not receive `_meta`; they are shared services and must
not rely on tenant context. A custom MCP is treated as `stateless` unless its
request explicitly declares `isolation: "tenant_aware"`; custom tenant-aware
servers are subject to the same obligations above.

## Publishing checklist

Before publishing a tenant-aware item, verify that its isolation tests cover
two users and two rooms, including attempts to read and mutate each other's
data. Set `isolation` to `tenant_aware` and provide a non-empty
`config_schema` describing required per-user configuration.
