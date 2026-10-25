# Legion managed MCP runner

`mcp_runner` is the operator-owned image used for the `node_npx_v1`
`managed_stdio` profile. It starts one pinned npm package with `npx` using a
direct argv (`shell: false`) and exposes a private `/mcp` Streamable HTTP
bridge on port `8080`.

The runner accepts its normalized profile through
`MCP_RUNNER_PROFILE_JSON`. A profile must contain an exact npm version and
`sha512` integrity before the child process starts; profiles without a pin are
resolved with `npm view` and written to `/data/.mcp-cache/locks`. The npm cache
and package data live under `/data`, which is mounted per user/item by
`mcp_host`.

The managed Node profile reserves at least 512 MB RAM and 512 MB storage:
package extraction and the first MCP parse can exceed a 256 MB cgroup limit,
and the pinned package cache is persistent. Before every child spawn the
runner removes only the derived `/data/.npm/_npx` extraction directory. npm's
tarball cache and other item data remain intact, so a failed install can be
retried without reusing a partial extraction.

Useful internal endpoints:

- `GET /healthz` — process health;
- `GET /readyz` — child process/readiness state;
- `GET /metrics` — bounded RSS and `/data` usage metrics;
- `GET /logs` — bounded, sanitized diagnostic entries;
- `POST /mcp` — JSON-RPC requests forwarded to the stdio child.

Build and publish the versioned operator image, then configure the same tag in
`MCP_HOST_RUNNER_IMAGE`:

```text
docker build -t legion/mcp-node-runner:1.0.0 mcp_runner
```

The image runs as UID/GID `10001`, and `mcp_host` additionally applies a
read-only root filesystem, dropped capabilities, no-new-privileges, a bounded
PID/file-descriptor budget, an item-scoped `/data` volume, and no host port
binding.
