# mcp_host

Internal supervisor for per-user MCP containers. It is the only service that
mounts Docker's socket; Hub calls it with `Authorization: Bearer $MCP_HOST_TOKEN`.

The legacy `personal_image` API remains available. Managed stdio requests use
the operator-owned `legion/mcp-node-runner:1.0.0` image and a normalized
`node_npx_v1` profile. The host chooses the runner image, private network and
port; a browser cannot provide any of them. The runner exposes only
`/mcp`/`/readyz` on the private Docker network and never receives a host port
binding.

Managed stdio secrets are not sent in Docker `Env` or the launch profile.
Hub issues mcp_host a one-time, item-scoped lease; the host consumes it and
bootstraps the runner over the private network. The runner accepts only the
allowlisted env names and redacts their values from child diagnostics.

Managed MCP data is stored in one named volume per user/item. `stop` removes
the process/container but preserves that volume; `DELETE` requires an explicit
confirmation and reports success only after Docker confirms the volume cleanup.

For production, set `MCP_HOST_STORAGE_QUOTA_MODE=enforced` and use a volume
driver that enforces `MCP_HOST_VOLUME_SIZE_OPTION` (the default local driver is
marked `best_effort` for development only).

## Local Hub deployment

The Hub development deployment includes `mcp-host` in
`hub/deployment/docker-compose.dev.yml`. Build the operator-owned runner once
from the repository root, then start the supervisor through the same Compose
project as Hub:

```text
docker build -t legion/mcp-node-runner:1.0.0 mcp_runner
docker compose -p deployment -f hub/deployment/docker-compose.dev.yml up -d --build mcp-host
```

The supervisor is reachable as `http://mcp-host:8320` from Hub. It creates
per-item runner containers on the Compose project's private `mcp-network`,
which is also attached to the MCP Gateway. If a custom Compose project name
is used, set `MCP_HOST_NETWORK` and `MCP_HOST_RUNNER_NETWORK` to the concrete
Docker network names (for example, `deployment_legion-network` and
`deployment_mcp-network`). The runner image is intentionally not exposed on
the host. The `-p deployment` value must match the project name used for the
already-running Hub stack; use the same value for all following commands.

The managed Node profile uses 512 MB minimum RAM and storage presets. The host allows
up to 60 seconds for a cold npm extraction and MCP protocol probe; this avoids
removing a healthy runner while its package is still parsing for the first
time.

Verify the supervisor before retrying an installation:

```text
docker compose -p deployment -f hub/deployment/docker-compose.dev.yml ps mcp-host
docker compose -p deployment -f hub/deployment/docker-compose.dev.yml exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://mcp-host:8320/health', timeout=5).read().decode())"
```
