import os


def _env_or_file(name: str, default: str = "") -> str:
    """``NAME_FILE`` (a Docker secret path) wins over ``NAME``, as in Hub and Market."""
    file_path = os.getenv(f"{name}_FILE")
    if file_path:
        with open(file_path, encoding="utf-8") as fh:
            return fh.read().strip()
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# No built-in default: without a token every call is refused (see app/auth.py).
MCP_HOST_TOKEN = _env_or_file("MCP_HOST_TOKEN")
MCP_HOST_PUBLIC_BASE_URL = os.getenv("MCP_HOST_PUBLIC_BASE_URL", "")
MCP_HOST_NETWORK = os.getenv("MCP_HOST_NETWORK", "legion-mcp")
MCP_HOST_PORT_RANGE = os.getenv("MCP_HOST_PORT_RANGE", "9100-9199")
MCP_HOST_CONTAINER_PREFIX = os.getenv("MCP_HOST_CONTAINER_PREFIX", "mcp-personal-")

# Managed stdio MCPs always use these operator-owned values. They are never
# accepted from a browser request.
MCP_HOST_RUNNER_IMAGE = os.getenv("MCP_HOST_RUNNER_IMAGE", "legion/mcp-node-runner:1.0.0")
MCP_HOST_RUNNER_PORT = _int("MCP_HOST_RUNNER_PORT", 8080)
MCP_HOST_RUNNER_NETWORK = os.getenv("MCP_HOST_RUNNER_NETWORK", MCP_HOST_NETWORK)
MCP_HOST_RUNNER_DATA_VOLUME_PREFIX = os.getenv("MCP_HOST_RUNNER_DATA_VOLUME_PREFIX", "mcp-data-")
MCP_HOST_RUNNER_NPM_REGISTRY = os.getenv("MCP_HOST_RUNNER_NPM_REGISTRY", "https://registry.npmjs.org/")
MCP_HOST_RUNNER_NPM_REGISTRY_HOSTS = os.getenv("MCP_HOST_RUNNER_NPM_REGISTRY_HOSTS", "registry.npmjs.org")
MCP_HOST_ALLOW_INSECURE_REGISTRY = os.getenv("MCP_HOST_ALLOW_INSECURE_REGISTRY", "0") == "1"
MCP_HOST_HUB_URL = os.getenv("MCP_HOST_HUB_URL", "http://hub:8000").rstrip("/")
MCP_HOST_SECRETS_TIMEOUT_SEC = float(os.getenv("MCP_HOST_SECRETS_TIMEOUT_SEC", "10"))
# These profiles are operator-owned policy names, never browser input.  The
# network layer should map them to separate egress policies in production.
MCP_HOST_INSTALL_EGRESS_PROFILE = os.getenv("MCP_HOST_INSTALL_EGRESS_PROFILE", "npm_registry_only")
MCP_HOST_RUNTIME_EGRESS_PROFILE = os.getenv("MCP_HOST_RUNTIME_EGRESS_PROFILE", "mcp_runtime_v1")
MCP_HOST_MAX_MEMORY_MB = _int("MCP_HOST_MAX_MEMORY_MB", 8192)
MCP_HOST_MAX_STORAGE_MB = _int("MCP_HOST_MAX_STORAGE_MB", 102400)
# A cold npx install may finish spawning before the package has completed its
# first parse/initialization. Keep the per-request timeout below the overall
# retry window so the host can retry without deleting a slow-but-healthy MCP.
MCP_HOST_PROBE_TIMEOUT_SEC = float(os.getenv("MCP_HOST_PROBE_TIMEOUT_SEC", "60"))
MCP_HOST_PROBE_REQUEST_TIMEOUT_SEC = float(os.getenv("MCP_HOST_PROBE_REQUEST_TIMEOUT_SEC", "30"))
MCP_HOST_MAX_LOG_BYTES = _int("MCP_HOST_MAX_LOG_BYTES", 8192)

# A Docker named volume is only a real quota when the configured driver
# enforces the size option. Production deployments must set this to enforced.
MCP_HOST_STORAGE_QUOTA_MODE = os.getenv("MCP_HOST_STORAGE_QUOTA_MODE", "best_effort")
MCP_HOST_VOLUME_DRIVER = os.getenv("MCP_HOST_VOLUME_DRIVER", "local")
MCP_HOST_VOLUME_SIZE_OPTION = os.getenv("MCP_HOST_VOLUME_SIZE_OPTION", "size")
