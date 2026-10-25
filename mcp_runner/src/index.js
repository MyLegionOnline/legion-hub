import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { StdioHttpBridge } from "./bridge.js";
import { cacheKey, safeError, validateArtifact, validateProfile, validateRegistry } from "./security.js";

const execFileAsync = promisify(execFile);
const DATA_DIR = process.env.MCP_RUNNER_DATA_DIR || "/data";
const PORT = Number(process.env.MCP_RUNNER_PORT || 8080);

async function resolveArtifact(profile) {
  if (profile.resolved_artifact) return validateArtifact(profile.resolved_artifact, { required: true });
  const registry = process.env.MCP_RUNNER_NPM_REGISTRY || "https://registry.npmjs.org/";
  const allowHttp = process.env.MCP_RUNNER_ALLOW_INSECURE_REGISTRY === "1" && process.env.NODE_ENV !== "production";
  const registryUrl = new URL(validateRegistry(registry, ["registry.npmjs.org", ...(process.env.MCP_RUNNER_NPM_REGISTRY_HOSTS || "").split(",").filter(Boolean)], { allowHttp }));
  const spec = `${profile.package_name}${profile.requested_version ? `@${profile.requested_version}` : ""}`;
  let stdout;
  try {
    ({ stdout } = await execFileAsync("npm", ["view", spec, "version", "dist.integrity", "--json", "--registry", registry], {
      shell: false,
      timeout: Number(process.env.MCP_RUNNER_RESOLVE_TIMEOUT_MS || 20_000),
      maxBuffer: 128 * 1024,
      env: { ...process.env, NPM_CONFIG_REGISTRY: registry, NPM_CONFIG_AUDIT: "false", NPM_CONFIG_FUND: "false" },
    }));
  } catch (error) {
    throw Object.assign(new Error("npm package registry is unavailable"), { code: "package_registry_unavailable", cause: error });
  }
  let metadata;
  try { metadata = JSON.parse(stdout); } catch { throw Object.assign(new Error("npm metadata was invalid"), { code: "package_resolution_failed" }); }
  const artifact = validateArtifact({ version: metadata.version, integrity: metadata["dist.integrity"] }, { required: true });
  return { ...artifact, resolved_at: new Date().toISOString() };
}

async function writeCacheLock(profile, artifact) {
  const key = cacheKey(profile, artifact);
  const lockDir = path.join(DATA_DIR, ".mcp-cache", "locks");
  const lockPath = path.join(lockDir, `${key}.json`);
  await fs.mkdir(lockDir, { recursive: true, mode: 0o700 });
  try {
    const existing = JSON.parse(await fs.readFile(lockPath, "utf8"));
    if (existing.package === profile.package_name && existing.version === artifact.version && existing.integrity === artifact.integrity) return existing;
  } catch {}
  const record = {
    package: profile.package_name,
    version: artifact.version,
    integrity: artifact.integrity,
    resolved_at: artifact.resolved_at || new Date().toISOString(),
    cache_key: key,
  };
  const tempPath = `${lockPath}.${process.pid}.tmp`;
  await fs.writeFile(tempPath, `${JSON.stringify(record)}\n`, { mode: 0o600 });
  await fs.rename(tempPath, lockPath);
  return record;
}

export async function cleanupCache(keepKeys = []) {
  const lockDir = path.resolve(DATA_DIR, ".mcp-cache", "locks");
  const safeRoot = path.resolve(DATA_DIR, ".mcp-cache");
  if (!lockDir.startsWith(`${safeRoot}${path.sep}`)) throw new Error("unsafe cache path");
  const keep = new Set(keepKeys);
  let entries = [];
  try { entries = await fs.readdir(lockDir, { withFileTypes: true }); } catch { return 0; }
  let removed = 0;
  for (const entry of entries) {
    if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/.test(entry.name)) continue;
    const key = entry.name.slice(0, -5);
    if (keep.has(key)) continue;
    await fs.unlink(path.join(lockDir, entry.name));
    removed += 1;
  }
  return removed;
}

// npx stores extracted package installations separately from npm's tarball cache.
// A crashed install can leave a partial directory there; remove only that derived
// cache before each child spawn so a retry gets a clean extraction target.
export function cleanupNpxInstallCache(dataDir = DATA_DIR) {
  const safeRoot = path.resolve(dataDir);
  const npxDir = path.resolve(safeRoot, ".npm", "_npx");
  if (!npxDir.startsWith(`${safeRoot}${path.sep}`)) throw new Error("unsafe npx cache path");
  fsSync.rmSync(npxDir, { recursive: true, force: true });
}

async function main() {
  const rawProfile = JSON.parse(process.env.MCP_RUNNER_PROFILE_JSON || "{}");
  const profile = validateProfile(rawProfile);
  if (!["mcp_runtime_v1", "none"].includes(process.env.MCP_RUNNER_EGRESS_PROFILE || "mcp_runtime_v1")) {
    throw Object.assign(new Error("runtime egress profile is not approved"), { code: "egress_policy_invalid" });
  }
  const artifact = await resolveArtifact(profile);
  const lock = await writeCacheLock(profile, artifact);
  if (process.env.MCP_RUNNER_CLEANUP_CACHE === "1") await cleanupCache([lock.cache_key]);
  const packageSpec = `${profile.package_name}@${artifact.version}`;
  const commandArgs = ["--yes", packageSpec, ...profile.command_args];
  const env = {
    ...process.env,
    MCP_RUNNER_PORT: String(PORT),
    MCP_RUNNER_DATA_DIR: DATA_DIR,
    NPM_CONFIG_CACHE: path.join(DATA_DIR, ".npm"),
    NPM_CONFIG_REGISTRY: process.env.MCP_RUNNER_NPM_REGISTRY || "https://registry.npmjs.org/",
    NPM_CONFIG_AUDIT: "false",
    NPM_CONFIG_FUND: "false",
  };
  delete env.MCP_RUNNER_PROFILE_JSON;
  const bridge = new StdioHttpBridge({
    command: "npx",
    args: commandArgs,
    env,
    port: PORT,
    correlationId: profile.correlation_id,
    metadata: {
      package: profile.package_name,
      resolved_artifact: lock,
      runner_version: process.env.MCP_RUNNER_VERSION || null,
    },
    dataDir: DATA_DIR,
    maxRestarts: Number(process.env.MCP_RUNNER_MAX_RESTARTS || 2),
    maxLogBytes: Number(process.env.MCP_RUNNER_LOG_MAX_BYTES || 8192),
    allowedEnvNames: profile.env_names,
    beforeSpawn: () => cleanupNpxInstallCache(DATA_DIR),
  });
  process.once("SIGTERM", () => void bridge.stop().finally(() => process.exit(0)));
  process.once("SIGINT", () => void bridge.stop().finally(() => process.exit(0)));
  await bridge.start();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    const safe = safeError(error);
    process.stderr.write(`${JSON.stringify({ error: safe.code, message: safe.message })}\n`);
    process.exitCode = 1;
  });
}

export { main, resolveArtifact, writeCacheLock };
