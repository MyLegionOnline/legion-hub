import crypto from "node:crypto";

const PACKAGE_NAME = /^(?:@[a-z0-9][a-z0-9._~-]*\/)?[a-z0-9][a-z0-9._~-]*$/;
const EXACT_VERSION = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const INTEGRITY = /^sha512-[A-Za-z0-9+/=]{20,}$/;
const ENV_NAME = /^[A-Z_][A-Z0-9_]{0,127}$/;
const MAX_ARGS = 64;
const MAX_ARG_LENGTH = 1024;

export class ProfileError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "ProfileError";
    this.code = code;
  }
}

function fail(code, message) {
  throw new ProfileError(code, message);
}

export function parsePackageSpec(value) {
  if (typeof value !== "string" || value.length === 0 || value.length > 214) {
    fail("package_spec_invalid", "The npm package specification is invalid.");
  }
  if (/[:\\/]|\s|[;&|`$()<>{}\[\]]/.test(value) && !value.startsWith("@")) {
    fail("package_spec_invalid", "Only registry package names and semver ranges are allowed.");
  }
  if (value.includes("://") || /^(?:git|file|http|https|workspace|npm):/i.test(value)) {
    fail("package_source_not_allowed", "Git, URL, file and workspace package sources are not allowed.");
  }

  let name = value;
  let requested = null;
  if (value.startsWith("@")) {
    const slash = value.indexOf("/");
    const separator = slash < 0 ? -1 : value.indexOf("@", slash);
    if (separator >= 0) {
      name = value.slice(0, separator);
      requested = value.slice(separator + 1);
    }
  } else {
    const separator = value.indexOf("@");
    if (separator >= 0) {
      name = value.slice(0, separator);
      requested = value.slice(separator + 1);
    }
  }
  if (!PACKAGE_NAME.test(name)) {
    fail("package_spec_invalid", "Only public npm registry package names are allowed.");
  }
  if (requested !== null && (requested.length === 0 || requested.length > 120 || !/^[0-9A-Za-z.*+^~<>=|\- ]+$/.test(requested))) {
    fail("package_spec_invalid", "The npm version or range is invalid.");
  }
  return { name, requested };
}

export function validateArgs(args) {
  if (!Array.isArray(args) || args.length > MAX_ARGS) {
    fail("command_args_invalid", "Too many MCP command arguments.");
  }
  return args.map((arg) => {
    if (typeof arg !== "string" || arg.length > MAX_ARG_LENGTH || /[\u0000-\u001f\u007f]/.test(arg)) {
      fail("command_args_invalid", "MCP command arguments must be bounded strings.");
    }
    if (/[;&|`$<>]/.test(arg)) {
      fail("command_args_invalid", "Shell metacharacters are not allowed in MCP arguments.");
    }
    return arg;
  });
}

export function validateEnvNames(names) {
  if (!Array.isArray(names) || names.length > 64) {
    fail("env_names_invalid", "Too many environment variable names.");
  }
  const result = [];
  for (const name of names) {
    if (typeof name !== "string" || !ENV_NAME.test(name) || name === "PATH" || name === "NODE_OPTIONS") {
      fail("env_names_invalid", "Only safe, explicitly bound environment names are allowed.");
    }
    if (!result.includes(name)) result.push(name);
  }
  return result;
}

export function validateRuntimeEnv(values, allowedNames = []) {
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    fail("secret_env_invalid", "The secret environment mapping is invalid.");
  }
  const allowed = new Set(validateEnvNames(allowedNames));
  const result = {};
  for (const [name, value] of Object.entries(values)) {
    if (!allowed.has(name) || typeof value !== "string" || value.length === 0 || value.length > 64 * 1024) {
      fail("secret_env_invalid", "The secret environment mapping is invalid.");
    }
    if (/[\u0000]/.test(value)) fail("secret_env_invalid", "The secret environment mapping is invalid.");
    result[name] = value;
  }
  return result;
}

export function validateArtifact(artifact, { required = false } = {}) {
  if (artifact === null || artifact === undefined) {
    if (required) fail("package_not_pinned", "The package must be pinned before the runner starts.");
    return null;
  }
  if (typeof artifact !== "object" || Array.isArray(artifact)) {
    fail("package_pin_invalid", "The resolved package metadata is invalid.");
  }
  if (!EXACT_VERSION.test(artifact.version || "") || !INTEGRITY.test(artifact.integrity || "")) {
    fail("package_pin_invalid", "The runner requires an exact npm version and sha512 integrity.");
  }
  return {
    version: artifact.version,
    integrity: artifact.integrity,
    resolved_at: typeof artifact.resolved_at === "string" ? artifact.resolved_at : null,
  };
}

export function validateRegistry(value, allowedHosts = ["registry.npmjs.org"], { allowHttp = false } = {}) {
  let parsed;
  try { parsed = new URL(value); } catch { fail("registry_not_allowed", "The npm registry is not approved."); }
  const hosts = new Set(allowedHosts.filter((host) => typeof host === "string").map((host) => host.toLowerCase()));
  const hostname = typeof parsed.hostname === "string" ? parsed.hostname.toLowerCase() : "";
  const validProtocol = parsed.protocol === "https:" || (allowHttp && parsed.protocol === "http:");
  if (!validProtocol || parsed.username || parsed.password || !hosts.has(hostname) || (parsed.protocol === "https:" && !["", "443"].includes(parsed.port))) {
    fail("registry_not_allowed", "The npm registry is not approved.");
  }
  return parsed.toString();
}

export function validateProfile(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    fail("launch_profile_invalid", "The managed MCP launch profile is invalid.");
  }
  if (input.mode !== "managed_stdio" || input.runtime_profile !== "node_npx_v1" || input.transport !== "stdio") {
    fail("runtime_profile_unsupported", "Only the node_npx_v1 managed stdio profile is supported.");
  }
  const { name, requested } = parsePackageSpec(input.package);
  const commandArgs = validateArgs(input.command_args || []);
  const envNames = validateEnvNames(input.env_names || []);
  const artifact = validateArtifact(input.resolved_artifact, { required: false });
  if (input.shell !== false) {
    fail("shell_execution_not_allowed", "Managed MCP commands always run with shell=false.");
  }
  return {
    mode: "managed_stdio",
    runtime_profile: "node_npx_v1",
    transport: "stdio",
    package: input.package,
    package_name: name,
    requested_version: requested,
    command_args: commandArgs,
    env_names: envNames,
    shell: false,
    resolved_artifact: artifact,
    correlation_id: typeof input.correlation_id === "string" && /^[A-Za-z0-9._-]{8,128}$/.test(input.correlation_id)
      ? input.correlation_id
      : crypto.randomUUID(),
  };
}

export function cacheKey(profile, artifact) {
  return crypto.createHash("sha256")
    .update(`${profile.package_name}\0${artifact.version}\0${artifact.integrity}`)
    .digest("hex");
}

export function safeError(error, fallback = "Managed MCP runner failed.") {
  if (error instanceof ProfileError) return { code: error.code, message: error.message };
  const code = typeof error?.code === "string" && /^[a-z0-9_]{3,64}$/.test(error.code)
    ? error.code
    : "runner_error";
  return { code, message: fallback };
}

export function sanitizeLog(value, redactions = []) {
  let output = String(value ?? "").replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ");
  for (const secret of redactions) {
    if (typeof secret === "string" && secret.length > 0) output = output.split(secret).join("[REDACTED]");
  }
  return output.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "").slice(0, 8192);
}
