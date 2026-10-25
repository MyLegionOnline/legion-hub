import assert from "node:assert/strict";
import { once } from "node:events";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import http from "node:http";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { JsonRpcFramer, StdioHttpBridge } from "../src/bridge.js";
import { cleanupNpxInstallCache, resolveArtifact } from "../src/index.js";
import { parsePackageSpec, validateArgs, validateProfile } from "../src/security.js";

const fixture = fileURLToPath(new URL("./stdio-fixture.js", import.meta.url));

function post(port, body) {
  return new Promise((resolve, reject) => {
    const request = http.request({ port, path: "/mcp", method: "POST", headers: { "content-type": "application/json" } }, (response) => {
      let payload = "";
      response.on("data", (chunk) => { payload += chunk; });
      response.on("end", () => resolve({ status: response.statusCode, body: payload ? JSON.parse(payload) : null, headers: response.headers }));
    });
    request.on("error", reject);
    request.end(JSON.stringify(body));
  });
}

test("framer accepts newline and content-length JSON-RPC messages", () => {
  const messages = [];
  const framer = new JsonRpcFramer((message) => messages.push(message), () => assert.fail("invalid frame"));
  framer.push(Buffer.from('{"jsonrpc":"2.0","id":1}\n'));
  const payload = JSON.stringify({ jsonrpc: "2.0", id: 2 });
  framer.push(Buffer.from(`Content-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`));
  assert.deepEqual(messages.map((message) => message.id), [1, 2]);
});

test("cleans only the derived npx install cache", async () => {
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "mcp-runner-cache-"));
  const staleDir = path.join(dataDir, ".npm", "_npx", "partial-install");
  const tarballCache = path.join(dataDir, ".npm", "_cacache");
  await fs.mkdir(staleDir, { recursive: true });
  await fs.mkdir(tarballCache, { recursive: true });
  await fs.writeFile(path.join(staleDir, "package.json"), "{}");
  await fs.writeFile(path.join(tarballCache, "keep-me"), "cached tarball");

  cleanupNpxInstallCache(dataDir);

  await assert.rejects(fs.access(path.join(dataDir, ".npm", "_npx")));
  await assert.doesNotReject(fs.access(path.join(tarballCache, "keep-me")));
  await fs.rm(dataDir, { recursive: true, force: true });
});

test("managed bridge forwards initialize and tools/list over HTTP", async (t) => {
  const bridge = new StdioHttpBridge({
    command: process.execPath,
    args: [fixture],
    port: 0,
    timeoutMs: 2000,
    maxRestarts: 0,
  });
  await bridge.start();
  t.after(() => bridge.stop());
  const address = bridge.server.address();
  const initialize = await post(address.port, { jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
  assert.equal(initialize.status, 200);
  assert.equal(initialize.body.result.serverInfo.name, "fixture");
  const tools = await post(address.port, { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
  assert.equal(tools.body.result.tools[0].name, "fixture_tool");
  assert.ok(initialize.headers["mcp-session-id"]);
  const notification = await post(address.port, { jsonrpc: "2.0", method: "notifications/initialized" });
  assert.equal(notification.status, 202);
});

test("profile and args reject shell escape and non-registry sources", () => {
  assert.deepEqual(parsePackageSpec("depwire-cli@^1.2.0"), { name: "depwire-cli", requested: "^1.2.0" });
  assert.throws(() => parsePackageSpec("file:../../evil"), /allowed|invalid/i);
  assert.throws(() => validateArgs(["$(touch /tmp/pwned)"]), /metacharacters/i);
  assert.throws(() => validateProfile({ mode: "managed_stdio", runtime_profile: "node_npx_v1", transport: "stdio", package: "depwire-cli", command_args: [], shell: true }), /shell=false/i);
});

test("package registry outage has a stable safe error", async () => {
  const names = [
    "MCP_RUNNER_NPM_REGISTRY",
    "MCP_RUNNER_NPM_REGISTRY_HOSTS",
    "MCP_RUNNER_ALLOW_INSECURE_REGISTRY",
    "MCP_RUNNER_RESOLVE_TIMEOUT_MS",
    "NODE_ENV",
  ];
  const previous = Object.fromEntries(names.map((name) => [name, process.env[name]]));
  process.env.MCP_RUNNER_NPM_REGISTRY = "http://127.0.0.1:9/";
  process.env.MCP_RUNNER_NPM_REGISTRY_HOSTS = "127.0.0.1";
  process.env.MCP_RUNNER_ALLOW_INSECURE_REGISTRY = "1";
  process.env.MCP_RUNNER_RESOLVE_TIMEOUT_MS = "1000";
  process.env.NODE_ENV = "test";
  try {
    await assert.rejects(
      resolveArtifact({ package_name: "fixture-mcp", requested_version: null }),
      (error) => error.code === "package_registry_unavailable",
    );
  } finally {
    for (const name of names) {
      if (previous[name] === undefined) delete process.env[name];
      else process.env[name] = previous[name];
    }
  }
});

test("secret bootstrap injects only allowlisted env and redacts child logs", async (t) => {
  const bridge = new StdioHttpBridge({
    command: process.execPath,
    args: [fixture],
    port: 0,
    timeoutMs: 2000,
    maxRestarts: 0,
    allowedEnvNames: ["DEPWIRE_API_KEY"],
  });
  await bridge.start();
  t.after(() => bridge.stop());
  assert.equal(bridge.ready, false);
  bridge.bootstrap({ DEPWIRE_API_KEY: "bootstrap-secret" });
  assert.equal(bridge.ready, true);
  assert.equal(bridge.env.DEPWIRE_API_KEY, "bootstrap-secret");
  bridge.log("test", "received bootstrap-secret from child");
  assert.ok(bridge.logs.at(-1).message.includes("[REDACTED]"));
  assert.throws(() => bridge.bootstrap({ DEPWIRE_API_KEY: "again" }), /already completed/i);
});

test("crash-loop is surfaced by readiness and does not restart forever", async (t) => {
  const bridge = new StdioHttpBridge({
    command: process.execPath,
    args: ["-e", "process.exit(17)"],
    port: 0,
    maxRestarts: 0,
  });
  await bridge.start();
  t.after(() => bridge.stop());
  await once(bridge.child, "exit");
  const address = bridge.server.address();
  const response = await fetch(`http://127.0.0.1:${address.port}/readyz`);
  const body = await response.json();
  assert.equal(response.status, 503);
  assert.equal(body.error, "runtime_crash_loop");
});

test("stopping a runner cancels in-flight JSON-RPC requests", async () => {
  const bridge = new StdioHttpBridge({
    command: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    port: 0,
    timeoutMs: 5000,
  });
  await bridge.start();
  const pending = bridge.request({ jsonrpc: "2.0", method: "initialize", params: {} });
  const rejection = assert.rejects(pending, /stopping|exited/i);
  await bridge.stop();
  await rejection;
});
