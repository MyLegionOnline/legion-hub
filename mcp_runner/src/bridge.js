import http from "node:http";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { sanitizeLog, validateRuntimeEnv } from "./security.js";

const DEFAULT_TIMEOUT_MS = 15_000;

class JsonRpcFramer {
  constructor(onMessage, onInvalid) {
    this.buffer = Buffer.alloc(0);
    this.onMessage = onMessage;
    this.onInvalid = onInvalid;
  }

  push(chunk) {
    this.buffer = Buffer.concat([this.buffer, Buffer.from(chunk)]);
    while (this.buffer.length) {
      const headerEnd = this.buffer.indexOf(Buffer.from("\r\n\r\n"));
      if (headerEnd >= 0 && /^content-length\s*:/im.test(this.buffer.subarray(0, headerEnd).toString("utf8"))) {
        const header = this.buffer.subarray(0, headerEnd).toString("utf8");
        const match = header.match(/content-length\s*:\s*(\d+)/i);
        if (!match) return this.invalid("invalid content-length");
        const length = Number(match[1]);
        const start = headerEnd + 4;
        if (this.buffer.length < start + length) return;
        const payload = this.buffer.subarray(start, start + length).toString("utf8");
        this.buffer = this.buffer.subarray(start + length);
        this.emit(payload);
        continue;
      }
      const newline = this.buffer.indexOf(0x0a);
      if (newline < 0) return;
      const payload = this.buffer.subarray(0, newline).toString("utf8").replace(/\r$/, "");
      this.buffer = this.buffer.subarray(newline + 1);
      if (payload.trim()) this.emit(payload);
    }
  }

  emit(payload) {
    try {
      const message = JSON.parse(payload);
      if (!message || typeof message !== "object") throw new Error("not an object");
      this.onMessage(message);
    } catch {
      this.invalid("invalid JSON-RPC message");
    }
  }

  invalid(message) {
    this.onInvalid(message);
    this.buffer = Buffer.alloc(0);
  }
}

export class StdioHttpBridge {
  constructor({ command, args, env, port = 8080, host = "0.0.0.0", correlationId = randomUUID(), metadata = {},
    spawnProcess = spawn, timeoutMs = DEFAULT_TIMEOUT_MS, maxRestarts = 2, maxLogBytes = 8192,
    dataDir = process.env.MCP_RUNNER_DATA_DIR || "/data", allowedEnvNames = [], runtimeEnv = {}, beforeSpawn = null } = {}) {
    this.command = command;
    this.args = args || [];
    this.baseEnv = env || process.env;
    this.env = { ...this.baseEnv, ...runtimeEnv };
    this.allowedEnvNames = [...allowedEnvNames];
    this.bootstrapped = this.allowedEnvNames.length === 0;
    this.port = port;
    this.host = host;
    this.correlationId = correlationId;
    this.metadata = metadata && typeof metadata === "object" ? { ...metadata } : {};
    this.spawnProcess = spawnProcess;
    this.timeoutMs = timeoutMs;
    this.maxRestarts = maxRestarts;
    this.maxLogBytes = maxLogBytes;
    this.dataDir = dataDir;
    this.beforeSpawn = typeof beforeSpawn === "function" ? beforeSpawn : null;
    this.child = null;
    this.server = null;
    this.ready = false;
    this.stopping = false;
    this.restartCount = 0;
    this.crashLoop = false;
    this.nextRequestId = 1;
    this.pending = new Map();
    this.logs = [];
    this.sessionId = randomUUID();
    this.framer = new JsonRpcFramer((message) => this.onMessage(message), (message) => this.log("bridge", message));
  }

  async start() {
    this.server = http.createServer((request, response) => {
      void this.handleRequest(request, response);
    });
    await new Promise((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(this.port, this.host, resolve);
    });
    if (this.bootstrapped) this.spawnChild();
  }

  bootstrap(values) {
    if (this.bootstrapped) throw new Error("secret bootstrap already completed");
    const runtimeEnv = validateRuntimeEnv(values, this.allowedEnvNames);
    this.env = { ...this.baseEnv, ...runtimeEnv };
    this.bootstrapped = true;
    this.spawnChild();
  }

  spawnChild() {
    if (this.stopping) return;
    try {
      this.beforeSpawn?.();
    } catch (error) {
      this.log("bridge", `pre-spawn hook failed: ${error?.message || "unknown error"}`);
      return;
    }
    const child = this.spawnProcess(this.command, this.args, { env: this.env, shell: false, stdio: ["pipe", "pipe", "pipe"] });
    this.child = child;
    this.ready = true;
    child.stdout?.on("data", (chunk) => this.framer.push(chunk));
    child.stderr?.on("data", (chunk) => this.log("stderr", chunk));
    child.on("error", (error) => this.log("child", error?.message || "child error"));
    child.on("exit", (code, signal) => this.onChildExit(code, signal));
  }

  onChildExit(code, signal) {
    this.ready = false;
    this.child = null;
    for (const pending of this.pending.values()) pending.reject(new Error("MCP child process exited"));
    this.pending.clear();
    if (this.stopping) return;
    if (this.restartCount >= this.maxRestarts) {
      this.crashLoop = true;
      this.log("supervisor", "runtime_crash_loop");
      return;
    }
    this.restartCount += 1;
    this.log("supervisor", `child exited (${code ?? "unknown"}/${signal ?? "none"}); restart ${this.restartCount}/${this.maxRestarts}`);
    setTimeout(() => this.spawnChild(), Math.min(1000 * (2 ** (this.restartCount - 1)), 5000)).unref?.();
  }

  onMessage(message) {
    if (message.id === undefined || message.id === null) return;
    const key = String(message.id);
    const pending = this.pending.get(key);
    if (!pending) return;
    this.pending.delete(key);
    clearTimeout(pending.timer);
    if (Object.prototype.hasOwnProperty.call(message, "error")) {
      const error = new Error("MCP server returned a JSON-RPC error");
      error.rpc = { code: message.error?.code, message: String(message.error?.message || "error").slice(0, 256) };
      pending.reject(error);
    } else {
      pending.resolve(message);
    }
  }

  request(message, timeoutMs = this.timeoutMs) {
    if (!this.child?.stdin?.writable) return Promise.reject(new Error("MCP child process is not ready"));
    const id = message.id === undefined ? this.nextRequestId++ : message.id;
    const request = { ...message, id };
    const key = String(id);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(key);
        reject(new Error("MCP request timed out"));
      }, timeoutMs);
      this.pending.set(key, { resolve, reject, timer });
      try {
        this.child.stdin.write(`${JSON.stringify(request)}\n`);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(key);
        reject(error);
      }
    });
  }

  notify(message) {
    if (!this.child?.stdin?.writable) throw new Error("MCP child process is not ready");
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  async handleRequest(request, response) {
    response.setHeader("X-Correlation-ID", this.correlationId);
    if (request.method === "OPTIONS") {
      response.writeHead(204);
      response.end();
      return;
    }
    if (request.url === "/healthz" && request.method === "GET") {
      this.json(response, 200, { status: "ok", correlation_id: this.correlationId, child_alive: Boolean(this.child) });
      return;
    }
    if (request.url === "/readyz" && request.method === "GET") {
      const status = this.ready && this.child ? 200 : 503;
      this.json(response, status, {
        status: this.ready ? "ready" : "not_ready",
        error: this.crashLoop ? "runtime_crash_loop" : undefined,
        correlation_id: this.correlationId,
      });
      return;
    }
    if (request.url === "/metrics" && request.method === "GET") {
      this.json(response, 200, await this.metrics());
      return;
    }
    if (request.url === "/metadata" && request.method === "GET") {
      this.json(response, 200, { ...this.metadata, correlation_id: this.correlationId });
      return;
    }
    if (request.url === "/bootstrap" && request.method === "POST") {
      if (this.bootstrapped) {
        this.json(response, 409, { error: "secret_bootstrap_already_completed", correlation_id: this.correlationId });
        return;
      }
      const body = await this.readBody(request);
      if (body === null) {
        this.json(response, 413, { error: "request_too_large", correlation_id: this.correlationId });
        return;
      }
      try {
        const payload = JSON.parse(body);
        this.bootstrap(payload?.env || {});
        this.json(response, 200, { status: "bootstrapped", correlation_id: this.correlationId });
      } catch {
        this.json(response, 400, { error: "secret_bootstrap_invalid", correlation_id: this.correlationId });
      }
      return;
    }
    if (request.url === "/logs" && request.method === "GET") {
      this.json(response, 200, { correlation_id: this.correlationId, entries: this.logs.slice(-32) });
      return;
    }
    if (request.url !== "/mcp" || request.method !== "POST") {
      response.setHeader("Allow", "GET,POST,OPTIONS");
      this.json(response, 405, { error: "method_not_allowed", correlation_id: this.correlationId });
      return;
    }
    const body = await this.readBody(request);
    if (body === null) {
      this.json(response, 413, { error: "request_too_large", correlation_id: this.correlationId });
      return;
    }
    let message;
    try {
      message = JSON.parse(body);
    } catch {
      this.json(response, 400, { error: "invalid_json", correlation_id: this.correlationId });
      return;
    }
    if (!message || typeof message !== "object" || typeof message.method !== "string") {
      this.json(response, 400, { error: "invalid_json_rpc", correlation_id: this.correlationId });
      return;
    }
    if (message.id === undefined || message.id === null) {
      try {
        this.notify(message);
        response.writeHead(202, { "Mcp-Session-Id": this.sessionId });
        response.end();
      } catch {
        this.json(response, 502, { error: "mcp_upstream_unavailable", correlation_id: this.correlationId });
      }
      return;
    }
    try {
      const result = await this.request(message);
      response.setHeader("Mcp-Session-Id", this.sessionId);
      this.json(response, 200, result);
    } catch (error) {
      const status = /timed out/i.test(error.message) ? 504 : 502;
      this.json(response, status, { error: "mcp_upstream_unavailable", correlation_id: this.correlationId });
    }
  }

  readBody(request) {
    return new Promise((resolve) => {
      let size = 0;
      const chunks = [];
      request.on("data", (chunk) => {
        size += chunk.length;
        if (size <= 1024 * 1024) chunks.push(chunk);
        else resolve(null);
      });
      request.on("end", () => { if (size <= 1024 * 1024) resolve(Buffer.concat(chunks).toString("utf8")); });
      request.on("error", () => resolve(null));
    });
  }

  json(response, status, body) {
    const payload = JSON.stringify(body);
    response.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(payload) });
    response.end(payload);
  }

  log(source, message) {
    const value = sanitizeLog(message, Object.values(this.env).filter((entry) => typeof entry === "string"));
    this.logs.push({ source, message: value.slice(0, this.maxLogBytes), at: new Date().toISOString(), correlation_id: this.correlationId });
    if (this.logs.length > 64) this.logs.splice(0, this.logs.length - 64);
  }

  async metrics() {
    return {
      correlation_id: this.correlationId,
      memory_usage_bytes: process.memoryUsage().rss,
      storage_used_bytes: await directorySize(this.dataDir),
      restart_count: this.restartCount,
      crash_loop: this.crashLoop,
    };
  }

  async stop() {
    this.stopping = true;
    this.ready = false;
    for (const pending of this.pending.values()) pending.reject(new Error("MCP runner is stopping"));
    this.pending.clear();
    const child = this.child;
    if (child && !child.killed) {
      child.kill("SIGTERM");
      await new Promise((resolve) => {
        const timer = setTimeout(() => { try { child.kill("SIGKILL"); } catch {} resolve(); }, 5000);
        child.once("exit", () => { clearTimeout(timer); resolve(); });
      });
    }
    if (this.server) await new Promise((resolve) => this.server.close(() => resolve()));
  }
}

export { JsonRpcFramer };

async function directorySize(root) {
  let total = 0;
  const pending = [{ path: root, depth: 0 }];
  let entriesSeen = 0;
  while (pending.length && entriesSeen < 20_000) {
    const current = pending.pop();
    let entries;
    try { entries = await fs.readdir(current.path, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      if (++entriesSeen > 20_000) break;
      const entryPath = path.join(current.path, entry.name);
      try {
        const stat = await fs.lstat(entryPath);
        if (stat.isDirectory() && current.depth < 12) pending.push({ path: entryPath, depth: current.depth + 1 });
        else if (stat.isFile()) total += stat.size;
      } catch {}
    }
  }
  return total;
}
