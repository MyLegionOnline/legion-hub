#!/usr/bin/env node

import readline from "node:readline";

const tools = [{
  name: "fixture_room_echo",
  description: "Returns the room identity received through MCP metadata.",
  inputSchema: { type: "object", properties: { message: { type: "string" } } },
}];

const output = (id, result) => process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`);
const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

input.on("line", (line) => {
  let request;
  try { request = JSON.parse(line); } catch { return; }
  if (request.method === "initialize") {
    output(request.id, {
      protocolVersion: "2025-06-18",
      serverInfo: { name: "legion-fixture", version: "1.0.0" },
      capabilities: { tools: {} },
    });
    return;
  }
  if (request.method === "tools/list") {
    output(request.id, { tools });
    return;
  }
  if (request.method === "tools/call") {
    const params = request.params || {};
    const meta = params._meta || {};
    const room = String(meta.room_id || "missing-room");
    const message = String((params.arguments || {}).message || "fixture-ok");
    output(request.id, {
      content: [{ type: "text", text: `room=${room}; message=${message}` }],
    });
    return;
  }
  output(request.id, {});
});
