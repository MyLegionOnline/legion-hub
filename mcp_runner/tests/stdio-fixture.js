import readline from "node:readline";

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  let request;
  try { request = JSON.parse(line); } catch { return; }
  if (request.method === "initialize") {
    process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: { protocolVersion: "2025-06-18", serverInfo: { name: "fixture", version: "1.0.0" }, capabilities: { tools: {} } } })}\n`);
  } else if (request.method === "tools/list") {
    process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: { tools: [{ name: "fixture_tool", description: "fixture", inputSchema: { type: "object" } }] } })}\n`);
  } else {
    process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: {} })}\n`);
  }
});
