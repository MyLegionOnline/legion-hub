import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";

const packageName = "fixture-mcp";
const packageVersion = "1.0.0";
const tarballPath = "/srv/fixture-mcp-1.0.0.tgz";
const tarball = fs.readFileSync(tarballPath);
const integrity = `sha512-${crypto.createHash("sha512").update(tarball).digest("base64")}`;

const metadata = {
  name: packageName,
  "dist-tags": { latest: packageVersion },
  versions: {
    [packageVersion]: {
      name: packageName,
      version: packageVersion,
      bin: { "fixture-mcp": "server.js" },
      dist: {
        integrity,
        tarball: "http://fixture-registry:8081/fixture-mcp/-/fixture-mcp-1.0.0.tgz",
      },
    },
  },
};

http.createServer((request, response) => {
  if (request.url === "/fixture-mcp") {
    const body = JSON.stringify(metadata);
    response.writeHead(200, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
    response.end(body);
    return;
  }
  if (request.url === "/fixture-mcp/-/fixture-mcp-1.0.0.tgz") {
    response.writeHead(200, { "content-type": "application/octet-stream", "content-length": tarball.length });
    response.end(tarball);
    return;
  }
  response.writeHead(404);
  response.end("not found");
}).listen(8081, "0.0.0.0");
