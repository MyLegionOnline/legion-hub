FROM node:24-bookworm-slim AS package
WORKDIR /fixture
COPY fixture-mcp/package.json fixture-mcp/server.js ./
RUN mkdir -p /out && npm pack --pack-destination /out

FROM node:24-bookworm-slim
WORKDIR /app
COPY --from=package /out/fixture-mcp-1.0.0.tgz /srv/fixture-mcp-1.0.0.tgz
COPY registry/registry.js ./registry.js
EXPOSE 8081
CMD ["node", "/app/registry.js"]
