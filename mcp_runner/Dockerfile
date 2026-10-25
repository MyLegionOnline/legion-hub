FROM node:24-bookworm-slim

ENV NODE_ENV=production \
    HOME=/tmp/legion-runner \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

WORKDIR /app
COPY package.json ./
COPY src ./src

RUN groupadd --system --gid 10001 legion-runner \
    && useradd --system --uid 10001 --gid 10001 --home-dir /tmp/legion-runner legion-runner \
    && mkdir -p /data /tmp/legion-runner \
    && chown -R 10001:10001 /app /data /tmp/legion-runner

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["node", "/app/src/index.js"]
