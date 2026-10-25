# SearXNG with our own engines. SearXNG loads engine modules only from its own
# package directory, which the container (cap_drop: ALL) cannot write to at
# start and which a single-file bind mount would break on Docker Desktop
# restarts — so the engine is baked into the image instead.
#
# After changing mojeek_api.py: `docker compose up -d --build searxng`.
FROM searxng/searxng@sha256:76b0bf285aca014c7191fc4d9234c4bfb358624ac33d8883833d496c059ec072

COPY mojeek_api.py /usr/local/searxng/searx/engines/mojeek_api.py
