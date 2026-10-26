# Release verification status

This is the release-maintainer record for the checks that cannot be inferred
from source review. Update it for the release candidate being tagged; do not
present an unchecked platform as supported.

## Checked locally on 2026-10-15

- Windows 11 with Docker Desktop's Linux engine 29.5.3 (amd64, 7.61 GiB Docker
  memory): `docker compose --env-file .env.example config --quiet` passed, both
  with and without the Caddy profile.
- The local self-hosted build passed for all stack images: backend, frontend,
  workspaced, workspace base and office, mcp-host, mcp-gateway and
  mcp-node-runner.
- The installers require Docker Engine 26 or newer, as the Compose stack uses
  Docker volume subpaths for room data. Docker Desktop satisfies that
  requirement in the checked environment.
- The `MARKET_ENABLED=false` default is present in the self-hosted Compose
  stack. The backend has an explicit startup guard for that switch; release
  testing must still run `hub/backend/tests/test_market_switch.py` in the
  release environment.

## Export checks on 2026-10-16

- Fresh file-only Hub and Connector exports passed their built-in public-tree
  checks. gitleaks v8.21.2 found no leaks in either exported tree. The Hub
  export contains no private Git history.
- On the Hub export, Docker Compose configuration passed with and without the
  Caddy profile. The exported install and no-plans scripts passed shell syntax
  checks. All exported workflow YAML files parsed successfully.
- On Windows with Node.js 22, Hub frontend lint and build passed.
- Tests mounted from the Hub export into local Python service images passed:
  `test_market_switch.py` 8 (Python 3.11), workspaced 260, mcp-gateway 107
  and mcp-host 7. The gateway test needs the whole exported tree because its
  SearXNG fixture is under `mcp_gateway/deployment/`.
- The Connector export built and passed 47 tests with one symlink test skipped
  on Windows (Node.js 22); its production dependency license check passed.
  A clean Linux Node.js 20 build and test run passed all 48 tests.

These checks do not replace the full Hub backend CI suite, a native arm64
build, or installation from a published release bundle.

## Checked on published release candidates (2026-10-26)

- `v0.1.0-rc.1`: the Windows PowerShell installer from a freshly unpacked
  `legion-selfhost.zip` on Windows 11 with Docker Desktop: all services
  healthy, sign-up and sign-in, a room started from the release workspace
  image.
- `v0.1.0-rc.1` to `rc.3`: `install.sh` from the release page on a clean
  Ubuntu 24.04 amd64 host (Docker Engine 29): all services healthy, the first
  account became the owner, six starter agents, a room with Node.js 24.
- `v0.1.0-rc.3` with `--domain` on the same host: Caddy obtained a Let's
  Encrypt certificate, HTTP redirects to HTTPS, sign-in and a room chat work
  over HTTPS.
- The release workflow built every image natively on amd64 and on
  `ubuntu-24.04-arm`; the published manifests of all eight images list
  `linux/amd64` and `linux/arm64`.
- Connector pairing from a fresh clone of `legion-connector` against the
  release Hub, under the dedicated user its installer creates; a direct chat
  with the external agent was answered from the connector's machine.
- With `MARKET_ENABLED=false`, no container opened a connection outside the
  Docker networks during two minutes after a backend restart.

The `v0.1.0` tag carries fixes made after `rc.3` (the installer's `.env`
permissions, the agent card after pairing, the connector's dedicated user);
each was checked on the same Ubuntu host before tagging.

## Not yet checked

- Apple Silicon. The arm64 images are built natively, but no installation on a
  Mac has been checked. Treat it as unverified.

## arm64 source review

The release workflow builds each image separately on an amd64 or arm64 runner
and combines their digests into a manifest only after both succeed. The reviewed
Dockerfiles contain no x86-only `--platform`, architecture test or binary
download. Their base images are the official multi-architecture Python, Node,
Debian, Nginx, PostgreSQL, Redis and Caddy images. This is a review result, not
a substitute for the required native build above.
