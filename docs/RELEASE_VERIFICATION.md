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

## Still required before the first public release

- Run the documented installer from a clean Ubuntu host and complete a
  start/login/room smoke test. No clean Ubuntu host was available for the
  2026-10-15 check.
- Run the documented Windows PowerShell installer from a freshly unpacked
  release bundle. The local Windows check above used `docker compose build`
  from the source checkout; it did not download a published release bundle.
- Run the arm64 GitHub Actions build and inspect the resulting multi-platform
  manifests. This must use the native `ubuntu-24.04-arm` job defined in
  `.github/workflows/_build-image.yml`, not an amd64 emulation result.
- Install the resulting release on Apple Silicon when a Mac is available.
  Apple Silicon is **not yet live-validated** as of this record.
- Run a connector pairing smoke test against the release Hub; it depends on
  the public connector release artifact and remains a P1 release check.

## arm64 source review

The release workflow builds each image separately on an amd64 or arm64 runner
and combines their digests into a manifest only after both succeed. The reviewed
Dockerfiles contain no x86-only `--platform`, architecture test or binary
download. Their base images are the official multi-architecture Python, Node,
Debian, Nginx, PostgreSQL, Redis and Caddy images. This is a review result, not
a substitute for the required native build above.
