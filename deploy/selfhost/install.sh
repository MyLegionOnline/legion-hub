#!/bin/sh
# Install Legion Hub on this machine (Linux, macOS, WSL): checks Docker, writes
# .env with fresh secrets, copies config/legion.toml, pulls the images and
# starts the stack.
#
#   ./install.sh [--domain hub.example.com] [--version v0.1.0] [--build] [--no-start]
#
#   --domain D   serve the Hub on https://D with an automatic certificate
#                (ports 80 and 443 must reach this machine, DNS must point here)
#   --version V  the release to run (default: latest)
#   --build      build the images from this checkout instead of pulling them
#   --no-start   write the configuration and stop
#
# Safe to run again: an existing .env and config/legion.toml are never
# overwritten, so your secrets and settings survive an update.
#
# Run from a download of this folder or, to fetch it first:
#   curl -fsSL https://github.com/MyLegionOnline/legion-hub/releases/latest/download/install.sh | sh
set -eu

REPO_URL="${LEGION_REPO_URL:-https://github.com/MyLegionOnline/legion-hub}"

DOMAIN=""
VERSION=""
BUILD=0
START=1
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="${2:?--domain needs a host name}"; shift 2 ;;
    --version) VERSION="${2:?--version needs a tag}"; shift 2 ;;
    --build) BUILD=1; shift ;;
    --no-start) START=0; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 1 ;;
  esac
done

log() { printf '\n==> %s\n' "$1"; }
die() { printf 'error: %s\n' "$1" >&2; exit 1; }

# ── Where the files are ────────────────────────────────────────────────────
# Next to this script when it is a real file; otherwise (piped from curl)
# download the bundle of the release into ~/legion.
SCRIPT_DIR=""
case "$0" in
  */install.sh|install.sh) SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" ;;
esac
if [ -z "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/docker-compose.yml" ]; then
  TARGET="${LEGION_DIR:-$HOME/legion}"
  log "Downloading the install bundle to $TARGET"
  command -v curl >/dev/null 2>&1 || die "curl is needed to download the bundle"
  command -v tar >/dev/null 2>&1 || die "tar is needed to unpack the bundle"
  mkdir -p "$TARGET"
  BUNDLE_VERSION="${VERSION:-latest}"
  if [ "$BUNDLE_VERSION" = "latest" ]; then
    BUNDLE_URL="$REPO_URL/releases/latest/download/legion-selfhost.tar.gz"
  else
    BUNDLE_URL="$REPO_URL/releases/download/$BUNDLE_VERSION/legion-selfhost.tar.gz"
  fi
  curl -fsSL "$BUNDLE_URL" | tar -xz -C "$TARGET"
  SCRIPT_DIR="$TARGET"
fi
cd "$SCRIPT_DIR"

# ── 1. Docker ──────────────────────────────────────────────────────────────
log "Checking Docker"
command -v docker >/dev/null 2>&1 \
  || die "Docker is not installed. Install Docker Engine (Linux) or Docker Desktop (macOS, Windows): https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 \
  || die "Docker is installed but not reachable. Start Docker Desktop, or add your user to the 'docker' group (then log in again)."
docker compose version >/dev/null 2>&1 \
  || die "Docker Compose v2 is missing ('docker compose'). Update Docker, or install the compose plugin."

ENGINE_MAJOR="$(docker version --format '{{.Server.Version}}' 2>/dev/null | cut -d. -f1 || true)"
case "$ENGINE_MAJOR" in
  ''|*[!0-9]*) echo "warning: could not read the Docker Engine version" ;;
  *) [ "$ENGINE_MAJOR" -ge 26 ] \
       || die "Docker Engine 26 or newer is required (found $ENGINE_MAJOR): room containers mount their folder of a shared volume by sub-path." ;;
esac
echo "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null) and Compose are ready."

# ── 2. .env with generated secrets ─────────────────────────────────────────
random_hex() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

# set_var NAME VALUE — replace NAME=… in .env, or add it.
set_var() {
  { grep -v "^$1=" .env || true; } > .env.tmp
  printf '%s=%s\n' "$1" "$2" >> .env.tmp
  mv .env.tmp .env
}

log "Checking .env"
if [ -f .env ]; then
  echo ".env already exists, leaving the secrets in it alone"
else
  cp .env.example .env
  chmod 600 .env 2>/dev/null || true
  for name in POSTGRES_PASSWORD SECRET_KEY WORKSPACED_TOKEN MCP_HOST_TOKEN MCP_GATEWAY_TOKEN; do
    set_var "$name" "$(random_hex)"
  done
  echo "Created .env with new secrets (it is private to you; keep a copy somewhere safe)."
fi
[ -z "$VERSION" ] || set_var LEGION_VERSION "$VERSION"
if [ -n "$DOMAIN" ]; then
  set_var HUB_DOMAIN "$DOMAIN"
  set_var HUB_PUBLIC_URL "https://$DOMAIN"
  set_var COMPOSE_PROFILES caddy
  echo "Configured https://$DOMAIN"
fi

# ── 3. config/legion.toml ──────────────────────────────────────────────────
log "Checking config/legion.toml"
mkdir -p config
if [ -f config/legion.toml ]; then
  echo "config/legion.toml already exists, leaving it alone"
else
  EXAMPLE=""
  for candidate in legion.example.toml ../../hub/legion.example.toml; do
    if [ -f "$candidate" ]; then EXAMPLE="$candidate"; break; fi
  done
  [ -n "$EXAMPLE" ] || die "legion.example.toml not found next to the installer"
  cp "$EXAMPLE" config/legion.toml
  echo "Created config/legion.toml (every key is commented out: the defaults apply)."
fi

if [ "$START" = "0" ]; then
  log "Configuration written. Start the Hub with:  docker compose up -d"
  exit 0
fi

# ── 4. Images and start ────────────────────────────────────────────────────
if [ "$BUILD" = "1" ]; then
  log "Building the images from this checkout (this takes a while)"
  docker compose build
else
  log "Pulling the images"
  docker compose pull || die "could not pull the images. Check the connection; for an unreleased checkout run again with --build."
fi

log "Starting Legion Hub"
docker compose up -d

PUBLIC_URL="$(grep '^HUB_PUBLIC_URL=' .env | cut -d= -f2-)"
log "Done"
echo "  Open ${PUBLIC_URL:-http://localhost:8080}"
echo "  The first account you register becomes the owner of this Hub."
echo ""
echo "Manage it from this folder:"
echo "  docker compose ps | logs -f backend | down | pull && docker compose up -d   (update)"
