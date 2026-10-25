# Installing Legion Hub

Legion Hub runs as a handful of Docker containers on one machine: your laptop,
a home server or a rented VPS. This page gets you from nothing to a working Hub
and then covers the cases people ask about: your own domain, running at home
behind a router, updates and backups.

What leaves your machine, and what you owe the people you let in, is on one
page: [SELF_HOSTING_PRIVACY.md](SELF_HOSTING_PRIVACY.md). Read it before you
open registration to anyone.

## What you need

| | |
|---|---|
| **Docker** | Docker Engine 26 or newer with Compose v2 (`docker compose`). Docker Desktop on Windows and macOS. The installer checks this |
| **CPU** | amd64 (Intel/AMD) or arm64 (Apple Silicon, Raspberry Pi 5, Graviton). The release workflow builds native images for both; see [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md) for the current validation status |
| **Memory** | The idle Hub takes about 0.5 GB. Every running room is a container with up to 2 GB (change it in `legion.toml`). **4 GB for Docker is the minimum, 8 GB is comfortable** for a few rooms |
| **Disk** | About 10 GB for the images, plus your rooms' files |
| **Ports** | 8080 on the machine (change `LEGION_PORT`); 80 and 443 only if you use your own domain |

## Linux, macOS, WSL

```bash
curl -fsSL https://github.com/MyLegionOnline/legion-hub/releases/latest/download/install.sh | sh
```

The script downloads the install bundle into `~/legion` (set `LEGION_DIR` to put
it elsewhere), writes `.env` with new random secrets, copies
`config/legion.toml`, pulls the images and starts the Hub. Then open
<http://localhost:8080>.

To inspect before you run it, download the bundle (`legion-selfhost.tar.gz` on the
[releases page](https://github.com/MyLegionOnline/legion-hub/releases)), unpack it and run
`./install.sh` yourself. `./install.sh --help` lists the options.

**macOS.** Install Docker Desktop, then run the command above. Open Docker Desktop
→ Settings → Resources and give it at least 4 GB of memory (8 GB is better).
Once a release has passed its arm64 build and installation check, Apple Silicon
runs its native arm64 images without emulation. The current pre-release
validation status is recorded in [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md).

**Linux.** Install Docker Engine from <https://docs.docker.com/engine/install/>
and make your user a member of the `docker` group. Packages named `docker.io`
in some distributions are too old: check `docker version` shows 26 or higher.

**Rootless Docker, Colima, Podman.** The Hub's helpers create room containers
through the Docker socket. If your socket is not `/var/run/docker.sock`, set
`DOCKER_SOCKET` in `.env` to its path before you start.

## Windows

1. Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
   with the WSL 2 backend and start it. In Settings → Resources (or in
   `%UserProfile%\.wslconfig`) allow at least 4 GB of memory.
2. In PowerShell:

   ```powershell
   irm https://github.com/MyLegionOnline/legion-hub/releases/latest/download/install.ps1 | iex
   ```

   or download `legion-selfhost.zip` from the releases page, unpack it and run
   `.\install.ps1` in that folder. If PowerShell refuses to run the script,
   allow it for this session with `Set-ExecutionPolicy -Scope Process Bypass`.
3. Open <http://localhost:8080>.

Keep the install folder somewhere in your user profile; it holds `.env`, which
has the Hub's secrets.

## First start

1. Open the address the installer printed and **register**. The first account
   becomes the **owner**; until you change it, nobody else can register
   (`closed`).
2. In Settings add a model provider (an API key from Anthropic, OpenAI or
   OpenRouter). You can instead put a key in `.env` as a fallback for everybody.
3. Create an agent, then a room, and send the first message. The first room
   start takes a few seconds while its container comes up.

**If other people will use your Hub.** Once you open registration (`invite` or
`open`) you are the operator of the Hub and the controller of its users' data. We
cannot see your Hub, we have no agreement with those users, and our Terms of
Service and Privacy Policy do not cover them. The Hub shows none of our documents
on its sign-in and registration pages: write your own and show them where people
register. What the Hub sends us when you link a Legion account, and what you owe
your users, is in [SELF_HOSTING_PRIVACY.md](SELF_HOSTING_PRIVACY.md). If you link
a Legion account or use Market tools, the Market's
[Terms of Service](https://market.mylegion.online/terms) and
[Privacy Policy](https://market.mylegion.online/privacy) apply to that account.

To let other people in, set the registration mode in `config/legion.toml`
(`invite` is the safest) and restart the backend. See
[CONFIGURATION.md](CONFIGURATION.md#who-may-register). Everyone you let in gets
a shell inside containers on this machine.

## Your own domain with HTTPS

On a server with a public address:

1. Point a DNS `A`/`AAAA` record (for example `hub.example.com`) at the machine
   and open ports 80 and 443.
2. Run the installer with the domain:

   ```bash
   ./install.sh --domain hub.example.com
   ```

   This writes `HUB_DOMAIN`, `HUB_PUBLIC_URL=https://hub.example.com` and
   `COMPOSE_PROFILES=caddy` to `.env` and starts [Caddy](https://caddyserver.com/),
   which gets and renews a Let's Encrypt certificate by itself. On an existing
   install, put those three lines in `.env` yourself and run
   `docker compose up -d`.

`HUB_PUBLIC_URL` must be exactly the address people type, including `https://`:
sign-in redirects and links are built from it.

## At home, behind a router

A home machine has no public address, so the Hub has to be reached some other
way. Two options that need no open port:

**Tailscale** (private; only your own devices):

```bash
tailscale serve --bg 8080
```

Tailscale prints an `https://<machine>.<tailnet>.ts.net` address. Put it in
`.env` as `HUB_PUBLIC_URL`, then `docker compose up -d`. Keep `LEGION_BIND=127.0.0.1`.

**Cloudflare Tunnel** (public address, Cloudflare in front):

```bash
cloudflared tunnel --url http://localhost:8080
```

For a permanent address create a named tunnel in the Cloudflare dashboard that
sends your hostname to `http://localhost:8080`, and set `HUB_PUBLIC_URL` to
`https://<your hostname>`. A public address means anyone can reach the sign-in
page: keep registration `closed` or `invite`.

**Only inside your home network:** set `LEGION_BIND=0.0.0.0` and
`HUB_PUBLIC_URL=http://<this machine's address>:8080`. Traffic is not
encrypted, so use this on a network you trust.

## Updating

```bash
cd ~/legion          # the install folder
docker compose pull
docker compose up -d
```

To move to a specific release, set `LEGION_VERSION=v0.1.0` in `.env` first.
Your data lives in Docker volumes and survives updates. Read the release notes
before a major step; database changes are applied by the Hub when it starts.

## Backups

Three things hold your data. Back all of them up, and keep `.env` and the keyring
volume somewhere other than the machine itself.

| What | Why it matters | How |
|---|---|---|
| The database | Users, rooms, agents, tasks, chat | `docker compose exec -T db pg_dump -U legion legion \| gzip > legion-db.sql.gz` |
| The keyring volume `legion_legion_secrets` | **Encrypts the secrets the Hub stores** (provider keys, room secrets). Without it those secrets cannot be read, even from a restored database | `docker run --rm -v legion_legion_secrets:/s -v "$PWD":/b alpine tar czf /b/legion-secrets.tgz -C /s .` |
| The rooms' files, volume `legion_workspace_data` | The files and git history of every room | `docker run --rm -v legion_workspace_data:/s -v "$PWD":/b alpine tar czf /b/legion-rooms.tgz -C /s .` |
| `.env` | The Hub's passwords and session key | copy the file |

The volume names above are for the default install folder; `docker volume ls`
shows yours. Restoring is the same commands the other way round on a fresh
install, before you first start the Hub (`psql` for the dump, `tar xzf` into the
volume).

## Stopping and removing

```bash
docker compose down            # stop; data stays
docker compose down -v         # stop AND delete the database, keyring and rooms' files
```

Room containers (`ws-<room id>`) and personal MCP containers (`mcp-*`) are not
part of the Compose project, so `docker compose down` does not remove them.
List them with `docker ps -a --filter name=ws-` and remove them with `docker rm -f`.

## Known limits of this release

- **Previews of web apps running in a room** (the Browser tab, where you start a
  room's app and open it through the Hub) are off. Each app is served on its own
  address, `svc-<id>.<your domain>`, so it needs wildcard DNS and a wildcard
  certificate, which this stack does not set up. Agents still run dev servers
  and check them from inside the room; you just cannot open them in the
  Browser tab. How the feature works, and what a self-hoster would have to add:
  [hub/BROWSER_SERVICES_REFERENCE.md](../hub/BROWSER_SERVICES_REFERENCE.md), section 7.
- **The Market's SSH tool for servers and its image-understanding tool** are
  not part of the Hub. An agent sees images through its own model when that
  model accepts image input, and you can connect your own MCP servers for
  anything else. To use the Market's tools from a self-hosted Hub, link a Legion
  account (see [SEARCH.md](SEARCH.md) for how Market tools are paid for).
- The Hub image is large (about 3 GB) because it contains a headless browser for
  screenshots.

## Troubleshooting

**`docker compose pull` says "denied" or "not found".** The release you asked
for does not exist or its images are not public yet. Check `LEGION_VERSION` in
`.env`; `latest` always works once a release exists.

**The backend keeps restarting.** `docker compose logs backend`. The Hub refuses
to start on an invalid `legion.toml` (the message names the key) or a mistyped
`LEGION_REGISTRATION`; fix it and `docker compose up -d`.

**"Room failed to start" or rooms never get a container.**
`docker compose logs workspaced`. The usual causes: the Docker socket path
(`DOCKER_SOCKET`), Docker older than 26, or Docker out of memory (raise it in
Docker Desktop). `docker ps -a --filter name=ws-` shows the room containers.

**Sign-in loops or links point to the wrong address.** `HUB_PUBLIC_URL` in `.env`
is not the address in your browser's bar. Fix it and `docker compose up -d`.

**I lost the owner's password.** A self-hosted Hub sends no reset emails; set a
new password from the server instead, in the folder with `docker-compose.yml`:

```bash
docker compose exec backend python -m app.reset_password owner@example.com
```

It asks for the new password twice (at least 8 characters). This works for any
user of the Hub, not only the owner. Sessions already signed in stay signed in.
