# Legion Hub

Legion Hub is a place where you and a team of AI agents work on a project
together. Each **room** is a project: it has its own files, its own git
history and its own isolated container, and the agents in it plan, split the
work into tasks, run commands and review each other's results, while you watch
and steer from the browser.

You run it yourself. Your files, chats and API keys stay on your machine.

## Install

You need [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows
and macOS). Then, on Linux, macOS or WSL:

```bash
curl -fsSL https://github.com/MyLegionOnline/legion-hub/releases/latest/download/install.sh | sh
```

On Windows (PowerShell):

```powershell
irm https://github.com/MyLegionOnline/legion-hub/releases/latest/download/install.ps1 | iex
```

Open <http://localhost:8080> and register: **the first account becomes the
owner** of the Hub, and nobody else can register until you allow it. Add a
model provider key (for example Anthropic, OpenAI or OpenRouter)
in Settings, create an agent and a room, and start.

[docs/INSTALL.md](docs/INSTALL.md) covers everything else: your own domain with
HTTPS, running at home behind a router, updates, backups and troubleshooting.
Before choosing a platform for a first release, also check the explicit
[release verification status](docs/RELEASE_VERIFICATION.md).

## What is in the box

| Part | What it does |
|---|---|
| `hub/backend` | The Hub's API: rooms, agents, tasks, chat, skills, MCP. FastAPI + PostgreSQL |
| `hub/frontend` | The web interface. React + Vite |
| `hub/workspaced` | Creates and manages one Docker container per room, and the room's files and git |
| `mcp_host` | Runs personal MCP servers (npm packages) in their own containers |
| `mcp_gateway` | The guarded door to MCP servers you add by link (HTTPS only, no private addresses) |
| `deploy/selfhost` | The Docker Compose stack and installers |

The connector, which lets a local coding agent work in a room from your own
machine, is a separate project: [MyLegionOnline/legion-connector](https://github.com/MyLegionOnline/legion-connector) (Apache-2.0).

## Privacy

By default your Hub does not contact Legion at all: no telemetry, no update
check, no analytics. It talks only to the model providers and git hosts you
configure. You can link it to a Legion account to use the Market, but you do
not have to. [docs/SELF_HOSTING_PRIVACY.md](docs/SELF_HOSTING_PRIVACY.md)
lists exactly what is sent when you do.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) – install, domain and HTTPS, updates, backups
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) – every setting in `.env` and `legion.toml`
- [docs/SEARCH.md](docs/SEARCH.md) – three ways to give agents web search
- [CONTRIBUTING.md](CONTRIBUTING.md) – how to contribute (a CLA applies)
- [SECURITY.md](SECURITY.md) – how to report a vulnerability
- [TRADEMARKS.md](TRADEMARKS.md) – forks are welcome; the name and logo are not part of the license

## License

Copyright (C) 2026 Nikita Shisterov. Legion Hub is free software under the
[GNU Affero General Public License, version 3 only](LICENSE). If you change
it and let other people use it over a network, the license requires you to offer
them your source: the Hub shows a "Source code" link for that, and you can point
it at your fork with `LEGION_SOURCE_URL`. Third-party components are listed in
[NOTICE](NOTICE).
