# Web search for your agents

A self-hosted Hub ships without a search engine, because every good one needs
either a key from its provider or a server that is not blocked by it. There are
four ways to give agents search; pick the one that suits you. All of them
work with a Hub that never contacts Legion except the last.

## 1. The built-in `web_search` tool (simplest)

Get an API key from [Brave Search](https://brave.com/search/api/) or
[Tavily](https://tavily.com/) and put it in `.env`:

```
WEB_SEARCH_PROVIDER=brave        # or tavily
WEB_SEARCH_API_KEY=<your key>
```

Then `docker compose up -d`. Agents get a `web_search` tool (title, URL and
snippet per result) that works together with `web_fetch` to read a page. Without
a key the tool does not exist. You pay the provider directly; check its terms for
your use.

## 2. Your own MCP server

Any MCP search server works: a [SearXNG](https://github.com/searxng/searxng)
you run yourself with an MCP adapter, or one of the many search MCP servers
published as Docker images. Install it for the owner from `config/legion.toml`:

```toml
[[personal_mcp]]
name = "search"
image = "ghcr.io/example/search-mcp:1.0"   # an MCP server that speaks HTTP
port = 3000
```

or by link if it runs elsewhere (HTTPS only):

```toml
[[personal_mcp]]
name = "search"
url = "https://search.example.com/mcp"
```

Restart the backend (`docker compose restart backend`). Credentials do not go in
the file: add them in Settings → Secrets and bind them to the server. Public
SearXNG instances and datacentre addresses are often blocked by the search
engines behind them, so a search server on a home connection usually works
better than one on a VPS.

## 3. An external MCP server with your own key

Some search providers offer a hosted MCP endpoint (Tavily, for example). Add it
by link in the Hub's MCP settings, with your key stored in Settings → Secrets. The
Hub's MCP gateway makes the call; it accepts HTTPS addresses only and refuses
private ones.

## 4. Search from the Legion Market, paid with credits

If you would rather not manage a provider account, link a Legion account
(`MARKET_ENABLED=true` in `.env`, then Settings → Profile → Legion account) and use the
search tool from the Market. Calls go through Legion's public gateway and are paid
from the account's credits. During the beta credits are granted by hand: ask at
<legal@mylegion.online>. This is the only option that sends anything to Legion;
[SELF_HOSTING_PRIVACY.md](SELF_HOSTING_PRIVACY.md) says exactly what, including that
the **search query itself** reaches us and the third-party service behind the tool.
