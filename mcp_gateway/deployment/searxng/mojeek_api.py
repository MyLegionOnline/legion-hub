# SPDX-License-Identifier: AGPL-3.0-only
"""Mojeek Search API (general web), keyed and paid per query.

SearXNG's bundled ``mojeek`` engine scrapes the HTML site from our datacenter
IP, which Mojeek's terms (3.5 d/e) reserve for authorised API users. This
engine talks to the official API instead:
https://www.mojeek.com/support/api/search/request_parameters.html

The key is read from the ``MOJEEK_API_KEY`` environment variable — SearXNG's
``settings.yml`` has no env interpolation and that file is committed. Without a
key :func:`setup` returns ``False`` and SearXNG drops the engine, so the rest of
the metasearch keeps working.

One page is one billed request, hence the low ``max_page``.

This file is copied into ``searx/engines/`` at container start (see the
``searxng`` service in the compose files).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from searx.exceptions import (
    SearxEngineAPIException,
    SearxEngineAccessDeniedException,
    SearxEngineTooManyRequestsException,
)

log = logging.getLogger("searx.engines.mojeek_api")

about = {
    "website": "https://www.mojeek.com",
    "wikidata_id": "Q60747299",
    "official_api_documentation": "https://www.mojeek.com/support/api/search/request_parameters.html",
    "use_official_api": True,
    "require_api_key": True,
    "results": "JSON",
}

categories = ["general", "web"]
paging = True
safesearch = True
time_range_support = True
language_support = False
max_page = 3

base_url = "https://www.mojeek.com/search"
page_size = 10  # Mojeek's default `t`; `s` is a 1-based result offset.

#: Longer title/snippet than Mojeek's defaults (56 / 160) cost nothing extra and
#: give the agent more to work with. Both are capped by the API at 127 / 511.
title_length = 127
snippet_length = 511

_api_key = ""
_since_days = {"day": 1, "week": 7, "month": 31, "year": 366}


def setup(engine_settings):
    global _api_key  # pylint: disable=global-statement
    _api_key = os.environ.get("MOJEEK_API_KEY", "").strip()
    if not _api_key:
        log.warning("MOJEEK_API_KEY is not set; the mojeek api engine is disabled")
        return False
    return True


def request(query, params):
    args = {
        "api_key": _api_key,
        "q": query,
        "fmt": "json",
        "s": (params["pageno"] - 1) * page_size + 1,
        "safe": min(params["safesearch"], 1),
        "tlen": title_length,
        "dlen": snippet_length,
    }
    if params["time_range"]:
        since = datetime.now(timezone.utc) - timedelta(days=_since_days[params["time_range"]])
        args["since"] = since.strftime("%Y%m%d")

    params["url"] = f"{base_url}?{urlencode(args)}"
    return params


def response(resp):
    body = resp.json().get("response") or {}

    status = body.get("status")
    if status != "OK":
        # Daily limit, bad key, exhausted deposit, ... come back as a message in
        # `status`. Raise so SearXNG suspends the engine and records the cause
        # instead of treating it as "no results".
        message = f"Mojeek API: {status or 'no status in response'}"
        lowered = message.lower()
        if "limit" in lowered or "rate" in lowered:
            raise SearxEngineTooManyRequestsException(message=message)
        if "key" in lowered or "auth" in lowered or "credit" in lowered or "balance" in lowered:
            # Default suspension is a day; after a top-up or a key fix the
            # operator wants search back in minutes.
            raise SearxEngineAccessDeniedException(suspended_time=600, message=message)
        raise SearxEngineAPIException(message)

    results = []
    for item in body.get("results") or []:
        url = item.get("url")
        title = item.get("title")
        if not url or not title:
            continue
        results.append({"url": url, "title": title, "content": item.get("desc") or ""})
    return results
