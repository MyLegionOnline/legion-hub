"""The SearXNG ``mojeek api`` engine (deployment/searxng/mojeek_api.py).

The engine imports ``searx.exceptions``, which only exists inside the SearXNG
image, so a stand-in module is injected before loading the file. What is pinned
here is our logic: the request URL, paging and error classification.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ENGINE_PATH = Path(__file__).resolve().parents[2] / "deployment" / "searxng" / "mojeek_api.py"


class _TooMany(Exception):
    def __init__(self, suspended_time=None, message="Too many request"):
        super().__init__(message)


class _Denied(Exception):
    def __init__(self, suspended_time=None, message="Access denied"):
        super().__init__(message)
        self.suspended_time = suspended_time


class _API(Exception):
    pass


@pytest.fixture()
def engine(monkeypatch):
    fake = types.ModuleType("searx.exceptions")
    fake.SearxEngineTooManyRequestsException = _TooMany
    fake.SearxEngineAccessDeniedException = _Denied
    fake.SearxEngineAPIException = _API
    monkeypatch.setitem(sys.modules, "searx", types.ModuleType("searx"))
    monkeypatch.setitem(sys.modules, "searx.exceptions", fake)
    spec = importlib.util.spec_from_file_location("mojeek_api_under_test", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _params(**kw):
    return {"pageno": 1, "safesearch": 0, "time_range": None, **kw}


def _query(engine, **kw):
    engine.setup({})
    url = engine.request("hello world", _params(**kw))["url"]
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://www.mojeek.com/search"
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}


class TestSetup:
    def test_without_a_key_the_engine_drops_itself(self, engine, monkeypatch):
        monkeypatch.delenv("MOJEEK_API_KEY", raising=False)
        assert engine.setup({}) is False

    def test_blank_key_counts_as_missing(self, engine, monkeypatch):
        monkeypatch.setenv("MOJEEK_API_KEY", "   ")
        assert engine.setup({}) is False

    def test_with_a_key_it_stays(self, engine, monkeypatch):
        monkeypatch.setenv("MOJEEK_API_KEY", "k")
        assert engine.setup({}) is True


class TestRequest:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        monkeypatch.setenv("MOJEEK_API_KEY", "secret-key")

    def test_basic_parameters(self, engine):
        q = _query(engine)
        assert q["api_key"] == "secret-key"
        assert q["q"] == "hello world"
        assert q["fmt"] == "json"
        assert q["s"] == "1"
        assert q["safe"] == "0"

    @pytest.mark.parametrize("pageno,offset", [(1, "1"), (2, "11"), (3, "21")])
    def test_pages_become_one_based_offsets(self, engine, pageno, offset):
        assert _query(engine, pageno=pageno)["s"] == offset

    @pytest.mark.parametrize("safesearch,safe", [(0, "0"), (1, "1"), (2, "1")])
    def test_strict_safesearch_is_clamped_to_mojeeks_flag(self, engine, safesearch, safe):
        assert _query(engine, safesearch=safesearch)["safe"] == safe

    def test_time_range_becomes_a_since_date(self, engine):
        assert len(_query(engine, time_range="week")["since"]) == 8
        assert "since" not in _query(engine)

    def test_longer_title_and_snippet_are_requested(self, engine):
        q = _query(engine)
        assert (q["tlen"], q["dlen"]) == ("127", "511")


class TestResponse:
    def test_maps_results_and_skips_incomplete_ones(self, engine):
        body = {
            "response": {
                "status": "OK",
                "results": [
                    {"url": "https://a.example", "title": "A", "desc": "snippet"},
                    {"url": "https://b.example", "title": "B"},
                    {"url": "", "title": "no url"},
                    {"url": "https://c.example", "title": ""},
                ],
            }
        }
        assert engine.response(_Resp(body)) == [
            {"url": "https://a.example", "title": "A", "content": "snippet"},
            {"url": "https://b.example", "title": "B", "content": ""},
        ]

    def test_ok_with_no_results_is_empty_not_an_error(self, engine):
        assert engine.response(_Resp({"response": {"status": "OK", "results": []}})) == []

    @pytest.mark.parametrize("status", ["Daily limit reached", "Rate limit exceeded"])
    def test_limits_are_reported_as_too_many_requests(self, engine, status):
        with pytest.raises(_TooMany):
            engine.response(_Resp({"response": {"status": status}}))

    @pytest.mark.parametrize("status", ["Invalid API key", "Insufficient credit", "Authentication failed"])
    def test_key_and_balance_problems_are_denials_suspended_for_minutes_not_a_day(self, engine, status):
        with pytest.raises(_Denied) as exc:
            engine.response(_Resp({"response": {"status": status}}))
        assert exc.value.suspended_time == 600

    def test_anything_else_is_an_api_error_never_silent_empty(self, engine):
        with pytest.raises(_API):
            engine.response(_Resp({"response": {"status": "Something odd"}}))
        with pytest.raises(_API):
            engine.response(_Resp({}))
