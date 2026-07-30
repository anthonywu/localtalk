"""Unit tests for web_search backends and routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localtalk.services.tools.online import ConnectivityCache, NetworkStatus
from localtalk.services.tools.web import (
    is_live_query,
    is_weather_query,
    make_web_search_tool,
    run_web_search,
    wikipedia_search,
)

pytestmark = pytest.mark.unit


class TestQueryClassification:
    def test_weather_is_live(self):
        assert is_weather_query("weather in San Francisco tomorrow")
        assert is_live_query("San Francisco weather tomorrow")

    def test_capital_not_live(self):
        assert not is_live_query("capital of Mongolia")
        assert not is_weather_query("capital of Mongolia")


class TestWikipediaSearch:
    def test_success(self):
        opensearch = MagicMock()
        opensearch.status_code = 200
        opensearch.raise_for_status = MagicMock()
        opensearch.json.return_value = [
            "paris",
            ["Paris"],
            ["capital"],
            ["https://en.wikipedia.org/wiki/Paris"],
        ]

        summary = MagicMock()
        summary.status_code = 200
        summary.json.return_value = {
            "extract": "Paris is the capital of France.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Paris"}},
        }

        client = MagicMock()
        client.get.side_effect = [opensearch, summary]
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=client):
            result = wikipedia_search("paris", max_results=1)

        assert result["ok"] is True
        assert result["hits"][0]["title"] == "Paris"
        assert "capital of France" in result["hits"][0]["snippet"]
        assert result["hits"][0]["source"] == "web"
        assert "English Wikipedia online" in result["cite"]
        assert result["hits"][0]["citation"]["kind"] == "web"


class TestRunWebSearchRouting:
    def test_weather_prefers_wttr_when_browser_present(self):
        session = MagicMock()
        with patch(
            "localtalk.services.tools.web.weather_lookup_via_browser",
            return_value={
                "ok": True,
                "hits": [{"title": "Weather for San Francisco", "snippet": "Sunny"}],
                "backend": "wttr.in",
                "cite": "According to wttr.in",
            },
        ) as weather:
            with patch("localtalk.services.tools.web.google_search_via_browser") as google:
                result = run_web_search(
                    "weather in San Francisco tomorrow",
                    browser_session=session,
                )
        weather.assert_called_once()
        google.assert_not_called()
        assert result["backend"] == "wttr.in"
        assert result["ok"] is True

    def test_general_query_uses_google_before_wikipedia(self):
        session = MagicMock()
        with patch(
            "localtalk.services.tools.web.google_search_via_browser",
            return_value={
                "ok": True,
                "hits": [{"title": "Result", "snippet": "info"}],
                "backend": "google",
                "cite": "According to Google Search",
            },
        ) as google:
            with patch("localtalk.services.tools.web.wikipedia_search") as wiki:
                result = run_web_search("best coffee shops near me", browser_session=session)
        google.assert_called_once()
        wiki.assert_not_called()
        assert result["backend"] == "google"

    def test_macbook_price_opens_product_page_before_google(self):
        """Price/product queries open Chrome/Safari and read the vendor page first."""
        session = MagicMock()
        with patch(
            "localtalk.services.tools.web.product_page_via_browser",
            return_value={
                "ok": True,
                "hits": [{"title": "Apple MacBook Pro", "snippet": "From $1599"}],
                "backend": "product_page",
            },
        ) as product:
            with patch("localtalk.services.tools.web.google_search_via_browser") as google:
                result = run_web_search(
                    "current MacBook Pro price July 2026",
                    browser_session=session,
                )
        product.assert_called_once()
        # headed=True so Playwright opens a visible Chrome/Safari window
        assert product.call_args.kwargs.get("headed") is True
        google.assert_not_called()
        assert result["backend"] == "product_page"
        assert result["ok"] is True

    def test_failure_includes_suggested_urls(self):
        with patch(
            "localtalk.services.tools.web.duckduckgo_lite_search",
            return_value={"ok": False, "error": "no results"},
        ):
            result = run_web_search("MacBook Pro price", browser_session=None)
        assert result["ok"] is False
        assert result.get("suggested_urls")
        assert any("apple.com/macbook-pro" in u for u in result["suggested_urls"])

    def test_live_query_skips_wikipedia_on_failure(self):
        with patch(
            "localtalk.services.tools.web.duckduckgo_lite_search",
            return_value={"ok": False, "error": "no results"},
        ):
            with patch("localtalk.services.tools.web.wikipedia_search") as wiki:
                result = run_web_search("weather tomorrow", browser_session=None)
        wiki.assert_not_called()
        assert result["ok"] is False
        assert any("skipped" in d for d in result.get("detail", []))

    def test_encyclopedic_falls_back_to_wikipedia(self):
        with patch(
            "localtalk.services.tools.web.duckduckgo_lite_search",
            return_value={"ok": False, "error": "no results"},
        ):
            with patch(
                "localtalk.services.tools.web.wikipedia_search",
                return_value={"ok": True, "hits": [{"title": "Paris"}], "backend": "wikipedia"},
            ) as wiki:
                result = run_web_search("capital of France", browser_session=None)
        wiki.assert_called_once()
        assert result["ok"] is True


class TestWebSearchTool:
    def test_offline_short_circuit(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(NetworkStatus(online=False, reachable=False, primary="none"))
        tool = make_web_search_tool(cache)
        with patch("localtalk.services.tools.web.run_web_search") as search:
            result = tool.handler({"query": "paris"})
        search.assert_not_called()
        assert result["ok"] is False
        assert "offline" in result["error"]

    def test_success_when_reachable(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(NetworkStatus(online=True, reachable=True, primary="wifi"))
        tool = make_web_search_tool(cache)
        with patch(
            "localtalk.services.tools.web.run_web_search",
            return_value={"ok": True, "hits": [{"title": "Paris"}], "count": 1},
        ) as search:
            result = tool.handler({"query": "paris"})
        search.assert_called_once()
        assert result["ok"] is True

    def test_passes_browser_session_getter(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(NetworkStatus(online=True, reachable=True, primary="wifi"))
        session = MagicMock()
        tool = make_web_search_tool(cache, browser_session_getter=lambda: session)
        with patch(
            "localtalk.services.tools.web.run_web_search",
            return_value={"ok": True, "hits": [], "count": 0},
        ) as search:
            tool.handler({"query": "weather SF"})
        assert search.call_args.kwargs["browser_session"] is session
