"""Unit tests for web_search Wikipedia backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localtalk.services.tools.online import ConnectivityCache, NetworkStatus
from localtalk.services.tools.web import make_web_search_tool, wikipedia_search

pytestmark = pytest.mark.unit


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


class TestWebSearchTool:
    def test_offline_short_circuit(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(NetworkStatus(online=False, reachable=False, primary="none"))
        tool = make_web_search_tool(cache)
        with patch("localtalk.services.tools.web.wikipedia_search") as search:
            result = tool.handler({"query": "paris"})
        search.assert_not_called()
        assert result["ok"] is False
        assert "offline" in result["error"]

    def test_success_when_reachable(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(NetworkStatus(online=True, reachable=True, primary="wifi"))
        tool = make_web_search_tool(cache)
        with patch(
            "localtalk.services.tools.web.wikipedia_search",
            return_value={"ok": True, "hits": [{"title": "Paris"}], "count": 1},
        ) as search:
            result = tool.handler({"query": "paris"})
        search.assert_called_once()
        assert result["ok"] is True
