"""Unit tests for knowledge citation helpers."""

from __future__ import annotations

import pytest

from localtalk.knowledge.citation import attach_citations, offline_citation, web_citation
from localtalk.knowledge.packs import DEFAULT_PACK_ID

pytestmark = pytest.mark.unit


class TestCitations:
    def test_offline_citation_with_article(self):
        cite = offline_citation(DEFAULT_PACK_ID, "Paris")
        assert cite["label"] == "Simple English Wikipedia"
        assert "Simple English Wikipedia" in cite["spoken_prefix"]
        assert "Paris" in cite["spoken_prefix"]
        assert "cite this source" in cite["instruction"].lower() or "briefly cite" in cite["instruction"]

    def test_web_citation(self):
        cite = web_citation("wikipedia", "Paris")
        assert "English Wikipedia online" in cite["spoken_prefix"]
        assert "Paris" in cite["spoken_prefix"]

    def test_attach_citations_to_get(self):
        result = attach_citations(
            {
                "ok": True,
                "action": "get",
                "title": "Paris",
                "text": "Paris is the capital.",
                "pack_id": DEFAULT_PACK_ID,
            }
        )
        assert result["cite"].startswith("According to")
        assert "Simple English Wikipedia" in result["cite"]
        assert result["citation"]["article"] == "Paris"

    def test_attach_citations_to_search_hits(self):
        result = attach_citations(
            {
                "ok": True,
                "action": "search",
                "hits": [{"title": "Paris", "pack_id": DEFAULT_PACK_ID, "source": "offline"}],
            }
        )
        assert result["hits"][0]["cite"]
        assert result["citation"]["label"] == "Simple English Wikipedia"

    def test_attach_citations_to_web_search(self):
        result = attach_citations(
            {
                "ok": True,
                "backend": "wikipedia",
                "hits": [{"title": "Paris", "snippet": "capital", "source": "web"}],
            }
        )
        assert "English Wikipedia online" in result["cite"]
        assert result["hits"][0]["citation"]["kind"] == "web"

    def test_failed_result_unchanged(self):
        result = attach_citations({"ok": False, "error": "missing"})
        assert result == {"ok": False, "error": "missing"}
