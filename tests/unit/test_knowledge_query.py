"""Unit tests for offline knowledge query (mocked libzim)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from localtalk.knowledge.packs import DEFAULT_PACK_ID
from localtalk.knowledge.query import KnowledgeQueryService, html_to_text
from localtalk.knowledge.store import InstalledPack, KnowledgeStore

pytestmark = pytest.mark.unit


class TestHtmlToText:
    def test_strips_tags_and_scripts(self):
        html = "<html><script>bad()</script><p>Hello <b>world</b></p><style>.x{}</style></html>"
        text = html_to_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "bad()" not in text
        assert ".x" not in text


class TestKnowledgeQueryService:
    def _installed(self, tmp_path: Path) -> InstalledPack:
        zim = tmp_path / "pack.zim"
        zim.write_bytes(b"zim")
        return InstalledPack(
            pack_id=DEFAULT_PACK_ID,
            filename=zim.name,
            path=zim,
            size_bytes=3,
            downloaded_at="2026-07-30T00:00:00+00:00",
            source_url="https://example.test/pack.zim",
        )

    def test_no_packs_installed(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        service = KnowledgeQueryService(store)
        result = service.query({"action": "search", "query": "paris"})
        assert result["ok"] is False
        assert "no offline knowledge packs" in result["error"]
        assert "acquire_knowledge" in result["hint"]

    def test_unknown_pack(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        service = KnowledgeQueryService(store)
        result = service.query({"action": "search", "query": "x", "pack": "nope"})
        assert result["ok"] is False
        assert "unknown pack" in result["error"]

    def test_search_uses_libzim(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        pack = self._installed(tmp_path)
        (tmp_path / f"{DEFAULT_PACK_ID}.json").write_text(
            (
                f'{{"pack_id":"{DEFAULT_PACK_ID}","filename":"{pack.filename}",'
                f'"size_bytes":3,"downloaded_at":"t","source_url":"u"}}'
            ),
            encoding="utf-8",
        )
        service = KnowledgeQueryService(store)

        archive = MagicMock()
        archive.has_fulltext_index = True
        entry = MagicMock()
        entry.title = "Paris"
        entry.get_item.return_value.content = b"<p>Paris is a city.</p>"
        archive.get_entry_by_path.return_value = entry

        search = MagicMock()
        search.getEstimatedMatches.return_value = 1
        search.getResults.return_value = ["A/Paris"]
        searcher = MagicMock()
        searcher.search.return_value = search

        with (
            patch("libzim.reader.Archive", return_value=archive),
            patch("libzim.search.Searcher", return_value=searcher),
            patch("libzim.search.Query") as query_cls,
        ):
            query_cls.return_value.set_query.return_value = MagicMock()
            result = service.query({"action": "search", "query": "paris", "max_results": 3})

        assert result["ok"] is True
        assert result["count"] == 1
        assert result["hits"][0]["title"] == "Paris"
        assert result["hits"][0]["source"] == "offline"
        assert "Paris is a city" in result["hits"][0]["snippet"]

    def test_get_article(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        pack = self._installed(tmp_path)
        (tmp_path / f"{DEFAULT_PACK_ID}.json").write_text(
            (
                f'{{"pack_id":"{DEFAULT_PACK_ID}","filename":"{pack.filename}",'
                f'"size_bytes":3,"downloaded_at":"t","source_url":"u"}}'
            ),
            encoding="utf-8",
        )
        service = KnowledgeQueryService(store)

        archive = MagicMock()
        archive.has_entry_by_path.side_effect = lambda p: p == "A/Paris"
        archive.has_entry_by_title.return_value = False
        entry = MagicMock()
        entry.title = "Paris"
        entry.path = "A/Paris"
        entry.get_item.return_value.content = b"<p>Capital of France.</p>"
        archive.get_entry_by_path.return_value = entry

        with patch("libzim.reader.Archive", return_value=archive):
            result = service.query({"action": "get", "query": "A/Paris"})

        assert result["ok"] is True
        assert result["title"] == "Paris"
        assert "Capital of France" in result["text"]
