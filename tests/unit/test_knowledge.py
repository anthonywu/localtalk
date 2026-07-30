"""Unit tests for offline knowledge pack catalog and cache store."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from localtalk.knowledge.packs import DEFAULT_PACK_ID, get_pack, list_packs, pack_ids
from localtalk.knowledge.store import KnowledgeStore, ResolvedDownload, is_safe_zim_filename

pytestmark = pytest.mark.unit

_SAMPLE_OPDS = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <name>wikipedia_en-simple_all</name>
    <flavour>maxi</flavour>
    <link rel="http://opds-spec.org/acquisition/open-access"
          type="application/x-zim"
          href="https://example.test/wikipedia_en-simple_all_maxi_2026-06.zim.meta4"
          length="3500000000"/>
  </entry>
  <entry>
    <name>wikipedia_en-simple_all</name>
    <flavour>nopic</flavour>
    <link rel="http://opds-spec.org/acquisition/open-access"
          type="application/x-zim"
          href="https://example.test/wikipedia_en-simple_all_nopic_2026-06.zim.meta4"
          length="990000000"/>
  </entry>
</feed>
"""


class TestKnowledgePacks:
    def test_default_pack_is_simple_wikipedia_nopic(self):
        assert DEFAULT_PACK_ID == "wikipedia_en_simple_all_nopic"
        pack = get_pack(DEFAULT_PACK_ID)
        assert pack is not None
        assert pack.recommended is True
        assert pack.flavour == "nopic"

    def test_catalog_includes_wiktionary_and_top(self):
        ids = pack_ids()
        assert "wikipedia_en_top_nopic" in ids
        assert "wiktionary_en_simple_all_nopic" in ids
        assert "wikipedia_en_physics_nopic" in ids
        assert list_packs()[0].id == DEFAULT_PACK_ID

    def test_unknown_pack_returns_none(self):
        assert get_pack("nope") is None


class TestKnowledgeStore:
    def test_cache_dir_honors_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        from localtalk.knowledge.store import default_cache_dir

        assert default_cache_dir() == tmp_path / "xdg" / "localtalk" / "knowledge"

    def test_list_status_marks_installed(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        zim = tmp_path / "wikipedia_en-simple_all_nopic_2026-06.zim"
        zim.write_bytes(b"zim-bytes")
        (tmp_path / f"{DEFAULT_PACK_ID}.json").write_text(
            json.dumps(
                {
                    "pack_id": DEFAULT_PACK_ID,
                    "filename": zim.name,
                    "size_bytes": zim.stat().st_size,
                    "downloaded_at": "2026-07-30T00:00:00+00:00",
                    "source_url": "https://example.test/pack.zim",
                }
            ),
            encoding="utf-8",
        )

        status = store.acquire("list")
        assert status["ok"] is True
        assert status["default_pack"] == DEFAULT_PACK_ID
        by_id = {p["id"]: p for p in status["packs"]}
        assert by_id[DEFAULT_PACK_ID]["installed"] is True
        assert by_id[DEFAULT_PACK_ID]["path"] == str(zim)
        assert by_id["wiktionary_en_simple_all_nopic"]["installed"] is False

    def test_acquire_unknown_pack(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        result = store.acquire("not_a_real_pack")
        assert result["ok"] is False
        assert "unknown pack" in result["error"]

    def test_acquire_already_installed_skips_download(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        zim = tmp_path / "pack.zim"
        zim.write_bytes(b"data")
        (tmp_path / f"{DEFAULT_PACK_ID}.json").write_text(
            json.dumps(
                {
                    "pack_id": DEFAULT_PACK_ID,
                    "filename": "pack.zim",
                    "size_bytes": 4,
                    "downloaded_at": "2026-07-30T00:00:00+00:00",
                    "source_url": "https://example.test/pack.zim",
                }
            ),
            encoding="utf-8",
        )

        with patch.object(store, "resolve_download") as resolve:
            result = store.acquire(None)
        resolve.assert_not_called()
        assert result["ok"] is True
        assert result["already_installed"] is True
        assert result["pack_id"] == DEFAULT_PACK_ID

    def test_resolve_download_picks_matching_flavour(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        pack = get_pack(DEFAULT_PACK_ID)
        assert pack is not None

        class _Resp:
            def read(self):
                return _SAMPLE_OPDS.encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch("urllib.request.urlopen", return_value=_Resp()):
            resolved = store.resolve_download(pack)

        assert resolved.filename == "wikipedia_en-simple_all_nopic_2026-06.zim"
        assert resolved.url.endswith(".zim")
        assert not resolved.url.endswith(".meta4")
        assert resolved.size_bytes == 990000000

    def test_acquire_downloads_and_writes_manifest(self, tmp_path):
        store = KnowledgeStore(cache_dir=tmp_path, console=Console())
        pack = get_pack("wiktionary_en_simple_all_nopic")
        assert pack is not None
        resolved = ResolvedDownload(
            pack_id=pack.id,
            filename="wiktionary_en_simple_all_nopic_2026-04.zim",
            url="https://example.test/wiktionary.zim",
            size_bytes=5,
        )

        def _fake_download(url, dest, expected_size=None):
            dest.write_bytes(b"hello")

        with (
            patch.object(store, "resolve_download", return_value=resolved),
            patch.object(store, "_download_file", side_effect=_fake_download),
        ):
            result = store.acquire(pack.id)

        assert result["ok"] is True
        assert result["already_installed"] is False
        dest = Path(result["path"])
        assert dest.is_file()
        assert dest.read_bytes() == b"hello"
        manifest = json.loads((tmp_path / f"{pack.id}.json").read_text(encoding="utf-8"))
        assert manifest["filename"] == resolved.filename
        assert manifest["source_url"] == resolved.url

    def test_is_safe_zim_filename(self):
        assert is_safe_zim_filename("wikipedia_en-simple_all_nopic_2026-06.zim")
        assert not is_safe_zim_filename("../evil.zim")
        assert not is_safe_zim_filename("nope.txt")
