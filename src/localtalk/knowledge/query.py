"""Query installed offline knowledge packs (Kiwix ZIM archives)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from localtalk.knowledge.citation import attach_citations
from localtalk.knowledge.packs import DEFAULT_PACK_ID, get_pack, pack_ids
from localtalk.knowledge.store import InstalledPack, KnowledgeStore

_DEFAULT_MAX_RESULTS = 3
_MAX_RESULTS_CAP = 5
_DEFAULT_MAX_CHARS = 2000
_WHITESPACE_RE = re.compile(r"\s+")


class _HTMLToText(HTMLParser):
    """Minimal HTML → plain text converter for ZIM article bodies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        elif tag in {"p", "div", "li", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Convert HTML article content to plain text suitable for LLM context."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return _WHITESPACE_RE.sub(" ", html)
    return parser.text()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0]
    return (cut or text[: max_chars - 1]).rstrip() + "…"


class KnowledgeQueryService:
    """Search and read articles from installed ZIM packs via libzim."""

    def __init__(self, store: KnowledgeStore, max_chars: int = _DEFAULT_MAX_CHARS):
        self.store = store
        self.max_chars = max_chars
        self._archives: dict[str, Any] = {}

    def query(self, args: dict) -> dict:
        """Dispatch a query_knowledge tool call."""
        action = str(args.get("action") or "search").lower().strip()
        query = str(args.get("query") or "").strip()
        pack_id = args.get("pack")
        pack_id = None if pack_id in (None, "", "all") else str(pack_id)
        max_results = args.get("max_results", _DEFAULT_MAX_RESULTS)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = _DEFAULT_MAX_RESULTS
        max_results = max(1, min(max_results, _MAX_RESULTS_CAP))

        if not query:
            return {"ok": False, "error": "query is required"}

        if pack_id is not None and get_pack(pack_id) is None:
            return {
                "ok": False,
                "error": f"unknown pack {pack_id!r}; known packs: {', '.join(pack_ids())}",
            }

        installed = self._packs_for_query(pack_id)
        if not installed:
            return {
                "ok": False,
                "error": "no offline knowledge packs installed",
                "hint": (
                    f"Call acquire_knowledge to download a pack first "
                    f"(recommended default: {DEFAULT_PACK_ID})."
                ),
                "installed": [],
            }

        if action == "search":
            return attach_citations(self.search(query, packs=installed, max_results=max_results))
        if action == "get":
            return attach_citations(self.get(query, packs=installed))
        return {"ok": False, "error": f"unknown action {action!r}; expected 'search' or 'get'"}

    def _packs_for_query(self, pack_id: str | None) -> list[InstalledPack]:
        if pack_id is not None:
            item = self.store.get_installed(pack_id)
            return [item] if item is not None else []
        installed = self.store.list_installed()
        # Prefer the default pack first when searching across all.
        installed.sort(key=lambda p: (0 if p.pack_id == DEFAULT_PACK_ID else 1, p.pack_id))
        return installed

    def _open_archive(self, pack: InstalledPack):
        path_key = str(pack.path)
        if path_key not in self._archives:
            try:
                from libzim.reader import Archive
            except ImportError as exc:
                raise RuntimeError(
                    "libzim is required to query offline knowledge packs. Install with: uv add libzim"
                ) from exc
            try:
                self._archives[path_key] = Archive(Path(pack.path))
            except Exception as exc:
                raise RuntimeError(f"failed to open ZIM for pack {pack.pack_id}: {exc}") from exc
        return self._archives[path_key]

    def search(
        self,
        query: str,
        packs: list[InstalledPack] | None = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> dict:
        """Full-text (or title-suggestion) search across installed packs."""
        packs = packs if packs is not None else self._packs_for_query(None)
        if not packs:
            return {
                "ok": False,
                "error": "no offline knowledge packs installed",
                "hint": f"Call acquire_knowledge (default: {DEFAULT_PACK_ID}).",
            }

        hits: list[dict] = []
        errors: list[str] = []
        for pack in packs:
            if len(hits) >= max_results:
                break
            try:
                archive = self._open_archive(pack)
                remaining = max_results - len(hits)
                hits.extend(self._search_pack(archive, pack, query, remaining))
            except RuntimeError as exc:
                errors.append(str(exc))

        if not hits and errors:
            return {"ok": False, "error": errors[0], "errors": errors}
        return {
            "ok": True,
            "action": "search",
            "query": query,
            "count": len(hits),
            "hits": hits,
            "errors": errors or None,
        }

    def _search_pack(self, archive, pack: InstalledPack, query: str, limit: int) -> list[dict]:
        from libzim.search import Query, Searcher
        from libzim.suggestion import SuggestionSearcher

        hits: list[dict] = []
        paths: list[str] = []

        if getattr(archive, "has_fulltext_index", False):
            search = Searcher(archive).search(Query().set_query(query))
            estimated = int(search.getEstimatedMatches())
            if estimated > 0:
                paths = list(search.getResults(0, min(limit, estimated)))

        if not paths:
            # Fall back to title suggestions when no FTS index or no FTS hits.
            try:
                suggestion = SuggestionSearcher(archive).suggest(query)
                count = int(suggestion.getEstimatedMatches())
                if count > 0:
                    paths = list(suggestion.getResults(0, min(limit, count)))
            except Exception:
                paths = []

        for path in paths[:limit]:
            title = path
            snippet = ""
            try:
                entry = archive.get_entry_by_path(path)
                title = entry.title or path
                item = entry.get_item()
                raw = bytes(item.content).decode("utf-8", errors="replace")
                snippet = _truncate(html_to_text(raw), 280)
            except Exception:
                pass
            pack_meta = get_pack(pack.pack_id)
            hits.append(
                {
                    "title": title,
                    "path": path,
                    "snippet": snippet,
                    "source": "offline",
                    "pack_id": pack.pack_id,
                    "source_label": pack_meta.title if pack_meta else pack.pack_id,
                }
            )
        return hits

    def get(self, title_or_path: str, packs: list[InstalledPack] | None = None) -> dict:
        """Fetch a single article by path or title from installed packs."""
        packs = packs if packs is not None else self._packs_for_query(None)
        if not packs:
            return {
                "ok": False,
                "error": "no offline knowledge packs installed",
                "hint": f"Call acquire_knowledge (default: {DEFAULT_PACK_ID}).",
            }

        errors: list[str] = []
        for pack in packs:
            try:
                archive = self._open_archive(pack)
                entry = self._resolve_entry(archive, title_or_path)
                if entry is None:
                    continue
                item = entry.get_item()
                raw = bytes(item.content).decode("utf-8", errors="replace")
                text = _truncate(html_to_text(raw), self.max_chars)
                pack_meta = get_pack(pack.pack_id)
                return {
                    "ok": True,
                    "action": "get",
                    "title": entry.title or title_or_path,
                    "path": entry.path,
                    "text": text,
                    "snippet": _truncate(text, 280),
                    "source": "offline",
                    "pack_id": pack.pack_id,
                    "source_label": pack_meta.title if pack_meta else pack.pack_id,
                    "truncated": len(html_to_text(raw)) > self.max_chars,
                }
            except RuntimeError as exc:
                errors.append(str(exc))

        return {
            "ok": False,
            "error": f"article not found: {title_or_path!r}",
            "errors": errors or None,
            "hint": "Try action='search' first to find matching titles/paths.",
        }

    def _resolve_entry(self, archive, title_or_path: str):
        path = title_or_path.strip().lstrip("/")
        if archive.has_entry_by_path(path):
            return archive.get_entry_by_path(path)
        # Common ZIM path prefixes
        for candidate in (path, f"A/{path}", f"A/{path.replace(' ', '_')}"):
            if archive.has_entry_by_path(candidate):
                return archive.get_entry_by_path(candidate)
        if archive.has_entry_by_title(title_or_path):
            return archive.get_entry_by_title(title_or_path)
        # Title with underscores / spaces variants
        alt = title_or_path.replace("_", " ")
        if alt != title_or_path and archive.has_entry_by_title(alt):
            return archive.get_entry_by_title(alt)
        return None
