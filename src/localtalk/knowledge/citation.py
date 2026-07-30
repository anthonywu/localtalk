"""Speakable citation helpers for offline and online knowledge tools."""

from __future__ import annotations

from typing import Any

from localtalk.knowledge.packs import get_pack

_CITE_INSTRUCTION = (
    "In your final spoken answer, briefly cite this source in natural speech "
    "(for example start with the spoken_prefix). Do not read URLs or pack ids aloud."
)


def offline_citation(pack_id: str | None, article_title: str | None = None) -> dict[str, Any]:
    """Build a citation payload for an installed offline pack hit."""
    pack = get_pack(pack_id) if pack_id else None
    label = pack.title if pack else (pack_id or "offline knowledge pack")
    if article_title:
        spoken_prefix = f"According to the {label} article on {article_title}"
    else:
        spoken_prefix = f"According to {label}"
    return {
        "kind": "offline",
        "label": label,
        "pack_id": pack_id,
        "article": article_title,
        "spoken_prefix": spoken_prefix,
        "instruction": _CITE_INSTRUCTION,
    }


_WEB_BACKEND_LABELS = {
    "wikipedia": "English Wikipedia online",
    "google": "Google Search",
    "duckduckgo": "DuckDuckGo",
    "wttr.in": "wttr.in weather",
    "product_page": "the product page",
}


def web_citation(backend: str = "wikipedia", article_title: str | None = None) -> dict[str, Any]:
    """Build a citation payload for an online web_search hit."""
    label = _WEB_BACKEND_LABELS.get(backend, f"{backend} online")
    if article_title and backend == "wikipedia":
        spoken_prefix = f"According to {label}, in the article on {article_title}"
    elif article_title and backend in {"google", "duckduckgo"}:
        spoken_prefix = f"According to {label}"
    else:
        spoken_prefix = f"According to {label}"
    return {
        "kind": "web",
        "label": label,
        "backend": backend,
        "article": article_title,
        "spoken_prefix": spoken_prefix,
        "instruction": _CITE_INSTRUCTION,
    }


def attach_citations(result: dict[str, Any]) -> dict[str, Any]:
    """Enrich a successful knowledge tool result with citation fields."""
    if not result.get("ok"):
        return result

    if result.get("action") == "get" or (result.get("text") and result.get("pack_id")):
        citation = offline_citation(result.get("pack_id"), result.get("title"))
        result = {**result, "citation": citation, "cite": citation["spoken_prefix"]}
        for hit_key in ("hits",):
            if hit_key in result and isinstance(result[hit_key], list):
                pass
        return result

    if result.get("action") == "search" and result.get("hits"):
        enriched_hits = []
        for hit in result["hits"]:
            citation = offline_citation(hit.get("pack_id"), hit.get("title"))
            enriched_hits.append({**hit, "citation": citation, "cite": citation["spoken_prefix"]})
        primary = enriched_hits[0]["citation"]
        return {
            **result,
            "hits": enriched_hits,
            "citation": primary,
            "cite": primary["spoken_prefix"],
        }

    if result.get("backend") and result.get("hits") is not None:
        # web_search shape
        backend = str(result.get("backend") or "wikipedia")
        enriched_hits = []
        for hit in result["hits"]:
            citation = web_citation(backend, hit.get("title"))
            enriched_hits.append({**hit, "citation": citation, "cite": citation["spoken_prefix"]})
        primary = (
            enriched_hits[0]["citation"]
            if enriched_hits
            else web_citation(backend)
        )
        return {
            **result,
            "hits": enriched_hits,
            "citation": primary,
            "cite": primary["spoken_prefix"],
        }

    return result
