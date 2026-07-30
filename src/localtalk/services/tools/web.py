"""Opt-in web_search Harmony tool (Wikipedia OpenSearch backend)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from localtalk.services.tools.base import ToolSpec, build_tool_description
from localtalk.services.tools.online import ConnectivityCache

TOOL_NAME = "web_search"
_USER_AGENT = "localtalk/0.6 (+https://github.com/anthonywu/localtalk)"
_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def wikipedia_search(
    query: str,
    *,
    max_results: int = 3,
    timeout_s: float = 8.0,
    console_print=None,
) -> dict[str, Any]:
    """Search English Wikipedia and return short page summaries."""
    import httpx

    max_results = max(1, min(int(max_results), 5))
    params = {
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": 0,
        "format": "json",
    }
    if console_print:
        console_print(f"[cyan]web_search → en.wikipedia.org query={query!r}[/cyan]")

    with httpx.Client(timeout=timeout_s, headers={"User-Agent": _USER_AGENT}) as client:
        response = client.get(_OPENSEARCH, params=params)
        response.raise_for_status()
        data = response.json()
        # OpenSearch: [query, [titles], [descriptions], [urls]]
        titles = list(data[1]) if isinstance(data, list) and len(data) > 1 else []
        urls = list(data[3]) if isinstance(data, list) and len(data) > 3 else []

        hits: list[dict[str, Any]] = []
        for idx, title in enumerate(titles[:max_results]):
            url = urls[idx] if idx < len(urls) else None
            snippet = ""
            try:
                summary_url = _SUMMARY.format(title=quote(title.replace(" ", "_"), safe="()_") )
                summary_resp = client.get(summary_url)
                if summary_resp.status_code == 200:
                    payload = summary_resp.json()
                    snippet = (payload.get("extract") or "")[:500]
                    url = payload.get("content_urls", {}).get("desktop", {}).get("page") or url
            except Exception:
                pass
            hits.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "source": "web",
                    "url": url,
                    "pack_id": None,
                }
            )

    return {
        "ok": True,
        "query": query,
        "count": len(hits),
        "hits": hits,
        "backend": "wikipedia",
    }


def make_web_search_tool(
    cache: ConnectivityCache,
    *,
    max_results_default: int = 3,
    timeout_s: float = 8.0,
    console_print=None,
) -> ToolSpec:
    def handler(args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        max_results = args.get("max_results", max_results_default)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = max_results_default

        status = cache.get(probe=False, force=False)
        if status is None:
            status = cache.get(probe=True, force=True)
        if status is not None and not status.reachable:
            return {
                "ok": False,
                "error": "offline or internet unreachable — web_search unavailable",
                "hint": "Use query_knowledge with an offline pack, or reconnect and try again.",
            }

        try:
            return wikipedia_search(
                query,
                max_results=max_results,
                timeout_s=timeout_s,
                console_print=console_print,
            )
        except Exception as exc:
            return {"ok": False, "error": f"web_search failed: {exc}"}

    def spoken_fallback(result: dict, args: dict) -> str:
        if not result.get("ok"):
            if "offline" in str(result.get("error", "")).lower():
                return "I can't reach the internet right now, so I couldn't search the web."
            return "Sorry, the web search didn't work."
        hits = result.get("hits") or []
        if not hits:
            return "I searched the web but didn't find a clear result."
        title = hits[0].get("title") or "a page"
        return f"I found information online about {title}."

    description = build_tool_description(
        TOOL_NAME,
        (
            "Search the public web for current or world knowledge (Wikipedia backend). "
            "Only available when the user launched LocalTalk with --enable-web. "
            "Use for facts that may need an online lookup; prefer offline query_knowledge "
            "when a local pack is installed. Returns short title/snippet results."
        ),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum number of results (default 3)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    return ToolSpec(name=TOOL_NAME, description=description, handler=handler, spoken_fallback=spoken_fallback)
