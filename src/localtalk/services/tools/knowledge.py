"""Offline knowledge Harmony tools: acquire_knowledge + query_knowledge."""

from __future__ import annotations

from localtalk.knowledge.packs import DEFAULT_PACK_ID, pack_ids
from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore
from localtalk.services.tools.base import ToolSpec, build_tool_description

ACQUIRE_TOOL_NAME = "acquire_knowledge"
QUERY_TOOL_NAME = "query_knowledge"


def make_acquire_knowledge_tool(store: KnowledgeStore, console_print) -> ToolSpec:
    """Build acquire_knowledge; ``console_print`` is a callable for status lines."""

    def handler(args: dict) -> dict:
        pack = args.get("pack")
        pack_id = None if pack is None else str(pack)
        console_print(f"[cyan]Acquiring knowledge pack: {pack_id or DEFAULT_PACK_ID}[/cyan]")
        return store.acquire(pack_id)

    def spoken_fallback(result: dict, args: dict) -> str:
        if args.get("pack") == "list" or result.get("packs") is not None:
            installed = [p["title"] for p in result.get("packs", []) if p.get("installed")]
            if installed:
                return (
                    "Here are the offline knowledge packs I can download. "
                    f"Already installed: {', '.join(installed)}. "
                    "The recommended default is Simple English Wikipedia."
                )
            return (
                "Here are the offline knowledge packs I can download. "
                "I recommend starting with Simple English Wikipedia, about one gigabyte."
            )
        if result.get("ok"):
            title = result.get("title") or "that knowledge pack"
            if result.get("already_installed"):
                return f"{title} is already downloaded and ready in your local cache."
            return f"Okay, I've downloaded {title} into your local cache."
        return "Sorry, I couldn't download that knowledge pack."

    pack_enum = ["list", *pack_ids()]
    description = build_tool_description(
        ACQUIRE_TOOL_NAME,
        (
            "Download an offline knowledge pack into the user's home cache so LocalTalk "
            "can use world knowledge without the internet (airplane mode). Call this when "
            "the user asks to download Wikipedia, get offline knowledge, install a "
            "knowledge base, or similar. "
            f"Default pack is {DEFAULT_PACK_ID} (Simple English Wikipedia without "
            "pictures, about 1 gigabyte) — recommend this first. Other packs: "
            "wikipedia_en_top_nopic (Best of English Wikipedia, about 2 gigabytes), "
            "wiktionary_en_simple_all_nopic (Simple English dictionary, about 25 "
            "megabytes), wikipedia_en_physics_nopic (physics articles, about 300 "
            "megabytes). Pass pack='list' to list available packs and what is already "
            "installed. Packs are stored under ~/.cache/localtalk/knowledge and kept "
            "for future sessions. Before a large download begins, LocalTalk automatically "
            "tells the user the pack name, size, and expected wait — you do not need to "
            "pre-announce the wait yourself."
        ),
        {
            "type": "object",
            "properties": {
                "pack": {
                    "type": "string",
                    "enum": pack_enum,
                    "description": (
                        f"Pack to download, or 'list' to show options. "
                        f"Omit or use {DEFAULT_PACK_ID} for the recommended default."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    )
    return ToolSpec(
        name=ACQUIRE_TOOL_NAME,
        description=description,
        handler=handler,
        spoken_fallback=spoken_fallback,
    )


def make_query_knowledge_tool(query_service: KnowledgeQueryService, console_print) -> ToolSpec:
    """Build query_knowledge for search/get over installed ZIM packs."""

    def handler(args: dict) -> dict:
        action = str(args.get("action") or "search")
        console_print(f"[cyan]Querying offline knowledge ({action})...[/cyan]")
        return query_service.query(args)

    def spoken_fallback(result: dict, args: dict) -> str:
        if not result.get("ok"):
            if "no offline knowledge packs" in str(result.get("error", "")):
                return (
                    "I don't have any offline knowledge packs installed yet. "
                    "Ask me to download Simple English Wikipedia first."
                )
            return "Sorry, I couldn't find that in the offline knowledge packs."
        if result.get("action") == "get" and result.get("text"):
            title = result.get("title") or "that article"
            cite = result.get("cite") or "According to the offline knowledge pack"
            return f"{cite}, I found details about {title}."
        hits = result.get("hits") or []
        if hits:
            titles = ", ".join(h.get("title", "result") for h in hits[:3])
            cite = result.get("cite") or "According to the offline knowledge packs"
            return f"{cite}, I found these results: {titles}."
        return "I searched the offline knowledge packs but found no matching articles."

    pack_enum = [*pack_ids(), "all"]
    description = build_tool_description(
        QUERY_TOOL_NAME,
        (
            "Search or read offline knowledge packs already downloaded into the local "
            "cache (Kiwix ZIM archives such as Simple English Wikipedia). Use this for "
            "factual, school, dictionary, or encyclopedia questions when offline packs "
            "may help. action='search' returns matching titles; action='get' returns "
            "article text. If no packs are installed, tell the user and call "
            f"acquire_knowledge (default {DEFAULT_PACK_ID}). Prefer search then get. "
            "When you answer from these results, you MUST cite the source in natural "
            "spoken English using the cite / citation.spoken_prefix field from the tool "
            "result (for example: 'According to Simple English Wikipedia, ...'). "
            "Do not read URLs or pack ids aloud."
        ),
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "get"],
                    "description": "search for matching articles, or get a specific article by title/path",
                },
                "query": {
                    "type": "string",
                    "description": "Search terms, or article title/path when action is get",
                },
                "pack": {
                    "type": "string",
                    "enum": pack_enum,
                    "description": "Optional pack id to restrict the query; omit or 'all' to search installed packs",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Max search hits (default 3)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    return ToolSpec(
        name=QUERY_TOOL_NAME,
        description=description,
        handler=handler,
        spoken_fallback=spoken_fallback,
    )
