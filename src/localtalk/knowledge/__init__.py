"""Offline knowledge packs (Kiwix ZIM archives) for airplane-mode world knowledge."""

from localtalk.knowledge.packs import (
    DEFAULT_PACK_ID,
    KNOWLEDGE_PACKS,
    KnowledgePack,
    get_pack,
    list_packs,
)
from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore, get_default_store

__all__ = [
    "DEFAULT_PACK_ID",
    "KNOWLEDGE_PACKS",
    "KnowledgePack",
    "KnowledgeQueryService",
    "KnowledgeStore",
    "get_default_store",
    "get_pack",
    "list_packs",
]
