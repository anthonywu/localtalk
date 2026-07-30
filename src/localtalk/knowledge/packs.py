"""Catalog of offline knowledge packs available for download."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgePack:
    """A downloadable offline knowledge pack (Kiwix ZIM archive)."""

    id: str
    title: str
    description: str
    catalog_name: str
    flavour: str
    approx_size_label: str
    recommended: bool = False


# Pack IDs are stable API names used by the acquire_knowledge tool.
# catalog_name/flavour map to the Kiwix OPDS catalog for resolving the latest ZIM URL.
DEFAULT_PACK_ID = "wikipedia_en_simple_all_nopic"

KNOWLEDGE_PACKS: tuple[KnowledgePack, ...] = (
    KnowledgePack(
        id=DEFAULT_PACK_ID,
        title="Simple English Wikipedia",
        description=(
            "Wikipedia written in simple English without pictures. "
            "Best first pack for school-age learners and general offline reference."
        ),
        catalog_name="wikipedia_en-simple_all",
        flavour="nopic",
        approx_size_label="about 1 gigabyte",
        recommended=True,
    ),
    KnowledgePack(
        id="wikipedia_en_top_nopic",
        title="Best of English Wikipedia",
        description=(
            "A curated selection of the most popular English Wikipedia articles without pictures. "
            "Deeper coverage for older students and general knowledge."
        ),
        catalog_name="wikipedia_en_top",
        flavour="nopic",
        approx_size_label="about 2 gigabytes",
    ),
    KnowledgePack(
        id="wiktionary_en_simple_all_nopic",
        title="Simple English Wiktionary",
        description=(
            "A compact dictionary in simple English. "
            "Useful for definitions, spelling, and word meanings."
        ),
        catalog_name="wiktionary_en_simple_all",
        flavour="nopic",
        approx_size_label="about 25 megabytes",
    ),
    KnowledgePack(
        id="wikipedia_en_physics_nopic",
        title="Wikipedia Physics",
        description="A selection of English Wikipedia articles on physics, without pictures.",
        catalog_name="wikipedia_en_physics",
        flavour="nopic",
        approx_size_label="about 300 megabytes",
    ),
)

_PACKS_BY_ID: dict[str, KnowledgePack] = {pack.id: pack for pack in KNOWLEDGE_PACKS}


def get_pack(pack_id: str) -> KnowledgePack | None:
    """Return a pack by id, or None if unknown."""
    return _PACKS_BY_ID.get(pack_id)


def list_packs() -> list[KnowledgePack]:
    """Return all known packs in catalog order."""
    return list(KNOWLEDGE_PACKS)


def pack_ids() -> list[str]:
    """Return stable pack ids for tool schemas."""
    return [pack.id for pack in KNOWLEDGE_PACKS]
