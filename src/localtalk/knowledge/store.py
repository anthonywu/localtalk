"""Download and cache offline knowledge packs under the user home cache."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from localtalk.knowledge.packs import DEFAULT_PACK_ID, KnowledgePack, get_pack, list_packs

_KIWIX_CATALOG_ENTRIES = "https://library.kiwix.org/catalog/v2/entries"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "localtalk/0.5 (+https://github.com/anthonywu/localtalk)"
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class InstalledPack:
    """Metadata for a pack present in the local cache."""

    pack_id: str
    filename: str
    path: Path
    size_bytes: int
    downloaded_at: str
    source_url: str


@dataclass(frozen=True)
class ResolvedDownload:
    """Resolved remote ZIM location for a pack."""

    pack_id: str
    filename: str
    url: str
    size_bytes: int | None


def default_cache_dir() -> Path:
    """Return ``~/.cache/localtalk/knowledge`` (honors ``XDG_CACHE_HOME``)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "localtalk" / "knowledge"


class KnowledgeStore:
    """Manage offline knowledge pack downloads in the user cache."""

    def __init__(self, cache_dir: Path | None = None, console: Console | None = None):
        self.cache_dir = (cache_dir or default_cache_dir()).expanduser()
        self.console = console or Console()

    def ensure_cache_dir(self) -> Path:
        """Create the cache directory if needed and return it."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def manifest_path(self, pack_id: str) -> Path:
        """Path to the JSON sidecar describing an installed pack."""
        return self.cache_dir / f"{pack_id}.json"

    def get_installed(self, pack_id: str) -> InstalledPack | None:
        """Return install metadata if the pack is present on disk."""
        manifest = self.manifest_path(pack_id)
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        filename = data.get("filename")
        if not filename:
            return None
        path = self.cache_dir / filename
        if not path.is_file():
            return None
        return InstalledPack(
            pack_id=pack_id,
            filename=filename,
            path=path,
            size_bytes=int(data.get("size_bytes") or path.stat().st_size),
            downloaded_at=str(data.get("downloaded_at") or ""),
            source_url=str(data.get("source_url") or ""),
        )

    def list_installed(self) -> list[InstalledPack]:
        """Return all packs that are present in the cache."""
        installed: list[InstalledPack] = []
        for pack in list_packs():
            item = self.get_installed(pack.id)
            if item is not None:
                installed.append(item)
        return installed

    def status_payload(self) -> dict:
        """Build a JSON-serializable status of available and installed packs."""
        packs = []
        for pack in list_packs():
            installed = self.get_installed(pack.id)
            packs.append(
                {
                    "id": pack.id,
                    "title": pack.title,
                    "description": pack.description,
                    "approx_size": pack.approx_size_label,
                    "recommended": pack.recommended,
                    "default": pack.id == DEFAULT_PACK_ID,
                    "installed": installed is not None,
                    "path": str(installed.path) if installed else None,
                    "size_bytes": installed.size_bytes if installed else None,
                }
            )
        return {
            "ok": True,
            "cache_dir": str(self.cache_dir),
            "default_pack": DEFAULT_PACK_ID,
            "packs": packs,
        }

    def resolve_download(self, pack: KnowledgePack) -> ResolvedDownload:
        """Resolve the latest ZIM download URL for a pack via the Kiwix catalog."""
        query = f"{_KIWIX_CATALOG_ENTRIES}?name={pack.catalog_name}&count=50"
        request = urllib.request.Request(query, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                xml_text = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach the Kiwix catalog: {exc}") from exc

        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", _ATOM_NS):
            name = (entry.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            flavour = (entry.findtext("atom:flavour", default="", namespaces=_ATOM_NS) or "").strip()
            if name != pack.catalog_name or flavour != pack.flavour:
                continue

            zim_url: str | None = None
            size_bytes: int | None = None
            for link in entry.findall("atom:link", _ATOM_NS):
                if link.get("type") != "application/x-zim":
                    continue
                href = link.get("href") or ""
                if not href:
                    continue
                zim_url = href.removesuffix(".meta4")
                length = link.get("length")
                if length and length.isdigit():
                    size_bytes = int(length)
                break
            if not zim_url:
                continue

            filename = Path(urlparse(zim_url).path).name
            if not is_safe_zim_filename(filename):
                continue
            return ResolvedDownload(
                pack_id=pack.id,
                filename=filename,
                url=zim_url,
                size_bytes=size_bytes,
            )

        raise RuntimeError(
            f"No Kiwix ZIM found for pack {pack.id!r} "
            f"(catalog name={pack.catalog_name!r}, flavour={pack.flavour!r})"
        )

    def acquire(self, pack_id: str | None = None) -> dict:
        """List packs or download one into the home cache.

        Args:
            pack_id: Pack to download, ``"list"`` to only report status, or
                ``None``/empty to download the default pack.

        Returns:
            JSON-serializable result for the Harmony tool response.
        """
        if pack_id in (None, "", "default"):
            pack_id = DEFAULT_PACK_ID
        if pack_id == "list":
            return self.status_payload()

        pack = get_pack(pack_id)
        if pack is None:
            known = ", ".join(p.id for p in list_packs())
            return {
                "ok": False,
                "error": f"unknown pack {pack_id!r}; known packs: {known}; or pass pack='list'",
                "default_pack": DEFAULT_PACK_ID,
            }

        existing = self.get_installed(pack.id)
        if existing is not None:
            return {
                "ok": True,
                "already_installed": True,
                "pack_id": pack.id,
                "title": pack.title,
                "path": str(existing.path),
                "size_bytes": existing.size_bytes,
                "cache_dir": str(self.cache_dir),
                "message": f"{pack.title} is already installed in the local cache.",
            }

        self.ensure_cache_dir()
        try:
            resolved = self.resolve_download(pack)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc), "pack_id": pack.id}

        dest = self.cache_dir / resolved.filename
        partial = self.cache_dir / f"{resolved.filename}.partial"
        self.console.print(
            f"[cyan]Downloading {pack.title} ({pack.approx_size_label}) to {dest}...[/cyan]"
        )
        try:
            self._download_file(resolved.url, partial, expected_size=resolved.size_bytes)
            partial.replace(dest)
        except Exception as exc:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
            return {
                "ok": False,
                "error": f"download failed: {exc}",
                "pack_id": pack.id,
                "url": resolved.url,
            }

        size_bytes = dest.stat().st_size
        downloaded_at = datetime.now(UTC).isoformat()
        manifest = {
            "pack_id": pack.id,
            "title": pack.title,
            "filename": resolved.filename,
            "size_bytes": size_bytes,
            "downloaded_at": downloaded_at,
            "source_url": resolved.url,
            "catalog_name": pack.catalog_name,
            "flavour": pack.flavour,
        }
        self.manifest_path(pack.id).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self.console.print(f"[green]Installed {pack.title} → {dest}[/green]")
        return {
            "ok": True,
            "already_installed": False,
            "pack_id": pack.id,
            "title": pack.title,
            "path": str(dest),
            "size_bytes": size_bytes,
            "cache_dir": str(self.cache_dir),
            "source_url": resolved.url,
            "message": f"Downloaded {pack.title} into the local cache.",
        }

    def _download_file(self, url: str, dest: Path, expected_size: int | None = None) -> None:
        """Stream a remote file to ``dest`` with a Rich progress bar."""
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = expected_size
            if total is None:
                length_header = response.headers.get("Content-Length")
                if length_header and length_header.isdigit():
                    total = int(length_header)
            with (
                Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                ) as progress,
                dest.open("wb") as out,
            ):
                task_id = progress.add_task("Downloading", total=total)
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    progress.update(task_id, advance=len(chunk))


_default_store: KnowledgeStore | None = None


def get_default_store(console: Console | None = None) -> KnowledgeStore:
    """Return a process-wide default store (optionally sharing a console)."""
    global _default_store
    if _default_store is None or (console is not None and _default_store.console is not console):
        _default_store = KnowledgeStore(console=console)
    return _default_store


_SAFE_ZIM_NAME = re.compile(r"^[A-Za-z0-9._-]+\.zim$")


def is_safe_zim_filename(name: str) -> bool:
    """Return True if ``name`` looks like a safe ZIM filename."""
    return bool(_SAFE_ZIM_NAME.match(name))
