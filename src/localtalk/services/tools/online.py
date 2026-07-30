"""macOS connectivity detection and check_online Harmony tool."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from localtalk.services.tools.base import ToolSpec, build_tool_description

TOOL_NAME = "check_online"
PrimaryKind = Literal["wifi", "ethernet", "other", "none"]


@dataclass
class InterfaceInfo:
    device: str
    type: PrimaryKind
    ipv4: bool = False
    ipv6: bool = False
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "type": self.type,
            "ipv4": self.ipv4,
            "ipv6": self.ipv6,
            "active": self.active,
        }


@dataclass
class NetworkStatus:
    online: bool
    reachable: bool
    primary: PrimaryKind
    interfaces: list[InterfaceInfo] = field(default_factory=list)
    probe: dict[str, Any] | None = None
    error: str | None = None
    checked_at: float = field(default_factory=time.time)

    def to_tool_result(self) -> dict[str, Any]:
        return {
            "ok": self.error is None,
            "online": self.online,
            "reachable": self.reachable,
            "primary": self.primary,
            "interfaces": [i.to_dict() for i in self.interfaces],
            "probe": self.probe,
            "error": self.error,
        }


def _run(cmd: list[str], timeout: float = 2.0) -> str:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _hardware_port_map() -> dict[str, PrimaryKind]:
    """Map device (en0) → wifi/ethernet from networksetup."""
    out = _run(["/usr/sbin/networksetup", "-listallhardwareports"])
    mapping: dict[str, PrimaryKind] = {}
    current_port = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            if "wi-fi" in current_port or "wifi" in current_port or "airport" in current_port:
                mapping[device] = "wifi"
            elif any(k in current_port for k in ("ethernet", "usb lan", "thunderbolt", "usb 10/100")):
                mapping[device] = "ethernet"
            else:
                mapping[device] = "other"
            current_port = ""
    return mapping


def _parse_nwi_better(text: str, port_map: dict[str, PrimaryKind]) -> list[InterfaceInfo]:
    """Parse ``scutil --nwi`` output into interface records."""
    by_dev: dict[str, InterfaceInfo] = {}
    section: str | None = None  # ipv4 / ipv6
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if "ipv4 network interface information" in lower:
            section = "ipv4"
            continue
        if "ipv6 network interface information" in lower:
            section = "ipv6"
            continue
        if "network information" == lower or stripped.startswith("REACH:"):
            section = None
            continue
        # Device line: "en0 : flags  ..."
        if ":" in stripped and not stripped.startswith(("Address", "Router", "DNS", "Orig", "Dest")):
            device = stripped.split(":", 1)[0].strip()
            if device and device.replace(".", "").replace("-", "").isalnum() and not device[0].isdigit():
                if device not in by_dev:
                    by_dev[device] = InterfaceInfo(
                        device=device,
                        type=port_map.get(device, "other"),
                        active=True,
                    )
                if section == "ipv4":
                    by_dev[device].ipv4 = True
                elif section == "ipv6":
                    by_dev[device].ipv6 = True
    return list(by_dev.values())


def _probe_reachability(url: str, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import httpx

        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            response = client.get(url)
        latency_ms = int((time.perf_counter() - started) * 1000)
        host = url.split("//", 1)[-1].split("/", 1)[0]
        ok = response.status_code in {200, 204}
        return {
            "attempted": True,
            "url_host": host,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "reachable": ok,
        }
    except Exception as exc:
        host = url.split("//", 1)[-1].split("/", 1)[0]
        return {
            "attempted": True,
            "url_host": host,
            "status_code": None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "reachable": False,
            "error": str(exc),
        }


def detect_connectivity(
    *,
    probe: bool = True,
    probe_timeout_s: float = 2.0,
    reachability_url: str = "https://connectivitycheck.gstatic.com/generate_204",
) -> NetworkStatus:
    """Detect macOS network interfaces and optionally probe internet reachability."""
    port_map = _hardware_port_map()
    nwi = _run(["/usr/sbin/scutil", "--nwi"])
    interfaces = _parse_nwi_better(nwi, port_map) if nwi else []

    # If scutil gave nothing useful, invent from port map + ifconfig-ish fallback:
    if not interfaces and port_map:
        for device, kind in port_map.items():
            interfaces.append(InterfaceInfo(device=device, type=kind, active=False))

    online = any(i.active and (i.ipv4 or i.ipv6) for i in interfaces) or bool(interfaces and nwi)
    # Refine online: any interface listed under IPv4/IPv6 sections counts
    online = any((i.ipv4 or i.ipv6) for i in interfaces) or (bool(nwi.strip()) and "No network" not in nwi)

    primary: PrimaryKind = "none"
    for kind in ("wifi", "ethernet", "other"):
        if any(i.type == kind and (i.ipv4 or i.ipv6 or i.active) for i in interfaces):
            primary = kind  # type: ignore[assignment]
            break
    if primary == "none" and interfaces:
        primary = interfaces[0].type

    probe_info = None
    reachable = False
    error = None
    if probe:
        probe_info = _probe_reachability(reachability_url, probe_timeout_s)
        reachable = bool(probe_info.get("reachable"))
        if not reachable and probe_info.get("error"):
            error = None  # soft: unreachable is a normal state, not a tool error
    else:
        reachable = online  # best-effort without probe

    return NetworkStatus(
        online=bool(online),
        reachable=reachable,
        primary=primary,
        interfaces=interfaces,
        probe=probe_info,
        error=error,
    )


def format_network_status_line(status: NetworkStatus, *, web_enabled: bool) -> str:
    """Human-readable startup status line."""
    if not status.online and not status.reachable:
        return "🌐 Network: offline — running fully local"
    medium = status.primary if status.primary != "none" else "network"
    if status.online and not status.reachable:
        return f"🌐 Network: interface up ({medium}) but internet unreachable — online tools unavailable"
    if status.reachable and web_enabled:
        return f"🌐 Network: online via {medium} — web search + browser tools available"
    if status.reachable and not web_enabled:
        return (
            f"🌐 Network: online via {medium} — online capabilities available "
            "(pass --enable-web for web search + browser)"
        )
    return f"🌐 Network: online via {medium}"


def format_privacy_banner_lines(status: NetworkStatus, *, web_enabled: bool) -> list[str]:
    """Status-aware privacy tip lines for the startup banner."""
    if not status.reachable:
        return [
            "✅ Everything runs 100% locally on your Mac",
            "✅ No tracking, no telemetry, no cloud APIs",
            "",
            "[yellow]📵 TIP: You can stay offline — LocalTalk works without WiFi.",
        ]
    if web_enabled:
        return [
            "✅ Core STT, LLM, and TTS still run locally on your Mac",
            "⚠️ Online tools enabled — web search and pages you browse leave this machine",
            "",
            "[yellow]TIP: Omit --enable-web next time to keep everything fully local.",
        ]
    return [
        "✅ Everything runs 100% locally on your Mac",
        "✅ No tracking, no telemetry, no cloud APIs (online tools off)",
        "",
        "[yellow]You're online, but online tools are off — pass --enable-web for web search + browser.",
    ]


class ConnectivityCache:
    """TTL cache for NetworkStatus shared by startup and the check_online tool."""

    def __init__(
        self,
        *,
        ttl_s: float = 45.0,
        probe_timeout_s: float = 2.0,
        reachability_url: str = "https://connectivitycheck.gstatic.com/generate_204",
    ):
        self.ttl_s = ttl_s
        self.probe_timeout_s = probe_timeout_s
        self.reachability_url = reachability_url
        self._status: NetworkStatus | None = None

    @property
    def status(self) -> NetworkStatus | None:
        return self._status

    def get(self, *, probe: bool = True, force: bool = False) -> NetworkStatus:
        if not force and self._status is not None and (time.time() - self._status.checked_at) < self.ttl_s:
            return self._status
        self._status = detect_connectivity(
            probe=probe,
            probe_timeout_s=self.probe_timeout_s,
            reachability_url=self.reachability_url,
        )
        return self._status

    def set(self, status: NetworkStatus) -> None:
        self._status = status


def make_check_online_tool(cache: ConnectivityCache) -> ToolSpec:
    def handler(args: dict) -> dict:
        probe = args.get("probe", True)
        if not isinstance(probe, bool):
            probe = str(probe).lower() not in {"false", "0", "no"}
        status = cache.get(probe=probe, force=True)
        return status.to_tool_result()

    def spoken_fallback(result: dict, args: dict) -> str:
        if not result.get("ok") and result.get("error"):
            return "Sorry, I couldn't check the network status."
        if result.get("reachable"):
            primary = result.get("primary") or "network"
            return f"Yes, you're online via {primary}."
        if result.get("online"):
            return "Your network interface is up, but the internet looks unreachable."
        return "You're offline right now."

    description = build_tool_description(
        TOOL_NAME,
        (
            "Check whether this Mac has a usable network path to the internet. "
            "Reports online/offline, primary interface type (wifi, ethernet, other, or none), "
            "and optionally a lightweight reachability probe. Use when the user asks about "
            "wifi/ethernet/online status, or before attempting an online lookup."
        ),
        {
            "type": "object",
            "properties": {
                "probe": {
                    "type": "boolean",
                    "description": (
                        "If true (default), also attempt a short HTTP reachability probe. "
                        "If false, only inspect local interface state."
                    ),
                },
            },
            "additionalProperties": False,
        },
    )
    return ToolSpec(name=TOOL_NAME, description=description, handler=handler, spoken_fallback=spoken_fallback)
