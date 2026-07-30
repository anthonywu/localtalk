"""Unit tests for macOS connectivity detection and check_online helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from localtalk.services.tools.online import (
    ConnectivityCache,
    InterfaceInfo,
    NetworkStatus,
    detect_connectivity,
    format_network_status_line,
    format_privacy_banner_lines,
    make_check_online_tool,
)

pytestmark = pytest.mark.unit

_NWI_WIFI = """
Network information

IPv4 network interface information
     en0 : flags      : 0x5 (IPv4,DNS)
         address    : 192.168.1.10

IPv6 network interface information
   No IPv6 states found.

   REACH : flags 0x00000002 (Reachable)
"""

_PORTS = """
Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:ff

Hardware Port: USB 10/100/1000 LAN
Device: en7
Ethernet Address: 11:22:33:44:55:66
"""


class TestDetectConnectivity:
    def test_wifi_online_reachable(self):
        with (
            patch("localtalk.services.tools.online._run", side_effect=[_PORTS, _NWI_WIFI]),
            patch(
                "localtalk.services.tools.online._probe_reachability",
                return_value={
                    "attempted": True,
                    "url_host": "connectivitycheck.gstatic.com",
                    "status_code": 204,
                    "latency_ms": 10,
                    "reachable": True,
                },
            ),
        ):
            status = detect_connectivity(probe=True)

        assert status.online is True
        assert status.reachable is True
        assert status.primary == "wifi"
        assert any(i.device == "en0" for i in status.interfaces)

    def test_probe_unreachable(self):
        with (
            patch("localtalk.services.tools.online._run", side_effect=[_PORTS, _NWI_WIFI]),
            patch(
                "localtalk.services.tools.online._probe_reachability",
                return_value={
                    "attempted": True,
                    "url_host": "connectivitycheck.gstatic.com",
                    "status_code": None,
                    "latency_ms": 5,
                    "reachable": False,
                    "error": "timeout",
                },
            ),
        ):
            status = detect_connectivity(probe=True)

        assert status.online is True
        assert status.reachable is False

    def test_offline(self):
        with patch("localtalk.services.tools.online._run", return_value=""):
            status = detect_connectivity(probe=False)
        assert status.reachable is False


class TestStatusFormatters:
    def test_startup_lines(self):
        online = NetworkStatus(online=True, reachable=True, primary="wifi")
        assert "web search + browser" in format_network_status_line(online, web_enabled=True)
        assert "--enable-web" in format_network_status_line(online, web_enabled=False)
        offline = NetworkStatus(online=False, reachable=False, primary="none")
        assert "offline" in format_network_status_line(offline, web_enabled=False)

    def test_privacy_banner(self):
        online = NetworkStatus(online=True, reachable=True, primary="ethernet")
        web_on = "\n".join(format_privacy_banner_lines(online, web_enabled=True))
        assert "Online tools enabled" in web_on
        web_off = "\n".join(format_privacy_banner_lines(online, web_enabled=False))
        assert "--enable-web" in web_off


class TestConnectivityCache:
    def test_ttl_reuse(self):
        cache = ConnectivityCache(ttl_s=60.0)
        canned = NetworkStatus(online=True, reachable=True, primary="wifi")
        with patch("localtalk.services.tools.online.detect_connectivity", return_value=canned) as detect:
            first = cache.get(probe=True, force=True)
            second = cache.get(probe=True, force=False)
        assert first is canned
        assert second is canned
        assert detect.call_count == 1


class TestCheckOnlineTool:
    def test_handler(self):
        cache = ConnectivityCache(ttl_s=60.0)
        cache.set(
            NetworkStatus(
                online=True,
                reachable=True,
                primary="wifi",
                interfaces=[InterfaceInfo(device="en0", type="wifi", ipv4=True, active=True)],
            )
        )
        tool = make_check_online_tool(cache)
        with patch.object(cache, "get", return_value=cache.status) as get:
            result = tool.handler({"probe": True})
        get.assert_called_once()
        assert result["ok"] is True
        assert result["primary"] == "wifi"
        assert "online via wifi" in tool.spoken_fallback(result, {})
