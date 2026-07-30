"""Unit tests for Playwright browser security + Harmony tools (mocked session)."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from localtalk.models.config import BrowserToolsConfig
from localtalk.services.browser.security import validate_public_http_url
from localtalk.services.browser.session import BrowserSession, browser_engine_status, probe_cdp
from localtalk.services.tools.browser import (
    CLICK,
    CLOSE,
    EXTRACT,
    NAVIGATE,
    SNAPSHOT,
    TYPE,
    make_browser_tools,
)

pytestmark = pytest.mark.unit


class TestValidatePublicHttpUrl:
    def test_https_ok(self):
        ok, err = validate_public_http_url("https://example.com/path")
        assert ok is True
        assert err is None

    def test_http_ok(self):
        ok, err = validate_public_http_url("http://example.com")
        assert ok is True

    def test_rejects_file(self):
        ok, err = validate_public_http_url("file:///etc/passwd")
        assert ok is False
        assert "scheme" in (err or "")

    def test_rejects_javascript(self):
        ok, err = validate_public_http_url("javascript:alert(1)")
        assert ok is False

    def test_rejects_localhost(self):
        ok, err = validate_public_http_url("http://localhost:8080/")
        assert ok is False
        assert "blocked" in (err or "").lower()

    def test_rejects_loopback_ip(self):
        ok, err = validate_public_http_url("http://127.0.0.1/")
        assert ok is False

    def test_rejects_private_ip(self):
        ok, err = validate_public_http_url("http://192.168.1.1/")
        assert ok is False

    def test_dns_failure_fails_closed(self, monkeypatch):
        def _raise_gaierror(*_a, **_k):
            raise socket.gaierror(8, "nodename nor servname provided")

        monkeypatch.setattr("localtalk.services.browser.security.socket.getaddrinfo", _raise_gaierror)
        ok, err = validate_public_http_url("https://unresolvable.invalid/")
        assert ok is False
        assert "resolve" in (err or "").lower()

    def test_empty(self):
        ok, err = validate_public_http_url("")
        assert ok is False


class TestBrowserSessionGuards:
    def test_navigate_blocked_without_playwright_launch(self):
        session = BrowserSession(BrowserToolsConfig(enabled=True, engine="chrome"))
        result = session.navigate("http://127.0.0.1/")
        assert result["ok"] is False
        assert "blocked" in result["error"].lower()

    def test_snapshot_requires_open_page(self):
        session = BrowserSession(BrowserToolsConfig())
        result = session.snapshot()
        assert result["ok"] is False
        assert "navigate" in result["error"]


class TestBrowserToolsRegistry:
    def test_make_browser_tools_names(self):
        session = MagicMock(spec=BrowserSession)
        specs = make_browser_tools(session)
        names = {s.name for s in specs}
        assert names == {NAVIGATE, SNAPSHOT, CLICK, TYPE, EXTRACT, CLOSE}

    def test_navigate_tool_dispatches(self):
        session = MagicMock(spec=BrowserSession)
        session.navigate.return_value = {"ok": True, "url": "https://example.com", "title": "Example"}
        specs = {s.name: s for s in make_browser_tools(session)}
        result = specs[NAVIGATE].handler({"url": "https://example.com"})
        session.navigate.assert_called_once_with("https://example.com")
        assert result["ok"] is True

    def test_click_tool_passes_selectors(self):
        session = MagicMock(spec=BrowserSession)
        session.click.return_value = {"ok": True}
        specs = {s.name: s for s in make_browser_tools(session)}
        specs[CLICK].handler({"text": "More", "role": None})
        session.click.assert_called_once()
        kwargs = session.click.call_args.kwargs
        assert kwargs["text"] == "More"


class TestBrowserToolsConfig:
    def test_defaults(self):
        cfg = BrowserToolsConfig()
        assert cfg.enabled is False
        assert cfg.engine == "chrome"
        assert cfg.attach is True  # CDP attach is default
        assert cfg.cdp_url == "http://127.0.0.1:9222"
        assert cfg.headed is False
        assert cfg.max_tool_rounds == 12

    def test_safari_engine(self):
        cfg = BrowserToolsConfig(engine="safari", enabled=True)
        assert cfg.engine == "safari"


class TestCdpProbe:
    def test_probe_cdp_success(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"Browser": "Chrome/140", "webSocketDebuggerUrl": "ws://x"}
        client = MagicMock()
        client.get.return_value = response
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        with patch("httpx.Client", return_value=client):
            result = probe_cdp("http://127.0.0.1:9222")
        assert result["ok"] is True
        assert "Chrome" in result["detail"]

    def test_probe_cdp_failure(self):
        client = MagicMock()
        client.get.side_effect = OSError("connection refused")
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        with patch("httpx.Client", return_value=client):
            result = probe_cdp("http://127.0.0.1:9222")
        assert result["ok"] is False


class TestBrowserSessionAttachClose:
    def test_close_attached_does_not_call_browser_close(self):
        session = BrowserSession(BrowserToolsConfig(attach=True))
        session._attached = True
        session._owns_page = True
        page = MagicMock()
        browser = MagicMock()
        playwright = MagicMock()
        session._page = page
        session._context = MagicMock()
        session._browser = browser
        session._playwright = playwright
        session.close()
        page.close.assert_called_once()
        browser.close.assert_not_called()
        playwright.stop.assert_called_once()
        assert session._attached is False
        assert session._page is None

    def test_browser_engine_status_prefers_cdp(self):
        import sys
        import types

        # Simulate playwright present without requiring the optional package
        fake_pw = types.ModuleType("playwright")
        with (
            patch(
                "localtalk.services.browser.session.probe_cdp",
                return_value={"ok": True, "detail": "CDP ready"},
            ),
            patch.dict(sys.modules, {"playwright": fake_pw}),
        ):
            status = browser_engine_status("chrome", attach=True, cdp_url="http://127.0.0.1:9222")
        assert status["ok"] is True
        assert status["mode"] == "attach"
