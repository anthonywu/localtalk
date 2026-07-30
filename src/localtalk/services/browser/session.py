"""Playwright-backed browser session using system Chrome or WebKit (Safari engine)."""

from __future__ import annotations

import threading
from typing import Any

from localtalk.models.config import BrowserToolsConfig
from localtalk.services.browser.security import validate_public_http_url


def browser_engine_status(engine: str) -> dict[str, Any]:
    """Check whether Playwright and the requested engine look usable (no full launch)."""
    try:
        import playwright  # noqa: F401
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "ok": False,
            "engine": engine,
            "error": "playwright not installed — uv pip install 'localtalk[browser]' or: uv pip install playwright",
        }

    try:
        with sync_playwright() as p:
            if engine == "chrome":
                # channel=chrome uses the installed Google Chrome app on macOS
                browser = p.chromium.launch(channel="chrome", headless=True)
                browser.close()
                return {"ok": True, "engine": "chrome", "detail": "system Google Chrome (Playwright channel)"}
            # safari → WebKit (Safari's engine; not Safari.app itself)
            browser = p.webkit.launch(headless=True)
            browser.close()
            return {
                "ok": True,
                "engine": "safari",
                "detail": "Playwright WebKit (Safari engine). If launch fails, run: playwright install webkit",
            }
    except Exception as exc:
        hint = ""
        if engine == "safari":
            hint = " Try: playwright install webkit"
        elif engine == "chrome":
            hint = " Ensure Google Chrome is installed at /Applications/Google Chrome.app"
        return {"ok": False, "engine": engine, "error": f"{exc}.{hint}"}


class BrowserSession:
    """Lazy, process-wide Playwright session (one page) shared by Harmony tools."""

    def __init__(self, config: BrowserToolsConfig, console_print=None):
        self.config = config
        self._console_print = console_print
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _log(self, message: str) -> None:
        if self._console_print:
            self._console_print(f"[cyan]{message}[/cyan]")

    @property
    def is_open(self) -> bool:
        return self._page is not None

    def ensure(self) -> dict[str, Any]:
        """Launch browser + page if needed. Returns {ok, ...}."""
        with self._lock:
            if self._page is not None:
                return {"ok": True, "already_open": True, **self._page_meta()}
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                return {
                    "ok": False,
                    "error": "playwright not installed — uv pip install 'localtalk[browser]'",
                }

            try:
                self._playwright = sync_playwright().start()
                headless = not self.config.headed
                if self.config.engine == "chrome":
                    self._log("browser → launching system Chrome")
                    self._browser = self._playwright.chromium.launch(channel="chrome", headless=headless)
                else:
                    self._log("browser → launching WebKit (Safari engine)")
                    self._browser = self._playwright.webkit.launch(headless=headless)
                self._context = self._browser.new_context(
                    java_script_enabled=True,
                    ignore_https_errors=False,
                )
                self._page = self._context.new_page()
                self._page.set_default_timeout(self.config.navigation_timeout_ms)
                return {"ok": True, "already_open": False, "engine": self.config.engine, **self._page_meta()}
            except Exception as exc:
                self.close()
                return {"ok": False, "error": f"failed to launch browser: {exc}"}

    def navigate(self, url: str) -> dict[str, Any]:
        ok, err = validate_public_http_url(url)
        if not ok:
            return {"ok": False, "error": err}

        ready = self.ensure()
        if not ready.get("ok"):
            return ready

        with self._lock:
            assert self._page is not None
            self._log(f"browser → navigate {url}")
            try:
                response = self._page.goto(
                    url, wait_until="domcontentloaded", timeout=self.config.navigation_timeout_ms
                )
                status = response.status if response is not None else None
                # Re-validate final URL after redirects
                final = self._page.url
                ok_final, err_final = validate_public_http_url(final)
                if not ok_final:
                    self._page.goto("about:blank")
                    return {"ok": False, "error": f"navigation blocked after redirect: {err_final}", "url": final}
                return {
                    "ok": True,
                    "url": final,
                    "title": self._page.title(),
                    "status": status,
                    "engine": self.config.engine,
                }
            except Exception as exc:
                return {"ok": False, "error": f"navigate failed: {exc}"}

    def snapshot(self, max_chars: int | None = None) -> dict[str, Any]:
        if not self.is_open:
            return {"ok": False, "error": "no browser page open — call browser_navigate first"}
        limit = max_chars if max_chars is not None else self.config.snapshot_max_chars
        limit = max(500, min(int(limit), 20000))

        with self._lock:
            assert self._page is not None
            self._log("browser → snapshot")
            try:
                text = self._build_snapshot_text()
                truncated = len(text) > limit
                if truncated:
                    text = text[: limit - 20] + "\n…[truncated]"
                return {
                    "ok": True,
                    "url": self._page.url,
                    "title": self._page.title(),
                    "snapshot": text,
                    "truncated": truncated,
                    "engine": self.config.engine,
                }
            except Exception as exc:
                return {"ok": False, "error": f"snapshot failed: {exc}"}

    def click(
        self,
        *,
        text: str | None = None,
        role: str | None = None,
        name: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_open:
            return {"ok": False, "error": "no browser page open — call browser_navigate first"}

        with self._lock:
            assert self._page is not None
            try:
                locator = self._resolve_locator(text=text, role=role, name=name, selector=selector)
                self._log(f"browser → click {self._locator_label(text, role, name, selector)}")
                locator.first.click(timeout=self.config.navigation_timeout_ms)
                return {"ok": True, **self._page_meta()}
            except Exception as exc:
                return {"ok": False, "error": f"click failed: {exc}"}

    def type_text(
        self,
        text: str,
        *,
        role: str | None = None,
        name: str | None = None,
        selector: str | None = None,
        submit: bool = False,
    ) -> dict[str, Any]:
        if not self.is_open:
            return {"ok": False, "error": "no browser page open — call browser_navigate first"}
        if not text:
            return {"ok": False, "error": "text is required"}

        with self._lock:
            assert self._page is not None
            try:
                if role or name or selector:
                    locator = self._resolve_locator(text=None, role=role, name=name, selector=selector)
                    self._log("browser → type into element")
                    locator.first.fill(text, timeout=self.config.navigation_timeout_ms)
                else:
                    self._log("browser → type into focused element / keyboard")
                    self._page.keyboard.type(text)
                if submit:
                    self._page.keyboard.press("Enter")
                return {"ok": True, "submitted": submit, **self._page_meta()}
            except Exception as exc:
                return {"ok": False, "error": f"type failed: {exc}"}

    def extract_text(self, max_chars: int | None = None) -> dict[str, Any]:
        if not self.is_open:
            return {"ok": False, "error": "no browser page open — call browser_navigate first"}
        limit = max_chars if max_chars is not None else self.config.extract_max_chars
        limit = max(200, min(int(limit), 20000))

        with self._lock:
            assert self._page is not None
            self._log("browser → extract_text")
            try:
                body = self._page.inner_text("body", timeout=self.config.navigation_timeout_ms)
                body = " ".join(body.split())
                truncated = len(body) > limit
                if truncated:
                    body = body[: limit - 20] + " …[truncated]"
                return {
                    "ok": True,
                    "url": self._page.url,
                    "title": self._page.title(),
                    "text": body,
                    "truncated": truncated,
                    "engine": self.config.engine,
                }
            except Exception as exc:
                return {"ok": False, "error": f"extract_text failed: {exc}"}

    def close(self) -> dict[str, Any]:
        with self._lock:
            self._log("browser → close")
            errors: list[str] = []
            for label, closer in (
                ("page", lambda: self._page.close() if self._page else None),
                ("context", lambda: self._context.close() if self._context else None),
                ("browser", lambda: self._browser.close() if self._browser else None),
                ("playwright", lambda: self._playwright.stop() if self._playwright else None),
            ):
                try:
                    closer()
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            if errors:
                return {"ok": True, "closed": True, "warnings": errors}
            return {"ok": True, "closed": True}

    def _page_meta(self) -> dict[str, Any]:
        if self._page is None:
            return {"url": None, "title": None}
        try:
            return {"url": self._page.url, "title": self._page.title()}
        except Exception:
            return {"url": None, "title": None}

    def _resolve_locator(
        self,
        *,
        text: str | None,
        role: str | None,
        name: str | None,
        selector: str | None,
    ):
        assert self._page is not None
        if selector:
            return self._page.locator(selector)
        if role:
            kwargs = {}
            if name:
                kwargs["name"] = name
            return self._page.get_by_role(role, **kwargs)
        if text:
            return self._page.get_by_text(text, exact=False)
        raise ValueError("provide text, role, or selector")

    @staticmethod
    def _locator_label(text, role, name, selector) -> str:
        if selector:
            return f"selector={selector!r}"
        if role:
            return f"role={role!r} name={name!r}"
        return f"text={text!r}"

    def _build_snapshot_text(self) -> str:
        """Compact page observation for the local LLM (title, URL, ARIA/links)."""
        assert self._page is not None
        parts: list[str] = [
            f"URL: {self._page.url}",
            f"Title: {self._page.title()}",
            "",
        ]
        # Prefer ARIA snapshot when available (Playwright ≥1.49)
        try:
            aria = self._page.locator("body").aria_snapshot()
            if aria:
                parts.append("ARIA snapshot:")
                parts.append(aria)
                return "\n".join(parts)
        except Exception:
            pass

        # Fallback: headings + links + buttons
        try:
            links = self._page.eval_on_selector_all(
                "a[href]",
                "els => els.slice(0, 40).map(a => ({text: (a.innerText||'').trim().slice(0,80), href: a.href}))",
            )
            buttons = self._page.eval_on_selector_all(
                "button, [role=button], input[type=submit]",
                "els => els.slice(0, 20).map(b => (b.innerText||b.value||'').trim().slice(0,80)).filter(Boolean)",
            )
            headings = self._page.eval_on_selector_all(
                "h1,h2,h3",
                "els => els.slice(0, 15).map(h => ({tag: h.tagName, text: (h.innerText||'').trim().slice(0,120)}))",
            )
            if headings:
                parts.append("Headings:")
                for h in headings:
                    parts.append(f"  {h.get('tag')}: {h.get('text')}")
            if links:
                parts.append("Links:")
                for link in links:
                    parts.append(f"  - {link.get('text')!r} → {link.get('href')}")
            if buttons:
                parts.append("Buttons:")
                for b in buttons:
                    parts.append(f"  - {b!r}")
        except Exception as exc:
            parts.append(f"(fallback snapshot partial: {exc})")
            try:
                parts.append(self._page.inner_text("body")[:3000])
            except Exception:
                pass
        return "\n".join(parts)
