"""Playwright browser session: CDP attach to Chrome (default) or launch Chrome/WebKit."""

from __future__ import annotations

import threading
from typing import Any

from localtalk.models.config import BrowserToolsConfig
from localtalk.services.browser.security import validate_public_http_url


def probe_cdp(cdp_url: str, timeout_ms: int = 2000) -> dict[str, Any]:
    """Return {ok, detail/error} if something answers at the CDP endpoint."""
    try:
        import httpx
    except ImportError:
        return {"ok": False, "error": "httpx not installed"}

    base = cdp_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout_ms / 1000.0) as client:
            # Chrome exposes /json/version when remote debugging is enabled
            response = client.get(f"{base}/json/version")
            if response.status_code != 200:
                return {"ok": False, "error": f"CDP HTTP {response.status_code} at {base}/json/version"}
            data = response.json()
            browser = data.get("Browser") or data.get("webSocketDebuggerUrl") or "chrome"
            return {"ok": True, "detail": f"CDP ready ({browser})", "cdp_url": cdp_url, "version": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cdp_url": cdp_url}


def browser_engine_status(
    engine: str, *, attach: bool = True, cdp_url: str = "http://127.0.0.1:9222"
) -> dict[str, Any]:
    """Check whether Playwright / CDP look usable (no long-lived session)."""
    # Probe CDP first (HTTP only) so attach readiness works even before Playwright import.
    if engine == "chrome" and attach:
        cdp = probe_cdp(cdp_url)
        if cdp.get("ok"):
            try:
                import playwright  # noqa: F401
            except ImportError:
                return {
                    "ok": False,
                    "engine": "chrome",
                    "error": "CDP is up but playwright is not installed — uv pip install 'localtalk[browser]'",
                }
            return {
                "ok": True,
                "engine": "chrome",
                "mode": "attach",
                "detail": f"will attach to Chrome via CDP at {cdp_url} — {cdp.get('detail')}",
                "cdp_url": cdp_url,
            }
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {
                "ok": False,
                "engine": "chrome",
                "error": (
                    f"CDP not reachable at {cdp_url} ({cdp.get('error')}) and playwright not installed. "
                    "Enable Chrome remote debugging or: uv pip install 'localtalk[browser]'"
                ),
            }
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome", headless=True)
                browser.close()
            return {
                "ok": True,
                "engine": "chrome",
                "mode": "launch-fallback",
                "detail": (
                    f"CDP not reachable at {cdp_url} ({cdp.get('error')}); "
                    "will launch a separate Chrome. Enable remote debugging "
                    "(chrome://inspect → Allow remote debugging) for attach mode."
                ),
                "cdp_url": cdp_url,
            }
        except Exception as exc:
            return {
                "ok": False,
                "engine": "chrome",
                "error": (
                    f"CDP unavailable ({cdp.get('error')}) and Chrome launch failed: {exc}. "
                    "Install Chrome or enable remote debugging on port 9222."
                ),
            }

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "ok": False,
            "engine": engine,
            "error": "playwright not installed — uv pip install 'localtalk[browser]'",
        }

    try:
        with sync_playwright() as p:
            if engine == "chrome":
                browser = p.chromium.launch(channel="chrome", headless=True)
                browser.close()
                return {"ok": True, "engine": "chrome", "mode": "launch", "detail": "system Google Chrome (launch)"}
            browser = p.webkit.launch(headless=True)
            browser.close()
            return {
                "ok": True,
                "engine": "safari",
                "mode": "launch",
                "detail": "Playwright WebKit (Safari engine). If launch fails: playwright install webkit",
            }
    except Exception as exc:
        hint = " Try: playwright install webkit" if engine == "safari" else " Ensure Google Chrome is installed"
        return {"ok": False, "engine": engine, "error": f"{exc}.{hint}"}


class BrowserSession:
    """Lazy Playwright session: CDP attach (Chrome default) or launch."""

    def __init__(self, config: BrowserToolsConfig, console_print=None):
        self.config = config
        self._console_print = console_print
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._attached = False  # True when connected over CDP (do not kill user Chrome)
        self._owns_page = False  # True when we created the tab (safe to close on disconnect)

    def _log(self, message: str) -> None:
        if self._console_print:
            self._console_print(f"[cyan]{message}[/cyan]")

    @property
    def is_open(self) -> bool:
        return self._page is not None

    @property
    def is_attached(self) -> bool:
        return self._attached

    def ensure(self, *, headed: bool | None = None) -> dict[str, Any]:
        """Attach or launch browser + page if needed."""
        with self._lock:
            want_headed = self.config.headed if headed is None else bool(headed)
            if self._page is not None:
                # Attached Chrome is already "visible"; only relaunch when we own a headless launch
                if want_headed and not self.config.headed and not self._attached:
                    self._log("browser → relaunching headed for visual page read")
                    self._close_unlocked()
                    self.config.headed = True
                else:
                    return {
                        "ok": True,
                        "already_open": True,
                        "headed": self.config.headed,
                        "attached": self._attached,
                        **self._page_meta(),
                    }

            if headed is not None and not (self.config.attach and self.config.engine == "chrome"):
                self.config.headed = want_headed

            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                return {
                    "ok": False,
                    "error": "playwright not installed — uv pip install 'localtalk[browser]'",
                }

            try:
                self._playwright = sync_playwright().start()

                # Default for Chrome: attach over CDP (user's real browser / DevTools)
                if self.config.engine == "chrome" and self.config.attach:
                    attached = self._try_attach_cdp()
                    if attached.get("ok"):
                        return attached
                    self._log(f"browser → CDP attach failed ({attached.get('error')}); falling back to launch")

                return self._launch_browser()
            except Exception as exc:
                self._close_unlocked()
                return {"ok": False, "error": f"failed to open browser: {exc}"}

    def _try_attach_cdp(self) -> dict[str, Any]:
        """Connect to a running Chrome via Chrome DevTools Protocol."""
        assert self._playwright is not None
        cdp_url = (self.config.cdp_url or "http://127.0.0.1:9222").rstrip("/")
        self._log(f"browser → attaching to Chrome via CDP {cdp_url}")
        try:
            browser = self._playwright.chromium.connect_over_cdp(cdp_url, timeout=4000)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        self._browser = browser
        self._attached = True
        # Prefer existing default context (cookies / logins); open a fresh tab we own
        if browser.contexts:
            self._context = browser.contexts[0]
        else:
            self._context = browser.new_context(
                java_script_enabled=True,
                ignore_https_errors=False,
                viewport={"width": 1280, "height": 900},
            )
        self._page = self._context.new_page()
        self._owns_page = True
        self._page.set_default_timeout(self.config.navigation_timeout_ms)
        # Attached Chrome is inherently "headed" (user's window)
        self.config.headed = True
        self._log("browser → attached to Chrome (new tab; your session/cookies available)")
        return {
            "ok": True,
            "already_open": False,
            "engine": "chrome",
            "headed": True,
            "attached": True,
            "cdp_url": cdp_url,
            **self._page_meta(),
        }

    def _launch_browser(self) -> dict[str, Any]:
        """Launch a separate Chrome or WebKit instance (automation profile)."""
        assert self._playwright is not None
        headless = not self.config.headed
        engine = self.config.engine
        self._attached = False
        if engine == "chrome":
            mode = "headed" if self.config.headed else "headless"
            self._log(f"browser → launching system Chrome ({mode})")
            self._browser = self._playwright.chromium.launch(channel="chrome", headless=headless)
        else:
            mode = "headed" if self.config.headed else "headless"
            self._log(f"browser → launching WebKit/Safari engine ({mode})")
            self._browser = self._playwright.webkit.launch(headless=headless)
        self._context = self._browser.new_context(
            java_script_enabled=True,
            ignore_https_errors=False,
            viewport={"width": 1280, "height": 900},
        )
        self._page = self._context.new_page()
        self._owns_page = True
        self._page.set_default_timeout(self.config.navigation_timeout_ms)
        return {
            "ok": True,
            "already_open": False,
            "engine": self.config.engine,
            "headed": self.config.headed,
            "attached": False,
            **self._page_meta(),
        }

    def navigate(self, url: str, *, headed: bool | None = None, wait_ms: int = 0) -> dict[str, Any]:
        ok, err = validate_public_http_url(url)
        if not ok:
            return {"ok": False, "error": err}

        ready = self.ensure(headed=headed)
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
                try:
                    self._page.wait_for_load_state("networkidle", timeout=min(8000, self.config.navigation_timeout_ms))
                except Exception:
                    pass
                if wait_ms > 0:
                    try:
                        self._page.wait_for_timeout(wait_ms)
                    except Exception:
                        pass
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
                    "headed": self.config.headed,
                    "attached": self._attached,
                }
            except Exception as exc:
                return {"ok": False, "error": f"navigate failed: {exc}"}

    def read_visible_text(self, max_chars: int | None = None) -> dict[str, Any]:
        """Read what a human would roughly see: main content + price-like lines."""
        if not self.is_open:
            return {"ok": False, "error": "no browser page open — call browser_navigate first"}
        limit = max_chars if max_chars is not None else self.config.extract_max_chars
        limit = max(200, min(int(limit), 20000))

        with self._lock:
            assert self._page is not None
            self._log("browser → read visible text")
            page = self._page
            try:
                text = ""
                for sel in ("main", "article", "[role=main]", "body"):
                    try:
                        loc = page.locator(sel).first
                        if loc.count() > 0:
                            text = loc.inner_text(timeout=self.config.navigation_timeout_ms) or ""
                            if len(text.strip()) > 40:
                                break
                    except Exception:
                        continue

                price_bits: list[str] = []
                try:
                    price_bits = page.eval_on_selector_all(
                        "[class*='price'], [data-price], .price, .current_price, "
                        "span:has-text('$'), p:has-text('From $'), div:has-text('From $')",
                        """(els) => els.slice(0, 30).map(e => (e.innerText||'').trim())
                           .filter(t => t && t.length < 120 && /\\$|from/i.test(t))""",
                    )
                except Exception:
                    price_bits = []

                body = " ".join((text or "").split())
                prices = []
                for bit in price_bits or []:
                    cleaned = " ".join(str(bit).split())
                    if cleaned and cleaned not in prices:
                        prices.append(cleaned)

                parts: list[str] = []
                if prices:
                    parts.append("Prices on page: " + "; ".join(prices[:15]))
                if body:
                    parts.append(body)
                combined = "\n".join(parts).strip()
                if not combined:
                    return {"ok": False, "error": "page had no visible text"}

                truncated = len(combined) > limit
                if truncated:
                    combined = combined[: limit - 20] + " …[truncated]"
                return {
                    "ok": True,
                    "url": page.url,
                    "title": page.title(),
                    "text": combined,
                    "prices": prices[:15],
                    "truncated": truncated,
                    "engine": self.config.engine,
                    "headed": self.config.headed,
                    "attached": self._attached,
                }
            except Exception as exc:
                return {"ok": False, "error": f"read_visible_text failed: {exc}"}

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
                    "attached": self._attached,
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
                return {"ok": True, "attached": self._attached, **self._page_meta()}
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
                return {"ok": True, "submitted": submit, "attached": self._attached, **self._page_meta()}
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
                    "attached": self._attached,
                }
            except Exception as exc:
                return {"ok": False, "error": f"extract_text failed: {exc}"}

    def close(self) -> dict[str, Any]:
        with self._lock:
            return self._close_unlocked()

    def _close_unlocked(self) -> dict[str, Any]:
        """Disconnect or close browser resources.

        When attached over CDP, never call browser.close() — that can quit the user's Chrome.
        Only close the tab we opened, then stop the Playwright driver connection.
        """
        self._log("browser → " + ("disconnect CDP" if self._attached else "close"))
        errors: list[str] = []
        if self._attached:
            if self._owns_page and self._page is not None:
                try:
                    self._page.close()
                except Exception as exc:
                    errors.append(f"page: {exc}")
            # Do not close context/browser — shared with the user's Chrome
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception as exc:
                    errors.append(f"playwright: {exc}")
        else:
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
        self._attached = False
        self._owns_page = False
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
        assert self._page is not None
        parts: list[str] = [
            f"URL: {self._page.url}",
            f"Title: {self._page.title()}",
            "",
        ]
        try:
            aria = self._page.locator("body").aria_snapshot()
            if aria:
                parts.append("ARIA snapshot:")
                parts.append(aria)
                return "\n".join(parts)
        except Exception:
            pass

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
