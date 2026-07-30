"""web_search Harmony tool — Google (browser) first, Wikipedia last resort."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, quote_plus

from localtalk.knowledge.citation import attach_citations
from localtalk.services.browser.session import BrowserSession
from localtalk.services.tools.base import ToolSpec, build_tool_description
from localtalk.services.tools.online import ConnectivityCache

TOOL_NAME = "web_search"
_USER_AGENT = "localtalk/0.6 (+https://github.com/anthonywu/localtalk)"
_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Queries that need live web results — never satisfied by Wikipedia alone.
_LIVE_RE = re.compile(
    r"\b("
    r"weather|forecast|temperature|humidity|rain|snow|storm|"
    r"news|headline|score|scores|stock|stocks|price|prices|"
    r"today|tonight|tomorrow|yesterday|this\s+week|right\s+now|currently|"
    r"live|breaking"
    r")\b",
    re.I,
)
_WEATHER_RE = re.compile(r"\b(weather|forecast|temperature|humidity|rain|snow|storm)\b", re.I)
_PLACE_STRIP_RE = re.compile(
    r"\b("
    r"weather|forecast|temperature|for|in|at|the|a|an|today|tonight|"
    r"tomorrow|yesterday|what|is|will|be|like|show|me|look|up|can|you|"
    r"please|current|right|now"
    r")\b",
    re.I,
)


def is_live_query(query: str) -> bool:
    return bool(_LIVE_RE.search(query or ""))


def is_weather_query(query: str) -> bool:
    return bool(_WEATHER_RE.search(query or ""))


def _place_from_weather_query(query: str) -> str:
    """Best-effort location phrase from a weather question."""
    cleaned = _PLACE_STRIP_RE.sub(" ", query or "")
    cleaned = re.sub(r"[^\w\s,.-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned or "San Francisco"


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
        titles = list(data[1]) if isinstance(data, list) and len(data) > 1 else []
        urls = list(data[3]) if isinstance(data, list) and len(data) > 3 else []

        hits: list[dict[str, Any]] = []
        for idx, title in enumerate(titles[:max_results]):
            url = urls[idx] if idx < len(urls) else None
            snippet = ""
            try:
                summary_url = _SUMMARY.format(title=quote(title.replace(" ", "_"), safe="()_"))
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
                    "source_label": "English Wikipedia online",
                    "url": url,
                    "pack_id": None,
                }
            )

    return attach_citations(
        {
            "ok": True,
            "query": query,
            "count": len(hits),
            "hits": hits,
            "backend": "wikipedia",
        }
    )


def weather_lookup_via_browser(
    session: BrowserSession,
    query: str,
    *,
    console_print=None,
) -> dict[str, Any]:
    """Open the browser on wttr.in and read the visible forecast."""
    place = _place_from_weather_query(query)
    path = quote(place.replace(" ", "+"), safe="+")
    url = f"https://wttr.in/{path}?T"
    result = visual_read_url(
        session,
        url,
        title=f"Weather for {place}",
        query=query,
        headed=True,
        console_print=console_print,
        backend="wttr.in",
    )
    if result.get("ok"):
        result["place"] = place
        text = (result.get("hits") or [{}])[0].get("snippet") or ""
        if "unknown location" in text.lower():
            return {"ok": False, "error": f"no weather data for {place!r}", "backend": "wttr.in"}
    return result


_BLOCKED_PAGE_RE = re.compile(
    r"(unusual traffic|captcha|before you continue to google|"
    r"enable javascript|consent\.google|verify you.?re not a robot|"
    r"our systems have detected)",
    re.I,
)

# Vendor pages for common product-price questions when SERPs fail.
_PRODUCT_PAGE_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"macbook\s*pro", re.I), "https://www.apple.com/macbook-pro/", "Apple MacBook Pro"),
    (re.compile(r"macbook\s*air", re.I), "https://www.apple.com/macbook-air/", "Apple MacBook Air"),
    (re.compile(r"\biphone\b", re.I), "https://www.apple.com/iphone/", "Apple iPhone"),
    (re.compile(r"\bipad\b", re.I), "https://www.apple.com/ipad/", "Apple iPad"),
    (re.compile(r"\bmac\s*mini\b", re.I), "https://www.apple.com/mac-mini/", "Apple Mac mini"),
    (re.compile(r"\bmac\s*studio\b", re.I), "https://www.apple.com/mac-studio/", "Apple Mac Studio"),
]


def _dismiss_consent_banners(page) -> None:
    """Best-effort click through Google/EU consent dialogs."""
    selectors = (
        "button#L2AGLb",  # Google "Accept all" (EU)
        'button:has-text("Accept all")',
        'button:has-text("I agree")',
        'button:has-text("Accept")',
        'button:has-text("Reject all")',  # sometimes still lands on results
    )
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=1200)
            page.wait_for_timeout(400)
            return
        except Exception:
            continue


def _page_body_text(page, timeout_ms: int) -> str:
    try:
        text = page.inner_text("body", timeout=timeout_ms) or ""
        return " ".join(text.split())
    except Exception:
        return ""


def _looks_blocked(text: str) -> bool:
    if not text:
        return True
    if _BLOCKED_PAGE_RE.search(text):
        return True
    # Consent walls are short and repetitive
    if len(text) < 280 and "cookie" in text.lower():
        return True
    return False


def google_search_via_browser(
    session: BrowserSession,
    query: str,
    *,
    max_results: int = 3,
    console_print=None,
) -> dict[str, Any]:
    """Run a Google web search in the local browser and scrape result cards + page text."""
    max_results = max(1, min(int(max_results), 5))
    # gbv=1 = basic HTML results (more scrape-friendly than the JS SERP)
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&gl=us&gbv=1"
    if console_print:
        console_print(f"[cyan]web_search → google.com query={query!r}[/cyan]")

    nav = session.navigate(url)
    if not nav.get("ok"):
        return {"ok": False, "error": nav.get("error") or "google navigate failed", "backend": "google"}

    page = session._page  # intentional: scrape SERP structure
    if page is None:
        return {"ok": False, "error": "no browser page", "backend": "google"}

    _dismiss_consent_banners(page)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass
    try:
        page.wait_for_timeout(800)
    except Exception:
        pass

    hits: list[dict[str, Any]] = []
    # Basic HTML (gbv=1) uses simpler anchors; modern SERP uses h3 cards.
    scrape_scripts = (
        (
            "a",
            """(els) => els
                .filter(a => a.querySelector('h3') || (a.textContent||'').length > 20)
                .slice(0, 12)
                .map(a => {
                    const h3 = a.querySelector('h3');
                    const title = (h3 ? h3.innerText : a.innerText || '').trim().slice(0, 200);
                    return { title, url: a.href || null, snippet: '' };
                })
                .filter(h => h.title && h.url && !h.url.includes('google.com/search'))""",
        ),
        (
            "div.g, div[data-sokoban-container], div.ezO2md",
            """(els) => els.slice(0, 10).map(el => {
                const h3 = el.querySelector('h3');
                const a = el.querySelector('a[href]');
                const snip = el.querySelector('.VwiC3b, .yXK7lf, [data-sncf], span.st, div[style*=\"-webkit-line-clamp\"]');
                return {
                    title: (h3 && h3.innerText || '').trim().slice(0, 200),
                    url: (a && a.href) || null,
                    snippet: (snip && snip.innerText || '').trim().slice(0, 500),
                };
            }).filter(h => h.title)""",
        ),
    )
    for selector, script in scrape_scripts:
        if hits:
            break
        try:
            raw_hits = page.eval_on_selector_all(selector, script)
            for item in (raw_hits or [])[:max_results]:
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                hits.append(
                    {
                        "title": title,
                        "snippet": (item.get("snippet") or "").strip(),
                        "source": "web",
                        "source_label": "Google Search",
                        "url": item.get("url"),
                        "pack_id": None,
                    }
                )
        except Exception:
            continue

    panel_text = _page_body_text(page, session.config.navigation_timeout_ms)
    if _looks_blocked(panel_text) and not hits:
        return {
            "ok": False,
            "error": "google page blocked (consent, captcha, or empty shell)",
            "backend": "google",
            "url": session._page.url if session._page else url,
        }

    # Body text is often enough for prices/widgets even when cards fail to parse.
    if panel_text and not _looks_blocked(panel_text) and len(panel_text) > 120:
        hits.insert(
            0,
            {
                "title": f"Google results for {query}",
                "snippet": panel_text[:1800],
                "source": "web",
                "source_label": "Google Search",
                "url": url,
                "pack_id": None,
            },
        )
        hits = hits[: max(max_results + 1, 2)]

    if not hits:
        return {
            "ok": False,
            "error": "google search returned no parseable results",
            "backend": "google",
        }

    return attach_citations(
        {
            "ok": True,
            "query": query,
            "count": len(hits),
            "hits": hits,
            "backend": "google",
        }
    )


def visual_read_url(
    session: BrowserSession,
    url: str,
    *,
    title: str | None = None,
    query: str = "",
    headed: bool = True,
    console_print=None,
    backend: str = "browser_visual",
) -> dict[str, Any]:
    """Open Chrome/Safari via Playwright, render the page, read visible text (prices)."""
    if console_print:
        engine = session.config.engine
        mode = "visible window" if headed else "headless"
        console_print(f"[cyan]web_search → open {engine} ({mode}) and read {url}[/cyan]")

    nav = session.navigate(url, headed=headed, wait_ms=1200)
    if not nav.get("ok"):
        return {"ok": False, "error": nav.get("error") or "navigate failed", "backend": backend}

    read = session.read_visible_text(max_chars=4000)
    if not read.get("ok"):
        # Fallback to plain extract_text
        read = session.extract_text(max_chars=4000)
    if not read.get("ok"):
        return {"ok": False, "error": read.get("error") or "visual read failed", "backend": backend}

    text = (read.get("text") or "").strip()
    if len(text) < 40:
        return {"ok": False, "error": "page had little visible text", "backend": backend}

    prices = read.get("prices") or []
    label = title or read.get("title") or url
    hits = [
        {
            "title": label,
            "snippet": text[:2000],
            "source": "web",
            "source_label": label,
            "url": read.get("url") or url,
            "pack_id": None,
            "prices": prices,
        }
    ]
    return attach_citations(
        {
            "ok": True,
            "query": query,
            "count": 1,
            "hits": hits,
            "backend": backend,
            "engine": read.get("engine") or session.config.engine,
            "headed": read.get("headed", session.config.headed),
            "prices": prices,
        }
    )


def product_page_via_browser(
    session: BrowserSession,
    query: str,
    *,
    console_print=None,
    headed: bool = True,
) -> dict[str, Any]:
    """Open a known vendor product page in a real browser and read visible prices/text."""
    match = next((item for item in _PRODUCT_PAGE_HINTS if item[0].search(query)), None)
    if match is None:
        return {"ok": False, "error": "no product page hint for query", "backend": "product_page"}

    _pat, url, label = match
    return visual_read_url(
        session,
        url,
        title=label,
        query=query,
        headed=headed,
        console_print=console_print,
        backend="product_page",
    )


def duckduckgo_browser_search(
    session: BrowserSession,
    query: str,
    *,
    max_results: int = 3,
    console_print=None,
) -> dict[str, Any]:
    """DuckDuckGo HTML results via the local browser (better than blocked Google)."""
    max_results = max(1, min(int(max_results), 5))
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    if console_print:
        console_print(f"[cyan]web_search → duckduckgo.com (browser) query={query!r}[/cyan]")

    nav = session.navigate(url)
    if not nav.get("ok"):
        return {"ok": False, "error": nav.get("error") or "ddg navigate failed", "backend": "duckduckgo"}

    page = session._page
    if page is None:
        return {"ok": False, "error": "no browser page", "backend": "duckduckgo"}

    hits: list[dict[str, Any]] = []
    try:
        raw_hits = page.eval_on_selector_all(
            "div.result, div.links_main, div.web-result",
            """(els) => els.slice(0, 10).map(el => {
                const a = el.querySelector('a.result__a, a[href]');
                const snip = el.querySelector('.result__snippet, a.result__snippet');
                return {
                    title: (a && a.innerText || '').trim().slice(0, 200),
                    url: (a && a.href) || null,
                    snippet: (snip && snip.innerText || '').trim().slice(0, 500),
                };
            }).filter(h => h.title)""",
        )
        for item in (raw_hits or [])[:max_results]:
            hits.append(
                {
                    "title": item.get("title") or "Result",
                    "snippet": item.get("snippet") or "",
                    "source": "web",
                    "source_label": "DuckDuckGo",
                    "url": item.get("url"),
                    "pack_id": None,
                }
            )
    except Exception:
        hits = []

    body = _page_body_text(page, session.config.navigation_timeout_ms)
    if body and len(body) > 120 and not _looks_blocked(body):
        hits.insert(
            0,
            {
                "title": f"DuckDuckGo results for {query}",
                "snippet": body[:1800],
                "source": "web",
                "source_label": "DuckDuckGo",
                "url": url,
                "pack_id": None,
            },
        )
        hits = hits[: max(max_results + 1, 2)]

    if not hits:
        return {"ok": False, "error": "duckduckgo browser search returned no results", "backend": "duckduckgo"}

    return attach_citations(
        {
            "ok": True,
            "query": query,
            "count": len(hits),
            "hits": hits,
            "backend": "duckduckgo",
        }
    )


def duckduckgo_lite_search(
    query: str,
    *,
    max_results: int = 3,
    timeout_s: float = 8.0,
    console_print=None,
) -> dict[str, Any]:
    """HTTP fallback via DuckDuckGo lite HTML (no browser, no API key)."""
    from html.parser import HTMLParser

    import httpx

    class _DDGParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.hits: list[dict[str, str]] = []
            self._in_result_link = False
            self._in_snippet = False
            self._cur_title = ""
            self._cur_url = ""
            self._cur_snip = ""

        def handle_starttag(self, tag, attrs):
            attrs_d = dict(attrs)
            cls = attrs_d.get("class", "")
            if tag == "a" and "result-link" in cls:
                self._in_result_link = True
                self._cur_title = ""
                self._cur_url = attrs_d.get("href") or ""
            elif tag == "td" and "result-snippet" in cls:
                self._in_snippet = True
                self._cur_snip = ""

        def handle_endtag(self, tag):
            if tag == "a" and self._in_result_link:
                self._in_result_link = False
            elif tag == "td" and self._in_snippet:
                self._in_snippet = False
                if self._cur_title:
                    self.hits.append(
                        {
                            "title": self._cur_title.strip(),
                            "url": self._cur_url,
                            "snippet": self._cur_snip.strip(),
                        }
                    )
                    self._cur_title = self._cur_url = self._cur_snip = ""

        def handle_data(self, data):
            if self._in_result_link:
                self._cur_title += data
            elif self._in_snippet:
                self._cur_snip += data

    max_results = max(1, min(int(max_results), 5))
    if console_print:
        console_print(f"[cyan]web_search → duckduckgo lite query={query!r}[/cyan]")

    with httpx.Client(timeout=timeout_s, headers={"User-Agent": _USER_AGENT}, follow_redirects=True) as client:
        response = client.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
        )
        response.raise_for_status()
        parser = _DDGParser()
        parser.feed(response.text)

    hits = [
        {
            "title": h["title"],
            "snippet": h["snippet"][:500],
            "source": "web",
            "source_label": "DuckDuckGo",
            "url": h["url"] or None,
            "pack_id": None,
        }
        for h in parser.hits[:max_results]
    ]
    if not hits:
        return {"ok": False, "error": "duckduckgo returned no results", "backend": "duckduckgo"}

    return attach_citations(
        {
            "ok": True,
            "query": query,
            "count": len(hits),
            "hits": hits,
            "backend": "duckduckgo",
        }
    )


def _suggested_urls(query: str) -> list[str]:
    urls = [f"https://www.google.com/search?q={quote_plus(query)}"]
    for pat, product_url, _label in _PRODUCT_PAGE_HINTS:
        if pat.search(query):
            urls.insert(0, product_url)
    if is_weather_query(query):
        place = _place_from_weather_query(query)
        urls.insert(0, f"https://wttr.in/{quote(place.replace(' ', '+'), safe='+')}?T")
    return urls[:4]


def run_web_search(
    query: str,
    *,
    max_results: int = 3,
    timeout_s: float = 8.0,
    browser_session: BrowserSession | None = None,
    console_print=None,
) -> dict[str, Any]:
    """Search the public web with a host-side fallback chain.

    Order:
    1. Weather → wttr.in (visible browser when possible)
    2. Product price queries → open vendor page in Chrome/Safari and read on-screen text
    3. Google Search (browser)
    4. DuckDuckGo HTML (browser)
    5. DuckDuckGo lite (HTTP)
    6. Wikipedia — non-live encyclopedic only
    """
    errors: list[str] = []
    live = is_live_query(query)
    # Price/product questions: open a real window so we can read the rendered page.
    want_visual = bool(re.search(r"\b(price|prices|cost|how much|msrp|\$)\b", query, re.I)) or any(
        pat.search(query) for pat, _u, _l in _PRODUCT_PAGE_HINTS
    )

    if browser_session is not None and is_weather_query(query):
        try:
            result = weather_lookup_via_browser(browser_session, query, console_print=console_print)
            if result.get("ok") and result.get("hits"):
                return result
            errors.append(f"wttr.in: {result.get('error')}")
        except Exception as exc:
            errors.append(f"wttr.in: {exc}")

    if browser_session is not None:
        # Product pages first for MacBook/iPhone/etc. prices — open headed Chrome/Safari
        # and read what is on screen (not a Wikipedia SERP).
        if want_visual or any(pat.search(query) for pat, _u, _l in _PRODUCT_PAGE_HINTS):
            try:
                result = product_page_via_browser(
                    browser_session,
                    query,
                    console_print=console_print,
                    headed=True,
                )
                if result.get("ok") and result.get("hits"):
                    return result
                if result.get("error") != "no product page hint for query":
                    errors.append(f"product_page: {result.get('error')}")
            except Exception as exc:
                errors.append(f"product_page: {exc}")

        try:
            # For price lookups, also force a visible Google window and read the SERP.
            if want_visual:
                ready = browser_session.ensure(headed=True)
                if not ready.get("ok"):
                    errors.append(f"browser: {ready.get('error')}")
            result = google_search_via_browser(
                browser_session,
                query,
                max_results=max_results,
                console_print=console_print,
            )
            if result.get("ok") and result.get("hits"):
                return result
            errors.append(f"google: {result.get('error')}")
        except Exception as exc:
            errors.append(f"google: {exc}")

        try:
            result = duckduckgo_browser_search(
                browser_session,
                query,
                max_results=max_results,
                console_print=console_print,
            )
            if result.get("ok") and result.get("hits"):
                return result
            errors.append(f"duckduckgo-browser: {result.get('error')}")
        except Exception as exc:
            errors.append(f"duckduckgo-browser: {exc}")

    try:
        result = duckduckgo_lite_search(
            query,
            max_results=max_results,
            timeout_s=timeout_s,
            console_print=console_print,
        )
        if result.get("ok") and result.get("hits"):
            return result
        errors.append(f"duckduckgo-http: {result.get('error')}")
    except Exception as exc:
        errors.append(f"duckduckgo-http: {exc}")

    # Wikipedia last, and only for non-live encyclopedic questions.
    if not live:
        try:
            result = wikipedia_search(
                query,
                max_results=max_results,
                timeout_s=timeout_s,
                console_print=console_print,
            )
            if result.get("ok") and result.get("hits"):
                return result
            errors.append(f"wikipedia: {result.get('error')}")
        except Exception as exc:
            errors.append(f"wikipedia: {exc}")
    else:
        errors.append("wikipedia: skipped for live/current/price query")

    suggestions = _suggested_urls(query)
    return {
        "ok": False,
        "error": "web_search failed on all backends",
        "detail": errors,
        "suggested_urls": suggestions,
        "hint": (
            "Call browser_navigate on one of suggested_urls, then browser_extract_text, "
            "and answer from the page. Do not just say you cannot find it."
        ),
    }


def make_web_search_tool(
    cache: ConnectivityCache,
    *,
    max_results_default: int = 3,
    timeout_s: float = 8.0,
    console_print=None,
    browser_session: BrowserSession | None = None,
    browser_session_getter=None,
) -> ToolSpec:
    """Build web_search. Pass browser_session or a zero-arg getter for lazy access."""

    def _session() -> BrowserSession | None:
        if browser_session_getter is not None:
            try:
                return browser_session_getter()
            except Exception:
                return browser_session
        return browser_session

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
            return run_web_search(
                query,
                max_results=max_results,
                timeout_s=timeout_s,
                browser_session=_session(),
                console_print=console_print,
            )
        except Exception as exc:
            return {"ok": False, "error": f"web_search failed: {exc}"}

    def spoken_fallback(result: dict, args: dict) -> str:
        if not result.get("ok"):
            if "offline" in str(result.get("error", "")).lower():
                return "I can't reach the internet right now, so I couldn't search the web."
            # Prefer next-step guidance over a dead-end apology
            suggestions = result.get("suggested_urls") or []
            if suggestions:
                return (
                    "The automatic search didn't return usable results. "
                    "I can open a product or search page next if you want me to keep going."
                )
            return "The web search didn't return usable results."
        hits = result.get("hits") or []
        if not hits:
            return "I searched the web but didn't get a clear snippet."
        snippet = (hits[0].get("snippet") or "").strip()
        if snippet:
            # Give TTS something factual when the model returns empty final text
            return snippet[:400]
        title = hits[0].get("title") or "a page"
        cite = result.get("cite") or "According to an online source"
        return f"{cite}, I found information about {title}."

    description = build_tool_description(
        TOOL_NAME,
        (
            "Search the public internet for an answer. Adult user on a private Mac — "
            "no topic bans; just look it up. Host routing: weather → wttr.in; product "
            "prices (MacBook, iPhone, etc.) → open system Chrome or Safari/WebKit via "
            "Playwright, load the vendor page, and read visible on-page text/prices; "
            "otherwise Google then DuckDuckGo in the browser. Wikipedia only for "
            "non-live encyclopedic facts. ALWAYS call this for weather, news, scores, "
            "prices, or today/tomorrow. Prefer query_knowledge only for stable offline "
            "pack facts. Cite using the cite field. Do not invent results."
        ),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in natural language"},
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
