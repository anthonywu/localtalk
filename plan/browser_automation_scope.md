# Scope: Local browser driving (Playwright vs Stagehand)

Status: **decided — Playwright MVP in progress**  
Related: [[plan/web_tools_harmony.md]] (HTTP tools + connectivity)  
Branch context: `release-0.6.0`

## Decision (2026-07-30)

- **Backend:** Playwright only (Stagehand deferred / out of scope for default path).
- **Browsers:** assume user has **system Google Chrome** and **Safari** installed.
  - `chrome` → `chromium.launch(channel="chrome")`
  - `safari` → Playwright **WebKit** (Safari engine; not controlling Safari.app UI)
- **Opt-in:** `--enable-web` / `LOCALTALK_ENABLE_WEB=1` enables **both** `web_search` and browser tools (same product intent)
- **Engine flag:** `--browser-engine {chrome,safari}` (default chrome; only meaningful with `--enable-web`)
- **Headed:** `--browser-headed` (default headless)
- **Dep:** optional extra `localtalk[browser]` → `playwright`


## Question

What would it take for LocalTalk’s gpt-oss Harmony agent to **drive a real local browser** (navigate, click, type, read pages) using either:

1. **Playwright** (deterministic browser automation), or  
2. **Stagehand** (AI `act` / `extract` / `observe` / `agent` over a browser, local or Browserbase)?

This is **not** the same as `web_search` / `fetch_url` (stateless HTTP). Browser driving implies a **long-lived session**, interactive pages, JS-rendered DOM, and multi-step tool loops.

## Current LocalTalk hooks (already in tree)

As of this branch, the LLM side already has most of the agent plumbing:

| Piece | Today |
| --- | --- |
| Tool registry | `services/tools/base.py` (`ToolSpec`, `ToolRegistry`) |
| Multi-round tool loop | `MLXLanguageModelService._run_tool_loop` (`max_tool_rounds`) |
| Opt-in web tools | `web_search` (Wikipedia), connectivity cache |
| Knowledge tools | `acquire_knowledge` / query path (parallel work) |

Browser automation would be **another tool family** on the same registry — not a new agent framework.

What is **missing** for browsers:

- Process lifecycle (launch / reuse / kill Chromium)
- Session state across voice turns (open tabs, cookies, URL)
- Async-friendly or thread-isolated browser I/O (Playwright is sync/async; Stagehand is async/session-API)
- Observation format the **local** model can reason over (a11y tree / markdown / truncated HTML — not full screenshots unless we add vision)
- Timeouts that fit **voice** UX (users wait while “Looking that up…” is fine for 2–5s; 30–60s agent runs are a different product)
- Packaging: browser binary install story for `uv tool install localtalk`

## Options compared

### A. Playwright (raw) — **best fit for LocalTalk’s privacy pitch**

**What it is:** Microsoft’s browser automation library. Python package `playwright` + one-time `playwright install chromium` (or system Chrome via channel).

**Dry-run deps (current venv):**

```text
playwright==1.61.0
  + greenlet, pyee
# Plus separate browser download (~150–300MB Chromium)
```

**How the agent would use it:** gpt-oss never talks Playwright Python; it calls **Harmony tools** that wrap a shared `BrowserSession`:

| Tool | Args (sketch) | Returns |
| --- | --- | --- |
| `browser_open` | `url?`, `headless?` | `{ok, url, title}` |
| `browser_navigate` | `url` | `{ok, url, title, status}` |
| `browser_snapshot` | `max_chars?` | accessibility-tree or simplified markdown of current page |
| `browser_click` | `ref` or `selector` / `text` | `{ok, url, title}` |
| `browser_type` | `ref`, `text`, `submit?` | `{ok}` |
| `browser_extract_text` | `selector?`, `max_chars` | plain text for TTS summarization |
| `browser_close` | — | `{ok}` |

This mirrors the **Playwright MCP** tool shape (refs from a11y snapshots) without running an MCP server — keep it **in-process**.

**Pros**

- Fully local; no second cloud LLM
- Explicit, auditable actions (console can log every click/nav)
- Small direct dependency tree
- Aligns with offline-first / privacy banner once user opts in
- Deterministic enough to test with mocked page fixtures

**Cons**

- gpt-oss must plan multi-step UI control from snapshots (harder than “click the login button” in English)
- Brittle sites (heavy SPAs, captchas, shadow DOM) need good snapshot + ref design
- You own retry / wait-for / cookie / multi-tab logic
- Voice latency: each step ≈ 1 LLM round-trip + browser time → raise `max_tool_rounds` (e.g. 8–15 for browser mode)

**Effort (staff estimate)**

| Slice | Size | Notes |
| --- | --- | --- |
| Optional extra `playwright` + install docs | S | `optional-dependencies.browser` or `--enable-browser` |
| `BrowserSession` wrapper (sync, one context) | M | lifecycle, timeout, headless default, URL allowlist |
| Harmony tools + registry wiring | M | snapshot format is the hard part |
| Startup status (“browser ready / not installed”) | S | like network status line |
| Voice UX: “Browsing…” spinner / cancel | M | tool calls can exceed normal LLM latency |
| Security (SSRF, file://, private nets, downloads) | M | same class as `fetch_url`, stricter for browser |
| Tests (mocked Playwright) | M | unit without launching Chrome in CI |
| **Total MVP (navigate + snapshot + extract + close)** | **~1.5–2.5 eng weeks** | |
| **Polish (click/type refs, multi-tab, persistent profile)** | **+1–2 weeks** | |

### B. Stagehand (v3 Python) — **local browser, cloud-ish brain**

**What it is (2026):** Browserbase’s agent SDK. Primitives: `navigate`, `act`, `extract`, `observe`, `execute` (agent). Python package `stagehand` (v3.x on PyPI) is primarily a **client** to a Stagehand server.

**Local mode (documented):**

```python
client = Stagehand(server="local", model_api_key=model_key, ...)
session = client.sessions.start(
    model_name="anthropic/claude-sonnet-4-6",  # or other cloud model
    browser={"type": "local", "launchOptions": {"headless": True}},
)
```

- Spawns an **embedded SEA binary** + **local Chrome/Chromium** (not always bundled; may need system Chrome / `CHROME_PATH`)
- **`MODEL_API_KEY` is required** even in local browser mode — `act` / `extract` / `observe` use a **cloud LLM** (or whatever Stagehand routes via that key)
- `BROWSERBASE_API_KEY` can be a dummy string in pure local mode; real Browserbase is optional for cloud browsers

**Dry-run deps:**

```text
stagehand==3.22.0
  + anyio, distro, httpx, pydantic, sniffio, typing-extensions
# Local mode also needs Chrome + embedded SEA runtime from the wheel
# Older stagehand-py==0.3.x pulled playwright + browserbase explicitly (legacy)
```

**How LocalTalk would use it:** fewer Harmony tools, higher-level:

| Tool | Maps to Stagehand | Notes |
| --- | --- | --- |
| `browser_open` | `sessions.start` + optional `navigate` | long-lived session id |
| `browser_act` | `sessions.act(instruction=...)` | NL action; **calls Stagehand’s model** |
| `browser_extract` | `sessions.extract(instruction, schema)` | structured JSON for voice summary |
| `browser_agent` | `sessions.execute(...)` | multi-step autonomous; **high latency / cost** |
| `browser_close` | `sessions.end` + client close | must run on shutdown |

gpt-oss becomes a **supervisor**: decide *what* to do; Stagehand’s model decides *how* to click.

**Pros**

- Much better at messy real websites (self-healing, NL targets)
- Fewer tool rounds for the local model
- Path to Browserbase cloud browsers later (optional)

**Cons (serious for LocalTalk)**

- **Breaks “100% local / no cloud APIs”** for the browser path unless Stagehand can be pointed at a **local** OpenAI-compatible endpoint (needs explicit validation — not the documented default)
- Second model bill + latency on every `act`/`extract`
- Two-process architecture (SEA + Chrome) harder to package in `uv tool install`
- Async/session API vs LocalTalk’s mostly sync service methods → `asyncio.run` bridges or a worker thread
- Harder to unit-test offline; heavier CI
- Risk of user surprise: voice assistant “went online to Anthropic/OpenAI to click a button”

**Effort**

| Slice | Size | Notes |
| --- | --- | --- |
| Optional dep + env (`MODEL_API_KEY`) + docs | S–M | privacy banner must change when enabled |
| Session manager around Stagehand client | M | start/end, timeouts, headless |
| Harmony tools (`act` / `extract` / close) | S–M | thinner than Playwright tool surface |
| Dual-LLM cost/latency UX | M | progress logs; hard caps on `execute` |
| Local-only model backend experiment | L / unknown | only if product insists on no cloud for act |
| Tests | M–L | integration-heavy |
| **Total MVP (local Chrome + act + extract)** | **~1–2 eng weeks** | assuming cloud model key OK |
| **Fully local Stagehand (local LLM for act)** | **unknown / likely L** | research spike first |

### C. Hybrid (recommended product shape if both matter)

```text
--enable-browser          # Playwright session + discrete tools (default local path)
--enable-browser-ai       # Stagehand act/extract (requires MODEL_API_KEY); optional
```

Or single flag with backend:

```text
--browser-backend=playwright|stagehand
```

**Recommendation for LocalTalk identity:** ship **Playwright MVP first**. Treat Stagehand as an **opt-in advanced backend** for users who already accept cloud models, not as the default.

## Architecture sketch (Playwright-first)

```text
src/localtalk/services/
  browser/
    session.py       # BrowserSession: launch, page, snapshot, close
    snapshot.py      # a11y → compact text for gpt-oss
    security.py      # URL allow/deny, private IP, scheme checks
  tools/
    browser.py       # Harmony ToolSpecs bound to BrowserSession
```

Lifecycle:

1. User starts `localtalk --enable-browser`
2. Startup: verify Playwright + Chromium installed; print status line  
   `🧭 Browser: ready (Chromium headless)` / `not installed — run playwright install chromium`
3. First tool that needs a page → lazy-launch `BrowserSession` (avoid boot cost if unused)
4. Session **survives across voice turns** until `browser_close`, idle TTL, or process exit
5. Assistant shutdown / SIGINT → always close browser (like audio teardown)

Observation design (critical):

- Prefer **accessibility tree** or **ARIA snapshot** over raw HTML (token budget + less junk)
- Cap at ~4–8k chars; include interactive elements with stable **refs** (`e12`) for click/type
- Never dump full screenshots into gpt-oss unless a vision model is added later

Tool loop:

- Bump `max_tool_rounds` when browser tools registered (e.g. 12)
- Console: `browser → navigate https://...`, `browser → click e12`, timings
- Spoken final answer only after model stops calling tools (same as today)

## Voice UX constraints

| Concern | Implication |
| --- | --- |
| Silence during long tools | Status line / TTS “One moment, I’m opening that page.” once, then quiet |
| Multi-step tasks | Prefer extract-once flows (“what’s the top HN story?”) over login+form wizards in v1 |
| Interrupt | Esc / Ctrl+C should cancel in-flight Playwright ops if possible |
| Headful vs headless | Default headless; `--browser-headed` for demos / debugging |
| Profile / cookies | v1: ephemeral context; later: optional persistent profile path |

## Security / privacy

Same bar as web tools, stricter because the browser can execute JS and access more surfaces:

- [ ] Opt-in flag; tools unregistered when off
- [ ] Block `file://`, `javascript:`, private RFC1918, link-local, metadata IPs (navigate + redirects)
- [ ] No automatic download of executables; deny download dir or sandbox downloads
- [ ] Default headless; no access to user’s real Chrome profile in v1 (credential exfil risk)
- [ ] Audit log every URL and action
- [ ] Stagehand path: explicit consent that page content may go to `MODEL_API_KEY` provider
- [ ] Startup privacy banner branch when browser and/or Stagehand AI enabled

## Packaging / install story

| Path | What users run |
| --- | --- |
| Playwright | `uv tool install 'localtalk[browser]'` then `playwright install chromium` (or document Nix: `playwright` + chromium) |
| Stagehand local | `uv tool install 'localtalk[browser-ai]'` + system Chrome + `MODEL_API_KEY` |
| CI unit tests | mock `BrowserSession`; never launch Chrome in default suite |
| Optional smoke | `@pytest.mark.hardware` or `integration` with real Chromium |

Twelve-factor: config via env (`LOCALTALK_ENABLE_BROWSER`, `LOCALTALK_BROWSER_HEADED`, `MODEL_API_KEY` only for Stagehand).

## What this is *not* (defer)

- Full “Open Operator” autonomous multi-minute agents on voice by default
- Computer-use vision models (screenshot → click coordinates)
- Controlling the user’s everyday Chrome profile / password manager
- Cloudflare Browser Rendering / remote Browserbase as the **only** path
- Replacing Kiwix offline knowledge with live browsing

## Decision matrix

| Criterion | Playwright | Stagehand local+cloud LLM |
| --- | --- | --- |
| Privacy / offline story | ✅ fits | ❌ conflicts unless local model proven |
| Reliability on messy sites | ⚠️ medium | ✅ stronger |
| Dep / install weight | medium (Chromium) | medium–high (SEA + Chrome + API key) |
| gpt-oss tool complexity | higher (many steps) | lower (NL act) |
| Latency / cost | local compute only | + cloud tokens per act |
| Testability | good with mocks | weaker offline |
| Fit with current registry | direct | direct but dual-LLM |

## Suggested phased delivery

### Phase 0 — Spike (1–2 days)

1. Script: launch Playwright Chromium, open HN, dump a11y snapshot size/quality for gpt-oss context.
2. Script: Stagehand `server="local"` + `extract` on example.com; measure latency and whether any local OpenAI-compatible base URL works.
3. Write findings into this doc’s “Spike results” section; pick backend.

### Phase 1 — Playwright MVP (after web tools land)

- `BrowserSession` + tools: open/navigate/snapshot/extract_text/close  
- `--enable-browser`, startup status, security guards  
- `max_tool_rounds` bump when browser enabled  
- Unit tests with fakes  

### Phase 2 — Interaction

- click/type by ref from snapshot  
- basic wait-for-load / timeout policies  
- headed mode flag  

### Phase 3 — Optional Stagehand backend

- Only if Phase 0 proves value and privacy UX is acceptable  
- `browser_act` / `browser_extract` thin wrappers  
- Banner + config for `MODEL_API_KEY`  

## Open questions

1. **Is cloud LLM for browser actions acceptable** as an explicit opt-in, or is pure-local non-negotiable?
2. **Headed browser** during voice demos — desirable for trust (“I can see what it’s doing”)?
3. **Session across turns** vs one-shot browse-and-close per user utterance?
4. Should browser tools share the `--enable-web` flag or be a **separate** `--enable-browser`?
5. Target Chromium via Playwright-managed binary vs system Google Chrome (`channel="chrome"`)?

## Recommendation (staff SWE)

- **Do Playwright first**, as an opt-in tool family parallel to `web_search`, using the existing multi-round Harmony loop.
- Treat **Stagehand as a later optional backend** for users who already use cloud models — not the default path for a product whose banner says “no cloud APIs.”
- Scope MVP to **read-mostly browsing** (navigate + snapshot + extract + speak) before interactive click/type.
- Budget **~2 weeks** for a solid Playwright MVP after connectivity/web tools stabilize; Stagehand only after a **1–2 day spike** on local-model viability.

## Spike results

_(fill after Phase 0)_

| Check | Playwright | Stagehand local |
| --- | --- | --- |
| Install friction on macOS | | |
| Cold start latency | | |
| Snapshot tokens for HN front page | | |
| Extract quality without cloud | n/a | |
| Works offline after install | | |
