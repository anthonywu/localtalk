# Plan: Harmony web tools + online detection

Status: design / ready for review (revised: startup auto-detect)  
Branch context: `release-0.6.0`  
Depends on: shipped mid-session tool path in `src/localtalk/services/mlx_llm.py` (`set_reasoning_level`, 0.5.0)

## Goal

Expose two (optionally three) Harmony function tools so gpt-oss can:

1. **Detect network availability** — including wifi vs ethernet vs none
2. **Look up knowledge on the public internet** when the user wants current/world facts
3. (Optional stretch) **Fetch a single URL** when search snippets are not enough

And at **every process startup**, proactively detect online/offline and print a status line that shows whether online capabilities are available right now.

Stay aligned with LocalTalk’s product identity: **offline-first, privacy-default, opt-in network search**. Platform target remains **macOS-only** (no Linux/Windows connectivity path).

## Non-goals (this iteration)

- Full offline RAG / Kiwix ZIM (`src/localtalk/knowledge/` stub exists; finish separately)
- Browser automation / multi-page crawl
- Paid search API keys as a hard requirement
- Streaming tool results into TTS mid-call
- Plugin marketplace
- Non-macOS network detection

## Current state (what we extend)

`MLXLanguageModelService` already has a working Harmony tool path:

| Piece | Location / behavior |
| --- | --- |
| Tool schema | module-level `_REASONING_TOOL = ToolDescription.new(...)` |
| Registration | `_build_prompt_messages` → `DeveloperContent.with_function_tools([_REASONING_TOOL])` |
| Stop tokens | `_init_harmony` registers `stop_tokens_for_assistant_actions()` so generation halts on `<|call|>` |
| Parse | `_extract_reasoning_tool_call` looks for `recipient == "functions.set_reasoning_level"` |
| Execute + confirm | `_handle_reasoning_tool_call` mutates state, posts tool result as `Role.TOOL`, re-renders, streams a spoken confirmation |
| History | user → tool-call → tool-result → final assistant message |
| Tests | `tests/unit/test_mlx_llm.py::TestReasoningToolCalls` |

This is a **single-tool, one-shot handler**. Internet tools need a **generic multi-tool, multi-round loop**.

`openai_harmony` already supports multiple tools via `DeveloperContent.with_function_tools([tool_a, tool_b, ...])`. Recipients are `functions.<tool_name>`.

`httpx` is already transitive (`0.28.1`). Promote to a **direct** dependency when we ship.

Broken WIP stub (do not couple online tools to this):

```text
src/localtalk/knowledge/__init__.py  # imports packs/store that do not exist yet
```

## Product / privacy constraints

1. **Web search is opt-in** (`--enable-web` / `LOCALTALK_ENABLE_WEB=1`). When off, do not register `web_search` / `fetch_url`.
2. **Startup connectivity probe is always on by default** (L1 + short L2). This is intentional ambient I/O so the user sees online/offline at launch. Escape hatch: `--skip-network-probe`.
3. **`check_online` tool always registered** (answers “am I on wifi?” without enabling search).
4. **When web tools enabled**, log every outbound host (query, host, status, bytes).
5. **SSRF guard** on fetch: no private/link-local/localhost/metadata, http(s) only.
6. **Hard caps**: timeouts, max results, max chars into model context.

### CLI flags

```text
localtalk --enable-web              # register web_search (+ fetch later)
localtalk --skip-network-probe      # skip startup L2 probe (CI / purists)
# LOCALTALK_ENABLE_WEB=1
```

## Startup connectivity status (required UX)

Today init lives in `VoiceAssistant` with a Rich `Live` panel (`core/assistant.py`): model load, VAD, audio devices, reasoning level, then `_print_privacy_banner()` (“you can disable WiFi…”).

**Requirement:** during that same init path, run the shared connectivity checker **once**, cache the result, and print network status **before** “Ready!”.

### When / what

- Near the end of the Live panel (after audio devices / with the reasoning line) so the final panel the user reads includes network status
- Hard timeout: `probe_timeout_s` (~2s) so airplane mode cannot stall startup
- Same implementation as the tool: `detect_connectivity(probe=True)` → `NetworkStatus`

```text
status = detect_connectivity(probe=True)
self.network_status = status   # cache; inject into tool registry / LLM service
```

### Status line matrix

Two orthogonal axes: (A) is the Mac reachable? (B) are web tools enabled?

| Reachable? | `--enable-web`? | Startup line(s) |
| --- | --- | --- |
| yes (wifi) | yes | `🌐 Network: online via wifi — web search available` |
| yes (ethernet) | yes | `🌐 Network: online via ethernet — web search available` |
| yes | no (default) | `🌐 Network: online via wifi — online capabilities available (pass --enable-web to allow lookups)` |
| L1 up, L2 fail | * | `🌐 Network: interface up (wifi) but internet unreachable — online tools unavailable` |
| offline | * | `🌐 Network: offline — running fully local` |

When online + web enabled, optional second line:

```text
   Capabilities: check_online, web_search
```

### Privacy banner (status-aware)

Do **not** claim “no cloud APIs” while advertising web search.

| State | Privacy tip |
| --- | --- |
| Offline | Current message: fully local, can stay offline |
| Online, web tools **off** | “You’re online, but web lookup is off — core assistant stays local. Pass `--enable-web` to allow knowledge search.” |
| Online, web tools **on** | “Web lookup is enabled — searches leave this machine. Core STT/LLM/TTS still run locally.” |

### Init panel sketch

```python
init_messages.append("🌐 Checking network...")
live.update(create_panel())
self.network_status = detect_connectivity(probe=config.web_tools.startup_probe)
init_messages.append(format_network_status_line(self.network_status, web_enabled=...))
if self.network_status.reachable and web_enabled:
    init_messages.append("   Capabilities: web search (Wikipedia), connectivity check")
elif self.network_status.reachable and not web_enabled:
    init_messages.append("   Online capabilities available — start with --enable-web to turn them on")
```

### Cache reuse

1. Startup stores `NetworkStatus` (+ timestamp).
2. `check_online` returns cache if age < `status_ttl_s` (~45s) unless force-refresh.
3. First `web_search`: if cache says unreachable/offline, short-circuit without HTTP.

## Tool surface

### 1. `check_online` (always registered)

**When:** user asks about wifi/ethernet/online; or model is unsure before search.

**Schema:**

```json
{
  "name": "check_online",
  "description": "Check whether this Mac has a usable network path to the internet. Reports online/offline, primary interface type (wifi, ethernet, other, or none), and optionally a lightweight reachability probe.",
  "parameters": {
    "type": "object",
    "properties": {
      "probe": {
        "type": "boolean",
        "description": "If true (default), also attempt a short HTTP reachability probe. If false, only inspect local interface state."
      }
    },
    "additionalProperties": false
  }
}
```

**Result shape:**

```json
{
  "ok": true,
  "online": true,
  "reachable": true,
  "primary": "wifi",
  "interfaces": [
    {"device": "en0", "type": "wifi", "ipv4": true, "ipv6": false, "active": true}
  ],
  "probe": {
    "attempted": true,
    "url_host": "connectivitycheck.gstatic.com",
    "status_code": 204,
    "latency_ms": 42
  },
  "error": null
}
```

**macOS-only implementation:**

| Layer | What it answers | How |
| --- | --- | --- |
| L1 interface | wifi/ethernet up with an address? | `scutil --nwi` + `networksetup -listallhardwareports` (Wi-Fi → wifi; Ethernet/USB LAN/Thunderbolt → ethernet) |
| L2 reachability | actual internet path? | short `httpx` GET to `https://connectivitycheck.gstatic.com/generate_204` (1–2s timeout) |

Avoid `netifaces` / `psutil`. No Linux/Windows branch.

### 2. `web_search` (only when `--enable-web`)

Returns short snippets for spoken summary. **First backend: Wikipedia** OpenSearch + page summary via `httpx` (stable, no API key, aligns with offline Wikipedia roadmap). Optional later: DDG / Brave.

```json
{
  "name": "web_search",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "max_results": {"type": "integer", "minimum": 1, "maximum": 5}
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

### 3. `fetch_url` (optional follow-up)

Single public http(s) page, SSRF-guarded, char-capped. Stdlib HTML extract first.

## Architecture

### Module layout

```text
src/localtalk/
  services/
    mlx_llm.py              # multi-round tool loop + registry wiring
    tools/
      __init__.py
      base.py               # ToolSpec, ToolRegistry, dispatch
      reasoning.py          # set_reasoning_level
      online.py             # detect_connectivity, format_network_status_line, check_online handler
      web.py                # web_search (+ optional fetch_url)
  models/
    config.py               # WebToolsConfig
  core/
    assistant.py            # startup probe + status lines + privacy banner
```

`online.py` is shared by **startup** and the **Harmony tool** — one implementation, two call sites.

### Generic tool loop

```text
messages_extra = [user_message]
for round in range(1, max_tool_rounds + 1):
    render → stream → parse
    call = extract_first_function_tool_call(parsed)
    if call is None: break
    result = registry.dispatch(call.name, call.args)
    messages_extra += [call_msg, tool_result_msg]
record_turn(... full exchange + final ...)
```

Defaults: `max_tool_rounds=3`. Host short-circuits `web_search` when cache says offline.

### Config

```python
class WebToolsConfig(BaseModel):
    enabled: bool = False
    max_tool_rounds: int = Field(default=3, ge=1, le=8)
    search_max_results: int = Field(default=3, ge=1, le=5)
    search_timeout_s: float = Field(default=8.0, ge=1.0, le=30.0)
    probe_timeout_s: float = Field(default=2.0, ge=0.5, le=10.0)
    status_ttl_s: float = Field(default=45.0, ge=0.0)
    startup_probe: bool = Field(default=True)  # False via --skip-network-probe
    fetch_max_chars: int = Field(default=2000, ge=200, le=8000)
    backend: Literal["wikipedia", "ddg"] = "wikipedia"
    reachability_url: str = "https://connectivitycheck.gstatic.com/generate_204"
```

### Prompt policy

Inject web-tool instructions into the developer/system prompt **only when** `web_tools.enabled`. Always document `check_online` + `set_reasoning_level`.

## Testing plan

Unit suite stays offline (mock subprocess + httpx).

| Test | Intent |
| --- | --- |
| `test_check_online_wifi` / `_ethernet` / `_offline` | L1 fixtures |
| `test_check_online_probe_unreachable` | L1 up, L2 fail |
| `test_startup_status_line_online_web_enabled` | “web search available” |
| `test_startup_status_line_online_web_disabled` | online + “pass --enable-web” |
| `test_startup_status_line_offline` | fully local |
| `test_startup_probe_skipped_when_disabled` | no HTTP when `startup_probe=False` |
| `test_connectivity_cache_reused_within_ttl` | no re-probe |
| `test_web_tools_not_registered_when_disabled` | no `web_search` in developer tools |
| `test_web_search_success_loop` / `_offline_short_circuit` | tool loop |
| `test_reasoning_tool_still_works_via_registry` | 0.5.0 regression |

Update `tests/TEST_CATALOG.md` for every case.

## Implementation PR stack

### PR1 — Tool registry + multi-round loop

- Extract `set_reasoning_level`; generic loop; no user-visible change

### PR2 — `check_online` + **startup network status**

- `services/tools/online.py` (`detect_connectivity`, formatter, handler)
- Wire into `VoiceAssistant` Live panel + status-aware privacy banner
- Cache on assistant → inject into LLM/tools
- `--skip-network-probe`
- Unit tests (fixtures + formatters)

### PR3 — `web_search` + `--enable-web`

- Wikipedia backend; prompt injection; startup line lists capabilities when enabled
- README / CHANGELOG privacy notes

### PR4 (optional) — `fetch_url` + extra backends

### Later — Offline knowledge bridge (Kiwix)

## Dependency impact

| Package | Action | Why |
| --- | --- | --- |
| `httpx` | promote to direct | probe + search + fetch |
| scrapers / BS4 / netifaces | defer or avoid | keep deps lean |

## Decisions (resolved / remaining)

| # | Decision | Status |
| --- | --- | --- |
| 1 | `check_online` always available | **Resolved:** yes (local inspect; probe per policy) |
| 2 | Startup auto-detect online/offline | **Resolved:** yes, always by default; print online capabilities when online |
| 3 | macOS only for connectivity | **Resolved:** yes |
| 4 | Web search default | **Resolved:** off until `--enable-web` |
| 5 | Search backend first ship | **Resolved (rec):** Wikipedia |
| 6 | Extract `services/tools/` in PR1 | **Rec:** yes |
| 7 | Failed search: admit + use parametric knowledge | **Rec:** yes (model-side) |

## Success criteria

- **Every startup** (unless `--skip-network-probe`): clear online/offline line; if online, states whether web search is active or how to enable online capabilities
- Default flags: no search traffic; one short startup probe only; STT/LLM/TTS unchanged
- `--enable-web` + reachable: “web search available”; spoken answers use tool results
- Airplane mode: startup says offline; tools fail soft without hanging
- Unit suite offline-clean

## File touch list

```text
src/localtalk/services/mlx_llm.py
src/localtalk/services/tools/*             # online.py shared by startup + tool
src/localtalk/models/config.py
src/localtalk/cli.py                       # --enable-web, --skip-network-probe
src/localtalk/core/assistant.py            # startup probe + status + privacy banner
prompts/default.txt                        # or dynamic append when web enabled
pyproject.toml                             # httpx direct
tests/unit/test_mlx_llm.py
tests/unit/test_tools_online.py
tests/unit/test_tools_web.py
tests/unit/test_assistant.py
tests/TEST_CATALOG.md
README.md / CHANGELOG.md
```

## Relationship to offline roadmap

README “Offline Knowledge Base” (Kiwix) remains the privacy-preserving long-term path. Online tools are the gated counterpart; share a future common `KnowledgeHit` shape with ZIM search.
