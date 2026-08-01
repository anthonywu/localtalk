# Marketing Site — localtalk on GitHub Pages

## Goal
A single-page marketing site for `localtalk`, statically hosted on GitHub Pages at
`https://anthonywu.github.io/localtalk/`, sourced from the existing `/docs` folder.

## Decisions (locked)
- **Stack:** single `index.html`, Tailwind via CDN (`cdn.tailwindcss.com`), no build step.
- **Hosting:** GitHub Pages, Source = `main` branch `/docs` folder (configured in repo Settings → Pages).
- **Aesthetic:** clean Apple-native light (SF-inspired type, generous whitespace, subtle shadows,
  Apple-blue accent), no dark mode toggle in v1.
- **Base path:** project-page URL has a `/localtalk/` prefix → all asset links **must be relative**
  (`./...`), never root-absolute. No external local assets needed (Tailwind is CDN; fonts use the
  system stack), so this is low-risk.

## Site structure (single page, top → bottom)
1. **Nav** — `💻🎤🔊 localtalk` wordmark + anchor links (Features, Install, Philosophy) +
   GitHub + PyPI links + a primary "Get started" button.
2. **Hero** — headline: privacy-first voice assistant, runs entirely offline on Apple Silicon.
   Subhead naming the audience (DIYers, educators, parents, learners). CTA: copy-to-clipboard
   install one-liner `uv tool install localtalk` + a `uvx localtalk` "try now" hint. A clean
   terminal mockup showing `$ localtalk`.
3. **Pillars** — 3–4 cards: **Fully offline** (turn off WiFi, it still works), **100% private**
   (conversations never leave the device), **Zero API keys** (no accounts, ever), **Apple-native**
   (AVFoundation, Foundation Models, Tingting).
4. **Features grid** — condensed from README: sentence-streamed TTS, Silero VAD auto-listen,
   live waveform, mid-session reasoning control ("think harder"), offline knowledge packs
   (Wikipedia-in-your-cache), online tools w/ voice toggle, dual type/speak input, Tingting +
   Qwen3-TTS Chinese voices, datetime-aware persona.
5. **Install / Quick start** — `uv tool install localtalk` then `localtalk`; note models download
   on first run; note macOS + Apple Silicon + Python 3.11+ + libsndfile.
6. **Design Philosophy** — the four principles condensed (Apple-native end to end; macOS-first
   latest-first w/ best-effort fallbacks; terminal-first terminal-only / no GUI; built for
   tinkerers & learners). Link back to README.
7. **Why "LocalTalk"** — the Apple LocalTalk networking homage (nice narrative hook).
8. **Footer** — MIT license, GitHub / PyPI links, short acknowledgments.

## Design system
- **Type:** `-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif`.
  Mono accents (`ui-monospace, SFMono-Regular`) for code/terminal.
- **Colors:** white / `#f5f5f7` canvas, near-black `#1d1d1f` text, Apple blue `#0071e3` primary CTA,
  subtle gray borders `#d2d2d7`.
- **Components:** rounded-2xl cards, soft `shadow-sm`, hover lift; sticky translucent nav w/ blur.
- **Motion:** minimal — button hover, subtle fade-in via Tailwind `transition`.

## Files to create
- `docs/index.html` — the whole site (Tailwind config inlined for custom colors/fonts).
- `docs/CNAME` — **not** added (project-page URL, not a custom domain).
- `docs/robots.txt` — optional, allow-all. (Nice-to-have, can defer.)

Note: `docs/WAVEFORMS.md` already exists and stays untouched; Pages serves `index.html` as the
homepage and the `.md` is simply an unlinked orphan.

## GitHub Pages setup (manual, one-time — I cannot set repo settings)
1. Repo → **Settings → Pages**.
2. **Source:** `Deploy from a branch`.
3. **Branch:** `main` / folder **`/docs`** → **Save**.
4. Site goes live at `https://anthonywu.github.io/localtalk/` within ~1 min.

## Out of scope for v1 (follow-ups)
- Dark mode / theme toggle.
- Product screenshots or a demo audio clip (needs assets we don't have yet).
- SEO sitemap / structured data.
- A `/docs` README rendering of WAVEFORMS.md.

## Open question
- Hero visual: terminal mockup only (zero asset deps, ships fastest) vs. a waveform motif banner
  (needs a tiny inline SVG, still no external asset). I'd default to **terminal mockup** — confirm?
