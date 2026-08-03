# Marketing Site — localtalk on GitHub Pages

## Goal

A single-page marketing site for `localtalk`, statically hosted on GitHub Pages at
`https://anthonywu.github.io/localtalk/`, sourced from `/docs`.

## Design north star: Apple HIG

The site is a **marketing surface**, not a macOS app — but visual and interaction
choices deliberately track [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
fundamentals so an Apple-native product feels at home.

### Fundamentals applied

| HIG idea | How the site expresses it |
| --- | --- |
| **Clarity** | One hero promise; type hierarchy mapped to SF-like scale (caption → hero); short body copy; SF-style line icons instead of emoji clutter |
| **Deference** | Content first: light chrome, thin separators, no busy backgrounds or decorative grids; nav is translucent and stays out of the way |
| **Depth** | Subtle elevation (`shadow-soft`) and grouped canvases (`#f5f5f7` / system grouped); cards over flat walls only when they group related content |

### Foundations checklist

**Typography**

- System stack: `-apple-system`, `BlinkMacSystemFont`, SF Pro Text/Display, Helvetica Neue fallbacks
- Mono: SF Mono / system monospace for install commands only
- Approximate marketing scale: caption 12, footnote 13, subhead 15, body 17, title3 20, large title 34, display/hero 48–56
- Prefer **sentence case**; avoid shouting ALL CAPS for body content (eyebrows use small weight + tracking, not dense caps blocks)
- `text-wrap: balance` / `pretty` on headlines and body where it helps

**Color**

- Semantic tokens via CSS variables: label, secondary, tertiary, fill, canvas, grouped, separator, link
- Link blue closer to system blue (`#0066cc` light / `#0a84ff` dark) — sufficient contrast on white/black
- **Private by default** language (not absolute “100% private”)
- **Light + dark** via `prefers-color-scheme` (no toggle chrome) — matches system appearance preference
- Do **not** use the Apple logo or 🍎; do not imply Apple affiliation (footer disclaimer)

**Layout**

- Comfortable measure: ~980px reading column, ~1100px wide grids
- Generous vertical rhythm (section padding ~80–96px)
- 8pt-ish spacing; cards at 18px corner radius (marketing soft radius, not iOS continuous corner claim)

**Accessibility (HIG + WCAG-minded)**

- Skip link to `#main`
- Visible `:focus-visible` rings (system blue)
- Minimum **44×44 pt** touch targets for buttons and icon controls
- Links that matter use **underline** (not color alone)
- `prefers-reduced-motion: reduce` disables entrance and decorative motion
- Terminal mockup has `sr-only` summary for assistive tech
- Copy buttons announce “Copied” via `aria-label` temporarily
- `color-scheme` + `theme-color` meta for browser chrome

**Motion**

- Purposeful, short (≤0.6s), ease `cubic-bezier(0.25, 0.1, 0.25, 1)`
- No parallax, no bounce, no hover-lift that shifts layout

**Interaction**

- Primary CTA: pill (`border-radius: 980px`), filled system blue
- Secondary: stroke pill
- Install bars: monospaced command + copy control (clipboard, with fallback)

### Explicit non-goals

- Pixel-perfect SF Symbols (inline SVG stand-ins only)
- Liquid Glass / full macOS 26+ material stack (static marketing page)
- Claiming App Store or Apple endorsement
- Custom dark-mode toggle UI (system preference only)

## Decisions (locked)

- **Stack:** single `index.html`, Tailwind via CDN, no build step; design tokens mostly in CSS variables for light/dark.
- **Hosting:** GitHub Pages, Source = branch `/docs` folder.
- **Base path:** project-page `/localtalk/` → relative asset URLs only.
- **Positioning copy:** see `plan/positioning.md` (beta 0.9.0, teaching tool + parental guidance, voice loop first).

## Site structure (top → bottom)

1. **Skip link** + **nav** — wordmark (SVG mark, not emoji) · Features · Install · Philosophy · Why · GitHub · Get started  
2. **Hero** — “never leaves your Mac” · install one-liner · terminal mockup (decorative + sr-only)  
3. **Pillars** — Offline · Private by default · Zero API keys · Built for Mac  
4. **Features** — Voice loop grid · Power features grid  
5. **Install** — one command · needs / dep / secrets  
6. **Philosophy** — four opinions + HIG nod (clarity / deference / depth)  
7. **Why LocalTalk** — name homage  
8. **Closing CTA** — teach the loop  
9. **Footer** — links, MIT, **not affiliated with Apple**

## Files

- `docs/index.html` — site  
- `docs/WAVEFORMS.md` — unlinked technical note (leave as-is)  
- `plan/marketing_site.md` — this design brief  
- `plan/positioning.md` — product message hierarchy  

## GitHub Pages setup (manual)

1. Repo → **Settings → Pages**  
2. Source: Deploy from a branch  
3. Branch + **`/docs`** → Save  
4. Live at `https://anthonywu.github.io/localtalk/`

## Follow-ups (optional)

- Real product screenshot or short silent demo GIF (highest conversion asset still missing)  
- Open Graph image asset  
- Structured data (`SoftwareApplication`)  
- Self-host critical CSS if CDN policy matters  
