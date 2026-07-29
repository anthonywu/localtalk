# Terminal Waveform Visualization

This document describes how the LocalTalk terminal app visualizes audio waveforms in real time.

There are **two independent waveform implementations**, both using the same core technique (Unicode block characters + Rich `Live` display) but serving different purposes:

| Implementation | File | Purpose | Triggered by |
|---|---|---|---|
| VAD recording waveform | `src/localtalk/services/audio_vad_auto.py` | Live audio + VAD status during speech capture | `record_with_vad_auto()` → `record_with_vad_automatic()` |
| Microphone-test waveform | `src/localtalk/services/audio.py` (`test_microphone`) | Diagnostic level metering | CLI `--test-mic` flag |

---

## Core Technique (shared by both)

### Unicode block ramp

Both implementations use the same 9-character Unicode block ramp, ordered from quietest to loudest:

```python
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"
```

- Index 0 = space (silence / lowest)
- Index 8 = `█` (maximum)

### Amplitude → block mapping (`level_to_block`)

Each audio chunk's peak amplitude (a float in `[0, 1]`) is converted to one block character:

```python
def level_to_block(level: float) -> str:
    level = min(1.0, max(0.0, level))   # clamp to [0, 1]
    level = level ** 0.5                 # square-root (perceptual) scaling
    index = int(level * (len(WAVEFORM_BLOCKS) - 1))  # 0–8
    return WAVEFORM_BLOCKS[index]
```

**Square-root scaling** (`level ** 0.5`) compresses the dynamic range so quiet sounds are more visible. Without it, normal speech (which rarely peaks above 0.2) would render as mostly the first two blocks. The mapping is:

| Input level | After √ | Block index | Character | Meaning |
|---|---|---|---|---|
| 0.0 | 0.0 | 0 | ` ` | silence |
| 0.04 | 0.2 | 1 | `▁` | very quiet |
| 0.25 | 0.5 | 4 | `▄` | mid-level |
| 0.5 | 0.71 | 5 | `▅` | moderate |
| 1.0 | 1.0 | 8 | `█` | maximum |

### Amplitude calculation

Both implementations compute **peak absolute amplitude** per chunk (not RMS):

```python
level = np.abs(audio_float32).max()
```

This gives the single largest sample value in the chunk, in the range `[0, 1]` for float32 audio.

### Rendering with Rich

Both use Rich's [`Live`](https://rich.readthedocs.io/en/stable/live.html) display with a borderless `Table` to create a continuously-updating terminal UI:

- `Live(refresh_per_second=15, ...)` — refreshes the display ~15 times per second.
- The display is a `Table(show_header=False, box=None, padding=0)` with indented rows.
- A `Text` object holds the waveform string, with per-character styling (color) applied via `waveform.append(block, style=...)`.

### History buffer

Both maintain a sliding-window history using `collections.deque` with a `maxlen`:

- VAD recording: `deque(maxlen=60)` storing `(level, is_speech)` tuples
- Mic test: `deque(maxlen=60)` storing `float` levels

As new audio chunks arrive, old samples fall off the left edge — the waveform scrolls right-to-left.

### Padding

If the history buffer hasn't filled yet (fewer samples than the display width), the remaining columns are padded with `▁` in `dim` style so the waveform always spans the full width.

---

## Implementation 1: VAD Recording Waveform

**File:** `src/localtalk/services/audio_vad_auto.py`
**Function:** `record_with_vad_automatic(audio_service)`
**Called from:** `AudioService.record_with_vad_auto()` ← `VoiceAssistant.process_voice_input()`

### Constants

```python
WAVEFORM_WIDTH  = 60   # display width in characters
WAVEFORM_HISTORY = 60  # number of chunks retained (matches width)
CHUNK_SIZE = 512       # samples per chunk (Silero VAD requires 512 @ 16kHz)
```

At 16 kHz / 512 samples per chunk, each history entry represents **32 ms** of audio. The 60-entry window shows the last **~1.9 seconds** of audio.

### History entries

Each entry is a `(level, is_speech)` tuple:

```python
level_history: deque[tuple[float, bool]] = deque(maxlen=60)
# ...
last_audio_level = np.abs(audio_float32).max()
is_speech = vad_prob > audio_service.config.vad_threshold
level_history.append((last_audio_level, is_speech))
```

The `is_speech` flag comes from the Silero VAD model's probability compared against `config.vad_threshold` (default 0.5).

### Color coding (`create_waveform`)

Each block is colored based on whether VAD classified that chunk as speech:

| Condition | Color | Meaning |
|---|---|---|
| `is_speech == True` | `bold green` | VAD detected speech |
| `level > 0.02` (but not speech) | `yellow` | Audio present but below VAD threshold |
| `level <= 0.02` | `dim` | Near-silence |

### Status display (`create_status_display`)

The full display is a 4-row table:

1. **Status line** — changes based on state:
   - Not yet spoken: `🎤 Listening... (speak now)` (cyan)
   - Currently speaking: `🎤 RECORDING YOUR SPEECH` (bold green)
   - Speech ended: `🤫 Processing...` (yellow)
2. **Waveform row** — the scrolling colorized waveform, indented 4 spaces
3. **Level/VAD indicator row** — a `●`/`○` glyph (green/dim) plus numeric values:
   ```
       ● Level: 0.342  VAD: 0.873
   ```
4. **Info row** — contextual:
   - Before speech: countdown to timeout (`Timeout in 2.1s`)
   - After speech starts: recording duration (`Recording: 1.5s / 120s max`)

### Display lifecycle

```
print("🎤 VAD is listening for your speech...")
print("   Waveform: green=speech detected, yellow=audio, dim=silence")

with Live(create_status_display(), refresh_per_second=15, transient=False) as live:
    with sd.InputStream(callback=audio_callback, blocksize=512):
        while not should_stop:
            live.update(create_status_display())
            time.sleep(0.05)  # ~20 Hz poll loop
        live.update(create_status_display())  # final frame
```

- `transient=False` — the last frame remains visible after the `Live` context exits (kept for debugging).
- The audio callback runs in sounddevice's audio thread; it updates `level_history` and the stop flags. The main thread polls and redraws at ~20 Hz, while Rich throttles rendering to 15 FPS.
- stdout/stderr are flushed before and after the `Live` context to avoid interleaved output.

---

## Implementation 2: Microphone-Test Waveform

**File:** `src/localtalk/services/audio.py`
**Function:** `AudioService.test_microphone(duration_seconds=5.0)`
**Called from:** CLI `--test-mic` flag (`src/localtalk/cli.py`)

### Constants (defined locally inside the method)

```python
WAVEFORM_WIDTH  = 60
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"
```

These are identical to the VAD module's constants but **duplicated as local variables** inside the method — they are not imported from `audio_vad_auto`.

### History entries

Each entry is a plain `float` (peak level only, no VAD classification):

```python
level_history: deque[float] = deque(maxlen=60)
level = np.abs(indata).max()
level_history.append(float(level))
```

### Color coding (`create_waveform`)

Color is based purely on amplitude thresholds (no VAD involvement):

| Level range | Color | Meaning |
|---|---|---|
| `< 0.01` | `dim red` | No signal |
| `0.01 – 0.05` | `yellow` | Very quiet |
| `0.05 – 0.2` | `green` | Good |
| `≥ 0.2` | `bold bright_green` | Strong |

### Status display (`create_display`)

A 2-row table:

1. **Waveform row** — scrolling colorized waveform, indented 4 spaces
2. **Level indicator row** — a glyph + status text + peak meter:
   ```
       ● Level: 0.215 (Good)  Peak: 0.342
   ```

The glyph and status text use the same thresholds as the color coding:

| Level range | Glyph | Status text |
|---|---|---|
| `< 0.01` | `○` (red) | No signal |
| `0.01 – 0.05` | `◐` (yellow) | Very quiet |
| `0.05 – 0.2` | `●` (green) | Good |
| `≥ 0.2` | `●` (bright green) | Strong |

### Display lifecycle

```
with Live(create_display(), refresh_per_second=15, transient=True) as live:
    with sd.InputStream(callback=audio_callback, blocksize=config.chunk_size):
        for _ in range(num_samples):
            live.update(create_display())
            time.sleep(config.chunk_size / config.sample_rate)
```

- `transient=True` — the waveform is erased when the `Live` context exits; only the summary results remain.
- The loop runs for a fixed number of iterations (`duration_seconds * sample_rate / chunk_size`), so the display lasts exactly the requested duration (default 5 seconds).

### Post-test summary

After the live display ends, a results block is printed:

```
━━━ Microphone Test Results ━━━
Peak level: 0.342
Audio detected: 87.3% of samples
```

Followed by a diagnosis:
- Peak `< 0.01` → ❌ no audio (with troubleshooting tips)
- Peak `< 0.05` → ⚠️ very low levels (with suggestions)
- Peak `≥ 0.05` → ✓ working properly

---

## Key Differences Between the Two Implementations

| Aspect | VAD recording | Mic test |
|---|---|---|
| **File** | `audio_vad_auto.py` | `audio.py` |
| **History type** | `deque[tuple[float, bool]]` | `deque[float]` |
| **Color basis** | VAD speech classification | Amplitude thresholds only |
| **Colors** | green / yellow / dim | bright_green / green / yellow / dim red |
| **`transient`** | `False` (last frame stays) | `True` (cleared after) |
| **Stop condition** | VAD logic (speech → silence) or timeout | Fixed duration (default 5s) |
| **Extra rows** | Status line + timeout/duration info | Peak meter + status text |
| **Glyph** | `●`/`○` (speech indicator) | `●`/`◐`/`○` (level indicator) |
| **Constants** | Module-level (importable) | Local method variables (duplicated) |

---

## Testing

Unit tests for the shared `level_to_block` function live in `tests/unit/test_audio_vad_auto.py`:

- **`TestLevelToBlock`** — verifies clamping (0, negative, >1), the full block index mapping, and the square-root scaling (e.g., 0.25 → √0.25 = 0.5 → block index 4).
- **`TestRecordWithVadAutomaticMocked`** — exercises the full recording function with a scripted `InputStream` and mocked VAD probabilities, verifying speech detection, audio contiguity, and normalization.

The mic-test waveform (`test_microphone`) has no unit tests because it requires a live audio device.

---

## Architecture Notes

- The waveform code is **embedded inside the recording functions** as closures, not factored into a reusable renderer. Both implementations duplicate the block ramp, the `level_to_block` function, and the `create_waveform` builder.
- The VAD implementation stores the module-level constants and `level_to_block` as importable symbols; the mic-test version redefines them locally.
- Rendering happens on the main thread (Rich `Live`), while audio capture happens on sounddevice's callback thread. They communicate through shared mutable state (`level_history`, flags) — no locking is used, which is acceptable because the deque append is atomic in CPython and the display only reads the history for rendering.
