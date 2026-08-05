# LocalTalk Tools — a friendly guide

> **Who this is for:** people who are new to building things with AI. If you've
> never heard of "function calling" or "tools" before, start here. We'll explain
> everything from scratch, no experience required.

## 1. What's a "tool"?

Imagine the AI is a smart friend sitting in a room with you. By itself, it can
only **talk** — it can't reach out and *do* anything.

A **tool** is like handing your friend a remote control:

> "Here's a button labeled *turn on timing stats*. Press it whenever that would help."

Now your friend can not only talk, but also flip switches in the app when you ask.

In LocalTalk, the AI (the language model) can't directly change your microphone
settings, switch the speaking voice, or search the web — those are things our
Python code does. **Tools are the bridge**: they let the AI press our buttons.

## 2. The four pieces of every tool

Every tool in LocalTalk is made of four parts. Think of a TV remote:

| Piece | Remote analogy | In our code |
|---|---|---|
| **Definition** | The label on the button (*"Volume Up"*) | The tool's name + description |
| **Parameters** | The instructions (*"hold to raise faster"*) | What inputs the tool needs |
| **Handler** | The wires that actually change the volume | The Python function that runs |
| **Spoken reply** | A little beep so you know it worked | What LocalTalk says out loud |

We glue all four together into one object called a **`ToolSpec`**. You'll see it
in `src/localtalk/services/tools/base.py`.

## 3. The shape of a tool (the "schema")

Here's the actual recipe, in plain English:

```python
ToolSpec(
    name="set_stats",                 # a unique button name
    description=...,                  # TEACHES the AI when to press it
    handler=some_function,            # the wires (runs when pressed)
    spoken_fallback=some_function,    # what to say out loud afterward
)
```

The two most important parts for beginners:

- **`description`** — this is the *only* hint the AI gets about when to use the
  tool. Write it like you're teaching a brand-new coworker: *"Use this when the
  user asks to turn timing stats on or off."* A vague description = the AI never
  uses your tool, or uses it at the wrong time.
- **`handler`** — a function that takes the AI's inputs and returns a small
  dictionary, always shaped like `{"ok": True, ...}` or `{"ok": False, "error": "..."}`.

### What's a "parameter"? (the form the AI fills in)

Parameters describe the **inputs** the tool needs, like a form. The AI reads this
form and fills it in. Example — the `set_stats` tool needs to know *on or off*:

```python
{
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",                                    # true or false
            "description": "True to show timing stats after each turn",
        }
    },
    "required": ["enabled"],       # the AI must fill this in
    "additionalProperties": False, # no surprise fields allowed
}
```

This shape is called **JSON Schema**. Don't let the name scare you — it's just a
way to describe a form: field names, types (`boolean`, `string`, `number`), and
which ones are required.

## 4. How a tool call flows (step by step)

You say: *"show me the timing stats."* Here's what happens:

1. **Whisper** transcribes your speech to text.
2. The **AI** looks at its menu of tools and thinks: *"`set_stats` fits — I'll
   call it with `enabled: true`."*
3. LocalTalk's **registry** receives that call and runs the matching **handler**.
4. The handler does the real work and returns `{"ok": True, "enabled": True}`.
5. The **`spoken_fallback`** turns that result into a sentence:
   *"Okay, timing stats is on."*
6. You hear it spoken aloud.

The AI never runs code itself — it only *asks* us to run a tool. We stay in control.

## 5. Two flavors of "thing the user can trigger"

This part trips people up, so read carefully:

- **LLM tools** — the **AI decides** to call these by reading the menu. Most tools
  work this way (e.g. `set_stats`, `set_tts_backend`, `voice_help`).
- **Direct voice commands** — LocalTalk **matches your words directly**, no AI
  round-trip. Faster, for things that must happen *before* the AI replies (e.g.
  *"speak Chinese"* switches the voice instantly so the answer is spoken in Chinese).

A feature can be **both**. Our `voice_help` ("tell me about premium voices") is:
saying *"help"* or *"usage"* triggers it instantly as a direct command, **and**
the AI can call the `voice_help` tool if you ask conversationally *"can you sound
more natural?"*. Both share one helper function, so there's a single source of truth.

## 6. Add your own tool — step by step

Let's say you want a brand-new tool: **`set_volume`** that changes playback volume.
Here's the whole recipe.

### Step 1 — Write the "factory"

A factory is just a function that *builds* your tool. Put it in
`src/localtalk/services/tools/settings.py` (the home for session-setting tools).
The factory takes a callback (the real work, owned by the assistant) and returns a
`ToolSpec`:

```python
def make_set_volume_tool(set_volume: SetInt) -> ToolSpec:
    def handler(args: dict) -> dict:            # the AI's inputs arrive here
        level = args.get("level")
        if level is None:
            return {"ok": False, "error": "level is required"}
        return set_volume(int(level))           # call the assistant's real function

    return ToolSpec(
        name="set_volume",
        description=build_tool_description(
            "set_volume",
            "Set playback volume from 0 to 100. Use when the user asks to change "
            "the volume, make it louder or quieter.",
            {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume 0–100"},
                },
                "required": ["level"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            f"Okay, volume set to {r['level']}." if r.get("ok")
            else f"Sorry, I couldn't change the volume. {r.get('error', '')}"
        ),
    )
```

### Step 2 — Do the real work in the assistant

Add the actual function to `src/localtalk/core/assistant.py`, returning the same
`{"ok": True/False, ...}` shape:

```python
def _tool_set_volume(self, level: int) -> dict:
    # ...actually change the volume here...
    return {"ok": True, "level": level}
```

### Step 3 — Hand it to the AI

In `assistant.py`, inside `bind_session_control(...)`, add one line so the LLM
services can reach your function:

```python
self.llm.bind_session_control({
    ...,
    "set_volume": self._tool_set_volume,
})
```

### Step 4 — Register it (⚠️ in BOTH places)

LocalTalk supports two AI engines — **MLX** and **Apple** — so you must register
your tool in **both** files, or it only works for one engine:

- `src/localtalk/services/mlx_llm.py`
- `src/localtalk/services/apple_llm.py`

In each, inside the registry-building block, add:

```python
if (set_volume := self.session_control.get("set_volume")) is not None:
    registry.register(make_set_volume_tool(set_volume))
```

> 💡 Why two places? Each engine builds its own tool menu. If you forget one,
> users on that engine won't have your tool. This is the #1 mistake beginners make.

### Step 5 — Test it

Add a test in `tests/unit/test_settings_tools.py` that calls your handler and
checks the result dict. That's it — ship it!

## 7. Worked example: `voice_help` (the simplest possible tool)

`voice_help` takes **no inputs** — it just tells the user about premium voices.
It's a great first tool to study because it's tiny. You'll find it in
`src/localtalk/services/tools/settings.py`:

```python
def make_voice_help_tool(voice_help: VoiceHelp) -> ToolSpec:
    def handler(args: dict) -> dict:
        return voice_help()          # no inputs needed — just ask for the info

    return ToolSpec(
        name="voice_help",
        description=build_tool_description(
            "voice_help",
            "Tell the user about available speech voices and how to get "
            "higher-quality ones. Use when the user asks about voice options...",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        ),
        handler=handler,
        spoken_fallback=lambda r, a: r.get("spoken") or "Sorry, I couldn't fetch voice info.",
    )
```

Notice:
- **No parameters** — `"properties": {}` and `"required": []` mean the AI calls it
  with no inputs.
- **`spoken_fallback` reads the result** — the handler returns
  `{"ok": True, "spoken": "..."}`, and we speak that `spoken` string.

The actual message is built once, in `assistant.py`'s `_tool_voice_help`, which is
shared with the *"help"* / *"usage"* direct command — one helper, two ways in.

## 8. Tips for great tools

- **Write the description for a newcomer.** Spell out *when* to use it and *what
  for*. The AI only has your words to go on.
- **Always return `{"ok": True/False, ...}`.** LocalTalk's spoken fallbacks and
  error handling depend on this shape.
- **Keep spoken replies short.** They're read aloud — one sentence is ideal.
- **Register in both engines** (MLX + Apple). 🚫 most common bug.
- **Prefer a direct command when speed matters.** If something must happen before
  the AI replies (like switching languages), match the words directly instead of
  waiting for the AI to call a tool.

## 9. Where things live (file map)

| Want to… | Look at |
|---|---|
| See the core types (`ToolSpec`, `ToolRegistry`) | `src/localtalk/services/tools/base.py` |
| See session-setting tools (`set_stats`, `voice_help`, …) | `src/localtalk/services/tools/settings.py` |
| See how tools get registered for an engine | `mlx_llm.py` and `apple_llm.py` |
| See how the assistant wires up its functions | `src/localtalk/core/assistant.py` → `bind_session_control` |
| See a direct (no-AI) voice command | `assistant.py` → `_handle_usage_command` |
| See the existing tool tests | `tests/unit/test_settings_tools.py` |

That's the whole system. Tools are just labeled buttons the AI can press — write
a clear label, wire it up, register it twice, and you're done. 🎉
