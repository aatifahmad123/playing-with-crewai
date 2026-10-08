# AI Newsroom

Three AI journalists. One verified story.

A [CrewAI](https://docs.crewai.com/) demo that turns a topic into a fact-checked
news article. A crew of three agents — a **Researcher**, a **Fact-Checker** and
an **Editor** — runs in sequence, each receiving the previous agent's output as
context, while a Gradio UI streams live progress and shows every intermediate
artifact.

```
Topic → 🔬 Researcher → 🔍 Fact-Checker → ✍️ Editor → Article
        (web search)     (verifies claims)  (writes prose)
```

## What makes it a *crew*, not one big prompt

- **Three distinct agents**, each with its own role, goal, backstory,
  temperature and iteration limit — the fact-checker runs cold (0.1) because it
  must classify claims conservatively; the editor runs warmer (0.5) because it
  writes prose.
- **Chained tasks** via CrewAI's `context` mechanism on a `Process.sequential`
  pipeline: research → fact-check → editing.
- **A real web search tool** (DuckDuckGo, no API key needed) available to the
  Researcher and Fact-Checker, so the article is grounded in current
  information — and agents are explicitly instructed to mark unverifiable
  claims `NEEDS_SOURCE` rather than invent citations.
- **Transparency UI**: watch each stage complete in real time, then read the
  research brief, the fact-check report, all cited sources and a
  "how this story was produced" breakdown with per-stage timing and token
  usage.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp env-example .env
# edit .env and paste your OpenAI API key (https://platform.openai.com/api-keys)

python app.py
```

Then open http://127.0.0.1:7860, type a topic, pick a style and length, and
press **Generate Story**.

## Configuration

All config lives in `.env` (see `.env.example`) and `config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `NEWSROOM_TIMEOUT_SECONDS` | `900` | Hard limit for one full pipeline run |
| `NEWSROOM_SEARCH_MAX_RESULTS` | `6` | Results per web search |
| `NEWSROOM_PORT` / `NEWSROOM_HOST` | `7860` / `127.0.0.1` | Gradio bind address |
| `NEWSROOM_OPEN_BROWSER` | `1` | Open a browser tab on launch |

The model is `gpt-4o-mini` served by OpenAI (set `OPENAI_MODEL_ID` in
`config.py` to change it).

## Project layout

```
app.py              Gradio UI: streams progress, fills output tabs
crew.py             Crew assembly, pipeline runner, Progress tracker
agents.py           The three agents (role, goal, backstory, LLM settings)
tasks.py            The three tasks and their context chaining
tools.py            web_search tool (DuckDuckGo via ddgs, no API key)
config.py           Settings, model IDs, temperatures, styles, lengths
compat.py           Compat shim for a CrewAI↔LiteLLM request-schema issue
utils/formatting.py Progress panel, source extraction, process rendering
```

## Notes on the compat shim

Recent CrewAI versions tag outgoing messages with a prompt-caching marker
(`cache_breakpoint`) that providers with strict request schemas (the original
trigger was Groq) reject as an unrecognized field. `compat.py` neutralizes that
marker before any agent is constructed — it only disables a caching
optimization that never worked through LiteLLM routing in the first place;
model behaviour is unchanged.

## Troubleshooting

- **401 / invalid API key** — check `OPENAI_API_KEY` in `.env`, restart the app.
- **429 rate limit** — wait a minute, or generate a shorter article.
- **"No sources were cited"** — the search tool likely couldn't reach the
  network during the run; check connectivity and regenerate.
