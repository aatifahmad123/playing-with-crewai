"""Presentation helpers for the AI Newsroom UI.

Everything the Gradio app renders beyond the raw agent output lives here:
progress panels, the crew-process diagram and source extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crew import PipelineResult, StageInfo

# --- Stage display metadata ----------------------------------------------------

STAGE_UI = {
    "research": {
        "icon": "🔬",
        "name": "Researcher",
        "running": "Researcher is researching…",
        "done": "Research complete",
        "summary": "Collected and structured information",
    },
    "fact_check": {
        "icon": "🔍",
        "name": "Fact-Checker",
        "running": "Fact-checker is checking claims…",
        "done": "Fact-check complete",
        "summary": "Reviewed factual claims",
    },
    "editing": {
        "icon": "✍️",
        "name": "Editor",
        "running": "Editor is writing…",
        "done": "Story published",
        "summary": "Produced final article",
    },
}

STAGE_ORDER = ("research", "fact_check", "editing")

_STATUS_RUNNING = "●"
_STATUS_WAITING = "○"
_STATUS_DONE = "✓"


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def render_progress(
    completed: int,
    elapsed_seconds: float | None = None,
    error: str | None = None,
    topic: str = "",
) -> str:
    """Build the live status panel shown while the crew is working.

    ``completed`` is the number of stages finished; the next stage is running.
    """

    if error:
        lines = ["## ✗ The newsroom hit a problem", "", f"**Error:** {error}", ""]
        for stage in STAGE_ORDER:
            ui = STAGE_UI[stage]
            if STAGE_ORDER.index(stage) < completed:
                lines.append(f"{_STATUS_DONE} {ui['done']}")
            else:
                lines.append(f"{_STATUS_WAITING} {ui['name']} waiting")
        return "\n".join(lines)

    lines = []
    if completed >= len(STAGE_ORDER):
        lines.append("## ✓ Story published")
    else:
        lines.append("## 🗞️ The crew is working…")
    if topic:
        lines.append(f"*Topic: “{topic}”*")
    lines.append("")

    for index, stage in enumerate(STAGE_ORDER):
        ui = STAGE_UI[stage]
        if index < completed:
            lines.append(f"{_STATUS_DONE} {ui['done']}")
        elif index == completed:
            suffix = ""
            if elapsed_seconds is not None:
                suffix = f" ({_fmt_seconds(elapsed_seconds)})"
            lines.append(f"{_STATUS_RUNNING} {ui['running']}{suffix}")
        else:
            lines.append(f"{_STATUS_WAITING} {ui['name']} waiting")

    if completed >= len(STAGE_ORDER):
        lines.append("")
        lines.append(
            "All three agents finished — see the tabs below. "
            "The Final Article is ready."
        )
    return "\n".join(lines)


# --- Source extraction -----------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"(?<![(\[])(https?://[^\s)\]>|,;\"']+)")
_URL_TRIM_CHARS = ".,;:'\""


@dataclass
class Source:
    title: str
    url: str


def extract_sources(*texts: str) -> list[Source]:
    """Pull cited URLs (markdown links first, then bare URLs) from agent output.

    Preserves first-seen order and de-duplicates by URL. Only http(s) links
    survive; anything the model may have hallucinated still has to look like
    a real URL to appear here — the UI always shows them as *agent-cited*
    sources.
    """

    seen: set[str] = set()
    sources: list[Source] = []

    for text in texts:
        if not text:
            continue
        for title, url in _MD_LINK_RE.findall(text):
            url = url.rstrip(_URL_TRIM_CHARS)
            if url not in seen:
                seen.add(url)
                sources.append(Source(title=title.strip() or url, url=url))
        for url in _BARE_URL_RE.findall(text):
            url = url.rstrip(_URL_TRIM_CHARS)
            if url not in seen:
                seen.add(url)
                sources.append(Source(title=_nice_host(url), url=url))

    return sources


def _nice_host(url: str) -> str:
    host_match = re.match(r"https?://([^/]+)", url)
    return host_match.group(1).replace("www.", "") if host_match else url


def render_sources(sources: list[Source]) -> str:
    if not sources:
        return (
            "### No sources were cited\n\n"
            "Neither the Researcher nor the Fact-Checker returned usable "
            "source URLs for this run. This can happen when the web search "
            "tool could not be reached — treat the article accordingly."
        )
    lines = [
        f"### {len(sources)} sources cited by the crew",
        "",
        "Every URL below appeared in the Researcher's brief or the "
        "Fact-Checker's report. They are agent-cited, not independently "
        "verified by this app.",
        "",
    ]
    for i, source in enumerate(sources, start=1):
        lines.append(f"{i}. [{source.title}]({source.url})")
    return "\n".join(lines)


# --- Crew process rendering --------------------------------------------------------

def render_crew_process(result: PipelineResult) -> str:
    """The 'proof of work' tab: who did what, in what order, and how long."""

    lines = [
        "## How this story was produced",
        "",
        "This app is not one big prompt. It is a **CrewAI crew** of three "
        "agents, each with its own role, goal and backstory, executing three "
        "chained tasks on a sequential process. Each task received the "
        "previous task's output as context.",
        "",
    ]

    arrows = []
    for stage in result.stages:
        ui = STAGE_UI.get(stage.label, {})
        icon = ui.get("icon", "•")
        name = ui.get("name", stage.label)
        seconds = (
            f" — {_fmt_seconds(stage.seconds)}"
            if stage.seconds is not None
            else ""
        )
        lines.append(f"### {icon} {name}{seconds}")
        lines.append(f"**Role:** {stage.agent_role}")
        lines.append("")
        lines.append(f"> {ui.get('summary', '')}")
        lines.append("")
        if stage.output_excerpt:
            excerpt = stage.output_excerpt
            if len(excerpt) > 600:
                excerpt = excerpt[:600].rsplit(" ", 1)[0] + " …"
            lines.append("**Output preview**")
            lines.append("")
            lines.append(f"> {excerpt.replace(chr(10), chr(10) + '> ')}")
            lines.append("")
        if stage.label != "editing":
            lines.append("↓")
            lines.append("")

    lines.append("### Totals")
    lines.append("")
    lines.append(f"- **Model:** `{result.model}` (OpenAI, all three agents)")
    lines.append(f"- **Total wall-clock time:** {_fmt_seconds(result.total_seconds)}")
    if result.token_usage:
        usage = result.token_usage
        lines.append(
            "- **LLM usage:** {requests} requests · {input_tokens:,} input "
            "tokens · {output_tokens:,} output tokens".format(**usage)
        )
    if result.stages and result.stages[-1].seconds is not None:
        lines.append(
            "- **Pipeline:** research → fact-check → editing "
            "(CrewAI `Process.sequential`, task `context` chaining)"
        )
    return "\n".join(lines)


def stage_summaries() -> list[tuple[str, str]]:
    """(stage, one-line summary) pairs, used before any run happens."""

    return [(s, STAGE_UI[s]["summary"]) for s in STAGE_ORDER]


def render_idle_crew_process() -> str:
    lines = [
        "## The crew",
        "",
        "Three agents will run in order once you generate a story. "
        "Execution details appear here after a run.",
        "",
    ]
    for stage, summary in stage_summaries():
        ui = STAGE_UI[stage]
        lines.append(f"### {ui['icon']} {ui['name']} — {summary}")
        lines.append("↓")
        lines.append("")
    lines.pop()  # drop the last arrow
    lines.pop()
    return "\n".join(lines)
