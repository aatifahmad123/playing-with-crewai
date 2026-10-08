"""AI Newsroom — Gradio UI.

A single-page app: the user submits a topic, a CrewAI crew of three agents
(researcher → fact-checker → editor) produces a story, and every intermediate
output is visible in tabs. The crew runs in a background thread while this
module's generator polls a shared ``Progress`` object to stream status
updates into the UI.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import gradio as gr

from config import ARTICLE_LENGTHS, ARTICLE_STYLES, PIPELINE_TIMEOUT_SECONDS, SETTINGS
from crew import PipelineResult, Progress, MissingApiKeyError, run_pipeline
from utils.formatting import (
    extract_sources,
    render_crew_process,
    render_idle_crew_process,
    render_progress,
    render_sources,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai-newsroom")

POLL_INTERVAL_SECONDS = 1.0

CUSTOM_CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
.masthead {text-align: center; padding: 18px 0 6px 0; border-bottom: 3px double #1a1a1a; margin-bottom: 4px;}
.masthead h1 {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 2.6rem !important;
    letter-spacing: 2px;
    margin: 0 !important;
    text-transform: uppercase;
}
.masthead p {
    font-family: Georgia, serif !important;
    font-style: italic;
    color: #555;
    margin: 4px 0 0 0 !important;
    font-size: 1.05rem;
}
.progress-panel {
    background: #fafafa;
    border: 1px solid #e2e2e2;
    border-radius: 10px;
    padding: 6px 14px;
    min-height: 60px;
    color: #1a1a1a;
}
/* The panel background is light, but Gradio 6's dark theme sets light text —
   force dark text inside the panel so it stays readable. (Gradio's CSS
   processor drops rules with the universal * selector, hence the explicit
   element list.) */
.progress-panel h1, .progress-panel h2, .progress-panel h3, .progress-panel h4,
.progress-panel p, .progress-panel li, .progress-panel strong, .progress-panel em,
.progress-panel code, .progress-panel span, .progress-panel div {
    color: #1a1a1a !important;
}
.article-body {font-family: Georgia, 'Times New Roman', serif !important; line-height: 1.65; font-size: 1.02rem;}
.article-body h1 {font-family: Georgia, serif !important; border-bottom: 1px solid #ddd; padding-bottom: 8px;}
.mono-report {font-size: 0.93rem;}
footer {display: none !important;}
"""

_INPUT_BANNER = (
    "### Welcome to the newsroom 👋\n\n"
    "Enter a topic below. A crew of three AI journalists — a **Researcher**, "
    "a **Fact-Checker** and an **Editor** — will collaborate to produce a "
    "verified news article. You can watch each agent work and read every "
    "intermediate output in the tabs below."
)


def _friendly_error(exc: Exception) -> str:
    """Translate common failures into a message a non-technical user can act on."""

    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, MissingApiKeyError):
        return str(exc)
    if "401" in lowered or "invalid api key" in lowered or "unauthorized" in lowered:
        return (
            "OpenAI rejected the API key (401). Check that OPENAI_API_KEY in "
            "your .env file is valid, then restart the app."
        )
    if "429" in lowered or "rate limit" in lowered:
        return (
            "OpenAI rate limit reached (429). Wait a minute and try again, or "
            "use a shorter article length."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return (
            "A model call timed out. This is usually temporary — try "
            "generating again, or choose a shorter length."
        )
    if "connection" in lowered or "network" in lowered or "temporary" in lowered:
        return (
            "A network problem interrupted a model or search call. Check "
            "your internet connection and try again."
        )
    return f"The pipeline failed before finishing: {message[:400]}"


def generate(topic: str, style: str, length: str):
    """Gradio generator: streams progress, then fills every output tab."""

    topic = (topic or "").strip()
    style = style or "Technology News"
    length = length or "Medium"

    if not SETTINGS.has_api_key:
        yield (
            render_progress(0, error="OPENAI_API_KEY is not configured. Add it to your .env file and restart the app."),
            _INPUT_BANNER,
            "_The Researcher's brief will appear here._",
            "_The Fact-Checker's report will appear here._",
            "_Sources cited by the crew will appear here._",
            render_idle_crew_process(),
        )
        return

    if not topic:
        gr.Warning("Please enter a topic first — e.g. “Impact of AI on software engineering jobs”.")
        yield (
            render_progress(0, error="No topic entered. Type a topic above and press Generate Story again."),
            _INPUT_BANNER,
            "_The Researcher's brief will appear here._",
            "_The Fact-Checker's report will appear here._",
            "_Sources cited by the crew will appear here._",
            render_idle_crew_process(),
        )
        return

    progress = Progress()
    holder: dict[str, object] = {}

    def worker() -> None:
        try:
            holder["result"] = run_pipeline(topic, style, length, progress=progress)
        except Exception as exc:  # noqa: BLE001 — reported to the user below
            logger.exception("pipeline failed")
            progress.fail(_friendly_error(exc))
            holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True, name="newsroom-crew")
    thread.start()

    run_started = time.monotonic()
    stage_started = run_started
    last_completed = 0
    timed_out = False

    while thread.is_alive():
        thread.join(timeout=POLL_INTERVAL_SECONDS)
        completed = progress.completed_count
        if completed > last_completed:
            stage_started = time.monotonic()
            last_completed = completed
        elapsed = time.monotonic() - stage_started
        yield (
            render_progress(completed, elapsed_seconds=elapsed, topic=topic),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        if time.monotonic() - run_started > PIPELINE_TIMEOUT_SECONDS:
            timed_out = True
            break

    if timed_out:
        gr.Warning("The crew took too long and was abandoned. Try a narrower topic or a shorter length.")
        yield (
            render_progress(progress.completed_count, error=f"The pipeline exceeded the {PIPELINE_TIMEOUT_SECONDS // 60}-minute limit and was stopped."),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    if "result" not in holder:
        error = progress.error or "Unknown error."
        gr.Warning("The pipeline failed — see the progress panel for details.")
        yield (
            render_progress(progress.completed_count, error=error, topic=topic),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    result: PipelineResult = holder["result"]  # type: ignore[assignment]
    sources = extract_sources(result.research, result.fact_check)

    yield (
        render_progress(3, topic=topic),
        result.article or "_The Editor returned an empty article._",
        result.research or "_The Researcher returned an empty brief._",
        result.fact_check or "_The Fact-Checker returned an empty report._",
        render_sources(sources),
        render_crew_process(result),
    )


def build_ui() -> gr.Blocks:
    # Gradio 6: custom CSS moves to launch(); Blocks() takes layout only.
    with gr.Blocks(title="AI Newsroom") as demo:
        with gr.Column(elem_classes=["masthead"]):
            gr.Markdown("# AI Newsroom")
            gr.Markdown("Three AI journalists. One verified story.")

        if not SETTINGS.has_api_key:
            gr.Markdown(
                "⚠️ **OPENAI_API_KEY is missing.** Copy `.env.example` to `.env`, "
                "add your OpenAI API key, and restart the app."
            )

        with gr.Row():
            with gr.Column(scale=5):
                topic = gr.Textbox(
                    label="Story topic",
                    placeholder="e.g. Impact of AI on software engineering jobs",
                    lines=1,
                    autofocus=True,
                )
            with gr.Column(scale=2):
                style = gr.Dropdown(
                    choices=list(ARTICLE_STYLES.keys()),
                    value="Technology News",
                    label="Article style",
                )
            with gr.Column(scale=2):
                length = gr.Radio(
                    choices=list(ARTICLE_LENGTHS.keys()),
                    value="Medium",
                    label="Length",
                )
        generate_btn = gr.Button("Generate Story", variant="primary", scale=1)

        progress_md = gr.Markdown(_INPUT_BANNER, elem_classes=["progress-panel"])

        with gr.Tabs(selected="tab-article") as tabs:
            with gr.Tab("📰 Final Article", id="tab-article"):
                article_md = gr.Markdown(
                    "_Your finished article will appear here._",
                    elem_classes=["article-body"],
                )
            with gr.Tab("🔬 Research Report", id="tab-research"):
                research_md = gr.Markdown(
                    "_The Researcher's brief will appear here._",
                    elem_classes=["mono-report"],
                )
            with gr.Tab("🔍 Fact Check", id="tab-factcheck"):
                factcheck_md = gr.Markdown(
                    "_The Fact-Checker's report will appear here._",
                    elem_classes=["mono-report"],
                )
            with gr.Tab("🔗 Sources", id="tab-sources"):
                sources_md = gr.Markdown("_Sources cited by the crew will appear here._")
            with gr.Tab("⚙️ Crew Process", id="tab-process"):
                process_md = gr.Markdown(render_idle_crew_process())

        generate_btn.click(
            fn=generate,
            inputs=[topic, style, length],
            outputs=[progress_md, article_md, research_md, factcheck_md, sources_md, process_md],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(max_size=8)
    open_browser = os.getenv("NEWSROOM_OPEN_BROWSER", "1") == "1" and bool(os.getenv("DISPLAY"))
    demo.launch(
        server_name=os.getenv("NEWSROOM_HOST", "127.0.0.1"),
        server_port=int(os.getenv("NEWSROOM_PORT", "7860")),
        inbrowser=open_browser,
        css=CUSTOM_CSS,
        footer_links=[],  # Gradio 6 replacement for show_api=False
    )


if __name__ == "__main__":
    main()
