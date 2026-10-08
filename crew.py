"""Crew assembly and pipeline execution for the AI Newsroom.

``run_pipeline`` builds a fresh crew per story (agents and tasks are cheap,
stateless objects) and runs it once with ``Crew.kickoff``. A ``Progress``
object is attached as the crew's ``task_callback`` so the UI can watch tasks
complete in real time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from crewai import Crew, Process
from crewai.tasks.task_output import TaskOutput

from config import ARTICLE_LENGTHS, ARTICLE_STYLES, SETTINGS, build_llm_kwargs
from tasks import build_tasks

# Task order labels used by the UI progress panel.
STAGE_LABELS = ("research", "fact_check", "editing")


class MissingApiKeyError(RuntimeError):
    """Raised when OPENAI_API_KEY is not configured."""


@dataclass
class StageInfo:
    """One executed task, with timing and a short output preview."""

    label: str
    agent_role: str
    agent_goal: str
    output_excerpt: str
    seconds: float | None = None


@dataclass
class PipelineResult:
    research: str
    fact_check: str
    article: str
    stages: list[StageInfo] = field(default_factory=list)
    total_seconds: float = 0.0
    model: str = ""
    token_usage: dict = field(default_factory=dict)


class Progress:
    """Thread-safe record of task completions, updated by the crew callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.completed_stages: list[str] = []
        self.error: str | None = None

    def record(self, output: TaskOutput) -> None:
        with self._lock:
            # Sequential process → completions arrive in stage order.
            if len(self.completed_stages) < len(STAGE_LABELS):
                self.completed_stages.append(STAGE_LABELS[len(self.completed_stages)])

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self.completed_stages)

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message


def _stage_info(task, label: str, seconds: float | None) -> StageInfo:
    excerpt = ""
    if task.output is not None and task.output.raw:
        excerpt = task.output.raw.strip()
    return StageInfo(
        label=label,
        agent_role=task.agent.role if task.agent else "Unknown agent",
        agent_goal=task.agent.goal if task.agent else "",
        output_excerpt=excerpt,
        seconds=seconds,
    )


def _task_seconds(task) -> float | None:
    if task.start_time is None or task.end_time is None:
        return None
    delta = task.end_time - task.start_time
    return round(delta.total_seconds(), 1)


def build_crew() -> Crew:
    """Assemble the newsroom: 3 agents, 3 chained tasks, sequential process."""

    research, fact_check, editing = build_tasks()
    return Crew(
        agents=[research.agent, fact_check.agent, editing.agent],
        tasks=[research, fact_check, editing],
        process=Process.sequential,
        verbose=True,
    )


def run_pipeline(
    topic: str,
    style: str = "Technology News",
    length: str = "Medium",
    progress: Progress | None = None,
) -> PipelineResult:
    """Run the full newsroom pipeline. Blocks until the story is done."""

    if not SETTINGS.has_api_key:
        raise MissingApiKeyError(
            "OPENAI_API_KEY is missing. Add it to your .env file "
            "(see .env.example) and restart the app."
        )

    progress = progress or Progress()
    crew = build_crew()
    crew.task_callback = progress.record

    style_guidance = ARTICLE_STYLES.get(style, ARTICLE_STYLES["General News"])
    length_guidance = ARTICLE_LENGTHS.get(length, ARTICLE_LENGTHS["Medium"])

    started = time.monotonic()
    try:
        crew.kickoff(
            inputs={
                "topic": topic.strip(),
                "style": style_guidance,
                "length": length_guidance,
            }
        )
    except MissingApiKeyError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI as a message
        progress.fail(str(exc))
        raise

    total_seconds = round(time.monotonic() - started, 1)

    research_task, fact_check_task, editing_task = crew.tasks

    def _raw(task) -> str:
        return task.output.raw.strip() if task.output and task.output.raw else ""

    stages = [
        _stage_info(research_task, "research", _task_seconds(research_task)),
        _stage_info(fact_check_task, "fact_check", _task_seconds(fact_check_task)),
        _stage_info(editing_task, "editing", _task_seconds(editing_task)),
    ]

    usage = {}
    metrics = crew.usage_metrics
    if metrics is not None:
        usage = {
            "requests": getattr(metrics, "successful_requests", 0),
            "input_tokens": getattr(metrics, "prompt_tokens", 0),
            "output_tokens": getattr(metrics, "completion_tokens", 0),
            "reasoning_tokens": getattr(metrics, "reasoning_tokens", 0),
        }

    return PipelineResult(
        research=_raw(research_task),
        fact_check=_raw(fact_check_task),
        article=_raw(editing_task),
        stages=stages,
        total_seconds=total_seconds,
        model=build_llm_kwargs("researcher")["model"],
        token_usage=usage,
    )
