"""The three AI Newsroom agents.

Each agent is a CrewAI ``Agent`` with a distinct role, goal and backstory.
All three run on the same OpenAI model (``gpt-4o-mini``) but with
role-appropriate temperatures and iteration limits. The Researcher and
Fact-Checker share the ``web_search`` tool; the Editor works purely on the
earlier agents' outputs.
"""

from __future__ import annotations

from crewai import Agent, LLM

from config import MAX_ITERATIONS, build_llm_kwargs
from tools import web_search


def _llm(agent_key: str) -> LLM:
    return LLM(**build_llm_kwargs(agent_key))


def researcher_agent() -> Agent:
    return Agent(
        role="Senior News Researcher",
        goal=(
            "Research the user's topic and produce a comprehensive, "
            "well-organized research brief that another journalist could "
            "write a story from without doing the research themselves."
        ),
        backstory=(
            "You are an experienced technology journalist and researcher. "
            "You investigate topics carefully, identify the most important "
            "developments, distinguish facts from speculation, and organize "
            "information so another journalist can use it directly. You are "
            "honest about what you could not find."
        ),
        tools=[web_search],
        llm=_llm("researcher"),
        max_iter=MAX_ITERATIONS["researcher"],
        verbose=True,
    )


def fact_checker_agent() -> Agent:
    return Agent(
        role="Investigative Fact Checker",
        goal=(
            "Review the researcher's report and determine which claims are "
            "reliable, questionable, unsupported, or potentially misleading, "
            "so the final article never presents speculation as fact."
        ),
        backstory=(
            "You are a meticulous fact-checking journalist. Your job is to "
            "challenge assumptions, identify unsupported claims, detect "
            "contradictions, and ensure the final article does not present "
            "speculation as fact. You are skeptical by nature and you show "
            "your reasoning."
        ),
        tools=[web_search],
        llm=_llm("fact_checker"),
        max_iter=MAX_ITERATIONS["fact_checker"],
        verbose=True,
    )


def editor_agent() -> Agent:
    return Agent(
        role="Senior News Editor",
        goal=(
            "Turn the research and fact-checking results into a polished, "
            "balanced, readable news article that survives editorial review."
        ),
        backstory=(
            "You are an experienced editor at a high-quality technology "
            "publication. You transform research into clear journalism while "
            "preserving factual accuracy and avoiding sensationalism. You "
            "would rather cut a dramatic claim than print an unsourced one."
        ),
        tools=[],  # the editor synthesizes the earlier outputs, it does not research
        llm=_llm("editor"),
        max_iter=MAX_ITERATIONS["editor"],
        verbose=True,
    )
