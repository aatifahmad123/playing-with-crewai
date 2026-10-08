"""The three AI Newsroom tasks and their context chaining.

The dependency chain is expressed with CrewAI's ``context`` mechanism:

    research_task
        ↓ (context)
    fact_check_task
        ↓ (context: research + fact check)
    editing_task

Because ``context`` must reference the *same* Task objects, ``build_tasks``
creates all three together and wires them up before returning them.

``{topic}``, ``{style}`` and ``{length}`` are CrewAI input placeholders that
are filled from ``crew.kickoff(inputs=...)``.
"""

from __future__ import annotations

from crewai import Task

from agents import editor_agent, fact_checker_agent, researcher_agent


def _research_task() -> Task:
    return Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Use the web_search tool — run several searches with different "
            "angles and queries until you genuinely understand the topic. "
            "Never fabricate information or sources; only report what the "
            "searches actually returned.\n\n"
            "Produce structured research notes containing these sections:\n"
            "1. Topic overview\n"
            "2. Key developments\n"
            "3. Important facts\n"
            "4. Important statistics (with numbers and dates)\n"
            "5. Key organizations and people\n"
            "6. Timeline (where relevant)\n"
            "7. Claims requiring verification\n"
            "8. Potentially misleading or uncertain claims\n"
            "9. Suggested article angles\n"
            "10. Sources (a list of every page you used, with title and URL)\n\n"
            "Rules:\n"
            "- Clearly distinguish established facts from uncertain or "
            "disputed information (label them).\n"
            "- Prefer recent, authoritative sources: official organizations, "
            "government sources, research papers, reputable news "
            "organizations, company announcements, academic institutions.\n"
            "- If a search fails or returns nothing, say so explicitly "
            "instead of guessing.\n"
            "- Do NOT write the final article. You are preparing raw "
            "material for a fact checker and an editor."
        ),
        expected_output=(
            "A markdown research brief with the ten sections listed above. "
            "Every factual claim that came from a search result must keep "
            "its source URL nearby so the fact checker can verify it. "
            "Uncertain or disputed items must be labelled as such."
        ),
        agent=researcher_agent(),
    )


def _fact_check_task(research: Task) -> Task:
    return Task(
        description=(
            "You are given a research brief about: {topic}\n\n"
            "Extract the most important factual claims from the brief and "
            "evaluate each one. Where useful, use the web_search tool to "
            "check a claim against live sources — especially statistics, "
            "dates, quotes and surprising assertions. Do not invent "
            "citations: if you cannot verify something, mark it "
            "NEEDS_SOURCE.\n\n"
            "Format every claim you review exactly like this:\n"
            "CLAIM: <the claim, in one sentence>\n"
            "STATUS: VERIFIED | QUESTIONABLE | UNSUPPORTED | NEEDS_SOURCE\n"
            "CONFIDENCE: HIGH | MEDIUM | LOW\n"
            "REASONING: <why you assigned this status>\n"
            "RECOMMENDED_CORRECTION: <what the article should say instead, "
            "or 'none'>\n\n"
            "Cover, at minimum, the claims most likely to end up in the "
            "final article. Distinguish fact, interpretation, prediction "
            "and opinion.\n\n"
            "End your report with these sections:\n"
            "- Verified facts (safe to print)\n"
            "- Claims that should be removed\n"
            "- Claims that need qualification\n"
            "- Corrections\n"
            "- Overall reliability assessment of the research brief\n\n"
            "Do NOT write the final article."
        ),
        expected_output=(
            "A markdown fact-check report: a series of CLAIM/STATUS/"
            "CONFIDENCE/REASONING/RECOMMENDED_CORRECTION blocks followed by "
            "the closing summary sections (verified facts, removals, "
            "qualifications, corrections, overall reliability assessment)."
        ),
        agent=fact_checker_agent(),
        context=[research],
    )


def _editing_task(research: Task, fact_check: Task) -> Task:
    return Task(
        description=(
            "Write the final news article about: {topic}\n\n"
            "You have two inputs: the research brief and the fact-check "
            "report. Use them as follows:\n"
            "- Do not include any claim the fact checker marked UNSUPPORTED "
            "unless it is clearly attributed as an unverified claim or "
            "prediction.\n"
            "- Resolve contradictions using the fact checker's recommended "
            "corrections.\n"
            "- Prefer the fact checker's VERIFIED facts over everything "
            "else.\n"
            "- Never invent facts, quotes, numbers or sources.\n"
            "- Clearly separate facts from predictions and opinions in the "
            "prose.\n\n"
            "Style: {style}\n"
            "Length: {length}\n\n"
            "The article must follow this exact markdown skeleton:\n"
            "# <Headline>\n\n"
            "<Subheadline — one sentence>\n\n"
            "## Introduction\n\n...\n\n"
            "## <Section 1>\n\n...\n\n"
            "## <Section 2>\n\n...\n\n"
            "## <Section 3>\n\n...\n\n"
            "## What the Evidence Says\n\n...\n\n"
            "## Fact Check Summary\n\n<short summary of what was verified, "
            "qualified or removed>\n\n"
            "Rules:\n"
            "- Neutral journalistic tone, no unnecessary hype.\n"
            "- Never mention the research process, agents, prompts or the "
            "fact-checking workflow inside the article — it must read like "
            "ordinary published journalism.\n"
            "- Output only the article markdown, nothing before or after."
        ),
        expected_output=(
            "A complete markdown news article following the exact skeleton "
            "above, in the requested style and length, ready for publication."
        ),
        agent=editor_agent(),
        context=[research, fact_check],
    )


def build_tasks() -> tuple[Task, Task, Task]:
    """Create (research_task, fact_check_task, editing_task) with context wired."""

    research = _research_task()
    fact_check = _fact_check_task(research)
    editing = _editing_task(research, fact_check)
    return research, fact_check, editing
