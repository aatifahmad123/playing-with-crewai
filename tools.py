"""CrewAI tools for the AI Newsroom.

A single, dependency-light web search tool built on DuckDuckGo (the ``ddgs``
package). It needs no API key, which keeps the demo simple while still giving
the Researcher and Fact-Checker access to *real*, current information instead
of asking the model to pretend it researched something.
"""

from __future__ import annotations

import logging

from crewai.tools import tool
from ddgs import DDGS

from config import WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web for current information about a topic or claim.

    Returns a numbered list of results, each with the page title, the full
    source URL and a short snippet. Use it to gather facts, statistics and
    recent developments, and to verify or refute specific claims. Always
    cite the URLs you actually saw — never invent a source.
    """

    query = (query or "").strip()
    if not query:
        return "Search failed: the search query was empty."

    try:
        results = DDGS().text(query, max_results=WEB_SEARCH_MAX_RESULTS)
    except Exception as exc:  # noqa: BLE001 — a tool must never crash the crew
        logger.warning("web_search failed for %r: %s", query, exc)
        return (
            f"Search failed for query '{query}' ({type(exc).__name__}). "
            "Tell the reader this search could not be run and treat any "
            "related claim as NEEDS_SOURCE."
        )

    if not results:
        return (
            f"No results found for '{query}'. Do not invent sources: mark "
            "any claim that relied on this search as NEEDS_SOURCE."
        )

    lines = [f"Web search results for: {query}", ""]
    for i, row in enumerate(results, start=1):
        title = (row.get("title") or "Untitled").strip()
        url = (row.get("href") or row.get("url") or "").strip()
        body = (row.get("body") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        lines.append(f"   Snippet: {body}")
        lines.append("")
    return "\n".join(lines)
