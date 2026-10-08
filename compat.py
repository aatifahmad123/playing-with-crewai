"""Compatibility shims for the installed CrewAI version.

CrewAI 1.15.25 tags the system and user messages it sends with
``cache_breakpoint: True`` so provider adapters can implement prompt caching.
For models routed through LiteLLM (like OpenAI) nobody strips that marker, and
providers with strict request schemas reject the whole call with an
"unrecognized field"-style error (the original trigger was Groq's
``property 'cache_breakpoint' is unsupported``).

``apply_crewai_compat()`` neutralizes the marker function so messages reach
the provider clean. This only disables a prompt-cache *optimization* that
never worked through LiteLLM routing in the first place — behaviour and
outputs are unchanged.

The shim is defensive: if a future CrewAI renames or removes the marker
function, it does nothing (and the bug it worked around is gone).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_crewai_compat() -> None:
    try:
        from crewai.llms import cache as llm_cache
    except ImportError:
        try:
            import crewai.llms.cache as llm_cache  # type: ignore[no-redef]
        except ImportError:
            return  # CrewAI internals changed; nothing to patch.

    original = getattr(llm_cache, "mark_cache_breakpoint", None)
    if original is None:
        return

    def mark_cache_breakpoint_noop(message: dict) -> dict:  # noqa: ANN001
        return dict(message)

    llm_cache.mark_cache_breakpoint = mark_cache_breakpoint_noop
    logger.info(
        "Applied CrewAI compat shim: cache_breakpoint message markers disabled."
    )
