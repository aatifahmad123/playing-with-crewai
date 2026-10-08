"""Central configuration for the AI Newsroom demo.

Loads environment variables from the project's ``.env`` file and exposes a
single validated ``Settings`` object. No secret ever leaves this module
except into the LLM client itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from compat import apply_crewai_compat

# Load the .env file that sits next to this module, regardless of the
# directory the app was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Must run before any Agent/LLM is constructed (see compat.py).
apply_crewai_compat()

# --- Model configuration -----------------------------------------------------

# CrewAI routes any "provider/model" string through LiteLLM, so the CrewAI
# model string for OpenAI is "openai/<model-id>".
OPENAI_MODEL_ID = "gpt-4o-mini"
CREWAI_MODEL = f"openai/{OPENAI_MODEL_ID}"

# Per-agent sampling temperatures. The fact-checker gets the coldest setting
# because it must classify claims conservatively; the editor gets the warmest
# because it writes prose.
TEMPERATURES = {
    "researcher": 0.3,
    "fact_checker": 0.1,
    "editor": 0.5,
}

# Bound each agent's ReAct loop so a confused agent cannot run forever.
MAX_ITERATIONS = {
    "researcher": 10,
    "fact_checker": 8,
    "editor": 5,
}

LLM_TIMEOUT_SECONDS = 180  # per LLM call
LLM_MAX_RETRIES = 4

# --- UI options ----------------------------------------------------------------

ARTICLE_STYLES = {
    "Technology News": "a technology publication (e.g. The Verge, Ars Technica) — precise, technically literate, focused on how the technology works and who it affects",
    "Business News": "a business publication (e.g. Bloomberg, Financial Times) — focused on markets, companies, jobs, money and economic consequences",
    "General News": "a general-interest news publication (e.g. Reuters, AP) — accessible to any reader, minimal jargon, broadly relevant",
}

ARTICLE_LENGTHS = {
    "Short": "roughly 400-600 words with 2-3 body sections",
    "Medium": "roughly 700-1000 words with 3-4 body sections",
    "Long": "roughly 1100-1500 words with 4-5 body sections",
}

# --- Run limits -----------------------------------------------------------------

PIPELINE_TIMEOUT_SECONDS = int(os.getenv("NEWSROOM_TIMEOUT_SECONDS", "900"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("NEWSROOM_SEARCH_MAX_RESULTS", "6"))


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings for the newsroom."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    model: str = CREWAI_MODEL
    model_id: str = OPENAI_MODEL_ID

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


SETTINGS = Settings()


def build_llm_kwargs(agent_key: str) -> dict:
    """LLM constructor keyword arguments for a given agent.

    Separate kwargs per agent so each one gets its own temperature while
    sharing the same model, key and retry policy.
    """

    return {
        "model": CREWAI_MODEL,
        "api_key": SETTINGS.openai_api_key,
        "temperature": TEMPERATURES[agent_key],
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
    }
