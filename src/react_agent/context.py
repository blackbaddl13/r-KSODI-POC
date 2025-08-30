"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    # Legacy/global prompt (Fallback – wird in graph.py nur genutzt, wenn LangSmith-Pull fehlschlägt)
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "Legacy/global system prompt. Role-specific prompts are loaded from LangSmith."
        },
    )

    # LangSmith Prompt IDs (können via ENV überschrieben werden: CAPTAIN_PROMPT_ID, OFFICER1_PROMPT_ID, OFFICER2_PROMPT_ID)
    captain_prompt_id: str = field(
        default="system_prompt_captain:latest",
        metadata={"description": "LangSmith prompt handle for the Captain (e.g., 'captain:latest')."},
    )
    officer1_prompt_id: str = field(
        default="system_prompt_officer1:latest",
        metadata={"description": "LangSmith prompt handle for the First Officer (e.g., 'officer1:latest')."},
    )
    officer2_prompt_id: str = field(
        default="system_prompt_officer2:latest",
        metadata={"description": "LangSmith prompt handle for the Second Officer (e.g., 'officer2:latest')."},
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={"description": "The default language model (provider/model-name)."},
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args (uppercased field names)."""
        for f in fields(self):
            if not f.init:
                continue
            env_key = f.name.upper()
            cur = getattr(self, f.name)
            os_val = os.environ.get(env_key)
            if os_val is not None:
                setattr(self, f.name, os_val)
            else:
                setattr(self, f.name, cur)
