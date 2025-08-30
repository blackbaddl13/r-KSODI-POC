"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    # Legacy/global prompt (weiter verfügbar, wird aber von rollenbasierten Prompts überlagert)
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "Legacy/global system prompt. Roles below are preferred."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={
            "description": "The default language model (provider/model-name)."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Role-specific Prompts (all with {system_time})
    captain_prompt: str = field(default=prompts.SYSTEM_PROMPT_CAPTAIN)
    officer1_prompt: str = field(default=prompts.SYSTEM_PROMPT_OFFICER1)
    officer2_prompt: str = field(default=prompts.SYSTEM_PROMPT_OFFICER2)

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue
            current = getattr(self, f.name)
            if current == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), current))
