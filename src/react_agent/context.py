"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    # Conversation loop limits (max exchanges between nodes)
    max_captain_officer1: int = field(
        default=3,
        metadata={
            "description": "Maximum Captain↔Officer1 exchanges allowed."
        },
    )
    max_officer1_officer2: int = field(
        default=2,
        metadata={
            "description": "Maximum Officer1↔Officer2 exchanges allowed."
        },
    )

    # Maximum recursion depth (assistant steps before stopping).
    max_depth: int = field(
        default=25,
        metadata={
            "description": "Maximum recursion depth per thread (assistant steps before stopping)."
        },
    )

    # Legacy/global prompt (fallback – used only if LangSmith pull fails)
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "Legacy/global system prompt. Role-specific prompts are loaded from LangSmith."
        },
    )

    # LangSmith prompt IDs (can be overridden via ENV: CAPTAIN_PROMPT_ID, OFFICER1_PROMPT_ID, OFFICER2_PROMPT_ID)
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

    # Global default model (used if node-specific model is not set)
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={"description": "The default language model (provider/model-name)."},
    )

    # Node-specific default models (overrides global default if set)
    captain_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-2024-05-13",
        metadata={"description": "Default model for the Captain node (overrides global if set)."},
    )
    officer1_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={"description": "Default model for the First Officer node (overrides global if set)."},
    )
    officer2_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5-mini",
        metadata={"description": "Default model for the Second Officer node (overrides global if set)."},
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    def __post_init__(self) -> None:
        """Fetch ENV vars for attributes that were not passed as args (priority: ENV > Studio > default)."""
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
