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
    max_phase_forge: int = field(
        default=3,
        metadata={"description": "Maximum Phase↔Forge exchanges allowed."},
    )

    # Maximum recursion depth
    max_depth: int = field(
        default=25,
        metadata={"description": "Maximum recursion depth per thread (assistant steps before stopping)."},
    )

    # Legacy/global prompt (fallback)
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={"description": "Legacy/global system prompt. Role-specific prompts are loaded from LangSmith."},
    )

    # LangSmith prompt IDs
    phase_prompt_id: str = field(
        default="ksodi_light_ethics:latest, role_definition_phase:latest, interaction_protocol_phase:latest, personalization_ksodi_light:latest",
        metadata={"description": "LangSmith prompt handle for Phase."},
    )
    forge_prompt_id: str = field(
        default="ksodi_light_ethics:latest, role_definition_forge:latest, interaction_protocol_forge:latest, personalization_ksodi_light:latest",
        metadata={"description": "LangSmith prompt handle for Forge."},
    )

    # Global default model
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={"description": "The default language model (provider/model-name)."},
    )

    # Node-specific default models
    phase_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-2024-05-13",
        metadata={"description": "Default model for the Phase node (overrides global if set)."},
    )
    forge_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={"description": "Default model for the Forge node (overrides global if set)."},
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    # Personalization / Prompt Vars
    ai_name: str = field(
        default="AI",
        metadata={"description": "Name the AI should be called by (used in prompts as {ai_name})."},
    )
    ai_language: str = field(
        default="English",
        metadata={"description": "Fallback language hint for prompts (used as {ai_language})."},
    )

    ai_role: str = field(
        default="Personal assistant",
        metadata={"description": "Fallback role hint for prompts (used as {ai_role})."},
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
