"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes."""

    is_last_step: IsLastStep = field(default=False)

    # --- New: recursion / delegation depth counter ---
    depth: int = field(default=0)
