# SPDX-License-Identifier: MIT
"""Tools for web search, time retrieval, and delegation routing."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, List, cast
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context

# --------- Real tools (used by Forge) ---------

@tool
async def search(query: str) -> dict[str, Any] | None:
    """Search the web (Tavily). Best for fresh/current info."""
    if not os.getenv("TAVILY_API_KEY"):
        return {"error": "TAVILY_API_KEY not set"}
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(dict[str, Any], result)


@tool
def get_time() -> str:
    """Return current time in Europe/Berlin (24h)."""
    return datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")


# Export: real tools set (Forge binds these directly)
TOOLS: List[Callable[..., Any]] = [search, get_time]


# --------- Delegation tools (Phase -> Forge) ---------

@tool
def delegate_phase_to_forge() -> str:
    """Delegate the task to Forge."""
    return "ok"


@tool
def handoff_to_phase() -> str:
    """Hand control back to Phase after tools are done."""
    return "ok"


# Exports for binding by role
DELEGATION_TOOLS_PHASE: List[Callable[..., Any]] = [delegate_phase_to_forge]
DELEGATION_TOOLS_FORGE: List[Callable[..., Any]] = [handoff_to_phase]
