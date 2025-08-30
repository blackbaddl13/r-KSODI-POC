"""This module provides tools for web search, time, and delegation routing."""

from __future__ import annotations
from typing import Any, Callable, List, Optional, cast
from datetime import datetime
import os

import pytz
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from react_agent.context import Context


# --------- Real tools (only Officer2 uses these) ---------

@tool(name="search", return_direct=False)
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search the web (Tavily). Best for fresh/current info."""
    if not os.getenv("TAVILY_API_KEY"):
        return {"error": "TAVILY_API_KEY not set"}
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any]), await wrapped.ainvoke({"query": query})

@tool(name="get_time", return_direct=False)
def get_time() -> str:
    """Return current time in Europe/Berlin (24h)."""
    tz = pytz.timezone("Europe/Berlin")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

# Export: real tools set for Officer2
TOOLS: List[Callable[..., Any]] = [search, get_time]


# --------- Delegation tools (routing via tool-calls) ---------
# Captain uses this one to delegate to Officer1
@tool(name="delegate_officer1", return_direct=False)
def delegate_officer1() -> str:
    """Delegate the task to First Officer."""
    return "ok"

# Officer1 uses this one to delegate to Officer2
@tool(name="delegate_officer2", return_direct=False)
def delegate_officer2() -> str:
    """Delegate the task to Second Officer."""
    return "ok"

# Exports for binding by role
DELEGATION_TOOLS_CAPTAIN: List[Callable[..., Any]] = [delegate_officer1]
DELEGATION_TOOLS_OFFICER1: List[Callable[..., Any]] = [delegate_officer2]
