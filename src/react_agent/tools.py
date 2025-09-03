"""Tools for web search, time retrieval, and delegation routing."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, List, cast

import pytz
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context


# --------- Real tools (used by Officer2) ---------

@tool
async def search(query: str) -> dict[str, Any] | None:
    """Search the web (Tavily). Best for fresh/current info."""
    if not os.getenv("TAVILY_API_KEY"):
        # Fail soft so the agent can continue without crashing
        return {"error": "TAVILY_API_KEY not set"}
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(dict[str, Any], result)


@tool
def get_time() -> str:
    """Return current time in Europe/Berlin (24h)."""
    tz = pytz.timezone("Europe/Berlin")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


# Export: real tools set for Officer2
TOOLS: List[Callable[..., Any]] = [search, get_time]


# --------- Delegation tools (routing via tool-calls) ---------

@tool
def delegate_officer1() -> str:
    """Delegate the task to First Officer."""
    return "ok"


@tool
def delegate_officer2(use: str, reason: str = "") -> str:
    """Delegate to Second Officer only when an external tool is needed.

    `use` must be one of: ``'get_time'`` or ``'search'``.
    """
    if use not in {"get_time", "search"}:
        return "reject"
    return f"ok:{use}"


# Exports for binding by role
DELEGATION_TOOLS_CAPTAIN: List[Callable[..., Any]] = [delegate_officer1]
DELEGATION_TOOLS_OFFICER1: List[Callable[..., Any]] = [delegate_officer2]
