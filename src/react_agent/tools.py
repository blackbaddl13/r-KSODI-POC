"""This module provides example tools for web scraping, time, and search functionality."""

from __future__ import annotations
from typing import Any, Callable, List, Optional, cast
from datetime import datetime
import os

import pytz
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from react_agent.context import Context


@tool(name="search", return_direct=False)
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search the web (Tavily). Best for fresh/current info."""
    if not os.getenv("TAVILY_API_KEY"):
        # Fail soft so the agent can continue without crashing
        return {"error": "TAVILY_API_KEY not set"}
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


@tool(name="get_time", return_direct=False)
def get_time() -> str:
    """Return current time in Europe/Berlin (24h)."""
    tz = pytz.timezone("Europe/Berlin")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


# Only Officer2 binds these tools in graph.py
TOOLS: List[Callable[..., Any]] = [search, get_time]
