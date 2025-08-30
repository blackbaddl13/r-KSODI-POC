"""This module provides example tools for web scraping, time, and search functionality."""

from typing import Any, Callable, List, Optional, cast
from datetime import datetime
import pytz

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context


# --- Web Search Tool ---
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results using Tavily."""
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# --- Time Tool (Berlin timezone, like in n8n) ---
def get_time() -> str:
    """Return the current time in Europe/Berlin (24h format)."""
    tz = pytz.timezone("Europe/Berlin")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


# --- Export Tools ---
TOOLS: List[Callable[..., Any]] = [
    search,
    get_time,
]
