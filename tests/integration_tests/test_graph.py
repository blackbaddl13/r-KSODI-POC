# SPDX-License-Identifier: MIT
from typing import Any, Awaitable, Callable, TypeVar, cast

import pytest
from langsmith import unit

from react_agent import graph
from react_agent.context import Context

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


ASYNCIO = cast(Callable[[F], F], pytest.mark.asyncio)
UNIT = cast(Callable[[F], F], unit)

@ASYNCIO
@UNIT
async def test_react_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        context=Context(system_prompt="You are a helpful AI assistant."),
    )
    assert "harrison" in str(res["messages"][-1].content).lower()
