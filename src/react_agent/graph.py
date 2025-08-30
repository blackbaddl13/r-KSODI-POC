from typing import Dict, Literal
from datetime import datetime, UTC

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.utils import load_chat_model
from react_agent.tools import (
    TOOLS,  # real tools for Officer2
    DELEGATION_TOOLS_CAPTAIN,
    DELEGATION_TOOLS_OFFICER1,
)


# --- Captain (GPT-4o-2024-05-13) ---
async def captain(state: State, runtime: Runtime[Context]) -> Dict:
    """Captain answers directly or delegates to Officer1 via tool-call."""
    model = load_chat_model("openai/gpt-4o-2024-05-13").bind_tools(DELEGATION_TOOLS_CAPTAIN)
    system = SystemMessage(
        content=runtime.context.captain_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --- First Officer (GPT-5) ---
async def officer1(state: State, runtime: Runtime[Context]) -> Dict:
    """Officer1 solves or delegates to Officer2 via tool-call."""
    model = load_chat_model("openai/gpt-5").bind_tools(DELEGATION_TOOLS_OFFICER1)
    system = SystemMessage(
        content=runtime.context.officer1_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --- Second Officer (GPT-5-mini, with REAL tools) ---
async def officer2(state: State, runtime: Runtime[Context]) -> Dict:
    """Officer2 has real tools (get_time, search)."""
    model = load_chat_model("openai/gpt-5-mini").bind_tools(TOOLS)
    system = SystemMessage(
        content=runtime.context.officer2_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --------- Routing via tool-calls only ---------

def _has_tool_call(msg: AIMessage, name: str) -> bool:
    return any(getattr(tc, "name", "") == name for tc in (msg.tool_calls or []))

def route_captain(state: State) -> Literal["__end__", "officer1"]:
    last = state.messages[-1]
    if isinstance(last, AIMessage) and _has_tool_call(last, "delegate_officer1"):
        return "officer1"
    return "__end__"

def route_officer1(state: State) -> Literal["captain", "officer2"]:
    last = state.messages[-1]
    if isinstance(last, AIMessage) and _has_tool_call(last, "delegate_officer2"):
        return "officer2"
    return "captain"

def route_officer2(state: State) -> Literal["tools", "officer1"]:
    # Officer2 may or may not call REAL tools; if yes, execute ToolNode then loop back.
    last = state.messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "officer1"


# --------- Build Graph ---------

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("captain", captain)
builder.add_node("officer1", officer1)
builder.add_node("officer2", officer2)

# ToolNode for REAL tools (used only by Officer2 path)
builder.add_node("tools", ToolNode(TOOLS))

# Flow
builder.add_edge("__start__", "captain")
builder.add_conditional_edges("captain", route_captain)
builder.add_conditional_edges("officer1", route_officer1)
builder.add_conditional_edges("officer2", route_officer2)

# ReAct cycle for Officer2 tools
builder.add_edge("tools", "officer2")

graph = builder.compile(name="Ship-Agent")
