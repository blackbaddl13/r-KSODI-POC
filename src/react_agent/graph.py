from typing import Dict, Literal, Iterable, Optional, Any
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
        content=runtime.context.captain_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --- First Officer (GPT-5) ---
async def officer1(state: State, runtime: Runtime[Context]) -> Dict:
    """Officer1 solves or delegates to Officer2 via tool-call; never ends."""
    model = load_chat_model("openai/gpt-5").bind_tools(DELEGATION_TOOLS_OFFICER1)
    system = SystemMessage(
        content=runtime.context.officer1_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --- Second Officer (GPT-5-mini, with REAL tools) ---
async def officer2(state: State, runtime: Runtime[Context]) -> Dict:
    """Officer2 uses real tools (get_time, search); never ends."""
    model = load_chat_model("openai/gpt-5-mini").bind_tools(TOOLS)
    system = SystemMessage(
        content=runtime.context.officer2_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
    )
    response = await model.ainvoke([system, *state.messages])
    return {"messages": [response]}


# --------- Tool-call inspection (robust across shapes) ---------
def _iter_tool_call_names(msg: AIMessage) -> Iterable[str]:
    calls = getattr(msg, "tool_calls", None) or []
    for tc in calls:
        name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
        if name:
            yield str(name)
            continue
        if isinstance(tc, dict):
            name = tc.get("name") or tc.get("tool")
            if name:
                yield str(name)
                continue
            fn = tc.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if name:
                    yield str(name)


def _iter_tool_calls(msg: AIMessage) -> Iterable[dict]:
    calls = getattr(msg, "tool_calls", None) or []
    for tc in calls:
        if isinstance(tc, dict):
            yield tc
        else:
            name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
            args: Any = getattr(tc, "args", None) or {}
            fn = getattr(tc, "function", None)
            if isinstance(fn, dict):
                name = fn.get("name", name)
                args = fn.get("arguments", args) or args
            yield {"name": name, "args": args}


def _get_tool_call(msg: AIMessage, expected: str) -> Optional[dict]:
    for tc in _iter_tool_calls(msg):
        if (tc.get("name") or "").strip() == expected:
            return tc
    return None


# --------- Routing via tool-calls only ---------
def route_captain(state: State) -> Literal["__end__", "delegation_tools_captain"]:
    """Captain may end or delegate (via ToolNode) to Officer1."""
    last = state.messages[-1]
    if isinstance(last, AIMessage) and _get_tool_call(last, "delegate_officer1"):
        return "delegation_tools_captain"
    return "__end__"


def route_officer1(state: State) -> Literal["captain", "delegation_tools_officer1"]:
    """Officer1 returns to Captain unless delegating (via ToolNode) to Officer2 with valid intent."""
    last = state.messages[-1]
    if isinstance(last, AIMessage):
        tc = _get_tool_call(last, "delegate_officer2")
        if tc:
            args = tc.get("args") or {}
            use = None
            if isinstance(args, dict):
                use = args.get("use")
            elif isinstance(args, str):
                if '"use"' in args:
                    try:
                        import json
                        use = json.loads(args).get("use")
                    except Exception:
                        pass
            if use in {"get_time", "search"}:
                return "delegation_tools_officer1"
    return "captain"


def route_officer2(state: State) -> Literal["tools", "officer1"]:
    """Officer2 executes real tools if requested, then loops; otherwise returns to Officer1."""
    last = state.messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "officer1"


# --------- Build Graph ---------
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("captain", captain)
builder.add_node("officer1", officer1)
builder.add_node("officer2", officer2)

# ToolNode for REAL tools (used by Officer2 path)
builder.add_node("tools", ToolNode(TOOLS))

# ToolNodes for Delegation (erzeugen die nötigen ToolMessages)
builder.add_node("delegation_tools_captain", ToolNode(DELEGATION_TOOLS_CAPTAIN))
builder.add_node("delegation_tools_officer1", ToolNode(DELEGATION_TOOLS_OFFICER1))

# Flow
builder.add_edge("__start__", "captain")
builder.add_conditional_edges("captain", route_captain)
builder.add_conditional_edges("officer1", route_officer1)
builder.add_conditional_edges("officer2", route_officer2)

# After delegation tool execution, continue to the right officer
builder.add_edge("delegation_tools_captain", "officer1")
builder.add_edge("delegation_tools_officer1", "officer2")

# ReAct cycle for Officer2 tools
builder.add_edge("tools", "officer2")

graph = builder.compile(name="Ship-Agent-DelegationToolNodes")
