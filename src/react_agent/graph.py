from typing import Dict, Literal, Iterable, Optional, Any
from datetime import datetime, UTC

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage  # NEW
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langsmith import Client  # NEW

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.utils import load_chat_model
from react_agent.tools import (
    TOOLS,
    DELEGATION_TOOLS_CAPTAIN,
    DELEGATION_TOOLS_OFFICER1,
)

# --- Config: depth limit ---
MAX_DEPTH = 5

# --- LangSmith client & helper (Prompts laden) ---
_ls = Client()

def _ls_messages(prompt_id: str, **vars) -> list[BaseMessage]:
    """
    Holt ChatPrompt aus LangSmith und rendert ihn zu Messages.
    Fallback: einfache SystemMessage mit Systemzeit.
    """
    try:
        prompt = _ls.pull_prompt(prompt_id)         # z.B. "system_prompt_captain:latest"
        value = prompt.invoke(vars)                 # Variablen einsetzen (system_time=...)
        return list(value.to_messages())            # -> [SystemMessage, ...]
    except Exception:
        return [SystemMessage(content=f"System time: {vars.get('system_time','')}")]


# --- Captain (GPT-4o-2024-05-13) ---
async def captain(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-4o-2024-05-13").bind_tools(DELEGATION_TOOLS_CAPTAIN)
    sys_msgs = _ls_messages(
        runtime.context.captain_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )
    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {"messages": [response], "depth": state.depth + 1}


# --- First Officer (GPT-5) ---
async def officer1(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-5").bind_tools(DELEGATION_TOOLS_OFFICER1)
    sys_msgs = _ls_messages(
        runtime.context.officer1_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )
    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {"messages": [response], "depth": state.depth + 1}


# --- Second Officer (GPT-5-mini, with REAL tools) ---
async def officer2(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-5-mini").bind_tools(TOOLS)
    sys_msgs = _ls_messages(
        runtime.context.officer2_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )
    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {"messages": [response], "depth": state.depth + 1}


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
            yield {
                "id": tc.get("id"),
                "name": tc.get("name") or tc.get("tool"),
                "args": tc.get("args") or tc.get("function", {}).get("arguments"),
            }
        else:
            name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
            args: Any = getattr(tc, "args", None) or {}
            fn = getattr(tc, "function", None)
            if isinstance(fn, dict):
                name = fn.get("name", name)
                args = fn.get("arguments", args) or args
            yield {"id": getattr(tc, "id", None), "name": name, "args": args}


def _get_tool_call(msg: AIMessage, expected: str) -> Optional[dict]:
    for tc in _iter_tool_calls(msg):
        if (tc.get("name") or "").strip() == expected:
            return tc
    return None


# --------- resolve pending tool_calls when stopping ---------
def resolve_pending(state: State) -> Dict:
    """
    Wenn die letzte AIMessage noch tool_calls enthält (z. B. bei Depth-Limit),
    erzeugen wir passende ToolMessages, damit das Protokoll gültig ist.
    """
    last = state.messages[-1]
    tool_msgs: list[ToolMessage] = []
    if isinstance(last, AIMessage):
        for tc in _iter_tool_calls(last):
            tc_id = tc.get("id") or ""
            name = tc.get("name") or "unknown_tool"
            tool_msgs.append(
                ToolMessage(
                    tool_call_id=str(tc_id),
                    content=f"Skipped '{name}' due to recursion limit (MAX_DEPTH={MAX_DEPTH}).",
                )
            )
    # kein depth++ hier, da kein Model-Schritt
    return {"messages": tool_msgs, "depth": state.depth}


# --------- Routing with depth limit ---------
def route_captain(state: State) -> Literal["__end__", "delegation_tools_captain"]:
    if state.depth >= MAX_DEPTH:
        return "__end__"
    last = state.messages[-1]
    if isinstance(last, AIMessage) and _get_tool_call(last, "delegate_officer1"):
        return "delegation_tools_captain"
    return "__end__"


def route_officer1(state: State) -> Literal["captain", "delegation_tools_officer1", "resolve_pending"]:
    # Bei Limit: offene tool_calls zuerst synthetisch beantworten
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "captain"
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


def route_officer2(state: State) -> Literal["tools", "officer1", "resolve_pending"]:
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "officer1"
    last = state.messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "officer1"


# --------- Build Graph ---------
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("captain", captain)
builder.add_node("officer1", officer1)
builder.add_node("officer2", officer2)

builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("delegation_tools_captain", ToolNode(DELEGATION_TOOLS_CAPTAIN))
builder.add_node("delegation_tools_officer1", ToolNode(DELEGATION_TOOLS_OFFICER1))

# node that resolves pending tool calls when stopping
builder.add_node("resolve_pending", resolve_pending)

builder.add_edge("__start__", "captain")
builder.add_conditional_edges("captain", route_captain)
builder.add_conditional_edges("officer1", route_officer1)
builder.add_conditional_edges("officer2", route_officer2)

builder.add_edge("delegation_tools_captain", "officer1")
builder.add_edge("delegation_tools_officer1", "officer2")
builder.add_edge("resolve_pending", "captain")  # nach synthetic replies finalisiert Captain
builder.add_edge("tools", "officer2")

graph = builder.compile(name="Ship-Agent-DelegationToolNodes")
