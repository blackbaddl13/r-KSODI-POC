"""LangGraph state machine for the Ship-Agent (Captain/Officer1/Officer2) with delegation."""

from datetime import UTC, datetime
from typing import Any, Iterable, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langsmith import Client

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import DELEGATION_TOOLS_CAPTAIN, DELEGATION_TOOLS_OFFICER1, TOOLS
from react_agent.utils import load_chat_model

# --- Config: depth limit (default; synced from Context.max_depth on first captain) ---
MAX_DEPTH: int = 25

# --- Conversation loop caps (synced from Context if present; checked against State counters) ---
# Captain <-> Officer1
MAX_C1_LOOPS: int = 3
# Officer1 <-> Officer2
MAX_O1O2_LOOPS: int = 2

# --- LangSmith client & helper ---
_ls = Client()


def _ls_messages(prompt_id: str, **kwargs: Any) -> list[BaseMessage]:
    """Load a LangSmith ChatPrompt by id and render it to messages.

    Fallback: returns a simple SystemMessage with current system time.
    """
    try:
        prompt = _ls.pull_prompt(prompt_id)  # e.g., "system_prompt_captain:latest"
        value = prompt.invoke(kwargs)        # inject variables (system_time=...)
        return list(value.to_messages())     # -> [SystemMessage, ...]
    except Exception:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]


def _ls_messages_multi(prompt_ids: str, **kwargs: Any) -> list[BaseMessage]:
    """Load one or more LangSmith ChatPrompts from a CSV handle list and concatenate their messages.

    Example: "firstOfficer:latest, firstOfficerAddition:latest, firstOfficerEmotions:1.0"
    """
    handles = [h.strip() for h in (prompt_ids or "").split(",") if h.strip()]
    msgs: list[BaseMessage] = []
    if not handles:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]
    for h in handles:
        msgs.extend(_ls_messages(h, **kwargs))
    return msgs


# --- Captain (uses context-configured model) ---
async def captain(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Captain step: binds delegation tools for Officer1 and produces the next AIMessage."""
    # Sync depth and loop caps from runtime context once per run (fallback to module defaults).
    try:
        md = int(getattr(runtime.context, "max_depth", 0))
        if md > 0:
            global MAX_DEPTH
            MAX_DEPTH = md
    except Exception:
        pass
    try:
        global MAX_C1_LOOPS, MAX_O1O2_LOOPS
        c1_cap = int(getattr(runtime.context, "max_captain_officer1", MAX_C1_LOOPS))
        o1o2_cap = int(getattr(runtime.context, "max_officer1_officer2", MAX_O1O2_LOOPS))
        MAX_C1_LOOPS = c1_cap if c1_cap >= 0 else MAX_C1_LOOPS
        MAX_O1O2_LOOPS = o1o2_cap if o1o2_cap >= 0 else MAX_O1O2_LOOPS
    except Exception:
        pass

    # --- Soft-Reset der Zähler zu Beginn eines neuen Human-Turns ---
    base_depth = state.depth
    base_c1 = getattr(state, "c1_loops", 0)
    base_o1o2 = getattr(state, "o1o2_loops", 0)
    try:
        last_msg = state.messages[-1] if state.messages else None
        if isinstance(last_msg, HumanMessage):
            base_depth = 0
            base_c1 = 0
            base_o1o2 = 0
    except Exception:
        pass

    # Resolve node-specific model with fallback to global default
    model_id = runtime.context.captain_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(DELEGATION_TOOLS_CAPTAIN)

    # Pull (evtl. gestückelte) system messages via LangSmith prompt handles
    sys_msgs = _ls_messages_multi(
        runtime.context.captain_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )

    # Invoke the model
    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {
        "messages": [response],
        "depth": base_depth + 1,
        "c1_loops": base_c1,
        "o1o2_loops": base_o1o2,
    }


# --- First Officer (uses context-configured model) ---
async def officer1(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """First Officer step: may delegate to Officer2 when external tools are needed."""
    # Resolve node-specific model with fallback to global default
    model_id = runtime.context.officer1_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(DELEGATION_TOOLS_OFFICER1)

    sys_msgs = _ls_messages_multi(
        runtime.context.officer1_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )

    # Increment Captain<->Officer1 loop counter because we just delegated to Officer1
    new_c1 = getattr(state, "c1_loops", 0) + 1

    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {"messages": [response], "depth": state.depth + 1, "c1_loops": new_c1}


# --- Second Officer (uses context-configured model, with REAL tools) ---
async def officer2(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Second Officer step: executes real tools (search/time) and advances the dialogue."""
    # Resolve node-specific model with fallback to global default
    model_id = runtime.context.officer2_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(TOOLS)

    sys_msgs = _ls_messages_multi(
        runtime.context.officer2_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
    )

    # Increment Officer1<->Officer2 loop counter because we just delegated to Officer2
    new_o1o2 = getattr(state, "o1o2_loops", 0) + 1

    response = await model.ainvoke([*sys_msgs, *state.messages])
    return {"messages": [response], "depth": state.depth + 1, "o1o2_loops": new_o1o2}


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
                name2 = fn.get("name")
                if name2:
                    yield str(name2)


def _iter_tool_calls(msg: AIMessage) -> Iterable[dict[str, Any]]:
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


def _get_tool_call(msg: AIMessage, expected: str) -> Optional[dict[str, Any]]:
    for tc in _iter_tool_calls(msg):
        if (tc.get("name") or "").strip() == expected:
            return tc
    return None


# --------- resolve pending tool_calls when stopping ---------
def resolve_pending(state: State) -> dict[str, Any]:
    """Synthesize ToolMessages for any pending tool_calls when we stop due to a limit.

    Ensures the transcript remains valid by replying to tool_calls before finalizing.
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
                    content="Skipped '{}' due to limit reached (depth or loop cap).".format(name),
                )
            )
    # no depth++ here, since no model step
    return {"messages": tool_msgs, "depth": state.depth}


# --------- Routing with depth + loop caps ---------
def route_captain(state: State) -> Literal["__end__", "delegation_tools_captain", "resolve_pending"]:
    """Captain routing: end or delegate to Officer1 via delegation tools."""
    # Depth cap
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "__end__"

    # Loop cap for Captain <-> Officer1
    if getattr(state, "c1_loops", 0) >= MAX_C1_LOOPS:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "__end__"

    last = state.messages[-1]
    if isinstance(last, AIMessage) and _get_tool_call(last, "delegate_officer1"):
        return "delegation_tools_captain"
    return "__end__"


def route_officer1(state: State) -> Literal["captain", "delegation_tools_officer1", "resolve_pending"]:
    """Officer1 routing: hand off to Officer2 only for allowed external tools."""
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "captain"

    # Respect max loop cap for Officer1 <-> Officer2
    if getattr(state, "o1o2_loops", 0) >= MAX_O1O2_LOOPS:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "captain"

    last = state.messages[-1]
    if isinstance(last, AIMessage):
        tc = _get_tool_call(last, "delegate_officer2")
        if tc:
            args = tc.get("args") or {}
            use: Optional[str] = None
            if isinstance(args, dict):
                use = args.get("use")
            elif isinstance(args, str) and '"use"' in args:
                try:
                    import json
                    use = json.loads(args).get("use")
                except Exception:
                    pass
            if use in {"get_time", "search"}:
                return "delegation_tools_officer1"
    return "captain"


def route_officer2(state: State) -> Literal["tools", "officer1", "resolve_pending"]:
    """Officer2 routing: execute tools if present, otherwise return to Officer1."""
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
builder.add_edge("resolve_pending", "captain")  # after synthetic replies captain finalizes
builder.add_edge("tools", "officer2")

graph = builder.compile(name="Ship-Agent-DelegationToolNodes")
