"""LangGraph state machine for KSODI-Light (Phase/Forge) — Forge runs real tools."""

from datetime import UTC, datetime
from typing import Any, Iterable, Literal, Optional
from uuid import uuid4

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
from react_agent.tools import DELEGATION_TOOLS_PHASE, DELEGATION_TOOLS_FORGE, TOOLS
from react_agent.utils import load_chat_model

# --- Config: depth limit (default; synced from Context.max_depth on first Phase step) ---
MAX_DEPTH: int = 25

# --- Conversation loop cap (synced from Context; checked against State counters) ---
# Phase <-> Forge
MAX_PHASE_FORGE_LOOPS: int = 3

# --- LangSmith client & helper ---
_ls = Client()


def _ls_messages(prompt_id: str, **kwargs: Any) -> list[BaseMessage]:
    try:
        prompt = _ls.pull_prompt(prompt_id)
        value = prompt.invoke(kwargs)
        return list(value.to_messages())
    except Exception:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]


def _ls_messages_multi(prompt_ids: str, **kwargs: Any) -> list[BaseMessage]:
    handles = [h.strip() for h in (prompt_ids or "").split(",") if h.strip()]
    msgs: list[BaseMessage] = []
    if not handles:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]
    for h in handles:
        msgs.extend(_ls_messages(h, **kwargs))
    return msgs


# --- Phase (strategic node; binds delegation tool to hand off to Forge) ---
async def phase(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Phase step: binds delegation tools for Forge and produces the next AIMessage."""
    # sync limits
    try:
        md = int(getattr(runtime.context, "max_depth", 0))
        if md > 0:
            global MAX_DEPTH
            MAX_DEPTH = md
    except Exception:
        pass
    try:
        global MAX_PHASE_FORGE_LOOPS
        cap = int(getattr(runtime.context, "max_phase_forge", MAX_PHASE_FORGE_LOOPS))
        MAX_PHASE_FORGE_LOOPS = cap if cap >= 0 else MAX_PHASE_FORGE_LOOPS
    except Exception:
        pass

    # soft reset on new Human turn
    base_depth = state.depth
    base_pf = getattr(state, "c1_loops", 0)  # reuse existing counter field
    try:
        last_msg = state.messages[-1] if state.messages else None
        if isinstance(last_msg, HumanMessage):
            base_depth = 0
            base_pf = 0
    except Exception:
        pass

    model_id = runtime.context.phase_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(DELEGATION_TOOLS_PHASE)

    sys_msgs = _ls_messages_multi(
        runtime.context.phase_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
        ai_name=getattr(runtime.context, "ai_name", "AI"),
        ai_language=getattr(runtime.context, "ai_language", "English"),
        ai_role=getattr(runtime.context, "ai_role", ""),
    )

    response = await model.ainvoke([*sys_msgs, *state.messages])
    try:
        response.name = "phase"
    except Exception:
        pass
    return {"messages": [response], "depth": base_depth + 1, "c1_loops": base_pf}


# --- Forge (execution node; REAL tools + post-tool synthesis + official handoff) ---
async def forge(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Forge step: executes real tools (search/time). After tool results, synthesize and handoff to Phase."""
    from langchain_core.messages import ToolMessage, AIMessage

    last = state.messages[-1] if state.messages else None
    if isinstance(last, ToolMessage):
        real_tool_names = {"search", "get_time"}
        last_tool = (getattr(last, "name", "") or "").strip()
        if last_tool in real_tool_names:
            model_id = runtime.context.forge_model or runtime.context.model
            summarizer = load_chat_model(model_id)
            sys_msgs = _ls_messages_multi(
                runtime.context.forge_prompt_id,
                system_time=datetime.now(tz=UTC).isoformat(),
                ai_name=getattr(runtime.context, "ai_name", "AI"),
                ai_language=getattr(runtime.context, "ai_language", "English"),
                ai_role=getattr(runtime.context, "ai_role", ""),
            )
            synth = await summarizer.ainvoke([*sys_msgs, *state.messages])

            try:
                content = synth.content if isinstance(synth.content, str) else str(synth.content)
            except Exception:
                content = ""

            handoff_msg = AIMessage(
                content=content,
                name="forge",
                tool_calls=[{
                    "id": f"call_{uuid4().hex}",
                    "name": "handoff_to_phase",
                    "args": {}
                }],
                additional_kwargs={"invisible": True, "handoff": True},
            )
            return {"messages": [handoff_msg], "depth": state.depth + 1, "c1_loops": state.c1_loops}

    model_id = runtime.context.forge_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(TOOLS)

    sys_msgs = _ls_messages_multi(
        runtime.context.forge_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
        ai_name=getattr(runtime.context, "ai_name", "AI"),
        ai_language=getattr(runtime.context, "ai_language", "English"),
        ai_role=getattr(runtime.context, "ai_role", ""),
    )

    new_pf = getattr(state, "c1_loops", 0) + 1  # reuse counter

    response = await model.ainvoke([*sys_msgs, *state.messages])

    try:
        response.name = "forge"
        response.additional_kwargs = {
            **getattr(response, "additional_kwargs", {}),
            "invisible": True
        }
    except Exception:
        pass

    return {"messages": [response], "depth": state.depth + 1, "c1_loops": new_pf}


# --------- Tool-call inspection (unchanged helpers) ---------
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
            name = getattr(tc, "name", None) or tc.get("tool_name", None)  # type: ignore[attr-defined]
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


def resolve_pending(state: State) -> dict[str, Any]:
    last = state.messages[-1]
    tool_msgs: list[ToolMessage] = []
    if isinstance(last, AIMessage):
        for tc in _iter_tool_calls(last):
            tc_id = tc.get("id") or ""
            name = tc.get("name") or "unknown_tool"
            tool_msgs.append(ToolMessage(tool_call_id=str(tc_id),
                                         content=f"Skipped '{name}' due to limit reached (depth or loop cap)."))
    return {"messages": tool_msgs, "depth": state.depth}


# --------- Routing with depth + loop caps ---------
def route_phase(state: State) -> Literal["__end__", "delegation_tools_phase", "resolve_pending"]:
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "__end__"

    if getattr(state, "c1_loops", 0) >= MAX_PHASE_FORGE_LOOPS:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "__end__"

    last = state.messages[-1]
    if isinstance(last, AIMessage) and _get_tool_call(last, "delegate_phase_to_forge"):
        return "delegation_tools_phase"
    return "__end__"


def route_forge(state: State) -> Literal["tools", "delegation_tools_forge", "phase", "resolve_pending"]:
    if state.depth >= MAX_DEPTH:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "resolve_pending"
        return "phase"
    last = state.messages[-1]
    if isinstance(last, AIMessage):
        if getattr(last, "tool_calls", None):
            if _get_tool_call(last, "handoff_to_phase"):
                return "delegation_tools_forge"
            return "tools"
    return "phase"


# --------- Build Graph ---------
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("phase", phase)
builder.add_node("forge", forge)

builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("delegation_tools_phase", ToolNode(DELEGATION_TOOLS_PHASE))
builder.add_node("delegation_tools_forge", ToolNode(DELEGATION_TOOLS_FORGE))
builder.add_node("resolve_pending", resolve_pending)

builder.add_edge("__start__", "phase")
builder.add_conditional_edges("phase", route_phase)
builder.add_conditional_edges("forge", route_forge)

builder.add_edge("delegation_tools_phase", "forge")
builder.add_edge("delegation_tools_forge", "phase")
builder.add_edge("resolve_pending", "phase")
builder.add_edge("tools", "forge")

graph = builder.compile(name="KSODI-Light—PhaseForge")
