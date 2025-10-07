# SPDX-License-Identifier: MIT
"""LangGraph state machine for KSODI-Light (Phase/Forge) — Forge runs real tools."""

from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Iterable, Literal
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langsmith import Client

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import DELEGATION_TOOLS_FORGE, DELEGATION_TOOLS_PHASE, TOOLS
from react_agent.utils import get_message_text, load_chat_model, strip_messages

# Limits (synced from Context at runtime)
MAX_DEPTH: int = 25
MAX_PHASE_FORGE_LOOPS: int = 3

_ls = Client()

# --- LangSmith prompt pull + strip (cached) ---
@lru_cache(maxsize=64)
def _pull_prompt_cached(prompt_id: str) -> Any:
    """Cache LangSmith prompt pulls to reduce API latency."""
    return _ls.pull_prompt(prompt_id)

def _ls_messages(prompt_id: str, **kwargs: Any) -> list[BaseMessage]:
    """Pull one LangSmith prompt, render with kwargs, strip meta headers."""
    try:
        prompt = _pull_prompt_cached(prompt_id)
        value = prompt.invoke(kwargs)
        return strip_messages(list(value.to_messages()))
    except Exception:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]

def _ls_messages_multi(prompt_ids: str, **kwargs: Any) -> list[BaseMessage]:
    """Comma-separated handles → merged messages (stripped)."""
    handles = [h.strip() for h in (prompt_ids or "").split(",") if h.strip()]
    if not handles:
        return [SystemMessage(content=f"System time: {kwargs.get('system_time', '')}")]
    msgs: list[BaseMessage] = []
    for h in handles:
        msgs.extend(_ls_messages(h, **kwargs))
    return strip_messages(msgs)  # idempotent

# --- Phase (non-streaming; safe TTFT-off) ---
async def phase(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Bind delegation tools and produce the next AIMessage (single step)."""
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

    # soft reset on new human turn
    base_depth = state.depth
    base_pf = getattr(state, "c1_loops", 0)
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

    resp = await model.ainvoke([*sys_msgs, *state.messages])
    try:
        resp.name = "phase"
    except Exception:
        pass

    return {"messages": [resp], "depth": base_depth + 1, "c1_loops": base_pf}

# --- Forge (non-streaming; tools + optional synthesis + handoff) ---
async def forge(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Execute real tools. After tool results, synthesize and hand off to Phase."""
    last = state.messages[-1] if state.messages else None

    # post-tool synthesis path for real tools
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
            content = get_message_text(synth) or ""

            handoff_msg = AIMessage(
                content=content,
                name="forge",
                tool_calls=[{"id": f"call_{uuid4().hex}", "name": "handoff_to_phase", "args": {}}],
                additional_kwargs={"invisible": True, "handoff": True},
            )
            return {"messages": [handoff_msg], "depth": state.depth + 1, "c1_loops": state.c1_loops}

    # normal forge
    model_id = runtime.context.forge_model or runtime.context.model
    model = load_chat_model(model_id).bind_tools(TOOLS)

    sys_msgs = _ls_messages_multi(
        runtime.context.forge_prompt_id,
        system_time=datetime.now(tz=UTC).isoformat(),
        ai_name=getattr(runtime.context, "ai_name", "AI"),
        ai_language=getattr(runtime.context, "ai_language", "English"),
        ai_role=getattr(runtime.context, "ai_role", ""),
    )

    new_pf = getattr(state, "c1_loops", 0) + 1
    resp = await model.ainvoke([*sys_msgs, *state.messages])
    try:
        resp.name = "forge"
        resp.additional_kwargs = {**getattr(resp, "additional_kwargs", {}), "invisible": True}
    except Exception:
        pass

    return {"messages": [resp], "depth": state.depth + 1, "c1_loops": new_pf}

# --- Tool-call inspection (unchanged helpers) ---
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


def _get_tool_call(msg: AIMessage, expected: str) -> dict[str, Any] | None:
    for tc in _iter_tool_calls(msg):
        if (tc.get("name") or "").strip() == expected:
            return tc
    return None

def resolve_pending(state: State) -> dict[str, Any]:
    """Resolve pending tool calls by skipping them with a ToolMessage. Otherwise API errors can occur."""
    last = state.messages[-1]
    tool_msgs: list[ToolMessage] = []
    if isinstance(last, AIMessage):
        for tc in _iter_tool_calls(last):
            tc_id = tc.get("id") or ""
            name = tc.get("name") or "unknown_tool"
            tool_msgs.append(ToolMessage(tool_call_id=str(tc_id),
                                         content=f"Skipped '{name}' due to limit reached (depth or loop cap)."))
    return {"messages": tool_msgs, "depth": state.depth}

# --- Routing ---
def route_phase(state: State) -> Literal["__end__", "delegation_tools_phase", "resolve_pending"]:
    """Decide next step after Phase."""
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
    """Decide next step after Forge."""
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

# --- Build Graph ---
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
