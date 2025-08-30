from typing import Dict, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.utils import load_chat_model
from react_agent.tools import TOOLS


# --- Captain (GPT-4o-2024-05-13) ---
async def captain(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-4o-2024-05-13")
    response = await model.ainvoke([*state.messages, HumanMessage(content=state.input)])
    return {"messages": [response]}


# --- First Officer (GPT-5) ---
async def officer1(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-5")
    response = await model.ainvoke([*state.messages])
    return {"messages": [response]}


# --- Second Officer (GPT-5-mini, with tools) ---
async def officer2(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-5-mini").bind_tools(TOOLS)
    response = await model.ainvoke([*state.messages])
    return {"messages": [response]}


# --- Routing Captain ---
def route_captain(state: State) -> Literal["__end__", "officer1"]:
    """Captain can finish directly or delegate to Officer1."""
    last = state.messages[-1].content.lower()
    if "off1" in last or "delegate" in last:
        return "officer1"
    return "__end__"


# --- Routing Officer1 ---
def route_officer1(state: State) -> Literal["captain", "officer2"]:
    """Officer1 can finish via Captain or delegate to Officer2."""
    last = state.messages[-1].content.lower()
    if "off2" in last or "delegate" in last:
        return "officer2"
    return "captain"


# --- Routing Officer2 ---
def route_officer2(state: State) -> Literal["tools", "officer1"]:
    """Officer2 may call tools or return to Officer1."""
    last = state.messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "officer1"


# --- Build Graph ---
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("captain", captain)
builder.add_node("officer1", officer1)
builder.add_node("officer2", officer2)
builder.add_node("tools", ToolNode(TOOLS))

# Flow
builder.add_edge("__start__", "captain")
builder.add_conditional_edges("captain", route_captain)
builder.add_conditional_edges("officer1", route_officer1)
builder.add_conditional_edges("officer2", route_officer2)

# Tools cycle
builder.add_edge("tools", "officer2")

graph = builder.compile(name="Ship-Agent")
