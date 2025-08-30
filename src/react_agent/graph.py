from typing import Dict, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State, InputState
from react_agent.utils import load_chat_model
from react_agent.tools import TOOLS

# --- Captain (GPT-4o) ---
async def captain(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-4o-mini")
    response = await model.ainvoke([*state.messages, HumanMessage(content=state.input)])
    return {"messages": [response]}

# --- First Officer (GPT-5) ---
async def officer1(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-5")
    response = await model.ainvoke([*state.messages])
    return {"messages": [response]}

# --- Second Officer (configurable) ---
async def officer2(state: State, runtime: Runtime[Context]) -> Dict:
    model = load_chat_model("openai/gpt-4o-mini")  # placeholder
    response = await model.ainvoke([*state.messages])
    return {"messages": [response]}

# --- Delegation Router ---
def delegation_response(state: State) -> Literal["captain", "officer2"]:
    # very simple: if "captain" in last output, route back to Captain
    last = state.messages[-1].content.lower()
    if "captain" in last:
        return "captain"
    return "officer2"

# --- Extract JSON Response ---
def extract_json(state: State) -> Dict:
    text = state.messages[-1].content.strip()
    return {"messages": [AIMessage(content=f"Extracted: {text}")]}

# --- Main Switch ---
def main_switch(state: State) -> Literal["officer1", "officer2", "extract_json"]:
    # very simple: keywords in the Captain's output
    last = state.messages[-1].content.lower()
    if "off1" in last or "delegate" in last:
        return "officer1"
    elif "json" in last:
        return "extract_json"
    return "officer2"

# --- Build Graph ---
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("captain", captain)
builder.add_node("officer1", officer1)
builder.add_node("officer2", officer2)
builder.add_node("delegation_response", delegation_response)
builder.add_node("extract_json", extract_json)

# Tools
builder.add_node("tools", ToolNode(TOOLS))

# Edges
builder.add_edge("__start__", "captain")
builder.add_conditional_edges("captain", main_switch)
builder.add_edge("officer1", "delegation_response")
builder.add_conditional_edges("delegation_response", delegation_response)
builder.add_edge("officer2", "officer1")
builder.add_edge("extract_json", "__end__")

graph = builder.compile(name="KSODI-Agent")
