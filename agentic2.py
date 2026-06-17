import os
from typing import TypedDict
from langgraph.graph import StateGraph,END
from dotenv import load_dotenv
from google import genai
import json
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AgentState(TypedDict):
    query:str
    known_facts:dict
    next_tool:str
    final_answer:str
    steps:int
def predict_flood():
    return {
        "severity": "high"
    }

def find_route():
    return {
        "route": "Dwarka Expressway",
        "travel_time": "65 minutes"
    }

def find_fuel():
    return {
        "fuel_station": "HP Petrol Pump"
    }

def find_hospital():
    return {
        "hospital": "Max Hospital"
    }
def planner_node(state):
    state["steps"] += 1
    if state["steps"]>5:
        state["next_tool"]="FINISH"
        return state
    prompt = f"""
    Query:
    {state["query"]}

    Known Facts:
    {state["known_facts"]}

    Available Tools:

    Flood tool -> severity

    Routing tool -> route, travel_time

    Fuel tool -> fuel_station

    Hospital tool -> hospital

    Rules:

    If severity exists, do not choose Flood tool.

    If route and travel_time exist,
    do not choose Routing tool.

    If fuel_station exists,
    do not choose Fuel tool.

    If hospital exists,
    do not choose Hospital tool.

    If enough information exists,
    return FINISH.

    Return JSON only.

    Example:

    {{
        "next_tool":"Flood tool"
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw = response.text
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    result = json.loads(raw)

    state["next_tool"] = result["next_tool"]

    print("\nPlanner:")
    print(state["next_tool"])

    return state
def tool_node(state):

    tool = state["next_tool"]

    print(f"\nExecuting: {tool}")

    if tool.lower() == "flood tool":
        result = predict_flood()

    elif tool.lower() == "routing tool":
        result = find_route()

    elif tool.lower() == "fuel tool":
        result = find_fuel()

    elif tool.lower() == "hospital tool":
        result = find_hospital()

    else:
        result = {}

    state["known_facts"].update(result)

    print("Observation:")
    print(result)

    return state
def answer_node(state):

    prompt = f"""
    User Query:
    {state["query"]}

    Known Facts:
    {state["known_facts"]}

    Generate final answer.
    Use known facts.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    state["final_answer"] = response.text

    return state
def route_after_planner(state):

    if state["next_tool"].upper() == "FINISH":
        return "answer"

    return "tool"
graph=StateGraph(AgentState)
graph.add_node("planner",planner_node)
graph.add_node("tool",tool_node)
graph.add_node("answer",answer_node)
graph.set_entry_point("planner")
graph.add_conditional_edges("planner",route_after_planner,{"tool":"tool","answer":"answer"})
graph.add_edge("tool","planner")
graph.add_edge("answer",END)
app=graph.compile()
initial_state = {
    "query": input("Enter prompt: "),
    "known_facts": { "hospital":"Max Hospital"},
    "next_tool": "",
    "final_answer": "",
    "steps": 0
}
result=app.invoke(initial_state)
print("\nFinal Answer:\n ")
print(result["final_answer"])