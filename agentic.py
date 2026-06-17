import os
import json
from dotenv import load_dotenv
from google import genai
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def predict_flood():
    return{
        "severity":"high"
    }
def find_route():
    return{
    "route":"Dwarka Expressway",
    "travel_time":"65 minutes"
}

def find_fuel():
    return{
        "fuel_station":"HP Petrol Pump"
    }
def find_hospital():
    return{
        "hospital":"Max Hospital"
    }

def extract_goal(state):
    prompt = f"""
            Extract 
            1.goal
            2.constraints from the given query  
            Query: {state["query"]}
            return json only
            example:
            {{"goal":"reach back Delhi safely",
            "constraints":[
            "motorcycle",
            "low fuel",
            "avoid floods"
            ]
            }}
            """
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    raw=response.text
    raw=raw.replace("```json","")
    raw=raw.replace("```","")
    raw=raw.strip()
    result = json.loads(raw)
    print(result)
    state["goal"]=result["goal"]
    state["constraints"]=result["constraints"]
    return state
def planner(state):
    prompt=f"""
    You are an AI Agent planner tool. You have been given the following information:
    1.Goal: {state["goal"]}
    2.Constraints: {state["constraints"]}
    3.Known Facts: {state["known_facts"]}
    Available Tools:
        1.Flood tool
        2.Routing tool
        3.Fuel tool
        4.Hospital tool
    Choose the best next tool
    If enough information exists, return FINISH
    Return JSON only
    Example: {{
    "next_tool":"Flood Tool"
    }} 
    Known Facts:
    {state["known_facts"]}
    
    Flood tool provides:
    severity
    
    Routing tool provides:
    route, travel_time
    
    Fuel tool provides:
    fuel_station
    
    Hospital tool provides:
    hospital
    
    Do not choose a tool if ALL of its outputs already exist in known_facts.
    
    If severity already exists, do not choose Flood tool.
    
    If route and travel_time already exist, do not choose Routing tool.
    
    If fuel_station already exists, do not choose Fuel tool.
    
    If hospital already exists, do not choose Hospital tool.
    """
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    print(response.text)
    raw=response.text
    raw=raw.replace("```json","")
    raw=raw.replace("```","")
    raw=raw.strip()
    result = json.loads(raw)
    state["next_tool"]=result["next_tool"]
    return state
def execute_tool(state):
    print(f"executing the tool: {state['next_tool']}")
    tool=state["next_tool"]
    if(tool=="Flood tool"):
        result=predict_flood()
    elif(tool=="Routing tool"):
        result=find_route()
    elif(tool=="Fuel tool"):
        result=find_fuel()
    elif(tool=="Hospital tool"):
        result = find_hospital()
    else:
        result={}
    state["known_facts"].update(result)
    print("observation:")
    print(result)
    return state
def generate_final_answer(state):
    prompt=f"""
        Query:{state["query"]}
        Goal:{state["goal"]}
        Constraints:{state["constraints"]}
        Known Facts:{state["known_facts"]}
        Generate a final response
        """
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    state["final_answer"]=response.text
    return state
def run_agent():
    state = {
        "query":"",
        "goal":"",
        "known_facts":{"hospital":"Max Hospital"},
        "constraints":[],
        "next_tool":"",
        "final_answer":""
    }
    state["query"]=input("enter your prompt: ")
    state["goal"] = state["query"]
    print("\nGoal:")
    print(state["goal"])
    print("\nConstraints:")
    print(state["constraints"])
    max_steps = 5
    steps = 0
    while True:
        steps+=1
        if steps > max_steps:
            print("Maximum steps reached")
            break
        state=planner(state)
        print(f"\nPlanner selected: ")
        print(f"\n{state['next_tool']}")
        if(state["next_tool"]=="FINISH"):
            break
        state=execute_tool(state)
    state=generate_final_answer(state)
    print(f"\nFinal Answer: ")
    print(f"{state['final_answer']}")
run_agent()