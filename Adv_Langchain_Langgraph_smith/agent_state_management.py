
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)


# 1. Define the agent's state
class AgentState(TypedDict):
    incident: str
    diagnosis: str
    retry_count: int
    status: str


# 2. First node: Analyze the incident
def analyze_incident(state: AgentState):
    response = llm.invoke(
        f"""Analyze this production incident.
        Identify the likely cause in 2-3 sentences.

        Incident: {state['incident']}"""
    )

    return {
        "diagnosis": response.content,
        "retry_count": state["retry_count"] + 1,
        "status": "analyzed"
    }


# 3. Second node: Generate recommendations
def recommend_fix(state: AgentState):
    response = llm.invoke(
        f"""Recommend a safe fix for this incident.

        Incident: {state['incident']}
        Diagnosis: {state['diagnosis']}"""
    )

    return {
        "status": "completed",
        "diagnosis": state["diagnosis"]
    }


# 4. Build the workflow
builder = StateGraph(AgentState)

builder.add_node("analyze", analyze_incident)
builder.add_node("recommend", recommend_fix)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "recommend")
builder.add_edge("recommend", END)

app = builder.compile()


# 5. Initialize the state
initial_state = {
    "incident": "ETL pipeline failed due to invalid timestamp format",
    "diagnosis": "",
    "retry_count": 0,
    "status": "started"
}


# 6. Execute the workflow
result = app.invoke(initial_state)

print("Incident:", result["incident"])
print("Diagnosis:", result["diagnosis"])
print("Retry count:", result["retry_count"])
print("Status:", result["status"])
