
import os
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
)


# ============================================================
# 2. SHARED STATE
# All agents read and update this shared incident state.
# ============================================================

class IncidentState(TypedDict):
    incident: str
    logs: str
    metrics: str
    deployment: str
    findings: list[str]
    next_agent: str
    decision_reason: str
    steps: int
    final_report: str


# ============================================================
# 3. SUPERVISOR OUTPUT SCHEMA
# The LLM must choose one valid next destination.
# ============================================================

class SupervisorDecision(BaseModel):
    next_agent: Literal[
        "logs",
        "metrics",
        "deployment",
        "report",
    ] = Field(description="The next agent to execute")

    reason: str = Field(
        description="Why this agent should execute next"
    )


supervisor_llm = llm.with_structured_output(
    SupervisorDecision
)


# ============================================================
# 4. SPECIALIST TOOLS
# Replace these demo functions with real APIs later.
# ============================================================

def search_logs(incident: str) -> str:
    """Demo log-search tool."""
    return (
        "Found 500 errors in the payment API. "
        "Database connection timeout occurred repeatedly."
    )


def get_metrics(incident: str) -> str:
    """Demo monitoring-metrics tool."""
    return (
        "API error rate: 40%. "
        "Average latency: 8 seconds. "
        "Database connection utilization: 98%."
    )


def get_deployment_history(incident: str) -> str:
    """Demo deployment-history tool."""
    return (
        "Version v2.8 was deployed 10 minutes before "
        "the incident. It introduced a database "
        "connection-pool configuration change."
    )


# ============================================================
# 5. SUPERVISOR AGENT
# Dynamically chooses the next specialist.
# ============================================================

def supervisor(state: IncidentState):
    print("\n" + "=" * 60)
    print("SUPERVISOR AGENT")
    print("=" * 60)

    decision = supervisor_llm.invoke([
        (
            "system",
            """
You are the supervisor of a production incident
investigation team.

Available agents:
- logs: investigate application and database errors
- metrics: investigate latency, resource utilization,
  traffic and error rates
- deployment: investigate recent releases and changes
- report: prepare the final incident report

Rules:
1. Inspect the incident and existing findings.
2. Delegate to a relevant agent whose investigation
   has not yet been completed.
3. Do not choose a completed agent again.
4. Choose report when enough evidence is available
   or all three specialist agents have run.
5. Never claim that an investigation ran unless its
   results are present in the supplied state.
6. The final report must distinguish evidence from
   hypotheses.
"""
        ),
        (
            "human",
            f"""
Incident:
{state['incident']}

Current findings:
{state['findings']}

Logs:
{state['logs'] or 'Not investigated'}

Metrics:
{state['metrics'] or 'Not investigated'}

Deployment history:
{state['deployment'] or 'Not investigated'}

Completed agents:
{[
    name for name, value in [
        ("logs", state["logs"]),
        ("metrics", state["metrics"]),
        ("deployment", state["deployment"]),
    ] if value
]}

Investigation steps completed: {state['steps']}

Choose the next agent.
"""
        ),
    ])

    # Safety guard: stop if the investigation step limit is reached.
    if state["steps"] >= 6:
        next_agent = "report"
        reason = "Maximum investigation step limit reached."
    else:
        next_agent = decision.next_agent
        reason = decision.reason

    # Do not allow the LLM to revisit a completed specialist.
    completed = {
        name
        for name, value in [
            ("logs", state["logs"]),
            ("metrics", state["metrics"]),
            ("deployment", state["deployment"]),
        ]
        if value
    }

    if next_agent in completed:
        remaining = [
            name
            for name in ("logs", "metrics", "deployment")
            if name not in completed
        ]

        next_agent = remaining[0] if remaining else "report"
        reason = "Routing guard selected an unfinished agent."

    print(f"Decision: {next_agent}")
    print(f"Reason: {reason}")

    return {
        "next_agent": next_agent,
        "decision_reason": reason,
    }


# ============================================================
# 6. SPECIALIST AGENTS
# Each agent owns a specific responsibility.
# ============================================================

def log_agent(state: IncidentState):
    print("\n[LOG AGENT] Investigating application logs...")

    result = search_logs(state["incident"])

    finding = f"Log investigation: {result}"
    print(finding)

    return {
        "logs": result,
        "findings": state["findings"] + [finding],
        "steps": state["steps"] + 1,
    }


def metrics_agent(state: IncidentState):
    print("\n[METRICS AGENT] Investigating monitoring data...")

    result = get_metrics(state["incident"])

    finding = f"Metrics investigation: {result}"
    print(finding)

    return {
        "metrics": result,
        "findings": state["findings"] + [finding],
        "steps": state["steps"] + 1,
    }


def deployment_agent(state: IncidentState):
    print("\n[DEPLOYMENT AGENT] Checking release history...")

    result = get_deployment_history(state["incident"])

    finding = f"Deployment investigation: {result}"
    print(finding)

    return {
        "deployment": result,
        "findings": state["findings"] + [finding],
        "steps": state["steps"] + 1,
    }


# ============================================================
# 7. FINAL REPORT AGENT
# Combines evidence from the specialist agents.
# ============================================================

def report_agent(state: IncidentState):
    print("\n[REPORT AGENT] Preparing incident report...")

    response = llm.invoke([
        (
            "system",
            """
You are a senior site reliability engineer.

Write an incident investigation report containing:
1. Incident summary
2. Evidence from logs
3. Evidence from metrics
4. Deployment findings
5. Most likely root cause and confidence
6. Recommended next steps
7. What still needs verification

Never invent facts or claim a hypothesis is proven.
Separate observed evidence from conclusions.
Do not execute remediation actions.
"""
        ),
        (
            "human",
            f"""
Incident: {state['incident']}

Log findings: {state['logs'] or 'Not investigated'}
Metrics findings: {state['metrics'] or 'Not investigated'}
Deployment findings: {
    state['deployment'] or 'Not investigated'
}

All findings:
{state['findings']}
"""
        ),
    ])

    report = response.content
    print("\n" + "=" * 60)
    print("FINAL INCIDENT REPORT")
    print("=" * 60)
    print(report)

    return {"final_report": str(report)}


# ============================================================
# 8. ROUTING
# The supervisor controls which agent executes next.
# ============================================================

def route_to_agent(state: IncidentState) -> str:
    return state["next_agent"]


# ============================================================
# 9. BUILD THE LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(IncidentState)

builder.add_node("supervisor", supervisor)
builder.add_node("logs", log_agent)
builder.add_node("metrics", metrics_agent)
builder.add_node("deployment", deployment_agent)
builder.add_node("report", report_agent)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "logs": "logs",
        "metrics": "metrics",
        "deployment": "deployment",
        "report": "report",
    },
)

# Each specialist returns control to the supervisor.
builder.add_edge("logs", "supervisor")
builder.add_edge("metrics", "supervisor")
builder.add_edge("deployment", "supervisor")

builder.add_edge("report", END)

graph = builder.compile()


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    initial_state: IncidentState = {
        "incident": (
            "Production payment API error rate increased "
            "to 40% after a recent deployment."
        ),
        "logs": "",
        "metrics": "",
        "deployment": "",
        "findings": [],
        "next_agent": "",
        "decision_reason": "",
        "steps": 0,
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print("\nInvestigation completed.")
    print("Specialist steps:", result["steps"])
