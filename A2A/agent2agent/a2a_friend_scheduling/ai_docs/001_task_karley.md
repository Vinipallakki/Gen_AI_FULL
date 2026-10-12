# Task: Create Karley's Agent

This document outlines the plan for creating Karley's agent, which will be responsible for managing her schedule and responding to pickleball game requests. The agent will be built using the Agent Development Kit (ADK) and will be accessible to the Host Agent via the Agent-to-Agent (A2A) protocol.

This plan is based on the structure of the `@google_adk` sample project.

## 1. Project Setup

-   **Directory:** All files for Karley's agent will be located in the `a2a_friend_scheduling/karley_agent/` directory.
-   **Dependencies:** Create a `pyproject.toml` file to manage the project's dependencies, including `a2a-google-adk`, `click`, `uvicorn`, and `python-dotenv`.

## 2. Agent Implementation

-   **`agent.py`:**
    -   Define a `KarleyAgent` class.
    -   Define a `Tool` for checking Karley's calendar. This will initially be a hardcoded schedule but can be replaced with a real calendar API later.
    -   Implement the agent's core logic to process incoming requests. The agent should be able to understand requests for availability and respond with open time slots.
    -   The agent's instructions will define its personality and capabilities, focusing on its role as a personal scheduling assistant for Karley.

-   **`agent_executor.py`:**
    -   Create a `KarleyAgentExecutor` class that inherits from `a2a.server.agent_execution.AgentExecutor`.
    -   This class will instantiate `KarleyAgent`.
    -   It will implement the `execute` method to handle the lifecycle of an incoming request, manage the task state (`working`, `input_required`, `failed`, `complete`), and stream responses back to the host.

-   **`__main__.py`:**
    -   This file will be the main entry point to run the agent as a server.
    -   It will use `click` to handle command-line arguments for `host` and `port`.
    -   It will define an `AgentCard` with metadata about Karley's agent (name, description, skills, etc.).
    -   It will instantiate `KarleyAgentExecutor`.
    -   It will set up and run a `A2AStarletteApplication` using `uvicorn`, passing it the `AgentCard` and a `DefaultRequestHandler` which in turn contains the `KarleyAgentExecutor`. The server will run on port `10002`.

## 3. A2A Integration

-   The agent will be configured to communicate using the A2A protocol, allowing it to understand and respond to messages from the Host Agent.
-   The server will expose an endpoint that the Host Agent can call to interact with Karley's agent.

## 4. Testing

-   Once the agent is running, we will test it by sending sample requests to its endpoint to ensure it responds correctly with Karley's availability.
-   We will also test the integration with the Host Agent to confirm that they can communicate successfully.
