# Project: Agent-to-Agent Friend Scheduling

## 1. Overview

The primary goal of this project is to build a multi-agent system that can schedule a game of pickleball among a group of friends. This project will serve as a demonstration of the Agent-to-Agent (A2A) communication protocol, where autonomous agents, built with different frameworks, can interact and collaborate to achieve a common goal.

## 2. Architecture

The system consists of a central Host Agent that acts as a proxy and communicates with several friend agents. Each agent represents a person and manages their schedule and preferences.

The high-level architecture is as follows:

-   **ADK Web (Client):** A web-based user interface for interacting with the Host Agent.
-   **Host Agent (ADK):** The central orchestrator, built using the Agent Development Kit (ADK). It routes messages and manages the overall scheduling task.
-   **Friend Agents:** Each friend has a personal agent responsible for their own calendar and scheduling preferences. These agents are built using different technologies to showcase interoperability.

The communication flow is as follows: The user (Brandon) initiates a request through the ADK Web Client to schedule a pickleball game. The Host Agent receives this request and then communicates with the agents of Karley, Nate, and Kaitlyn to find a suitable time.

## 3. Agents

There will be four main agents in this system. To be accessible to the Host Agent, each friend agent will run as a separate server, exposing an API that the Host Agent can call.

### 3.1. Host Agent (Brandon's Agent)

-   **Framework:** Agent Development Kit (ADK)
-   **Role:** This is the central agent that orchestrates the scheduling process. It acts as a proxy between the user and the friend agents. It will manage Brandon's calendar and preferences.
-   **Status:** The basic structure is already in place. However, the internal instructions need to be reviewed and corrected.
-   **Port:** `10001`

### 3.2. Friend Agents (Sub-Agents)

To be accessible to the Host Agent, each friend agent will run as a separate server, exposing an API that the Host Agent can call.

#### 3.2.1. Karley's Agent

-   **Framework:** Agent Development Kit (ADK)
-   **Role:** Manages Karley's schedule and preferences for playing pickleball.
-   **Example:** Based on `@google_adk` sample.
-   **Port:** `10002`

#### 3.2.2. Nate's Agent

-   **Framework:** CrewAI
-   **Role:** Manages Nate's schedule and preferences. This will demonstrate how a CrewAI-based agent can be integrated into the A2A network.
-   **Example:** Based on `@crewai` sample.
-   **Port:** `10003`

#### 3.2.3. Kaitlyn's Agent

-   **Framework:** LangGraph
-   **Role:** Manages Kaitlyn's schedule and preferences. This will showcase integration with a LangGraph-based agent.
-   **Example:** Based on `@langgraph` sample.
-   **Port:** `10004`

## 4. Pickleball Scheduling Tools

The Host Agent will be equipped with a specialized set of tools for scheduling pickleball games. These tools will enable the Host Agent to manage the entire scheduling process, from checking friend availability to booking the court.

### 4.1. Core Functionality

-   **Check Availability:** The tools will allow the Host Agent to query the friend agents (Karley, Nate, and Kaitlyn) for their available time slots.
-   **Find Common Slots:** The tools will compare the availability of all friends to find mutually convenient times for a game.
-   **Book Court:** Once a suitable time is found, the tools will interact with a database to reserve the pickleball court.

### 4.2. Database

-   **Technology:** A simple SQLite database will be used to manage court reservations.
-   **Schema:** The database will contain a `reservations` table. This table will track reserved time slots for the single pickleball court.

## 5. Development Plan

1.  **Finalize Master Plan:** Flesh out this document with any more details.
2.  **Develop Pickleball Scheduling Tools:** Create the Python scripts for the scheduling tools, including the SQLite database interaction.
3.  **Review and Refine Host Agent:** Correct the "raw instructions" for the host agent and integrate the new scheduling tools.
4.  **Develop Karley's Agent (ADK):** Create the agent and expose it as a server.
5.  **Develop Nate's Agent (CrewAI):** Create the agent and expose it as a server.
6.  **Develop Kaitlyn's Agent (LangGraph):** Create the agent and expose it as a server.
7.  **Integration and Testing:** Ensure all agents can communicate with the host agent and a pickleball game can be successfully scheduled.
