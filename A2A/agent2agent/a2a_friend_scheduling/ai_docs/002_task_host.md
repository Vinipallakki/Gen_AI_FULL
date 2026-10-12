# Task: Refactor the Host Agent

This document outlines the plan for refactoring the `host` agent (previously the `RoutingAgent`) to act as the central orchestrator for scheduling pickleball games with friend agents.

## 1. Project Structure and Naming

-   **Directory Cleanup:** The agent's code currently resides in `a2a_friend_scheduling/host_agent/host`. To simplify, move the contents of the inner `host` directory (`agent.py`, `remote_agent_connection.py`, etc.) up into `host_agent` and delete the now-empty `host` folder.
-   **Rename `host_agent` to `host`:** Rename the top-level `host_agent` directory to just `host` for consistency with the other agent directories (`karley_agent`, etc.).
-   **Rename `RoutingAgent`:** The class `RoutingAgent` in `agent.py` should be renamed to `HostAgent` to more accurately reflect its new role.

## 2. Develop Pickleball Scheduling Tools

A new set of tools is required for the Host Agent to manage the scheduling process. These will be created in a new file: `host/pickleball_tools.py`.

-   **Database Setup (`db.py`):**
    -   Create a file `host/db.py` to manage the SQLite database connection.
    -   It will include functions to initialize the database and create a `reservations` table with columns for `id`, `datetime`, `friend_name`, and `court_id`.
-   **Pickleball Tools (`pickleball_tools.py`):**
    -   **`check_friends_availability`:** This tool will take a list of friend names and a proposed date. It will use the `send_message` function to query each friend's agent for their availability on that date.
    -   **`find_common_timeslots`:** This tool will process the availability responses from the friend agents and identify overlapping time slots.
    -   **`book_pickleball_court`:** This tool will take a confirmed time, date, and list of friends, and it will create a reservation in the SQLite database.

## 3. Refactor the Host Agent (`agent.py`)

-   **Update Initialization:**
    -   Modify the `create` method to accept a list of friend agent URLs from environment variables (e.g., `KARLEY_AGENT_URL`, `NATE_AGENT_URL`). It should be configured to connect to all three friend agents, not just a single hardcoded one.
-   **Update Instructions:**
    -   Completely rewrite the `root_instruction` method. The new instructions will guide the agent on its role as a pickleball scheduler.
    -   It should direct the agent to first use `check_friends_availability`, then `find_common_timeslots`, and finally `book_pickleball_court`.
-   **Integrate Tools:**
    -   The `create_agent` method must be updated to include the new pickleball tools in its `tools` list, alongside the existing `send_message` tool.

## 4. Create a Server Entry Point (`__main__.py`)

-   The host agent currently lacks a standalone server entry point.
-   Create a new file, `host/__main__.py`, that is responsible for:
    -   Loading environment variables.
    -   Instantiating the `HostAgent`.
    -   Running the agent as an A2A server on its designated port, `10001`.
    -   This file will be similar in structure to `karley_agent/__main__.py`.
