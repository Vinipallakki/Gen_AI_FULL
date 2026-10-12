# Simple A2A Agent Demo

This document describes a simple agent and client demonstrating the Agent to Agent (A2A) SDK.

This application contains a simple agent and a test client to invoke it.

## Setup and Deployment

### Prerequisites

Before running the application locally, ensure you have the following installed:

1. **uv:** The Python package management tool used in this project. Follow the installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2. **python 3.13** Python 3.13 is required to run a2a-sdk 

## 1. Install dependencies

This will create a virtual environment in the `.venv` directory and install the required packages.

```bash
uv venv
source .venv/bin/activate
```

## 2. Run the Agent
Open a terminal and run the server with the dummy agent:

```bash
uv run .
```

The agent will be running on `http://localhost:9999`.

## 3. Run the Test Client
Open a new terminal and run the test client:

```bash
uv run --active test_client.py
```

You will see the client interact with the agent in the terminal output.

## References
- https://github.com/google/a2a-python
- https://codelabs.developers.google.com/intro-a2a-purchasing-concierge#1
