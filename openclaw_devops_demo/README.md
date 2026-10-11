# OpenClaw-Style DevOps Agent Demo

This project demonstrates the core architecture of a tool-using AI agent:

User -> Gateway/Entry Point -> Agent -> Tools -> Memory -> Response

## Project Structure

```
openclaw_devops_demo/
├── main.py
├── agent.py
├── tools.py
├── memory.py
└── README.md
```

## Run

No external packages are required.

```bash
python main.py
```

Try:

```text
Check my payment application
```

The flow is:

1. Agent receives the request.
2. Agent calls `check_application_status()`.
3. If the application is DOWN, it calls `get_application_logs()`.
4. The incident is stored in memory.
5. Ask:

```text
What was the previous issue?
```

The agent retrieves the previous incident.

## Architecture

```text
USER
  |
  v
ENTRY POINT / GATEWAY
  |
  v
DEVOPS AGENT
  |-------------------|
  v                   v
STATUS TOOL        MEMORY
  |
  v
APP DOWN?
  |
 YES
  |
  v
LOG TOOL
  |
  v
ANALYZE + STORE INCIDENT
  |
  v
RESPONSE
```

## Important Note

This is a simplified local demo showing the architecture and flow. It is intentionally lightweight and does not require a real OpenClaw installation or external LLM.

To connect this concept to a production system, replace the command-line entry point with an OpenClaw channel/Gateway integration and register the functions as callable tools.
