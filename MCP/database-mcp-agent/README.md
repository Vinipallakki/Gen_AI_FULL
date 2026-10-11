# Database MCP Agent --- Python, OpenAI, MCP, and SQLite

A learning project that lets an OpenAI-powered agent inspect a SQLite
database and answer questions by calling tools exposed by a local
**Model Context Protocol (MCP)** server.

Features: - Discover database tables. - Inspect a table's columns. - Run
read-only SQL queries. - Answer natural-language questions using
database results.

> OpenAI decides when to request a tool. The MCP client communicates
> with the MCP server. Python functions in `db_utils.py` perform the
> database operations. MCP standardizes tool discovery and
> communication; it does not generate SQL or make the model smarter by
> itself.

## 1. Architecture

``` text
User
  |
  v
client.py <----> OpenAI model
  |
  | MCP client over stdio
  v
server.py (MCP tools)
  |
  v
db_utils.py
  |
  v
db/sample.db (SQLite)
```

`client.py` starts `server.py` as a subprocess, so you normally do
**not** need to start the MCP server separately.

## 2. Project structure

``` text
database-mcp-agent/
├── db/
│   └── sample.db
├── server.py
├── client.py
├── db_utils.py
├── seed_db.py
├── test_db_utils.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 3. Requirements and setup

-   Python 3.10 or newer
-   An OpenAI API key
-   Internet access for OpenAI API requests

### `requirements.txt`

``` text
mcp[cli]>=1.28,<2
openai
python-dotenv
pytest
```

Install dependencies from the project root:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the environment's
Python executable directly instead.

### `.env`

Create this file in the project root:

``` dotenv
OPENAI_API_KEY=put_your_real_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Replace the placeholder with your real API key. Never commit `.env` to
Git.

## 4. Complete source code

Create the files below with the following contents.

### `seed_db.py`

This script creates the database and inserts example customers,
products, and orders. Running it again resets the sample data.

``` python
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent / "db"
DB_PATH = DB_DIR / "sample.db"


def seed_database() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );

            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                price REAL NOT NULL CHECK (price >= 0)
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                order_date TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );

            INSERT INTO customers (customer_id, name, city) VALUES
                (1, 'Rahul', 'Bengaluru'),
                (2, 'Priya', 'Mysuru'),
                (3, 'Amit', 'Hyderabad'),
                (4, 'Sneha', 'Bengaluru');

            INSERT INTO products (product_id, product_name, price) VALUES
                (1, 'Laptop', 60000),
                (2, 'Phone', 25000),
                (3, 'Headphones', 3000);

            INSERT INTO orders
                (order_id, customer_id, product_id, quantity, order_date)
            VALUES
                (1, 1, 1, 1, '2026-09-01'),
                (2, 2, 2, 2, '2026-09-03'),
                (3, 1, 3, 3, '2026-09-05'),
                (4, 3, 1, 1, '2026-09-08'),
                (5, 4, 2, 1, '2026-09-10'),
                (6, 2, 3, 2, '2026-10-02');
            """
        )

    print(f"Database seeded successfully: {DB_PATH}")


if __name__ == "__main__":
    seed_database()
```

### `db_utils.py`

SQLite operations live here. The connection is opened read-only,
SQLite's authorizer denies write operations, and query results are
capped.

``` python
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "db" / "sample.db"
MAX_ROWS = 200
QUERY_TIMEOUT_SECONDS = 3
MAX_QUERY_LENGTH = 10_000

DENIED_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
}


def authorize(action, arg1, arg2, database_name, trigger_name):
    """Allow table_info; deny writes and other PRAGMA operations."""
    if action == sqlite3.SQLITE_PRAGMA:
        if (arg1 or "").lower() == "table_info":
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    if action in DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


def open_readonly_db() -> sqlite3.Connection:
    """Open the sample database in read-only mode with safety settings."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run `python seed_db.py` first."
        )

    connection = sqlite3.connect(
        f"{DB_PATH.as_uri()}?mode=ro",
        uri=True,
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(authorize)
    connection.set_progress_handler(lambda: 1, 100_000)
    return connection


def list_tables() -> dict[str, Any]:
    """Return user-created table names."""
    sql = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    with closing(open_readonly_db()) as connection:
        rows = connection.execute(sql).fetchall()

    return {"tables": [row["name"] for row in rows]}


def describe_table(table_name: str) -> dict[str, Any]:
    """Return a table's column names, types, nullability, and primary-key flags."""
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")

    available_tables = list_tables()["tables"]
    if table_name not in available_tables:
        raise ValueError(
            f"Unknown table '{table_name}'. Available tables: {available_tables}"
        )

    quoted_name = '"' + table_name.replace('"', '""') + '"'
    sql = f"PRAGMA table_info({quoted_name})"

    with closing(open_readonly_db()) as connection:
        rows = connection.execute(sql).fetchall()

    return {
        "table": table_name,
        "columns": [
            {
                "name": row["name"],
                "type": row["type"],
                "not_null": bool(row["notnull"]),
                "primary_key": bool(row["pk"]),
            }
            for row in rows
        ],
    }


def run_read_query(sql: str) -> dict[str, Any]:
    """Execute one read-only SELECT/WITH query and return bounded results."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query cannot be empty")

    sql = sql.strip()
    if len(sql) > MAX_QUERY_LENGTH:
        raise ValueError(f"SQL query exceeds {MAX_QUERY_LENGTH} characters")

    if sql.endswith(";"):
        sql = sql[:-1].rstrip()

    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed")

    match = re.match(r"([A-Za-z]+)", sql)
    first_word = match.group(1).upper() if match else ""
    if first_word not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT or WITH queries are allowed")

    with closing(open_readonly_db()) as connection:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(MAX_ROWS + 1)
        columns = [item[0] for item in (cursor.description or [])]

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]

    return {
        "columns": columns,
        "rows": [dict(row) for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
```

### `server.py`

The `@mcp.tool()` decorator registers each function as an MCP tool.

``` python
import logging

from mcp.server.fastmcp import FastMCP

from db_utils import (
    describe_table as db_describe_table,
    list_tables as db_list_tables,
    run_read_query,
)

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("ReadOnly Database MCP Server")


@mcp.tool()
def list_tables() -> dict:
    """List all user-created tables in the SQLite database."""
    return db_list_tables()


@mcp.tool()
def describe_table(table_name: str) -> dict:
    """Describe a table's columns, data types, and primary-key status."""
    return db_describe_table(table_name)


@mcp.tool()
def query_database(sql: str) -> dict:
    """Run one read-only SELECT or WITH SQL query against the database."""
    return run_read_query(sql)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### `client.py`

This client discovers MCP tools, converts their schemas to OpenAI tool
definitions, and executes requested tools. Tool results are sent back to
OpenAI so the model can produce an answer or request another tool.

``` python
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_DIR / "server.py"
MAX_TOOL_ROUNDS = 6

load_dotenv(PROJECT_DIR / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to the .env file in the project root."
    )

openai_client = AsyncOpenAI(api_key=api_key)


def convert_mcp_tools_to_openai(mcp_tools):
    """Convert MCP tool schemas to OpenAI Chat Completions tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


def get_text_from_mcp_result(result) -> str:
    """Convert MCP content blocks into text."""
    parts = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(str(item))

    output = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"MCP tool returned an error: {output}"
    return output or "{}"


async def ask_agent(session: ClientSession, tools: list[dict], question: str) -> str:
    """Ask OpenAI a question and run any MCP tools requested by the model."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a database assistant. Use database tools for database "
                "facts. Inspect schemas when needed. Never claim a query succeeded "
                "unless you received its result. Explain calculations clearly. "
                "The database tools are read-only."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return assistant_message.content or "(The model returned an empty response.)"

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                result = await session.call_tool(tool_name, arguments=arguments)
                tool_output = get_text_from_mcp_result(result)
            except Exception as exc:
                tool_output = json.dumps(
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "tool": tool_name,
                    }
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    return (
        "Stopped after reaching the maximum tool-call rounds. "
        "Inspect tool errors or try a more specific question."
    )


async def main() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_DIR),
        env=os.environ.copy(),
    )

    print("Starting database MCP client...")

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed_tools = await session.list_tools()
            tools = convert_mcp_tools_to_openai(listed_tools.tools)

            print("Connected to database MCP server.")
            print("Available tools:", [tool["function"]["name"] for tool in tools])
            print("Type 'exit' to stop.")

            while True:
                try:
                    question = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if question.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                if not question:
                    continue

                try:
                    answer = await ask_agent(session, tools, question)
                    print(f"\nAgent: {answer}")
                except Exception as exc:
                    print(f"\nApplication error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
```

### `test_db_utils.py`

``` python
import pytest

from db_utils import describe_table, list_tables, run_read_query


def test_list_tables():
    assert list_tables()["tables"] == ["customers", "orders", "products"]


def test_describe_orders_table():
    result = describe_table("orders")
    names = [column["name"] for column in result["columns"]]

    assert names == [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_date",
    ]


def test_read_query():
    result = run_read_query("SELECT COUNT(*) AS total FROM orders")
    assert result["rows"][0]["total"] == 6


def test_reject_write_query():
    with pytest.raises(ValueError):
        run_read_query("DELETE FROM customers")


def test_reject_multiple_statements():
    with pytest.raises(ValueError):
        run_read_query("SELECT * FROM customers; SELECT * FROM orders")


def test_unknown_table():
    with pytest.raises(ValueError):
        describe_table("not_a_real_table")
```

### `.gitignore`

``` gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
db/sample.db
```

The database is generated by `seed_db.py`, so it does not need to be
committed. Remove `db/sample.db` from `.gitignore` only if you
intentionally want to distribute the generated database file.

## 5. Run the project

Run all commands from the project root.

### Step 1 --- Create the database

``` powershell
python seed_db.py
```

### Step 2 --- Test database utilities

``` powershell
python -c "from db_utils import list_tables; print(list_tables())"
python -c "from db_utils import describe_table; print(describe_table('orders'))"
python -c "from db_utils import run_read_query; print(run_read_query('SELECT * FROM orders'))"
```

Expected table names: `customers`, `orders`, and `products`.

### Step 3 --- Run tests

``` powershell
python -m pytest -v
```

### Step 4 --- Start the agent

``` powershell
python client.py
```

The client starts `server.py` automatically. Example questions:

``` text
List all tables in my database.
Describe the orders table.
Calculate total revenue by customer.
Which product has the highest total sales revenue?
How many orders are in the database?
```

## 6. Revenue example

Revenue is calculated as `quantity * product price` for each order.

``` sql
SELECT
    c.name,
    SUM(o.quantity * p.price) AS revenue
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
JOIN products AS p ON p.product_id = o.product_id
GROUP BY c.customer_id, c.name
ORDER BY revenue DESC;
```

Expected results for the seeded data:

  Customer     Revenue
  ---------- ---------
  Rahul          69000
  Amit           60000
  Priya          56000
  Sneha          25000

Laptop has the highest total sales revenue: **120000**.

## 7. How tool calling works

1.  The user asks a question in `client.py`.
2.  OpenAI decides whether tools are needed.
3.  The client calls an MCP tool using `session.call_tool()`.
4.  `server.py` routes the request to `db_utils.py`.
5.  `db_utils.py` executes the database operation and returns the
    result.
6.  The client sends the result back to OpenAI.
7.  OpenAI produces a final answer or requests another tool, up to the
    configured round limit.

## 8. Troubleshooting

-   **`OPENAI_API_KEY is missing`:** Confirm `.env` is in the project
    root and contains a valid `OPENAI_API_KEY`.
-   **`Database not found`:** Run `python seed_db.py`.
-   **`No module named 'mcp'`:** Run `pip install -r requirements.txt`
    in the Python environment you use to run the project.
-   **`describe_table` fails:** Remove imports from unrelated projects.
    The SQLite authorizer must allow `SQLITE_PRAGMA` only for
    `table_info` and deny other PRAGMA operations.
-   **The agent reaches the tool-round limit:** Inspect tool error
    messages and test the SQL directly before increasing
    `MAX_TOOL_ROUNDS`.

## 9. Security notes

This is a learning project, not a production-ready SQL execution
service.

-   Keep the database read-only for this agent.
-   Never commit `.env` or share your API key.
-   The read-only URI, SQLite authorizer, query-only setting, and row
    cap are defense-in-depth measures, not a complete security sandbox.
-   A simple `SELECT`/`WITH` prefix check is not a full SQL parser.
    Production deployments need stricter query validation,
    database-level least privilege, execution limits, and an explicit
    policy for permitted tables and operations.
-   Treat database contents as untrusted input. Do not follow
    instructions that appear inside database records.

## 10. Interview explanation

> "OpenAI handles reasoning and decides when to call a tool. MCP
> standardizes tool discovery and communication between the client and
> server. The MCP server exposes functions such as `list_tables`,
> `describe_table`, and `query_database`. The actual SQLite work is
> implemented in `db_utils.py`. The client returns tool results to the
> model so it can formulate the final response."

## 11. Can this be built without MCP?

Yes. You can use OpenAI function calling and execute Python functions
directly inside the application. MCP is useful when you want a
standardized interface and want to reuse the same tools from multiple
MCP-compatible clients.
