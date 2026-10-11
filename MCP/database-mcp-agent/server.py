
import logging

from mcp.server.fastmcp import FastMCP

from db_utils import (
    list_tables as db_list_tables,
    describe_table as db_describe_table,
    run_read_query,
)

# MCP protocol uses stdout for communication.
# Send application logs to stderr instead.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database-mcp")

mcp = FastMCP("ReadOnly Database MCP Server")


@mcp.tool()
def list_tables() -> dict:
    """List all accessible tables in the local SQLite database."""

    try:
        return {"tables": db_list_tables()}
    except Exception:
        logger.exception("Failed to list tables")
        return {"error": "Unable to list database tables."}


@mcp.tool()
def describe_table(table_name: str) -> dict:
    """Get column names, data types, nullability, and primary keys
    for a table in the database."""

    try:
        return {
            "table": table_name,
            "columns": db_describe_table(table_name),
        }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception:
        logger.exception("Failed to describe table")
        return {"error": "Unable to describe the requested table."}


@mcp.tool()
def query_database(sql: str) -> dict:
    """Execute a read-only SELECT or WITH SQL query.
    Use list_tables and describe_table to discover the schema first.
    Returns column names, up to 200 rows, and a truncation indicator."""

    try:
        return run_read_query(sql)
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    except Exception:
        logger.exception("Database query failed")
        return {"error": "Query failed. Check the SQL and table schema."}


if __name__ == "__main__":
    mcp.run(transport="stdio")
