


import re
import sqlite3
from pathlib import Path
from contextlib import closing


DB_PATH = (
    Path(__file__).resolve().parent
    / "db"
    / "sample.db"
)

MAX_ROWS = 200
QUERY_TIMEOUT_SECONDS = 3

# SQLite operations that must not be allowed.
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
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DROP_VTABLE,
}


def open_readonly_db():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            "Database not found. Run python seed_db.py first."
        )

    conn = sqlite3.connect(
        f"{DB_PATH.as_uri()}?mode=ro",
        uri=True,
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA query_only = ON")
    
    def authorize(action, arg1, arg2, db_name, trigger):
        # Permit only the internal schema-inspection PRAGMA.
        if action == sqlite3.SQLITE_PRAGMA:
            if (arg1 or "").lower() == "table_info":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        if action in DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY

        return sqlite3.SQLITE_OK


    conn.set_authorizer(authorize)


    # Interrupt long-running SQL operations.
    conn.set_progress_handler(
        lambda: 1,
        100_000,
    )

    return conn


def list_tables():
    with open_readonly_db() as conn:
        rows = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        return [row["name"] for row in rows]


def describe_table(table_name: str):
    if table_name not in list_tables():
        raise ValueError("Unknown table.")

    # The name is validated against actual database metadata.
    # Double quotes safely delimit the SQLite identifier.
    safe_name = '"' + table_name.replace('"', '""') + '"'

    with open_readonly_db() as conn:
        rows = conn.execute(
            f"PRAGMA table_info({safe_name})"
        ).fetchall()

        return [
            {
                "column": row["name"],
                "type": row["type"],
                "not_null": bool(row["notnull"]),
                "primary_key": bool(row["pk"]),
            }
            for row in rows
        ]


def run_read_query(sql: str):
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    query = sql.strip()

    # Only SELECT and WITH queries are accepted.
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are permitted.")

    # Permit at most one statement, with an optional trailing semicolon.
    query = query.rstrip()
    if query.endswith(";"):
        query = query[:-1].rstrip()

    if ";" in query:
        raise ValueError("Multiple SQL statements are not permitted.")

    with open_readonly_db() as conn:
        cursor = conn.execute(query)

        if cursor.description is None:
            raise ValueError("The query must return rows.")

        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS + 1)

        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        return {
            "columns": columns,
            "rows": [dict(row) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
