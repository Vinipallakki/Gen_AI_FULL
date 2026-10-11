
import pytest

from db_utils import (
    list_tables,
    describe_table,
    run_read_query,
)


def test_list_tables():
    tables = list_tables()

    assert "customers" in tables
    assert "products" in tables
    assert "orders" in tables


def test_describe_customers():
    columns = describe_table("customers")
    names = [column["column"] for column in columns]

    assert "customer_id" in names
    assert "name" in names
    assert "city" in names


def test_read_query():
    result = run_read_query(
        "SELECT name, city FROM customers ORDER BY customer_id"
    )

    assert result["row_count"] == 4
    assert result["rows"][0]["name"] == "Rahul"


def test_unknown_table():
    with pytest.raises(ValueError):
        describe_table("unknown_table")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET city = 'Delhi'",
        "DROP TABLE customers",
        "SELECT * FROM customers; SELECT * FROM orders",
        "PRAGMA table_info(customers)",
    ],
)
def test_reject_unsafe_queries(sql):
    with pytest.raises((ValueError, Exception)):
        run_read_query(sql)


def test_cte_read_query():
    result = run_read_query("""
        WITH city_counts AS (
            SELECT city, COUNT(*) AS total
            FROM customers
            GROUP BY city
        )
        SELECT city, total
        FROM city_counts
        ORDER BY city
    """)

    assert result["row_count"] >= 1
    assert "city" in result["columns"]


def test_query_result_limit():
    result = run_read_query("""
        SELECT 1 AS number
        FROM customers
        CROSS JOIN customers AS c2
        LIMIT 250
    """)

    assert result["row_count"] == 200
    assert result["truncated"] is True
