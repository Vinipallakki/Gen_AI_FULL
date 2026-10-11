
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "sample.db"


def seed_database():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                price REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                FOREIGN KEY(customer_id)
                    REFERENCES customers(customer_id),
                FOREIGN KEY(product_id)
                    REFERENCES products(product_id)
            );
        """)

        # Seed only when the tables are empty.
        if conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO customers VALUES (?, ?, ?)",
                [
                    (1, "Rahul", "Bengaluru"),
                    (2, "Priya", "Mysuru"),
                    (3, "Amit", "Hyderabad"),
                    (4, "Sneha", "Bengaluru"),
                ],
            )

        if conn.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO products VALUES (?, ?, ?)",
                [
                    (1, "Laptop", 60000),
                    (2, "Phone", 25000),
                    (3, "Headphones", 3000),
                ],
            )

        if conn.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 1, 1, 1, "2026-09-01"),
                    (2, 2, 2, 2, "2026-09-03"),
                    (3, 1, 3, 3, "2026-09-05"),
                    (4, 3, 1, 1, "2026-09-08"),
                    (5, 4, 2, 1, "2026-09-10"),
                    (6, 2, 3, 2, "2026-10-02"),
                ],
            )

    print(f"Database ready: {DB_PATH}")


if __name__ == "__main__":
    seed_database()
