import sqlite3


DB_NAME = "employees.db"


def create_database():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            experience INTEGER NOT NULL,
            salary INTEGER NOT NULL
        )
    """)

    employees = [
        (101, "Ravi", "IT", 5, 800000),
        (102, "Priya", "HR", 3, 600000),
        (103, "Arun", "Finance", 7, 1000000),
        (104, "Sneha", "IT", 4, 750000),
        (105, "Kiran", "Sales", 6, 850000)
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO employees
        (id, name, department, experience, salary)
        VALUES (?, ?, ?, ?, ?)
    """, employees)

    connection.commit()
    connection.close()

    print("Database created successfully.")


if __name__ == "__main__":
    create_database()