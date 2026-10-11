from mcp.server import MCPServer


# ============================================================
# MCP SERVER    
# ============================================================

mcp = MCPServer("Employee Server")


# ============================================================
# EMPLOYEE DATA
# ============================================================

employees = {
    101: {
        "name": "Ravi",
        "department": "IT",
        "experience": 5
    },
    102: {
        "name": "Priya",
        "department": "HR",
        "experience": 3
    },
    103: {
        "name": "Arun",
        "department": "Finance",
        "experience": 7
    },
    104: {
        "name": "Sneha",
        "department": "IT",
        "experience": 4
    }
}


# ============================================================
# TOOL 1
# ============================================================

@mcp.tool()
def get_employee(employee_id: int) -> str:
    """Get employee details using employee ID."""

    employee = employees.get(employee_id)

    if employee is None:
        return "Employee not found"

    return (
        f"ID: {employee_id}\n"
        f"Name: {employee['name']}\n"
        f"Department: {employee['department']}\n"
        f"Experience: {employee['experience']} years"
    )


# ============================================================
# TOOL 2
# ============================================================

@mcp.tool()
def get_employees_by_department(
    department: str
) -> str:
    """Get all employees belonging to a department."""

    result = []

    for employee_id, employee in employees.items():

        if employee["department"].lower() == department.lower():

            result.append(
                f"ID: {employee_id}, "
                f"Name: {employee['name']}, "
                f"Experience: {employee['experience']} years"
            )

    if not result:
        return "No employees found"

    return "\n".join(result)


# ============================================================
# TOOL 3
# ============================================================

@mcp.tool()
def get_employee_count() -> str:
    """Get total number of employees."""

    return f"Total employees: {len(employees)}"


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()