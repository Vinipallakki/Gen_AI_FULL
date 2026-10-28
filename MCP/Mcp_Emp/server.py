
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Employee Server")

EMPLOYEES = [
    {"id": 101, "name": "Rahul", "department": "IT", "salary": 80000},
    {"id": 102, "name": "Priya", "department": "HR", "salary": 65000},
    {"id": 103, "name": "Arun", "department": "Finance", "salary": 70000},
]


@mcp.tool()
def get_employee(employee_id: int) -> dict:
    """Get employee details using employee ID."""
    for employee in EMPLOYEES:
        if employee["id"] == employee_id:
            return {
                "name": employee["name"],
                "department": employee["department"],
                "salary": employee["salary"],
            }

    return {"error": "Employee not found"}


@mcp.tool()
def get_all_employees() -> list:
    """Return all employee records."""
    return EMPLOYEES


@mcp.tool()
def get_employee_count() -> int:
    """Return the total number of employees."""
    return len(EMPLOYEES)


@mcp.tool()
def get_employees_by_department(department: str) -> list:
    """Return employees belonging to a department."""
    return [
        employee
        for employee in EMPLOYEES
        if employee["department"].lower() == department.lower()
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
