
from fastapi import FastAPI

app = FastAPI(title="Employee API")

employees = [
    {"id": 1, "name": "Rahul", "department": "IT"},
    {"id": 2, "name": "Priya", "department": "HR"},
    {"id": 3, "name": "Arun", "department": "Finance"}
]

@app.get("/")
def home():
    return {"message": "Employee API is running!"}

@app.get("/employees")
def get_employees():
    return employees

@app.get("/employees/{employee_id}")
def get_employee(employee_id: int):
    for employee in employees:
        if employee["id"] == employee_id:
            return employee

    return {"error": "Employee not found"}
