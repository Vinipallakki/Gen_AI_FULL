from mcp.server.fastmcp import FastMCP

import requests
import sqlite3
import ast
import operator


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("Multi Tool Server")


# ============================================================
# TOOL 1 - WIKIPEDIA
# ============================================================

@mcp.tool()
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for general knowledge and factual information.
    """

    url = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    headers = {
        "User-Agent": "MCP-MultiTool-Agent/1.0"
    }

    params = {
        "q": query,
        "limit": 3
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        pages = data.get("pages", [])

        if not pages:
            return "No Wikipedia results found."

        results = []

        for page in pages:

            title = page.get("title", "")
            description = page.get("description", "")
            excerpt = page.get("excerpt", "")

            results.append(
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Excerpt: {excerpt}"
            )

        return "\n\n".join(results)

    except Exception as e:

        return f"Wikipedia error: {str(e)}"


# ============================================================
# TOOL 2 - WEATHER
# ============================================================

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    """

    try:

        # ----------------------------------------------------
        # First get latitude and longitude
        # ----------------------------------------------------

        geocode_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geocode_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(
            geocode_url,
            params=geocode_params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:

            return f"Could not find city: {city}"

        location = results[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location.get("name", city)
        country = location.get("country", "")

        # ----------------------------------------------------
        # Get weather
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "wind_speed_10m"
            ),
            "timezone": "auto"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data.get(
            "current",
            {}
        )

        temperature = current.get(
            "temperature_2m"
        )

        humidity = current.get(
            "relative_humidity_2m"
        )

        feels_like = current.get(
            "apparent_temperature"
        )

        wind_speed = current.get(
            "wind_speed_10m"
        )

        return (
            f"City: {city_name}, {country}\n"
            f"Temperature: {temperature} °C\n"
            f"Feels like: {feels_like} °C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} km/h"
        )

    except Exception as e:

        return f"Weather error: {str(e)}"


# ============================================================
# TOOL 3 - SQLITE DATABASE
# ============================================================

@mcp.tool()
def get_employee(employee_id: int) -> str:
    """
    Get employee information from SQLite database.
    """

    connection = sqlite3.connect(
        "employees.db"
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            department,
            experience,
            salary
        FROM employees
        WHERE id = ?
    """, (employee_id,))

    employee = cursor.fetchone()

    connection.close()

    if not employee:

        return "Employee not found."

    (
        employee_id,
        name,
        department,
        experience,
        salary
    ) = employee

    return (
        f"Employee ID: {employee_id}\n"
        f"Name: {name}\n"
        f"Department: {department}\n"
        f"Experience: {experience} years\n"
        f"Salary: ₹{salary:,}"
    )


# ============================================================
# TOOL 4 - PYTHON CALCULATOR
# ============================================================

operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_calculate(expression):

    node = ast.parse(
        expression,
        mode="eval"
    ).body

    return evaluate_node(node)


def evaluate_node(node):

    if isinstance(
        node,
        ast.Constant
    ):

        if isinstance(
            node.value,
            (int, float)
        ):

            return node.value

        raise ValueError(
            "Only numbers are allowed."
        )

    if isinstance(
        node,
        ast.BinOp
    ):

        left = evaluate_node(
            node.left
        )

        right = evaluate_node(
            node.right
        )

        operation = operators.get(
            type(node.op)
        )

        if operation is None:

            raise ValueError(
                "Operator not allowed."
            )

        return operation(
            left,
            right
        )

    if isinstance(
        node,
        ast.UnaryOp
    ):

        operand = evaluate_node(
            node.operand
        )

        operation = operators.get(
            type(node.op)
        )

        if operation is None:

            raise ValueError(
                "Operator not allowed."
            )

        return operation(
            operand
        )

    raise ValueError(
        "Invalid mathematical expression."
    )


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Perform a mathematical calculation.
    """

    try:

        result = safe_calculate(
            expression
        )

        return f"Result: {result}"

    except Exception as e:

        return f"Calculation error: {str(e)}"


# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":

    mcp.run()