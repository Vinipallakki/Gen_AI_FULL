import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ---------------------------------------
# Logging
# ---------------------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------------------
# LLM
# ---------------------------------------
llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0
)

# ---------------------------------------
# Primary Tool
# ---------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1)
)
def primary_weather_api(city: str):

    logging.info("Calling Primary Weather API...")

    response = requests.get(
        f"https://api.weatherapi.com/v1/current.json?q={city}",
        timeout=5
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------
# Fallback Tool
# ---------------------------------------
def fallback_weather_api(city: str):

    logging.info("Calling Fallback Weather API...")

    return {
        "location": city,
        "temperature": "Unknown",
        "condition": "Weather service unavailable"
    }


# ---------------------------------------
# Reliable Tool
# ---------------------------------------
@tool
def weather_tool(city: str) -> str:
    """Get current weather."""

    try:

        result = primary_weather_api(city)

        # Output Validation
        if "current" not in result:
            raise Exception("Invalid Weather Response")

        temp = result["current"]["temp_c"]
        condition = result["current"]["condition"]["text"]

        return f"{city}: {temp}°C, {condition}"

    except Exception as e:

        logging.error(e)

        result = fallback_weather_api(city)

        return (
            f"{result['location']}: "
            f"{result['temperature']}, "
            f"{result['condition']}"
        )


# ---------------------------------------
# Agent
# ---------------------------------------
question = input("Ask: ")

prompt = f"""
You are a helpful travel assistant.

If weather information is needed,
use the weather tool.

User Question:
{question}
"""

tool_result = weather_tool.invoke({"city": "Bangalore"})

response = llm.invoke(
    f"""
    User Question:
    {question}

    Tool Output:
    {tool_result}

    Answer the user.
    """
)

print("\nFinal Answer\n")
print(response.content)