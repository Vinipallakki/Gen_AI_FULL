from tenacity import retry, stop_after_attempt
from langchain_openai import ChatOpenAI
from langchain.tools import tool

llm = ChatOpenAI(model="gpt-5.5", temperature=0)

# Shared State
state = {
    "city": "Bangalore",
    "weather": None,
    "status": None,
    "error": None
}

# -------------------------------
# Primary Tool
# -------------------------------
@tool
@retry(stop=stop_after_attempt(3))
def weather_api(city: str):
    """Primary Weather API"""

    # Simulate failure
    raise Exception("Primary API Timeout")


# -------------------------------
# Backup Tool
# -------------------------------
@tool
def backup_weather(city: str):
    """Backup Weather API"""

    return f"Backup API: 26°C Cloudy in {city}"


# -------------------------------
# Recovery Agent
# -------------------------------
def weather_agent(state):

    try:

        result = weather_api.invoke(
            {"city": state["city"]}
        )

        state["weather"] = result
        state["status"] = "SUCCESS"

    except Exception as e:

        state["error"] = str(e)

        print("Primary Tool Failed")

        try:

            result = backup_weather.invoke(
                {"city": state["city"]}
            )

            state["weather"] = result
            state["status"] = "FALLBACK_SUCCESS"

        except Exception:

            recovery_prompt = f"""
            Primary and Backup weather tools failed.

            Error:
            {state['error']}

            Generate a safe response for the user.
            """

            answer = llm.invoke(recovery_prompt)

            state["weather"] = answer.content
            state["status"] = "LLM_RECOVERY"

    return state


result = weather_agent(state)

print(result)