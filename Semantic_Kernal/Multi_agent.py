import os
import asyncio
import uuid
import requests

from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIPromptExecutionSettings,
)

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise Exception("OPENAI_API_KEY not found in .env")

# ==========================================================
# Create Kernel
# ==========================================================

kernel = Kernel()

kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4o",
        api_key=api_key,
    )
)

# ==========================================================
# Utility Plugin
# ==========================================================

class UtilityPlugin:

    # ------------------------------------------------------
    # Calculator
    # ------------------------------------------------------

    @kernel_function(
        description="Use this tool whenever the user asks to perform a mathematical calculation."
    )
    def calculate(self, expression: str) -> str:

        print("\n>>> Calculator Tool Called")

        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Answer: {result}"

        except Exception as e:
            return f"Calculation Error: {e}"

    # ------------------------------------------------------
    # Weather
    # ------------------------------------------------------

    @kernel_function(
        description="Use this tool whenever the user asks for the current weather of a city."
    )
    def weather(self, city: str) -> str:

        print("\n>>> Weather Tool Called")

        try:

            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
                timeout=10,
            ).json()

            if "results" not in geo:
                return "City not found."

            latitude = geo["results"][0]["latitude"]
            longitude = geo["results"][0]["longitude"]

            weather = requests.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}"
                f"&longitude={longitude}"
                f"&current=temperature_2m,wind_speed_10m",
                timeout=10,
            ).json()

            current = weather["current"]

            return (
                f"City: {city}\n"
                f"Temperature: {current['temperature_2m']} °C\n"
                f"Wind Speed: {current['wind_speed_10m']} km/h"
            )

        except Exception as e:
            return str(e)

    # ------------------------------------------------------
    # Joke
    # ------------------------------------------------------

    @kernel_function(
        description="Use this tool whenever the user asks for a joke."
    )
    def random_joke(self) -> str:

        print("\n>>> Joke Tool Called")

        try:

            data = requests.get(
                "https://official-joke-api.appspot.com/random_joke",
                timeout=10,
            ).json()

            return (
                f"{data['setup']}\n\n"
                f"{data['punchline']}"
            )

        except Exception as e:
            return str(e)

    # ------------------------------------------------------
    # UUID Generator
    # ------------------------------------------------------

    @kernel_function(
        description="Use this tool whenever the user asks for a UUID or unique identifier."
    )
    def generate_uuid(self) -> str:

        print("\n>>> UUID Tool Called")

        return str(uuid.uuid4())
    # ==========================================================
# Register Plugin
# ==========================================================

kernel.add_plugin(
    UtilityPlugin(),
    plugin_name="UtilityPlugin",
)

# ==========================================================
# Chat History (Memory)
# ==========================================================

history = ChatHistory()

history.add_system_message(
    """
You are a helpful AI assistant.

You have access to tools.

Rules:
1. Use calculate() for ALL mathematical calculations.
2. Use weather() whenever the user asks about weather.
3. Use random_joke() whenever the user asks for a joke.
4. Use generate_uuid() whenever the user asks for a UUID or unique identifier.

If a tool exists for the task, always use the tool instead of answering from memory.

Remember previous conversation.
"""
)

# ==========================================================
# Function Calling Settings
# ==========================================================

settings = OpenAIPromptExecutionSettings()

settings.function_choice_behavior = FunctionChoiceBehavior.Auto()


chat_service = kernel.get_service(service_id="chat")

# ==========================================================
# Chat Function
# ==========================================================

async def chat():

    print("=" * 60)
    print("Semantic Kernel Agent Started")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        question = input("\nYou : ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        history.add_user_message(question)

        try:

            response = await chat_service.get_chat_message_content(
                chat_history=history,
                settings=settings,
                kernel=kernel,
            )

            history.add_message(response)

            print("\nAssistant :", response.content)

        except Exception as e:
            print("\nERROR:", e)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    try:
        asyncio.run(chat())

    except KeyboardInterrupt:
        print("\nApplication Closed.")

    except Exception as e:
        print("\nFatal Error:", e)
    