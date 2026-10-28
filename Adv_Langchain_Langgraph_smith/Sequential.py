# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage, ToolMessage

# # -------------------------------------------------------
# # Tool 1
# # -------------------------------------------------------
# @tool
# def calculator(number: int) -> str:
#     """
#     Calculate the square of a given number.

#     Use this tool whenever the user asks about
#     squares or square calculations.
#     """
#     return f"The square of {number} is {number * number}"


# # -------------------------------------------------------
# # Tool 2
# # -------------------------------------------------------
# @tool
# def weather(city: str) -> str:
#     """
#     Get the current weather of a city.

#     Use this tool whenever the user asks
#     about weather, temperature or climate.
#     """
#     weather_data = {
#         "bangalore": "28°C, Cloudy",
#         "mumbai": "31°C, Sunny",
#         "delhi": "38°C, Hot"
#     }

#     return weather_data.get(
#         city.lower(),
#         "Weather information not available."
#     )


# # -------------------------------------------------------
# # LLM
# # -------------------------------------------------------
# llm = ChatOpenAI(
#     model="gpt-5.5",
#     api_key="",
#     temperature=0
# )

# # Bind tools
# llm_with_tools = llm.bind_tools(
#     [calculator, weather]
# )

# # -------------------------------------------------------
# # Take User Input
# # -------------------------------------------------------
# question = input("Ask your question: ")

# messages = [
#     HumanMessage(content=question)
# ]

# # -------------------------------------------------------
# # Step 1 : LLM decides the tool
# # -------------------------------------------------------
# ai_message = llm_with_tools.invoke(messages)

# # If no tool is required
# if not ai_message.tool_calls:
#     print("\nAnswer:")
#     print(ai_message.content)
#     exit()

# print("\nTool Selected by LLM:")
# print(ai_message.tool_calls[0]["name"])

# tool_messages = []

# # -------------------------------------------------------
# # Step 2 : Execute Tool
# # -------------------------------------------------------
# available_tools = {
#     "calculator": calculator,
#     "weather": weather
# }

# for tool_call in ai_message.tool_calls:

#     tool_name = tool_call["name"]

#     tool = available_tools[tool_name]

#     tool_result = tool.invoke(tool_call["args"])

#     tool_messages.append(
#         ToolMessage(
#             content=tool_result,
#             tool_call_id=tool_call["id"]
#         )
#     )

# # -------------------------------------------------------
# # Step 3 : Send Tool Result back to LLM
# # -------------------------------------------------------
# final_response = llm_with_tools.invoke(
#     messages + [ai_message] + tool_messages
# )

# print("\nFinal Answer:")
# print(final_response.content)

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

# --------------------------------------------------
# 2. Get user input
# --------------------------------------------------

user_input = input(
    "Enter your role and experience: "
)

# --------------------------------------------------
# STEP 1: SKILL ANALYZER
# --------------------------------------------------

print("\nSTEP 1: Analyzing skills...")

skill_prompt = f"""
You are a technical skill analyzer.

Analyze this job profile:
{user_input}

Extract the likely technical skills.
Return only a concise list of skills.
"""

skill_response = llm.invoke([
    HumanMessage(content=skill_prompt)
])

skills = skill_response.content

print("\nExtracted Skills:")
print(skills)

# --------------------------------------------------
# STEP 2: LEARNING PLAN GENERATOR
# --------------------------------------------------

print("\nSTEP 2: Generating learning plan...")

learning_prompt = f"""
You are a technical career mentor.

Job profile:
{user_input}

Skills identified:
{skills}

Create a practical 5-day learning plan
based on these skills.

Use the identified skills to tailor the plan.
"""

learning_response = llm.invoke([
    HumanMessage(content=learning_prompt)
])

learning_plan = learning_response.content

print("\nLearning Plan:")
print(learning_plan)

# --------------------------------------------------
# STEP 3: INTERVIEW QUESTION GENERATOR
# --------------------------------------------------

print("\nSTEP 3: Generating interview questions...")

interview_prompt = f"""
You are a technical interviewer.

Job profile:
{user_input}

Identified skills:
{skills}

Learning plan:
{learning_plan}

Generate 5 relevant technical interview questions.
Include questions that test practical understanding.
"""

interview_response = llm.invoke([
    HumanMessage(content=interview_prompt)
])

interview_questions = interview_response.content

print("\nInterview Questions:")
print(interview_questions)

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 50)
print("SEQUENTIAL FLOW COMPLETED")
print("=" * 50)

print("\nSkills:\n", skills)
print("\nLearning Plan:\n", learning_plan)
print("\nInterview Questions:\n", interview_questions)