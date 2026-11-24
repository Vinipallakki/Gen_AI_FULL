# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# # Load variables from .env
# load_dotenv()

# # Create OpenAI client
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# # Take input from user
# prompt = input("Enter your question: ")

# # Take temperature from user
# temperature = float(input("Enter temperature (0 to 2): "))

# # Send request to LLM
# response = client.responses.create(
#     model="gpt-4.1",
#     input=prompt,
#     temperature=temperature
# )

# # Print response
# print("\nAI Response:")
# print(response.output_text)


import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# System instruction
system_instruction = """
You are a helpful AI assistant.
Answer questions clearly and concisely.
If you are not sure about something, say that you are not sure.
i need only in 7 bullet points and in simple language.
if you know kannada explain me in Hindi as well
"""

# Take input from user
prompt = input("Enter your question: ")

# Take temperature from user
temperature = float(input("Enter temperature (0 to 2): "))

# Send request to LLM
response = client.responses.create(
    model="gpt-4.1",
    instructions=system_instruction,
    input=prompt,
    temperature=temperature
)

# Print response
print("\nAI Response:")
print(response.output_text)