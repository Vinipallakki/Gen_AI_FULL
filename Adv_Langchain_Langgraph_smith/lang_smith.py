
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# Load environment variables
load_dotenv()

print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("API Key Exists:", bool(os.getenv("LANGSMITH_API_KEY")))

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ------------------- Tools -------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


@tool
def word_count(text: str) -> str:
    """Count the number of words."""
    return str(len(text.split()))


tools = [calculator, word_count]

# ------------------- Prompt -------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ------------------- Agent -------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ------------------- Run -------------------

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = executor.invoke({"input": question})

    print("\nAgent:", response["output"])



# import os
# import ast
# import operator
# import time
# from pathlib import Path

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# # Load .env from this file's directory
# load_dotenv(Path(__file__).resolve().parent / ".env")

# required = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
# missing = [key for key in required if not os.getenv(key)]

# if missing:
#     raise ValueError(
#         f"Missing environment variables: {', '.join(missing)}"
#     )

# # Enable LangSmith tracing
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ.setdefault(
#     "LANGSMITH_PROJECT", "langsmith-agent-demo"
# )

# print("Tracing:", os.getenv("LANGSMITH_TRACING"))
# print("Project:", os.getenv("LANGSMITH_PROJECT"))

# # LLM
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )

# # Safe arithmetic evaluator: no unrestricted eval()
# OPS = {
#     ast.Add: operator.add,
#     ast.Sub: operator.sub,
#     ast.Mult: operator.mul,
#     ast.Div: operator.truediv,
#     ast.Pow: operator.pow,
#     ast.Mod: operator.mod,
#     ast.USub: operator.neg,
#     ast.UAdd: operator.pos,
# }

# def calculate_node(node):
#     if isinstance(node, ast.Expression):
#         return calculate_node(node.body)

#     if isinstance(node, ast.Constant) and type(node.value) in (int, float):
#         return node.value

#     if isinstance(node, ast.BinOp) and type(node.op) in OPS:
#         left = calculate_node(node.left)
#         right = calculate_node(node.right)

#         # Prevent excessively large exponentiation
#         if isinstance(node.op, ast.Pow) and abs(right) > 10:
#             raise ValueError("Exponent is too large")

#         return OPS[type(node.op)](left, right)

#     if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
#         return OPS[type(node.op)](calculate_node(node.operand))

#     raise ValueError("Only basic arithmetic is allowed")


# @tool
# def calculator(expression: str) -> str:
#     """Calculate a basic arithmetic expression."""
#     tree = ast.parse(expression, mode="eval")
#     result = calculate_node(tree)
#     return str(result)


# @tool
# def word_count(text: str) -> str:
#     """Count the number of words in text."""
#     return str(len(text.split()))


# tools = [calculator, word_count]

# # Prompt
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a helpful assistant. "
#         "Use the calculator tool for arithmetic and "
#         "word_count for counting words. Explain results clearly."
#     ),
#     ("human", "{input}"),
#     MessagesPlaceholder("agent_scratchpad"),
# ])

# # Agent
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt,
# )

# executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
# )

# # Interactive loop
# print("\nLangSmith Agent Demo")
# print("Type 'exit' to stop.\n")

# while True:
#     try:
#         question = input("You: ").strip()

#         if question.lower() == "exit":
#             break

#         if not question:
#             continue

#         start = time.perf_counter()

#         result = executor.invoke({"input": question})

#         elapsed = time.perf_counter() - start

#         print("\nAgent:", result["output"])
#         print(f"Execution time: {elapsed:.2f} seconds")
#         print("Check LangSmith Tracing for this run.\n")

#     except KeyboardInterrupt:
#         print("\nExiting agent.")
#         break

#     except Exception as exc:
#         print(f"\nRequest failed: {type(exc).__name__}: {exc}\n")
