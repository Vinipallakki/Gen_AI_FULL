from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.sqlite import SqliteSaver


# -----------------------------------------
# OpenAI API Key (Hardcoded)
# -----------------------------------------

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your actual OpenAI API key
# -----------------------------------------
# LLM
# -----------------------------------------

llm = ChatOpenAI(
    model="gpt-5.5",
    api_key=OPENAI_API_KEY,
    temperature=0
)

# -----------------------------------------
# State
# -----------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------------------
# Chatbot Node
# -----------------------------------------

def chatbot(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }

# def adds():
#     return {
#         "messages": [HumanMessage(content="Hello! How can I assist you today?")]
#     }

# -----------------------------------------
# Build Graph
# -----------------------------------------

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "chatbot")
graph.add_edge("chatbot", END)



# -----------------------------------------
# Persistent Memory
# -----------------------------------------

with SqliteSaver.from_conn_string("chat_memory.db") as memory:

    app = graph.compile(
        checkpointer=memory
    )

    config = {
        "configurable": {
            "thread_id": "user_003"
        }
    }

    print("=" * 50)
    print("Persistent Memory Chatbot")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:

        user_input = input("\nYou : ")

        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        print("\nBot :", result["messages"][-1].content)