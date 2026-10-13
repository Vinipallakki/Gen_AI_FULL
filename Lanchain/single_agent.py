import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.retrievers import WikipediaRetriever

# ============= STEP 1: Load .env =============
load_dotenv()

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ============= STEP 2: Initialize LLM =============
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0,
    api_key=api_key
)

# ============= STEP 3: Define Tools =============
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression supplied as a string."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        retriever = WikipediaRetriever()
        results = retriever.invoke(query)
        return results[0].page_content if results else "No results found"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_info(query: str) -> str:
    """Get current information about a topic."""
    return f"Information about {query} retrieved."

# ============= STEP 4: Setup Memory =============
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# ============= STEP 5: Create Prompt Template =============
prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a helpful assistant with tools at your disposal. "
        "Use tools to answer questions accurately. "
        "Remember previous conversations and context."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ============= STEP 6: Build the Agent =============
tools = [calculate, wiki_search, get_current_info]



agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=10
)

# ============= STEP 7: Run Agent =============
if __name__ == "__main__":
    print("Agent is ready! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")