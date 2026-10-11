import asyncio
import json
import os

from dotenv import load_dotenv

from openai import OpenAI

from mcp import (
    ClientSession,
    StdioServerParameters
)

from mcp.client.stdio import stdio_client


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)


# ============================================================
# MCP SERVER CONFIGURATION
# ============================================================

server_params = StdioServerParameters(

    command="python",

    args=[
        "server.py"
    ]
)


# ============================================================
# CONVERT MCP TOOLS TO OPENAI TOOLS
# ============================================================

def convert_mcp_tools(
    mcp_tools
):

    openai_tools = []

    for tool in mcp_tools:

        openai_tools.append({

            "type": "function",

            "name": tool.name,

            "description": (
                tool.description
                or ""
            ),

            "parameters": (
                tool.inputSchema
            ),

            "strict": False
        })

    return openai_tools


# ============================================================
# AGENT
# ============================================================

async def ask_agent(
    session,
    question,
    openai_tools
):

    response = client.responses.create(

        model="gpt-5",

        instructions="""
You are an AI agent connected to an MCP server.

You have access to these MCP tools:

1. wikipedia_search
2. get_weather
3. get_employee
4. calculate

Choose the correct tool based on the user's question.

Rules:

- Use Wikipedia for general knowledge.
- Use get_weather for weather questions.
- Use get_employee for employee questions.
- Use calculate for mathematical calculations.
- Do not invent employee information.
- If a tool is needed, use the tool.
""",

        tools=openai_tools,

        input=question
    )


    # ========================================================
    # TOOL CALL LOOP
    # ========================================================

    while True:

        function_calls = [

            item

            for item in response.output

            if item.type == "function_call"

        ]


        # ----------------------------------------------------
        # No tool call
        # ----------------------------------------------------

        if not function_calls:

            return response.output_text


        tool_outputs = []


        # ----------------------------------------------------
        # Execute MCP tools
        # ----------------------------------------------------

        for tool_call in function_calls:

            print(
                f"\n[MCP TOOL SELECTED]"
            )

            print(
                f"Tool: {tool_call.name}"
            )


            arguments = json.loads(
                tool_call.arguments
            )


            print(
                f"Arguments: {arguments}"
            )


            # ================================================
            # THIS IS THE MCP CALL
            # ================================================

            result = await session.call_tool(

                tool_call.name,

                arguments
            )


            # ------------------------------------------------
            # Extract result
            # ------------------------------------------------

            text_result = ""

            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    text_result += (
                        content.text
                    )


            print(
                f"Tool Result: {text_result}"
            )


            # ------------------------------------------------
            # Send result back to LLM
            # ------------------------------------------------

            tool_outputs.append({

                "type":
                    "function_call_output",

                "call_id":
                    tool_call.call_id,

                "output":
                    text_result
            })


        # ====================================================
        # ASK LLM AGAIN
        # ====================================================

        response = client.responses.create(

            model="gpt-5",

            instructions="""
Use the MCP tool results to answer
the user's question clearly.
""",

            tools=openai_tools,

            input=tool_outputs,

            previous_response_id=response.id
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)

    print(
        "        MCP MULTI-TOOL AI AGENT"
    )

    print("=" * 70)


    # ========================================================
    # START MCP SERVER + CONNECT
    # ========================================================

    async with stdio_client(
        server_params
    ) as (
        read,
        write
    ):

        async with ClientSession(
            read,
            write
        ) as session:

            # -----------------------------------------------
            # Initialize MCP
            # -----------------------------------------------

            await session.initialize()


            # -----------------------------------------------
            # Discover MCP tools
            # -----------------------------------------------

            tools_result = (
                await session.list_tools()
            )


            print(
                "\nMCP Tools discovered:"
            )


            for tool in tools_result.tools:

                print(
                    f" - {tool.name}"
                )


            # -----------------------------------------------
            # Convert to LLM tool definitions
            # -----------------------------------------------

            openai_tools = (
                convert_mcp_tools(
                    tools_result.tools
                )
            )


            print(
                "\nType your question."
            )

            print(
                "Type 'exit' to quit.\n"
            )


            # =================================================
            # CHAT LOOP
            # =================================================

            while True:

                question = input(
                    "You: "
                )


                if question.lower() == "exit":

                    break


                try:

                    answer = await ask_agent(

                        session,

                        question,

                        openai_tools
                    )


                    print(
                        f"\nAgent: {answer}\n"
                    )


                except Exception as e:

                    print(
                        f"\nError: {e}\n"
                    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )