import asyncio
import json
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing from .env"
    )


if not GITHUB_TOKEN:
    raise ValueError(
        "GITHUB_PERSONAL_ACCESS_TOKEN is missing from .env"
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

llm = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# GITHUB MCP SERVER
# ============================================================

GITHUB_MCP_URL = (
    "https://api.githubcopilot.com/mcp/"
)


# ============================================================
# CONVERT MCP TOOLS TO OPENAI TOOLS
# ============================================================

def convert_mcp_tools(mcp_tools):

    openai_tools = []

    for tool in mcp_tools:

        openai_tools.append(
            {
                "type": "function",

                "name": tool.name,

                "description": (
                    tool.description
                    or ""
                ),

                "parameters": tool.input_schema,

                "strict": False
            }
        )

    return openai_tools


# ============================================================
# ASK AGENT
# ============================================================

async def ask_agent(
    session,
    question,
    openai_tools
):

    # --------------------------------------------------------
    # First request to LLM
    # --------------------------------------------------------

    response = llm.responses.create(

        model="gpt-5",

        instructions="""
        You are a helpful GitHub AI assistant.

        You are connected to GitHub through
        the Model Context Protocol (MCP).

        Use the available MCP tools whenever
        the user asks for information from GitHub.

        Important rules:

        1. Use MCP tools for GitHub information.
        2. Do not invent repository information.
        3. Do not invent issue information.
        4. Do not invent pull request information.
        5. If the user asks about a repository,
        use the appropriate GitHub MCP tool.
        6. If the user asks about issues,
        use the issue-related MCP tool.
        7. If the user asks about pull requests,
        use the pull-request MCP tool.
        """,

        tools=openai_tools,

        input=question
    )


    # ========================================================
    # MCP TOOL CALL LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Find function calls from the LLM
        # ----------------------------------------------------

        function_calls = [

            item

            for item in response.output

            if item.type == "function_call"

        ]


        # ----------------------------------------------------
        # If no tool call, return final answer
        # ----------------------------------------------------

        if not function_calls:

            return response.output_text


        tool_outputs = []


        # ====================================================
        # EXECUTE MCP TOOLS
        # ====================================================

        for tool_call in function_calls:

            print()
            print("=" * 60)
            print("MCP TOOL SELECTED")
            print("=" * 60)

            print(
                f"Tool: {tool_call.name}"
            )


            # ------------------------------------------------
            # Parse arguments
            # ------------------------------------------------

#"{owner}/{repo_name/pull}/{pull_number}"
#owner : repo_name = pull_number : None


            arguments = json.loads(
                tool_call.arguments
            )


            print(
                f"Arguments: {arguments}"
            )


            print(
                "\nCalling GitHub MCP Server..."
            )


            # =================================================
            # THIS IS THE ACTUAL MCP CALL
            # =================================================

            result = await session.call_tool(

                tool_call.name,

                arguments
            )


            # =================================================
            # READ MCP RESPONSE
            # =================================================

            output_text = ""


            for content in result.content:

                if hasattr(
                    content,
                    "text"
                ):

                    output_text += (
                        content.text
                    )


            print(
                "\nMCP Result received."
            )


            # ------------------------------------------------
            # Send result back to LLM
            # ------------------------------------------------

            tool_outputs.append(

                {
                    "type":
                        "function_call_output",

                    "call_id":
                        tool_call.call_id,

                    "output":
                        output_text
                }

            )


        # ====================================================
        # SEND TOOL RESULT BACK TO LLM
        # ====================================================

        response = llm.responses.create(

            model="gpt-5",

            instructions="""
            Use the results returned by the
            GitHub MCP server to answer the
            user's question.

            Be accurate and concise.
            Do not invent information.
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
    print("       GITHUB MCP CLIENT + AI AGENT")
    print("=" * 70)

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=30.0
    ) as http_client:

        async with streamable_http_client(
            GITHUB_MCP_URL,
            http_client=http_client
        ) as (
            read_stream,
            write_stream
        ):

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                print(
                    "\nConnecting to GitHub MCP Server..."
                )

                await session.initialize()

                print(
                    "Connected to GitHub MCP Server!"
                )

                # Discover MCP tools
                tools_result = await session.list_tools()

                print(
                    "\nGitHub MCP Tools Available:"
                )

                print("-" * 50)

                for tool in tools_result.tools:
                    print(
                        f"- {tool.name}"
                    )

                print("-" * 50)

                # Convert MCP tools to OpenAI format
                openai_tools = convert_mcp_tools(
                    tools_result.tools
                )

                print(
                    "\nGitHub MCP Agent is ready!"
                )

                print(
                    "Ask questions about GitHub."
                )

                print(
                    "Type 'exit' to stop.\n"
                )

                while True:

                    question = input(
                        "You: "
                    )

                    if question.lower().strip() == "exit":
                        print("\nGoodbye!")
                        break

                    if not question.strip():
                        continue

                    try:

                        answer = await ask_agent(
                            session,
                            question,
                            openai_tools
                        )

                        print(
                            f"\nAgent: {answer}\n"
                        )

                    except Exception as error:

                        print(
                            f"\nERROR: {error}\n"
                        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )