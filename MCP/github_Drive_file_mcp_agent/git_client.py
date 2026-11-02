
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# Default GitHub owner
GITHUB_OWNER = os.getenv(
    "GITHUB_OWNER",
    "vinipallakki"
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing from .env")

if not GITHUB_TOKEN:
    raise ValueError(
        "GITHUB_PERSONAL_ACCESS_TOKEN missing from .env"
    )


# ============================================================
# 2. OPENAI CLIENT
# ============================================================

llm = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# 3. GITHUB MCP CONFIGURATION
# ============================================================

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

GITHUB_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-MCP-Toolsets": "repos,issues,pull_requests"
}


# ============================================================
# 4. CONVERSATION MEMORY
# ============================================================

# Stores the latest OpenAI response ID.
# This allows GPT to remember previous terminal messages.

previous_response_id = None


# ============================================================
# 5. WRITE OPERATIONS
# ============================================================

WRITE_TOOLS = {

    "create_repository",
    "create_branch",
    "create_or_update_file",
    "delete_file",
    "fork_repository",
    "push_files",

    "issue_write",
    "add_issue_comment",
    "update_issue_comment",
    "sub_issue_write",

    "create_pull_request",
    "update_pull_request",
    "merge_pull_request",
    "update_pull_request_branch",
    "pull_request_review_write",
    "add_comment_to_pending_review",
    "add_reply_to_pull_request_comment"

}


# ============================================================
# 6. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = f"""
You are an intelligent GitHub AI Agent.

You are connected to GitHub through the
GitHub Model Context Protocol (MCP).

Your default GitHub owner is:

{GITHUB_OWNER}

============================================================
YOUR RESPONSIBILITIES
============================================================

1. Repository Management
   - Search repositories
   - List repositories
   - Create repositories
   - Create branches
   - Fork repositories
   - Read repository files
   - Create and update files
   - Push multiple files
   - Delete files

2. Issue Management
   - Create issues
   - Search issues
   - Read issues
   - Update issues
   - Add comments
   - Update comments

3. Pull Request Management
   - Create pull requests
   - Read pull requests
   - Update pull requests
   - Review pull requests
   - Merge pull requests
   - Add review comments

============================================================
IMPORTANT CONVERSATION RULES
============================================================

1. Remember information from previous messages
   in the current conversation.

2. If the user provides a repository name,
   reuse it in subsequent messages.

3. If the user does not provide an owner,
   use the default owner: {GITHUB_OWNER}.

4. Never ask for information already provided.

5. If the user says "yes, use default branch",
   understand that as the repository's default branch.

6. If the source branch is not known,
   retrieve the repository information before
   creating the new branch.

7. If sufficient information is available,
   execute the appropriate MCP tool.

8. Never invent repository names, branch names,
   issue numbers or pull request numbers.

9. Use MCP tools for all GitHub operations.

10. Do not claim an operation succeeded unless
    GitHub confirms its successful execution.

============================================================
OUTPUT RULES
============================================================

1. If the user requests repository names only,
   return repository names only.

2. If the user requests a summary,
   provide a concise summary.

3. Explain successful write operations clearly.

4. If an operation fails, explain the actual error.

5. Do not repeatedly call a failing tool
   with the same arguments.

============================================================
SAFETY
============================================================

Write operations require human approval
from the Python client.

Never attempt to bypass human approval.
"""


# ============================================================
# 7. CONVERT MCP TOOLS INTO OPENAI TOOLS
# ============================================================

def convert_mcp_tools(mcp_tools):  #search_repository, list_repositories, create_repository, create_branch, fork_repository, read_file, create_or_update_file, delete_file, push_files, issue_write, add_issue_comment, update_issue_comment, sub_issue_write, create_pull_request, update_pull_request, merge_pull_request, update_pull_request_branch, pull_request_review_write, add_comment_to_pending_review, add_reply_to_pull_request_comment

    openai_tools = []

    for tool in mcp_tools:   

        # {
        #     "name": tool.name,
        #     "description": tool.description or "",
        #     "parameters": tool.input_schema,
        #     input_schema: tool.input_schema,
        #     "properties": tool.input_schema.get("properties", {}),
        #     "branch": tool.input_schema.get("branch", {}),
        #     "repository": tool.input_schema.get("repository", {}),
        #     "owner": tool.input_schema.get("owner", {}),
        # "type": "function",
        # }

        openai_tools.append({

            "type": "function",

            "name": tool.name,

            "description": tool.description or "",

            "parameters": tool.input_schema,

            "strict": False

        })

    return openai_tools


# ============================================================
# 8. HUMAN APPROVAL
# ============================================================

def request_approval(tool_name, arguments):

    print("\n")
    print("=" * 65)
    print("             GITHUB WRITE OPERATION")
    print("=" * 65)

    print("Tool:", tool_name)

    print("\nArguments:")

    print(
        json.dumps(
            arguments,
            indent=4
        )
    )

    print("\nThis operation will modify GitHub.")

    approval = input(
        "\nApprove this operation? (yes/no): "
    ).strip().lower()

    if approval in ("yes", "y"):

        print("\nApproval granted.")

        return True

    print("\nOperation cancelled by user.")

    return False


# ============================================================
# 9. EXECUTE MCP TOOL
# ============================================================

async def execute_mcp_tool(session, tool_call):

    print("\n")
    print("=" * 65)
    print("                  MCP TOOL SELECTED")
    print("=" * 65)

    print("Tool:", tool_call.name)

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    arguments = json.loads(
        tool_call.arguments
    )

    print("\nArguments:")

    print(
        json.dumps(
            arguments,
            indent=4
        )
    )

    # --------------------------------------------------------
    # Human approval for write operations
    # --------------------------------------------------------

    if tool_call.name in WRITE_TOOLS:

        approved = request_approval(
            tool_call.name,
            arguments
        )

        if not approved:

            return json.dumps({

                "status": "cancelled",

                "message": (
                    "The user declined this GitHub operation. "
                    "Do not retry unless the user explicitly "
                    "requests it again."
                ),

                "tool": tool_call.name

            })

    # --------------------------------------------------------
    # Execute actual MCP tool
    # --------------------------------------------------------

    print("\nCalling GitHub MCP Server...")

    result = await session.call_tool(

        tool_call.name,

        arguments

    )

    # --------------------------------------------------------
    # Check MCP errors
    # --------------------------------------------------------

    is_tool_error = getattr(
        result,
        "is_error",
        getattr(result, "isError", False)
    )

    if is_tool_error:

        error_text = "\n".join(

            content.text

            for content in result.content

            if hasattr(content, "text")

        )

        raise RuntimeError(
            error_text or "GitHub MCP tool execution failed."
        )

    # --------------------------------------------------------
    # Extract response
    # --------------------------------------------------------

    output_parts = []

    for content in result.content:

        if hasattr(content, "text"):

            output_parts.append(
                content.text
            )

    output_text = "\n".join(
        output_parts
    )

    # --------------------------------------------------------
    # Structured response fallback
    # --------------------------------------------------------

    if not output_text:

        structured_content = getattr(
            result,
            "structured_content",
            getattr(result, "structuredContent", None)
        )

        if structured_content is not None:

            output_text = json.dumps(
                structured_content,
                default=str
            )

    print("\nMCP Result received.")

    print("\nResult:")

    print(output_text[:3000])

    return output_text


# ============================================================
# 10. ASK GITHUB AI AGENT
# ============================================================

async def ask_agent(
    session,
    question,
    openai_tools
):

    global previous_response_id

    # --------------------------------------------------------
    # FIRST OPENAI REQUEST
    # --------------------------------------------------------

    print("\nThinking...")

    request_args = {

        "model": "gpt-5",

        "instructions": SYSTEM_INSTRUCTIONS,

        "tools": openai_tools,

        "input": question

    }

    # Continue previous conversation
    if previous_response_id:

        request_args["previous_response_id"] = (
            previous_response_id
        )

    response = llm.responses.create(
        **request_args
    )

    previous_response_id = response.id

    # ========================================================
    # MCP TOOL CALL LOOP
    # ========================================================

    max_tool_rounds = 10

    for round_number in range(max_tool_rounds):

        # ----------------------------------------------------
        # Find function calls
        # ----------------------------------------------------

        function_calls = [

            item

            for item in response.output

            if item.type == "function_call"

        ]

        # ----------------------------------------------------
        # No tool calls: return final response
        # ----------------------------------------------------

        if not function_calls:

            previous_response_id = response.id

            return response.output_text

        # ----------------------------------------------------
        # Execute MCP tools
        # ----------------------------------------------------

        tool_outputs = []

        for tool_call in function_calls:

            try:

                output_text = await execute_mcp_tool(

                    session,

                    tool_call

                )

            except Exception as error:

                print("\nMCP TOOL ERROR:")

                print(type(error).__name__)

                print(str(error))

                output_text = json.dumps({

                    "status": "error",

                    "tool": tool_call.name,

                    "error": str(error)

                })

            # ------------------------------------------------
            # Prepare result for OpenAI
            # ------------------------------------------------

            tool_outputs.append({

                "type": "function_call_output",

                "call_id": tool_call.call_id,

                "output": output_text

            })

        # ----------------------------------------------------
        # Send tool results back to OpenAI
        # ----------------------------------------------------

        response = llm.responses.create(

            model="gpt-5",

            instructions=SYSTEM_INSTRUCTIONS,

            tools=openai_tools,

            input=tool_outputs,

            previous_response_id=response.id

        )

        previous_response_id = response.id

    return (
        "Maximum MCP tool execution rounds reached. "
        "Please narrow down your request."
    )


# ============================================================
# 11. MAIN FUNCTION
# ============================================================

async def main():

    print("=" * 70)

    print("             GITHUB MCP CLIENT + AI AGENT")

    print("=" * 70)

    print("\nDefault GitHub Owner:", GITHUB_OWNER)

    # --------------------------------------------------------
    # HTTP CLIENT
    # --------------------------------------------------------

    async with httpx.AsyncClient(

        headers=GITHUB_HEADERS,

        follow_redirects=True,

        timeout=60.0

    ) as http_client:

        try:

            # ================================================
            # CONNECT TO GITHUB MCP SERVER
            # ================================================

            async with streamable_http_client(

                GITHUB_MCP_URL,

                http_client=http_client

            ) as (read_stream, write_stream):

                async with ClientSession(

                    read_stream,

                    write_stream

                ) as session:

                    print(
                        "\nConnecting to GitHub MCP Server..."
                    )

                    # Initialize MCP
                    await session.initialize()

                    print(
                        "\nConnected to GitHub MCP Server!"
                    )

                    # ========================================
                    # DISCOVER MCP TOOLS
                    # ========================================

                    tools_result = await session.list_tools()

                    print(
                        "\nGitHub MCP Tools Available:"
                    )

                    print("-" * 60)

                    for tool in tools_result.tools:

                        print(
                            f"- {tool.name}"
                        )

                    print("-" * 60)

                    # ========================================
                    # CONVERT MCP TO OPENAI TOOLS
                    # ========================================

                    openai_tools = convert_mcp_tools(

                        tools_result.tools

                    )

                    print(
                        "\nTotal MCP Tools:",
                        len(openai_tools)
                    )

                    print(
                        "\nGitHub MCP Agent is ready!"
                    )

                    print(
                        "\nYou can perform READ and WRITE operations."
                    )

                    print(
                        "\nWrite operations require your approval."
                    )

                    print(
                        "\nType 'exit' to stop."
                    )

                    # ========================================
                    # USER INTERACTION LOOP
                    # ========================================

                    while True:

                        question = input(
                            "\nYou: "
                        ).strip()

                        if question.lower() == "exit":

                            print("\nGoodbye!")

                            break

                        if not question:

                            continue

                        try:

                            answer = await ask_agent(

                                session,

                                question,

                                openai_tools

                            )

                            print(
                                f"\nAgent: {answer}"
                            )

                        except Exception as error:

                            print(
                                "\nAGENT ERROR:",
                                repr(error)
                            )

        except Exception as error:

            print(
                "\nGITHUB MCP CONNECTION FAILED"
            )

            print(
                "Error Type:",
                type(error).__name__
            )

            print(
                "Error Details:",
                repr(error)
            )

            raise


# ============================================================
# 12. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
