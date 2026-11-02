# imports
# server configuration
# mcp connection
# mcp session
# tool discoveryexit
# tool execution


import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters  #establish connection with the server, manage the session, and call tools on the server
from mcp.client.stdio import stdio_client # establish a connection to the server using standard input/output (stdio) streams.


# ============================================================
# SERVER CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent   #mcp/mcp_emp

SERVER_FILE = BASE_DIR / "server.py"  #mcp/mcp_emp/server.py


server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_FILE)]
)

# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)
    print("        EMPLOYEE MCP CLIENT")
    print("=" * 60)

    print(
        f"\nStarting server:\n{SERVER_FILE}"
    )

    # ========================================================
    # CONNECT TO MCP SERVER
    # ========================================================

    async with stdio_client(
        server_params
    ) as (read, write): #read and write streams for communication with the server read = receiving data from the server, write = sending data to the server


        print(
            "\nSTDIO connection established."
        )

        # ====================================================
        # CREATE MCP SESSION
        # ====================================================

        async with ClientSession( #initialize a client session with the server using the read and write streams for communication discovering available tools and calling them as needed.
            read,
            write
        ) as session:


            print(
                "Initializing MCP session..."
            )


            await session.initialize()


            print(
                "MCP session initialized!"
            )


            # =================================================
            # DISCOVER MCP TOOLS
            # =================================================

            tools_result = (
                await session.list_tools()
            )


            print(
                "\nAvailable MCP Tools:"
            )

            print("-" * 60)


            for tool in tools_result.tools:

                print(
                    f"- {tool.name}"
                )

                if tool.description:

                    print(
                        f"  {tool.description}"
                    )


            print("-" * 60)


            # =================================================
            # MENU
            # =================================================

            print(
                "\nMCP Employee Assistant"
            )

            print(
                "Available commands:"
            )

            print(
                "  1. count"
            )

            print(
                "  2. employee"
            )

            print(
                "  3. department"
            )

            print(
                "  4. exit"
            )


            # =================================================
            # INTERACTIVE LOOP
            # =================================================

            while True:


                question = input(
                    "\nYou: "
                )


                question = (
                    question
                    .lower()
                    .strip()
                )


                # =============================================
                # EXIT
                # =============================================

                if question == "exit":

                    print(
                        "\nGoodbye!"
                    )

                    break


                # =============================================
                # EMPLOYEE COUNT
                # =============================================

                elif question in [
                    "count",
                    "1",
                    "get_employee_count"
                ]:


                    print(
                        "\nCalling MCP Tool:"
                    )

                    print(
                        "get_employee_count"
                    )


                    result = await session.call_tool(

                        "get_employee_count",

                        {}
                    )


                    print(
                        "\nAssistant:",
                        result.content[0].text
                    )


                # =============================================
                # EMPLOYEE DETAILS
                # =============================================

                elif question in [
                    "employee",
                    "2",
                    "get_employee"
                ]:


                    employee_id = input(
                        "Enter employee ID: "
                    )


                    try:

                        employee_id = int(
                            employee_id
                        )

                    except ValueError:

                        print(
                            "Employee ID must be a number."
                        )

                        continue


                    print(
                        "\nCalling MCP Tool:"
                    )

                    print(
                        "get_employee"
                    )


                    result = await session.call_tool(

                        "get_employee",

                        {
                            "employee_id":
                                employee_id
                        }
                    )


                    print(
                        "\nAssistant:",
                        result.content[0].text
                    )


                # =============================================
                # DEPARTMENT
                # =============================================

                elif question in [
                    "department",
                    "3",
                    "get_employees_by_department"
                ]:


                    department = input(
                        "Enter department: "
                    )


                    print(
                        "\nCalling MCP Tool:"
                    )

                    print(
                        "get_employees_by_department"
                    )


                    result = await session.call_tool(

                        "get_employees_by_department",

                        {
                            "department":
                                department
                        }
                    )


                    print(
                        "\nAssistant:",
                        result.content[0].text
                    )


                # =============================================
                # UNKNOWN COMMAND
                # =============================================

                else:

                    print(
                        "\nUnknown command."
                    )

                    print(
                        "Use: count, employee, "
                        "department, or exit."
                    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )