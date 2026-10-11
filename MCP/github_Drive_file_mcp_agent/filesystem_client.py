import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# FILESYSTEM MCP SERVER
# ============================================================

server_params = StdioServerParameters(

    command="cmd",

    args=[
        "/c",
        "npx",
        "-y",
        "@modelcontextprotocol/server-filesystem",
        r"C:\Users\Admin\Desktop\MicroDegree\Gen-AI-20042026\Gen_AI\MCP\github_Drive_file_mcp_agent\data"
    ]
)

# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)

    print(
        "       FILESYSTEM MCP CLIENT"
    )

    print("=" * 70)


    # ========================================================
    # START FILESYSTEM MCP SERVER
    # ========================================================

    print(
        "\nStarting Filesystem MCP Server..."
    )


    async with stdio_client(
        server_params
    ) as (
        read_stream,
        write_stream
    ):


        # ====================================================
        # CREATE MCP SESSION
        # ====================================================

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:


            # ================================================
            # INITIALIZE MCP
            # ================================================

            await session.initialize()


            print(
                "Connected to Filesystem MCP Server!"
            )


            # ================================================
            # DISCOVER TOOLS
            # ================================================

            result = await session.list_tools()


            print(
                "\nFilesystem MCP Tools:"
            )

            print("-" * 50)


            for tool in result.tools:

                print(
                    f"- {tool.name}"
                )


            print("-" * 50)


            # ================================================
            # ASK USER
            # ================================================

            print(
                "\nFilesystem MCP Client is ready."
            )


            while True:

                question = input(
                    "\nEnter command "
                    "(list/read/search/exit): "
                )


                if question.lower() == "exit":

                    break


                # ============================================
                # LIST DIRECTORY
                # ============================================

                if question.lower() == "list":

                    result = await session.call_tool(

                        "list_directory",

                        {
                            "path": "."
                        }

                    )


                    print(
                        "\nFiles:"
                    )

                    for content in result.content:

                        if hasattr(
                            content,
                            "text"
                        ):

                            print(
                                content.text
                            )


                # ============================================
                # READ EMPLOYEE FILE
                # ============================================

                elif question.lower() == "read":

                    result = await session.call_tool(

                        "read_file",

                        {
                            "path": "employee.txt"
                        }

                    )


                    print(
                        "\nFile Content:"
                    )

                    for content in result.content:

                        if hasattr(
                            content,
                            "text"
                        ):

                            print(
                                content.text
                            )


                # ============================================
                # SEARCH FILES
                # ============================================

                elif question.lower() == "search":

                    result = await session.call_tool(

                        "search_files",

                        {
                            "path": ".",
                            "pattern": "*.txt"
                        }

                    )


                    print(
                        "\nSearch Result:"
                    )

                    for content in result.content:

                        if hasattr(
                            content,
                            "text"
                        ):

                            print(
                                content.text
                            )


                else:

                    print(
                        "Use: list, read, search, or exit"
                    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())