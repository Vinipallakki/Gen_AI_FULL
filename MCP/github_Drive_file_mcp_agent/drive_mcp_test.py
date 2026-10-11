import asyncio
import json
import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# GOOGLE DRIVE MCP SERVER
# ============================================================

DRIVE_MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"


# ============================================================
# LOAD GOOGLE ACCESS TOKEN
# ============================================================

with open("google_token.json", "r") as file:
    token = json.load(file)

ACCESS_TOKEN = token["access_token"]


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)
    print("       GOOGLE DRIVE MCP CLIENT")
    print("=" * 70)


    # ========================================================
    # HTTP HEADERS
    # ========================================================

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }


    # ========================================================
    # HTTP CLIENT
    # ========================================================

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=60.0
    ) as http_client:


        # ====================================================
        # CONNECT TO GOOGLE DRIVE MCP SERVER
        # ====================================================

        print(
            "\nConnecting to Google Drive MCP Server..."
        )


        async with streamable_http_client(
            DRIVE_MCP_URL,
            http_client=http_client
        ) as (
            read_stream,
            write_stream
        ):


            # =================================================
            # MCP CLIENT SESSION
            # =================================================

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:


                # =============================================
                # INITIALIZE MCP
                # =============================================

                await session.initialize()


                print(
                    "Connected to Google Drive MCP Server!"
                )


                # =============================================
                # DISCOVER TOOLS
                # =============================================

                tools_result = (
                    await session.list_tools()
                )


                print(
                    "\nAvailable MCP Tools:"
                )

                print("-" * 50)


                for tool in tools_result.tools:

                    print(
                        f"- {tool.name}"
                    )


                print("-" * 50)


                # =============================================
                # SEARCH DRIVE
                # =============================================

                print(
                    "\nSearching Google Drive..."
                )


                result = await session.call_tool(

                    "search_files",

                    {
                        "query": "owner = 'me'",

                        "pageSize": 10,

                        "excludeContentSnippets": True
                    }
                )


                # =============================================
                # DISPLAY RESULT
                # =============================================

                print(
                    "\nSearch Result:"
                )

                print("=" * 60)


                for content in result.content:

                    if hasattr(content, "text"):

                        print(
                            content.text
                        )


                print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())