"""
Test 3 — Raw MCP client (teaching view)
========================================
This script uses the mcp library's own client directly, with no LangChain
abstraction. Every step of the MCP protocol is visible:

  1. sse_client()      — opens the SSE transport connection
  2. ClientSession()   — creates a protocol session over that connection
  3. initialize()      — MCP handshake: client and server exchange capabilities
  4. list_tools()      — client asks "what tools do you have?"
  5. call_tool()       — client invokes a specific tool by name

This is what MultiServerMCPClient (Test 2) does under the hood on every
agent run. Understanding these steps makes it easy to debug MCP servers,
write new clients, or implement your own transport.

Prerequisites:
    pip install mcp
    docker-compose up mcp_llm

Run:
    python test_3_raw_mcp_client.py
"""

import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession


async def main():
    print("Step 1 — Open SSE transport connection to the server")
    async with sse_client("http://localhost:8000/sse") as (read, write):
        print("           connection open\n")

        print("Step 2 — Create an MCP ClientSession over the connection")
        async with ClientSession(read, write) as session:
            print("           session created\n")

            print("Step 3 — MCP handshake: initialize()")
            init = await session.initialize()
            print(f"           server name:    {init.server_info.name}")
            print(f"           server version: {init.server_info.version}\n")

            print("Step 4 — list_tools(): ask the server what it can do")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"           tool: {tool.name!r}")
                print(f"           desc: {tool.description}")
                print(f"           args: {tool.inputSchema}\n")

            print("Step 5 — call_tool(): invoke generate_text directly")
            result = await session.call_tool(
                "generate_text",
                {"prompt": "What is the capital of France?"},
            )
            print(f"           response: {result.content[0].text}")


asyncio.run(main())
