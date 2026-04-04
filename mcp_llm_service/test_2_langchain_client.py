"""
Test 2 — LangChain MCP client (production view)
================================================
This is how the agent connects to the mcp_llm server in production.
MultiServerMCPClient discovers the tools advertised by the server and wraps
them as standard LangChain tools — the agent never needs to know it's talking
to an MCP server.

Prerequisites:
    pip install langchain-mcp-adapters langchain-core
    docker-compose up mcp_llm

Run:
    python test_2_langchain_client.py
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # As of langchain-mcp-adapters 0.1.0, the context manager was removed —
    # create the client directly and call get_tools().
    client = MultiServerMCPClient({
        "llm": {"url": "http://localhost:8000/sse", "transport": "sse"}
    })

    # The client connects, runs the MCP handshake, and returns the server's
    # tools as LangChain BaseTool objects ready to be handed to an agent.
    tools = await client.get_tools()
    print(f"Tools discovered: {[t.name for t in tools]}")
    print(f"generate_text description: {tools[0].description}\n")

    # Calling the tool looks identical to calling any LangChain tool.
    # The MCP transport is completely hidden from the caller.
    result = await tools[0].ainvoke({"prompt": "What is the capital of France?"})
    print(f"Response: {result}")


asyncio.run(main())
