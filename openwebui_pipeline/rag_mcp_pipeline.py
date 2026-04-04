"""
RAG MCP Agent — core agent logic

This module contains the run_agent() function that:
  1. Connects to both MCP servers (LLM + RAG) via SSE
  2. Builds a LangGraph ReAct agent with the discovered tools
  3. Runs the conversation and returns the final answer

It is called by main.py on every incoming chat request.
"""

import os
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
LLM_MCP_URL = os.getenv("LLM_MCP_URL", "http://mcp_llm:8000/sse")
RAG_MCP_URL = os.getenv("RAG_MCP_URL", "http://mcp_rag:8001/sse")

# Each key is a logical name for the server; the agent sees its tools prefixed
# with this name so you can tell at a glance which server a tool came from.
MCP_SERVERS = {
    "llm": {"url": LLM_MCP_URL, "transport": "sse"},
    "rag": {"url": RAG_MCP_URL, "transport": "sse"},
}

# The system prompt teaches the agent *how* to use its two MCP tools together.
# This is the core of the RAG pattern: retrieve context first, then generate.
SYSTEM_PROMPT = """You are a helpful assistant with access to two tools:

- search_knowledge_base: searches a vector database and returns relevant context
- generate_text: sends a prompt to an LLM and returns the generated response

Follow these steps for every question:
1. Call search_knowledge_base with the user's question to retrieve relevant context.
2. If context was found, call generate_text with a prompt that includes both
   the context and the question so the answer is grounded in the knowledge base.
3. If no context was found, call generate_text with just the question and answer
   from general knowledge.

Always return the final generated answer to the user."""


async def run_agent(messages: list[dict]) -> str:
    """Connect to both MCP servers, build a ReAct agent, and answer the question.

    Opens an SSE connection to each MCP server, fetches the available tools,
    then runs a LangGraph ReAct loop:
        think → choose tool → observe result → think → ... → final answer

    Args:
        messages: Conversation history as [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        The agent's final text response.
    """
    # MultiServerMCPClient connects to all servers and merges their tool lists.
    # As of langchain-mcp-adapters 0.1.0, the context manager was removed —
    # create the client directly and call get_tools().
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    logger.info("Agent loaded %d MCP tools: %s", len(tools), [t.name for t in tools])

    # The backbone LLM handles reasoning and tool selection.
    # temperature=0 keeps tool-call decisions deterministic.
    backbone_llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    # create_react_agent wires the LLM and tools into a LangGraph ReAct loop.
    # state_modifier prepends the system prompt to every invocation.
    agent = create_react_agent(backbone_llm, tools, prompt=SYSTEM_PROMPT)

    # Convert the dict-based history into LangChain message objects.
    # System messages are dropped — SYSTEM_PROMPT handles that role.
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    result = await agent.ainvoke({"messages": lc_messages})

    # The last message in the result is always the agent's final answer.
    return result["messages"][-1].content
