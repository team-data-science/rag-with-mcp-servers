#!/bin/bash
# Test 1 — MCP Inspector (browser UI)
#
# The MCP Inspector is a browser-based tool that connects to any running MCP
# server and lets you browse its tools and call them without writing any code.
#
# --- Why we pin to version 0.13.0 ---
# Inspector 0.14+ added OAuth discovery: before connecting it probes
# /.well-known/oauth-protected-resource and /.well-known/oauth-authorization-server
# Our MCP servers are internal Docker services with no auth, so those endpoints
# return 404 and the connection fails.
#
# This is an inspector-side issue, not a problem with our server implementation.
# FastMCP is the official high-level API inside the mcp Python SDK (from Anthropic)
# and is the recommended way to build MCP servers. Any MCP server — FastMCP or
# hand-rolled — would fail the same way against inspector 0.14+.
# Pinning to 0.13.0 skips the OAuth probing and connects cleanly.
#
# --- Connection details ---
# Always use the /sse path — the root URL (/) is not a valid MCP endpoint.
# FastMCP only registers the SSE handler at /sse.
#
# Prerequisites:
#   - Node.js installed (npx comes with it)
#   - mcp_llm container running:  docker-compose up mcp_llm
#
# Usage:
#   chmod +x test_1_inspector.sh
#   ./test_1_inspector.sh

echo "Opening MCP Inspector for mcp_llm ..."
echo ""
echo "  Transport type : SSE"
echo "  URL            : http://localhost:8000/sse"
echo ""
echo "A browser tab will open. Select 'Tools' to see generate_text and call it."
echo ""

npx @modelcontextprotocol/inspector@0.13.0
