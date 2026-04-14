# RAG Agent with MCP Servers

A teaching project that demonstrates how to build a modern AI agent using the **Model Context Protocol (MCP)**, **LangChain**, **LangGraph**, and **Open WebUI**. The agent answers questions by retrieving relevant context from a vector database and generating grounded responses — a pattern called Retrieval-Augmented Generation (RAG).

Everything runs locally in Docker. No cloud API keys required.

---

## What You Will Learn

- What MCP is and how to build real MCP servers in Python
- How an AI agent discovers and calls tools at runtime
- How RAG works: retrieve context first, then generate an answer
- How LangGraph's ReAct loop orchestrates multi-step reasoning
- How to connect a local chat UI (Open WebUI) to a custom agent backend

---

## Architecture Overview

```
Browser
   ↓
Open WebUI (port 3000)          — chat interface, user accounts, conversation history
   ↓  OpenAI-compatible API
openwebui_pipeline (port 9099)  — FastAPI server, hosts the LangGraph agent
   ↓
LangGraph ReAct Agent           — reasons, selects tools, observes results
   ├──→ MCP LLM Server (port 8000)   — tool: generate_text   (uses Ollama/qwen2.5:7b)
   └──→ MCP RAG Server (port 8001)   — tool: search_knowledge_base (uses Qdrant)
                                                    ↓
                                              Qdrant (port 6333)  — vector database
```

---

## Project Structure

```
rag-with-mcp-servers/
│
├── mcp_llm_service/                 # MCP Server 1 — LLM tool
│   ├── server.py                    # FastMCP server, exposes generate_text tool
│   ├── requirements.txt             # mcp[cli], langchain-ollama
│   ├── Dockerfile
│   ├── test_1_inspector.sh          # Test with browser UI (MCP Inspector)
│   ├── test_2_langchain_client.py   # Test the way the agent uses it
│   └── test_3_raw_mcp_client.py     # Test at raw MCP protocol level
│
├── mcp_rag_server/                  # MCP Server 2 — RAG tool
│   ├── server.py                    # FastMCP server, exposes search_knowledge_base tool
│   ├── requirements.txt             # mcp[cli], langchain-qdrant, langchain-huggingface
│   └── Dockerfile
│
├── openwebui_pipeline/              # The Agent
│   ├── main.py                      # FastAPI server, OpenAI-compatible API for Open WebUI
│   ├── rag_mcp_pipeline.py          # LangGraph ReAct agent, connects to both MCP servers
│   ├── requirements.txt             # langchain-mcp-adapters, langgraph, langchain-ollama
│   └── Dockerfile
│
├── bulk_importer/                   # One-time data loader
│   ├── voltedge_creator.py          # Reads voltedge_qa.json, generates embeddings, stores in Qdrant
│   └── voltedge_qa.json             # VoltEdge Q&A knowledge base (fictional company — unknown to Qwen)
│
└── docker-compose.yml               # Wires all services together
```

---

## Key Libraries Explained

### `mcp` — Model Context Protocol SDK
The official Python SDK from Anthropic for building MCP servers and clients.

**`FastMCP`** is the high-level API inside this SDK. It lets you turn any Python function into an MCP tool with a single decorator:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(input: str) -> str:
    """Description of what this tool does — the agent reads this."""
    return do_something(input)

mcp.run(transport="sse")   # starts an HTTP server on SSE transport
```

The docstring is not just documentation — the agent uses it at runtime to decide when to call the tool.

---

### `langchain-mcp-adapters` — MCP Client for LangChain
Connects a LangChain agent to one or more MCP servers. It runs the MCP handshake (`initialize` → `list_tools`) and wraps each discovered tool as a standard LangChain `BaseTool` object. The agent never needs to know it is talking to an MCP server.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "llm": {"url": "http://mcp_llm:8000/sse", "transport": "sse"},
    "rag": {"url": "http://mcp_rag:8001/sse", "transport": "sse"},
})
tools = await client.get_tools()
# → [generate_text, search_knowledge_base]
```

---

### `langgraph` — Agent Orchestration
LangGraph builds stateful, multi-step agents. We use the **ReAct** (Reason + Act) pattern via `create_react_agent`:

```
Think  →  choose tool  →  call tool  →  observe result  →  think again  →  ...  →  final answer
```

This loop continues until the agent decides it has enough information to give a final answer. Each step is visible in the logs.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
result = await agent.ainvoke({"messages": [HumanMessage(content="...")]})
```

---

### `langchain-ollama` — Local LLM Integration
Connects LangChain to a locally running Ollama instance. Used in two places:

1. **`mcp_llm_service/server.py`** — `ChatOllama` generates the final text response (uses `qwen2.5:7b`)
2. **`openwebui_pipeline/rag_mcp_pipeline.py`** — `ChatOllama` drives the agent's reasoning (uses `qwen2.5:7b`)

---

### `langchain-qdrant` + `langchain-huggingface` — RAG Stack
Used in `mcp_rag_server/server.py` to search the vector database:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = QdrantVectorStore(client=qdrant_client, collection_name="voltedge-qa", embedding=embeddings)
docs = store.similarity_search(question, k=3)
```

The same embedding model is used in `bulk_importer/voltedge_creator.py` when loading data. This is critical — the vectors must be in the same embedding space to be comparable.

---

## How a Question Gets Answered — Step by Step

1. **You type a question** in Open WebUI at `http://localhost:3000`

2. **Open WebUI** assembles the full conversation history and sends it to our pipeline via the OpenAI chat completions API:
   ```
   POST http://pipelines:9099/chat/completions
   {"messages": [{"role": "user", "content": "What is X?"}]}
   ```

3. **`openwebui_pipeline/main.py`** receives the request and calls `run_agent(messages)`

4. **`rag_mcp_pipeline.py`** creates a `MultiServerMCPClient`, connects to both MCP servers, and retrieves the tool list

5. **The ReAct agent starts its loop:**

   - **Think:** "I should search the knowledge base first"
   - **Act:** calls `search_knowledge_base("What is X?")`
   - The call goes via SSE to `mcp_rag:8001`, which embeds the question, queries Qdrant, and returns matching text
   - **Observe:** "Found 3 relevant passages: ..."
   - **Think:** "I have context, now I'll generate a grounded answer"
   - **Act:** calls `generate_text("Use this context: ... Answer: What is X?")`
   - The call goes via SSE to `mcp_llm:8000`, which sends the prompt to Ollama/qwen2.5:7b
   - **Observe:** "The answer is ..."
   - **Think:** "I have the final answer"

6. **The answer string** travels back through `main.py` → Open WebUI → your browser

---

## The MCP Protocol Handshake

When the agent connects to an MCP server, three things happen in order. You can observe all three in `mcp_llm_service/test_3_raw_mcp_client.py`:

| Step | MCP Call | What happens |
|---|---|---|
| 1 | `initialize()` | Client and server exchange capabilities and version info |
| 2 | `list_tools()` | Client asks "what tools do you have?" — server returns names, descriptions, and argument schemas |
| 3 | `call_tool()` | Client invokes a specific tool by name with arguments |

This handshake is what makes MCP powerful: the agent discovers tools dynamically at runtime. You can add a new tool to an MCP server and the agent will find it on the next connection — no code changes needed in the agent.

---

## Testing the MCP Servers

Three approaches are provided for `mcp_llm_service/`, from easiest to most instructive:

### Test 1 — Browser UI (MCP Inspector)
```bash
docker-compose up mcp_llm
./mcp_llm_service/test_1_inspector.sh
# Connect with: Transport=SSE, URL=http://localhost:8000/sse
# Click Tools → List Tools → Call generate_text
```
> **Note:** Pin to version `0.13.0`. Newer versions probe for OAuth endpoints (`/.well-known/oauth-*`) that FastMCP does not implement, causing connection failures.

### Test 2 — LangChain Client (how the agent uses it)
```bash
pip install langchain-mcp-adapters
python mcp_llm_service/test_2_langchain_client.py
```
The MCP transport is completely hidden — tools look like regular LangChain tools.

### Test 3 — Raw MCP Client (see the protocol directly)
```bash
pip install mcp
python mcp_llm_service/test_3_raw_mcp_client.py
```
Every protocol step is labelled and printed. This is what `MultiServerMCPClient` does under the hood.

---

## Setup & Running

### Prerequisites
- Docker Desktop
- Ollama running locally with the following model pulled:
  ```bash
  ollama pull qwen2.5:7b     # used by both mcp_llm (text generation) and the agent backbone (tool calling)
  ```

### 1 — Load data into Qdrant
```bash
cd bulk_importer
pip install qdrant-client sentence-transformers
python voltedge_creator.py
```

### 2 — Start everything
```bash
docker-compose up -d --build
```

### 3 — Open the chat UI
Go to `http://localhost:3000`, create an account, select **rag-mcp-agent** as the model, and start chatting.

### Useful commands
```bash
# Watch all logs
docker-compose logs -f

# Watch a specific service
docker-compose logs -f pipelines
docker-compose logs -f mcp_rag

# Rebuild a single service after code changes
docker-compose up -d --build pipelines

# Stop everything
docker-compose down

# Stop and delete all data (Qdrant + Open WebUI)
docker-compose down -v
```

---

## Environment Variables

All configuration lives in `docker-compose.yml`. Key variables:

| Service | Variable | Default | Description |
|---|---|---|---|
| `mcp_llm` | `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Where to reach Ollama |
| `mcp_llm` | `MODEL_NAME` | `qwen2.5:7b` | Model used for text generation |
| `mcp_rag` | `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection |
| `mcp_rag` | `QDRANT_COLLECTION` | `voltedge-qa` | Collection to search |
| `mcp_rag` | `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must match voltedge_creator.py |
| `pipelines` | `MODEL_NAME` | `qwen2.5:7b` | Agent backbone model (needs tool calling support) |
| `pipelines` | `LLM_MCP_URL` | `http://mcp_llm:8000/sse` | LLM MCP server endpoint |
| `pipelines` | `RAG_MCP_URL` | `http://mcp_rag:8001/sse` | RAG MCP server endpoint |

---

## Why This Architecture?

### Why MCP instead of direct API calls?
Without MCP, the agent would have hard-coded HTTP calls to specific endpoints. With MCP, the agent asks "what can you do?" at runtime and gets back a structured description. You can swap, add, or update tools without changing the agent code.

### Why qwen2.5:7b for everything?
`qwen2.5:7b` handles both the agent's tool-calling reasoning and the final text generation. It has strong structured output support (needed for tool calls) while also producing coherent prose. Using a single model simplifies the setup — no need to pull and manage multiple models locally.

### Why Open WebUI instead of building a custom frontend?
Open WebUI handles user authentication, conversation history, and the chat interface out of the box. Our agent code stays focused on intelligence, not plumbing.

### Why LangGraph instead of a simple loop?
LangGraph makes the agent's reasoning steps explicit and inspectable. Each think/act/observe cycle is a node in a graph. This makes it easier to debug, extend, and teach.
