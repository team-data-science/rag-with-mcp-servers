"""
RAG MCP Agent — OpenAI-compatible API server

Open WebUI connects to this service as if it were an OpenAI endpoint:
  OPENAI_API_BASE_URL=http://pipelines:9099

The /v1/models endpoint tells Open WebUI which "models" (agents) are available.
The /v1/chat/completions endpoint receives every user message and returns the
agent's response in OpenAI format.

This lets us keep all agent logic in agent.py while Open WebUI handles the chat
UI, user management, and conversation history — without any dependency conflicts.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging
import time
from rag_mcp_pipeline import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG MCP Agent")

MODEL_ID = "rag-mcp-agent"


# --- OpenAI-compatible request/response models ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-rag-mcp"
    object: str = "chat.completion"
    created: int = 0
    model: str = MODEL_ID
    choices: List[Choice]


# --- Endpoints ---

def models_response():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rag-mcp",
        }]
    })

@app.get("/v1/models")
async def list_models_v1():
    """OpenAI-standard path — used by most clients."""
    return models_response()

@app.get("/models")
async def list_models():
    """Non-versioned path — newer Open WebUI versions call this."""
    return models_response()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Receive a conversation from Open WebUI, run it through the RAG MCP agent,
    and return the response in OpenAI chat completion format.

    Open WebUI sends the full conversation history on every request — no
    separate memory management needed on our side.
    """
    logger.info("Received %d messages", len(request.messages))

    # Convert Pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    answer = await run_agent(messages)

    return ChatCompletionResponse(
        created=int(time.time()),
        choices=[Choice(message=Message(role="assistant", content=answer))],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099)
