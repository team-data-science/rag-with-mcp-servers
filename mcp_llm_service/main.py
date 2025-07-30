from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from llm import ask_llm

class MCPMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class MCPRequest(BaseModel):
    model: str
    messages: List[MCPMessage]
    max_tokens: int
    temperature: float

class MCPResponse(BaseModel):
    messages: List[MCPMessage]

app = FastAPI(title="MCP LLM Service")

@app.post("/generate", response_model=MCPResponse)
async def generate(request: MCPRequest):
    try:
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
        answer = ask_llm(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        response_messages = request.messages + [MCPMessage(role="assistant", content=answer)]
        return MCPResponse(messages=response_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
