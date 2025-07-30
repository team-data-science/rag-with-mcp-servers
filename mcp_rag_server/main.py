from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import requests
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_core.runnables.base import RunnableLambda
# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# If you want to see full payload dumps, uncomment:
# logger.setLevel(logging.DEBUG)

class MCPMessage(BaseModel):
    role: str
    content: str

class MCPRequest(BaseModel):
    model: str
    messages: List[MCPMessage]
    max_tokens: int
    temperature: float

class MCPResponse(BaseModel):
    messages: List[MCPMessage]

app = FastAPI(title="MCP RAG Server")

logger.info("Initializing Qdrant client...")
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None)
)
logger.info("Qdrant client initialized")

# Ensure the Qdrant collection exists
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "q-and-a")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
try:
    collections = qdrant.get_collections().collections
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' does not exist. Creating it...")
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}' with vector size {VECTOR_SIZE}.")
    else:
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")
except Exception as e:
    logger.error(f"Error ensuring Qdrant collection exists: {e}", exc_info=True)

def retrieve_context(input_dict):
    try:
        question = input_dict["question"]
        logger.info(f"Retrieving context for question: {question!r}")

        # 1) embed
        from llm import get_embedding
        vector = get_embedding(question)
        logger.info(f"→ Generated embedding vector of length {len(vector)}")

        # 2) search
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=int(os.getenv("RAG_K", "3"))
        )

        # 3) log each hit in detail
        context_texts = []
        for i, h in enumerate(hits, start=1):
            score = getattr(h, 'score', None)
            payload = h.payload or {}
            answer = payload.get("answer") or payload.get("text") or "<no-answer-field>"
            logger.info(f"→ Hit {i}: id={h.id!r}, score={score!r}, payload_keys={list(payload.keys())}")
            logger.debug(f"   full payload: {payload}")
            context_texts.append(answer)

        logger.info(f"Retrieved {len(context_texts)} context item(s) from Qdrant")

        if not context_texts:
            logger.warning("No context found in Qdrant — falling back to empty placeholder")
            context_texts = ["No relevant context found in the database."]

        return {"question": question, "context": context_texts}

    except Exception as e:
        logger.error(f"Error in retrieve_context: {e}", exc_info=True)
        return {"question": input_dict.get("question", ""), "context": ["Error retrieving context from database."]}

def build_rag_prompt(input_dict):
    try:
        question = input_dict["question"]
        context = input_dict["context"]
        logger.info(f"Building RAG prompt for question: {question!r}")

        ctx = "\n\n".join(context)
        prompt = f"Use the following context to answer the question:\n\n{ctx}\n\nQuestion: {question}\nAnswer:"
        logger.info(f"Built prompt of length: {len(prompt)}")
        return {"question": question, "prompt": prompt}

    except Exception as e:
        logger.error(f"Error in build_rag_prompt: {e}", exc_info=True)
        return {"question": input_dict.get("question", ""), "prompt": f"Question: {input_dict.get('question','')}\nAnswer:"}

def llm_mcp_call(input_dict):
    try:
        prompt = input_dict["prompt"]
        logger.info(f"Calling LLM with prompt: {prompt[:200]!r}")

        llm_url = os.getenv("LLM_SERVICE_URL", "http://mcp_llm:8000/generate")
        payload = {
            "model": "mistral",
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7
        }

        logger.info(f"Sending request to LLM service: {llm_url}")
        resp = requests.post(llm_url, json=payload, timeout=30)
        logger.info(f"LLM service response status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"LLM service response: {data}")
            assistant_content = next((m["content"] for m in data.get("messages", []) if m["role"] == "assistant"), None)
            if assistant_content:
                logger.info(f"Generated LLM response: {assistant_content[:100]!r}")
                return assistant_content
            else:
                logger.error("No assistant message found in LLM response")
                return "I couldn't generate a response. Please try again."
        else:
            logger.error(f"LLM service returned error {resp.status_code}: {resp.text}")
            return "Sorry, the language model is currently unavailable."

    except requests.exceptions.Timeout:
        logger.error("Timeout calling LLM service")
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error calling LLM service")
        return "Sorry, I couldn't connect to the language model. Please try again."
    except Exception as e:
        logger.error(f"Error in llm_mcp_call: {e}", exc_info=True)
        return "Sorry, there was an error generating the response."

# Set up RAG sequence
logger.info("Setting up RAG sequence...")
rag_sequence = (
    RunnableLambda(retrieve_context)
    | RunnableLambda(build_rag_prompt)
    | RunnableLambda(llm_mcp_call)
)
logger.info("RAG sequence setup complete")

@app.post("/ask", response_model=MCPResponse)
async def ask(request: MCPRequest):
    try:
        logger.info(f"Received /ask request with {len(request.messages)} messages")

        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            logger.error("No user message found in request")
            raise HTTPException(status_code=400, detail="No user message found")

        question = user_messages[-1].content
        logger.info(f"Processing question: {question[:100]!r}")

        # Run the RAG pipeline
        answer = rag_sequence.invoke({"question": question})
        logger.info(f"RAG pipeline completed, answer: {answer[:100]!r}")

        # Create response
        response_messages = request.messages + [MCPMessage(role="assistant", content=answer)]
        logger.info(f"Created response with {len(response_messages)} messages")

        return MCPResponse(messages=response_messages)

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting MCP RAG Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
