import os
import logging
from mcp.server.fastmcp import FastMCP
from langchain_ollama import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "mistral")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# FastMCP creates the MCP server and (for SSE transport) the HTTP layer.
# The server name appears in MCP client tool listings.
mcp = FastMCP("llm-server", host=HOST, port=PORT)


@mcp.tool()
def generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a text response from the LLM given a plain-text prompt.

    Args:
        prompt:      The full prompt to send to the model.
        model:       Ollama model name (e.g. 'mistral', 'llama3').
        max_tokens:  Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        The model's text response as a string.
    """
    logger.info("generate_text called: model=%s, max_tokens=%d", model, max_tokens)

    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        num_predict=max_tokens,
        temperature=temperature,
    )

    result = llm.invoke(prompt)
    logger.info("Response (first 100 chars): %r", result.content[:100])
    return result.content


if __name__ == "__main__":
    # transport="sse" starts an HTTP server so Docker containers and remote
    # MCP clients can connect.  stdio transport is used for local CLI tools.
    logger.info("Starting LLM MCP server on %s:%d", HOST, PORT)
    mcp.run(transport="sse")
