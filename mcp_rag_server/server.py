import os
import logging
from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "q-and-a")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# Load the embedding model once at startup — this is the same model used by
# bulk_importer/writer.py so vectors are in the same space.
logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Embedding model loaded")

# Connect to Qdrant once at startup.
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# FastMCP creates the MCP server and (for SSE transport) the HTTP layer.
mcp = FastMCP("rag-server", host=HOST, port=PORT)


@mcp.tool()
def search_knowledge_base(question: str, top_k: int = 3) -> str:
    """Search the vector knowledge base for context relevant to a question.

    Embeds the question using the same model that was used during ingestion,
    queries Qdrant for the nearest neighbours, and returns the matching
    answers concatenated as a single string ready to be used as RAG context.

    Args:
        question: The user's question to search for.
        top_k:    Number of results to retrieve (default 3).

    Returns:
        Relevant context as a newline-separated string, or a message if
        nothing was found.
    """
    logger.info("search_knowledge_base called: question=%r, top_k=%d", question, top_k)

    # content_payload_key tells LangChain which Qdrant payload field to map
    # to Document.page_content.  bulk_importer stores {"question": ..., "answer": ...}
    # so we point at "answer" — the text we want returned as context.
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        content_payload_key="answer",
    )

    docs = store.similarity_search(question, k=top_k)
    logger.info("Found %d document(s)", len(docs))

    if not docs:
        return "No relevant context found in the knowledge base."

    context = "\n\n".join(doc.page_content for doc in docs)
    logger.info("Returning %d chars of context", len(context))
    return context


if __name__ == "__main__":
    # transport="sse" starts an HTTP server so Docker containers and remote
    # MCP clients can connect.  stdio transport is used for local CLI tools.
    logger.info("Starting RAG MCP server on %s:%d", HOST, PORT)
    mcp.run(transport="sse")
