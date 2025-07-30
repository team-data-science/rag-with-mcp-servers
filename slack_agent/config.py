from dotenv import load_dotenv
import os

# Load only the essential environment variables
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "host.docker.internal:11434")

# Hard‑coded constants (not loaded from .env)
RAG_SERVER_URL = "http://mcp_rag:8001/ask"
MODEL_NAME = "mistral"
MAX_TOKENS = 512
TEMPERATURE = 0.7
