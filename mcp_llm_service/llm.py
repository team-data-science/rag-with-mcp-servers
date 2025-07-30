import os
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Read from env, fallback to host.docker.internal for Docker-Desktop
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "host.docker.internal:11434")

def ask_llm(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    # Build URL and payload
    url = f"http://{OLLAMA_API_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    # Log what we’re about to send
    logger.info(f"Asking LLM at URL: {url}")
    logger.info(f"Payload: model={model}, max_tokens={max_tokens}, temperature={temperature}")
    logger.debug(f"Full payload JSON: {payload}")

    try:
        resp = requests.post(url, json=payload, timeout=30)
        logger.info(f"Received HTTP {resp.status_code} from Ollama")
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Error during LLM request: {e}", exc_info=True)
        raise

    try:
        data = resp.json()
        logger.debug(f"Ollama response JSON: {data}")
    except ValueError:
        logger.error(f"Failed to parse JSON from response: {resp.text}")
        raise

    # Extract the text
    if "response" in data:
        answer = data["response"]
    elif "completion" in data:
        answer = data["completion"]
    elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        answer = data["choices"][0].get("text", "")
    else:
        answer = ""
        logger.warning("No completion field found in Ollama response")

    logger.info(f"LLM answered (first 100 chars): {answer[:100]!r}")
    return answer



'''
# Ollama HTTP API URL (configured via environment, but defaulted here)
OLLAMA_API_URL = "host.docker.internal:11434"



def ask_llm(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    url = f"http://{OLLAMA_API_URL}/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("completion") or data.get("response") or ""
'''