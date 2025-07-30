import os
import time
import logging
import requests
import json
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, RAG_SERVER_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

web_client = WebClient(token=SLACK_BOT_TOKEN)
socket_client = SocketModeClient(app_token=SLACK_APP_TOKEN, web_client=web_client)

def fetch_history(channel: str, thread_ts: str, limit: int = 6):
    logger.info(f"Fetching history for channel {channel}, thread {thread_ts}")
    try:
        resp = web_client.conversations_history(
            channel=channel,
            latest=thread_ts,
            limit=limit,
            inclusive=True
        )
        msgs = resp.get("messages", [])
        msgs.reverse()
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in msgs:
            if msg.get("subtype"):
                continue
            role = "assistant" if msg.get("bot_id") else "user"
            messages.append({"role": role, "content": msg.get("text", "")})
        logger.info(f"Fetched {len(messages)} messages from history")
        return messages
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return [{"role": "system", "content": "You are a helpful assistant."}]

def call_mcp_rag_server(messages: list) -> str:
    """Call the MCP RAG server using the new MCP message structure"""
    logger.info(f"Calling MCP RAG server with {len(messages)} messages")
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    
    try:
        logger.info(f"Sending request to {RAG_SERVER_URL}")
        resp = requests.post(RAG_SERVER_URL, json=payload, timeout=30)
        logger.info(f"RAG server response status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"RAG server response: {data}")
            
            # Find the assistant's response in the returned messages
            assistant_messages = [msg for msg in data.get("messages", []) if msg["role"] == "assistant"]
            if assistant_messages:
                answer = assistant_messages[-1]["content"]
                logger.info(f"Extracted assistant response: {answer[:100]}...")
                return answer
            else:
                logger.error(f"No assistant message found in response: {data}")
                return "Sorry, I couldn't process your request."
        else:
            logger.error(f"RAG server returned error status {resp.status_code}: {resp.text}")
            return "Sorry, I couldn't fetch an answer right now."
            
    except requests.exceptions.Timeout:
        logger.error("Timeout calling MCP RAG server")
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error calling MCP RAG server")
        return "Sorry, I couldn't connect to the server. Please try again."
    except Exception as e:
        logger.error(f"Error calling MCP RAG server: {e}")
        return "Sorry, I couldn't fetch an answer right now."

def handle_message(payload):
    try:
        event = payload.get("event", {})
        channel = event.get("channel")
        user = event.get("user")
        ts = event.get("ts")
        event_id = payload.get("event_id", "unknown")
        retry_attempt = payload.get("retry_attempt", 0)
        
        logger.info(f"Processing event {event_id} (retry attempt: {retry_attempt}) from user {user} in channel {channel}")
        
        # Ignore messages with no user, subtype, or from any bot (including this app)
        if not user or event.get("subtype") or event.get("bot_id"):
            logger.info(f"Skipping event {event_id} - from bot or system message")
            return
        
        thread_ts = event.get("thread_ts", ts)
        user_message = event.get("text", "")
        
        logger.info(f"User message: {user_message}")
        
        # Get conversation history and add the new user message
        messages = fetch_history(channel, thread_ts)
        messages.append({"role": "user", "content": user_message})
        
        # Call the MCP RAG server
        answer = call_mcp_rag_server(messages)
        
        # Post the response
        logger.info(f"Posting response to channel {channel}, thread {thread_ts}")
        web_client.chat_postMessage(
            channel=channel,
            text=answer,
            thread_ts=thread_ts
        )
        logger.info(f"Successfully posted response for event {event_id}")
        
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        # Try to send an error message to the user
        try:
            channel = event.get("channel")
            thread_ts = event.get("thread_ts", event.get("ts"))
            web_client.chat_postMessage(
                channel=channel,
                text="Sorry, something went wrong processing your message. Please try again.",
                thread_ts=thread_ts
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")

def events_api_handler(client: SocketModeClient, req: SocketModeRequest):
    try:
        logger.info(f"Received Socket Mode request: {req.type}, envelope_id: {req.envelope_id}")
        
        # Always acknowledge the event first
        resp = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(resp)
        logger.info(f"Acknowledged Socket Mode request: {req.envelope_id}")
        
        # Then process the event
        event = req.payload.get("event", {})
        if event.get("type") == "message":
            handle_message(req.payload)
        else:
            logger.info(f"Ignoring non-message event: {event.get('type')}")
            
    except Exception as e:
        logger.error(f"Error in events_api_handler: {e}", exc_info=True)
        # Try to acknowledge the event even if processing failed
        try:
            resp = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(resp)
            logger.info(f"Acknowledged Socket Mode request after error: {req.envelope_id}")
        except Exception as ack_error:
            logger.error(f"Failed to acknowledge Socket Mode request: {ack_error}")

if __name__ == "__main__":
    logger.info("Starting Slack bot...")
    # Register the message handler
    socket_client.socket_mode_request_listeners.append(events_api_handler)
    socket_client.connect()
    logger.info("Slack bot connected and listening...")
    while True:
        time.sleep(1)
