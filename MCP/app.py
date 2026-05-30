from flask import Flask, request, jsonify
import asyncio
import atexit
import logging
import threading
import uuid

from chat_session import ChatSession
from config import Configuration
from flask_cors import CORS
from llm_client import LLMClient
from mcp_server import Server
from prompts import build_system_message

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

chat_session = None
system_message = None
conversation_histories = {}
MAX_HISTORY_MESSAGES = 12

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop_lock = threading.Lock()

def get_history(session_id: str):
    return conversation_histories.setdefault(session_id, [])

def save_turn(session_id: str, user_message: str, assistant_message: str):
    history = get_history(session_id)
    history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ])
    conversation_histories[session_id] = history[-MAX_HISTORY_MESSAGES:]

async def initialize_system():
    """
    Initializes MCP servers and builds the system prompt.
    Runs only once when Flask starts.
    """
    global chat_session, system_message

    config = Configuration()
    server_config = config.load_config("servers_config.json")

    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]

    # Initialize servers
    for server in servers:
        await server.initialize()

    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)

    # Build tool description (same as your CLI)
    all_tools = []
    for server in servers:
        tools = await server.list_tools()
        all_tools.extend(tools)

    logging.info("Loaded MCP tools: %s", [tool.name for tool in all_tools])
    system_message = build_system_message(all_tools, include_conversation_guidance=True)

    return system_message


async def process_chat(message: str, user_id: str = "web-user", session_id: str | None = None):
    if chat_session is None or system_message is None:
        raise RuntimeError("Chat system has not been initialized")

    trace_id = uuid.uuid4().hex
    session_id = session_id or trace_id
    history = get_history(session_id)

    messages = [
        {"role": "system", "content": system_message},
        *history,
        {"role": "user", "content": message}
    ]

    llm_response = chat_session.llm_client.get_response(
        messages,
        name="mcp-http-tool-selection",
        metadata={"entrypoint": "flask"},
        trace_id=trace_id,
        user_id=user_id,
        session_id=session_id,
    )
    logging.info("LLM tool-selection response: %s", llm_response)

    result = await chat_session.process_llm_response(llm_response)

    if result != llm_response:
        final_response = chat_session.write_tool_response(
            user_message=message,
            tool_request=llm_response,
            tool_result=result,
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
        )
        logging.info("LLM final response: %s", final_response)

        save_turn(session_id, message, final_response)
        return final_response

    save_turn(session_id, message, llm_response)
    return llm_response


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    message = data.get("message")
    user_id = data.get("user_id", "web-user")
    session_id = data.get("session_id") or uuid.uuid4().hex

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        with loop_lock:
            response = loop.run_until_complete(process_chat(message, user_id, session_id))
    except Exception:
        logging.exception("Error while processing chat request")
        return jsonify({"error": "failed to process chat request"}), 500

    return jsonify({
        "response": str(response),
        "session_id": session_id,
    })

def cleanup_system():
    if chat_session is None or loop.is_closed():
        return

    try:
        with loop_lock:
            loop.run_until_complete(chat_session.cleanup_servers())
        chat_session.llm_client.flush_traces()
        chat_session.llm_client.close()
    except Exception:
        logging.exception("Error while cleaning up MCP servers")
    finally:
        loop.close()

if __name__ == "__main__":

    with loop_lock:
        loop.run_until_complete(initialize_system())
    atexit.register(cleanup_system)

    app.run(host="0.0.0.0", port=5000, threaded=False)
