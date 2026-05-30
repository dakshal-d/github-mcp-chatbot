from __future__ import annotations

import asyncio
import logging

from chat_session import ChatSession
from config import Configuration
from llm_client import LLMClient
from mcp_server import Server


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def run() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
