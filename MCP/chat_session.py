from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from llm_client import LLMClient
from mcp_server import Server, Tool
from prompts import build_system_message


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def write_tool_response(
        self,
        user_message: str,
        tool_request: str,
        tool_result: str,
        trace_id: str,
        user_id: str,
        session_id: str,
    ) -> str:
        """Convert raw tool output into a user-facing assistant response."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a response writer for an MCP assistant. "
                    "Convert raw tool execution output into a clear, concise message for the user. "
                    "Do not mention internal tool-call JSON, MCP protocol objects, TextContent, or stack traces. "
                    "If the tool succeeded, summarize what was done and include useful identifiers or links. "
                    "If the tool failed, explain what failed, include the actionable error detail, and suggest the next step. "
                    "Do not claim success when the tool output indicates an error."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{user_message}\n\n"
                    f"Tool call selected by the assistant:\n{tool_request}\n\n"
                    f"Raw tool execution result:\n{tool_result}\n\n"
                    "Write the final response to show to the user."
                ),
            },
        ]

        response = self.llm_client.get_response(
            messages,
            name="mcp-tool-response-writer",
            metadata={"agent": "tool-response-writer"},
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
        )

        if response.strip():
            return response

        logging.warning("Tool response writer returned empty output. Falling back to raw tool result.")
        return tool_result

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        def _extract_json_object(text: str) -> dict[str, Any] | None:
            """Extract a tool-call JSON object from a model response."""
            import re

            cleaned = text.strip()
            fence_match = re.search(r"```(?:\s*json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
            if fence_match:
                cleaned = fence_match.group(1).strip()

            candidates = [cleaned]
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                candidates.append(cleaned[first_brace : last_brace + 1])

            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

            return None

        def _normalize_tool_call(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
            tool_name = (
                tool_call.get("tool")
                or tool_call.get("tool_name")
                or tool_call.get("name")
            )
            arguments = (
                tool_call.get("arguments")
                or tool_call.get("args")
                or tool_call.get("parameters")
                or {}
            )

            if not isinstance(tool_name, str):
                return None
            if not isinstance(arguments, dict):
                logging.warning("Tool arguments were not a JSON object: %s", arguments)
                return None

            return tool_name, arguments

        tool_call = _extract_json_object(llm_response)
        if tool_call is None:
            logging.info("LLM response did not contain a tool-call JSON object.")
            return llm_response

        normalized_tool_call = _normalize_tool_call(tool_call)
        if normalized_tool_call is None:
            logging.info("JSON response was not a valid tool call: %s", tool_call)
            return llm_response

        tool_name, arguments = normalized_tool_call
        logging.info("Executing tool: %s", tool_name)
        logging.info("With arguments: %s", arguments)

        for server in self.servers:
            tools = await server.list_tools()
            matching_tool = next((tool for tool in tools if tool.name.lower() == tool_name.lower()), None)
            if matching_tool:
                try:
                    result = await server.execute_tool(matching_tool.name, arguments)

                    if isinstance(result, dict) and "progress" in result:
                        progress = result["progress"]  # type: ignore
                        total = result["total"]  # type: ignore
                        percentage = (progress / total) * 100  # type: ignore
                        logging.info(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                    logging.info("Tool Result: %s", result)
                    return f"Tool execution result: {result}"

                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    logging.error(error_msg)
                    return error_msg

        available_tools = []
        for server in self.servers:
            available_tools.extend(tool.name for tool in await server.list_tools())

        logging.warning("No server found with tool: %s. Available tools: %s", tool_name, available_tools)
        return f"No server found with tool: {tool_name}"

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools: list[Tool] = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            logging.info("Loaded MCP tools: %s", [tool.name for tool in all_tools])
            system_message = build_system_message(all_tools)

            messages = [{"role": "system", "content": system_message}]
            session_id = uuid.uuid4().hex

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    trace_id = uuid.uuid4().hex

                    llm_response = self.llm_client.get_response(
                        messages,
                        name="mcp-tool-selection",
                        trace_id=trace_id,
                        user_id="cli-user",
                        session_id=session_id,
                    )
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        final_response = self.write_tool_response(
                            user_message=user_input,
                            tool_request=llm_response,
                            tool_result=result,
                            trace_id=trace_id,
                            user_id="cli-user",
                            session_id=session_id,
                        )
                        logging.info("\nFinal response: %s", final_response)
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            self.llm_client.flush_traces()
            self.llm_client.close()
            await self.cleanup_servers()
