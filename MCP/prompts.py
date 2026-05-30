from __future__ import annotations

from mcp_server import Tool


def build_system_message(tools: list[Tool], include_conversation_guidance: bool = False) -> str:
    """Build the system prompt that teaches the LLM how to use available MCP tools."""
    tools_description = "\n".join([tool.format_for_llm() for tool in tools])

    prompt = (
        "You are a helpful assistant with access to these tools:\n\n"
        f"{tools_description}\n"
        "Choose the appropriate tool based on the user's question. "
        "If no tool is needed, reply directly.\n\n"
        "IMPORTANT: When you need to use a tool, you must ONLY respond with "
        "the exact JSON object format below, nothing else:\n"
        "{\n"
        '    "tool": "tool-name",\n'
        '    "arguments": {\n'
        '        "argument-name": "value"\n'
        "    }\n"
        "}\n\n"
        "After receiving a tool's response:\n"
        "1. Transform the raw data into a natural, conversational response\n"
        "2. Keep responses concise but informative\n"
        "3. Focus on the most relevant information\n"
        "4. Use appropriate context from the user's question\n"
        "5. Avoid simply repeating the raw data\n\n"
        "Please use only the tools that are explicitly defined above."
    )

    if include_conversation_guidance:
        prompt += (
            "\n\nUse the prior conversation messages to understand follow-up answers, "
            "missing details, pronouns, and corrections. If the latest user message "
            "is unrelated to the prior conversation, answer it as a new request."
        )

    return prompt
