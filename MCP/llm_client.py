from __future__ import annotations

import logging
import uuid
from typing import Any

# Gemini client kept for rollback.
# import httpx
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from utils import flush_langfuse, get_or_create_trace, safe_generation, safe_generation_end


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        # Gemini code kept for rollback.
        # self.model: str = "gemini-flash-latest"
        # self.client = httpx.Client(timeout=60.0)

    def _build_gemini_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Convert OpenAI-style chat messages into Gemini generateContent payload."""
        system_messages: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                system_messages.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096,
                "topP": 1,
            },
        }

        if system_messages:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}],
            }

        return payload

    def get_response(
        self,
        messages: list[dict[str, str]],
        name: str = "mcp-llm-call",
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.
            name: Langfuse observation name for this generation.
            metadata: Additional metadata to attach to the Langfuse observation.
            trace_id: Langfuse trace id. Pass the same id for multi-step LLM calls.
            user_id: Langfuse user id.
            session_id: Langfuse session id.

        Returns:
            The LLM's response as a string.

        Raises:
            APIError: If the request to the LLM fails.
        """
        trace = None
        generation = None
        trace_id = trace_id or uuid.uuid4().hex
        session_id = session_id or trace_id
        generation_metadata = {
            "provider": "groq",
            "app": "mcp-client",
            **(metadata or {}),
        }

        try:
            trace = get_or_create_trace(trace_id, user_id, session_id)
            generation = safe_generation(
                trace,
                name,
                self.model,
                {
                    "messages": messages,
                    "metadata": generation_metadata,
                },
            )
        except Exception as e:
            logging.warning("Langfuse trace setup failed: %s", e)

        try:
            completion_args: dict[str, Any] = {
                "messages": messages,
                "model": self.model,
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1,
                "stream": False,
                "stop": None,
            }
            response = self.client.chat.completions.create(**completion_args)
            content = response.choices[0].message.content or ""

            # Gemini code kept for rollback.
            # url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            # response = self.client.post(
            #     url,
            #     headers={
            #         "Content-Type": "application/json",
            #         "X-goog-api-key": self.api_key,
            #     },
            #     json=self._build_gemini_payload(messages),
            # )
            # response.raise_for_status()
            # data = response.json()
            # parts = data["candidates"][0]["content"].get("parts", [])
            # content = "".join(part.get("text", "") for part in parts)

            safe_generation_end(
                generation,
                output=content,
                metadata={
                    **generation_metadata,
                    "trace_id": trace_id,
                },
            )
            return content

        except (APIConnectionError, APIStatusError, APIError) as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, APIStatusError):
                logging.error(f"Status code: {e.status_code}")
                logging.error(f"Response details: {e.response.text}")

            # Gemini error handling kept for rollback.
            # except (httpx.HTTPError, KeyError, IndexError) as e:
            #     if isinstance(e, httpx.HTTPStatusError):
            #         logging.error(f"Status code: {e.response.status_code}")
            #         logging.error(f"Response details: {e.response.text}")

            safe_generation_end(
                generation,
                output=error_message,
                metadata={
                    **generation_metadata,
                    "trace_id": trace_id,
                },
                level="ERROR",
                status_message=error_message,
            )
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

    def flush_traces(self) -> None:
        """Flush queued Langfuse observations."""
        flush_langfuse()

    def close(self) -> None:
        """Close the underlying LLM HTTP client."""
        self.client.close()
