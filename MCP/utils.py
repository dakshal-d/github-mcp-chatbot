
import os
from langfuse import Langfuse, propagate_attributes
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY_DEV_SQL") or os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY_DEV_SQL") or os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=(
        os.getenv("LANGFUSE_HOST_DEV_SQL")
        or os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
    ),
)
# --- LANGFUSE HELPERS ---
 
def get_or_create_trace(trace_id: str, user_id: str, session_id: str):
    """Initializes or retrieves a Langfuse trace for observability."""
    return {
        "id": trace_id,
        "user_id": user_id,
        "session_id": session_id,
        "name": "mcp-chat-workflow",
        "environment": "dev_env",
    }
 
def safe_generation(trace, name, model, input_payload):
    """Safely creates a generation block in Langfuse trace."""
    try:
        with propagate_attributes(
            user_id=trace["user_id"],
            session_id=trace["session_id"],
            trace_name=trace["name"],
            metadata={"environment": trace["environment"]},
        ):
            return langfuse.start_observation(
                trace_context={"trace_id": trace["id"]},
                as_type="generation",
                name=name,
                model=model,
                input=input_payload,
                metadata={
                    "trace_name": trace["name"],
                    "user_id": trace["user_id"],
                    "session_id": trace["session_id"],
                    "environment": trace["environment"],
                },
            )
    except Exception as e:
        print(f"[Langfuse] generation failed: {e}")
        return None

def safe_generation_end(generation, output=None, metadata=None, level=None, status_message=None):
    """Safely completes a Langfuse generation."""
    if generation is None:
        return

    try:
        payload = {"output": output}
        if metadata:
            payload["metadata"] = metadata
        if level:
            payload["level"] = level
        if status_message:
            payload["status_message"] = status_message

        generation.update(**payload)
        generation.end()
    except Exception as e:
        print(f"[Langfuse] generation end failed: {e}")

def flush_langfuse():
    """Flushes queued Langfuse events."""
    try:
        langfuse.flush()
    except Exception as e:
        print(f"[Langfuse] flush failed: {e}")
