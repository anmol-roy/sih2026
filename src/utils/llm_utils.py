"""
Utility for safely extracting string content from LLM responses across providers.
Handles str, BaseMessage (AIMessage), list of content blocks, dicts, etc.
"""
from typing import Any


def extract_llm_text(response: Any) -> str:
    """
    Extract clean string text from an LLM response or content property.
    """
    if response is None:
        return ""

    # If response has a .content attribute (e.g. AIMessage)
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
                else:
                    parts.append(str(part))
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text")))
            else:
                parts.append(str(part))
        return "".join(parts).strip()

    return str(content).strip()
