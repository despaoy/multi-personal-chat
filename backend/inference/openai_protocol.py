"""Small protocol boundaries shared by native OpenAI-compatible callers."""

from urllib.parse import urlsplit


def nonthinking_parameters(base_url: str, *, local_template: bool = True) -> dict:
    """Use the official DeepSeek switch only for its exact service host."""
    if urlsplit(base_url).hostname == "api.deepseek.com":
        return {"thinking": {"type": "disabled"}}
    return {"chat_template_kwargs": {"enable_thinking": False}} if local_template else {}


def completed_chat_content(payload: object) -> str:
    """Partial, filtered or reasoning-only responses are not usable answers."""
    if not isinstance(payload, dict):
        raise ValueError("invalid chat completion")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("invalid chat completion choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("chat completion did not finish normally")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat completion has no usable content")
    return content.strip()
