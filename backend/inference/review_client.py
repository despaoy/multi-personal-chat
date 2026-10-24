"""Resolve optional reviewers against the actual selected low-level provider.

No chat generation, RAG, memory writes or persona assembly can be entered here.
"""


from collections.abc import Mapping, Sequence


def copy_review_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("reviewer messages must be mappings")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("reviewer message has an invalid role or content")
        copied.append({"role": role, "content": content})
    if not copied:
        raise ValueError("reviewer messages cannot be empty")
    return copied


class OpenAICompatibleReviewClient:
    def __init__(self, provider):
        self.provider = provider

    async def generate(self, messages, *, lora_name, temperature, max_tokens, stream, enable_thinking):
        if lora_name is not None or stream or enable_thinking:
            raise ValueError("reviewers require a deterministic non-streaming base-model request")
        copied = copy_review_messages(messages)
        content, _cost = await self.provider.async_complete(copied, temperature=temperature, max_tokens=max_tokens)
        return content


async def get_context_review_client():
    from inference.model_manager import get_model_manager
    from inference.request_context import get_request_context

    context = get_request_context()
    provider = context.provider if context is not None else get_model_manager().get_current_provider()
    selected = context.provider_name if context is not None else provider.name
    if selected == "vllm":
        from inference.vllm_client import get_vllm_client

        return await get_vllm_client()
    if selected == "openai_compat":
        return OpenAICompatibleReviewClient(provider)
    raise RuntimeError("unsupported selected context-review provider: " + selected)


async def review_context(messages, *, max_tokens):
    """Run a deterministic base-model review on the selected request connection."""
    client = await get_context_review_client()
    return await client.generate(
        messages=messages, lora_name=None, temperature=0.0, max_tokens=max_tokens,
        stream=False, enable_thinking=False,
    )
