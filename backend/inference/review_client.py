"""Resolve optional reviewers against the actual selected low-level provider.

No chat generation, RAG, memory writes or persona assembly can be entered here.
"""


class OpenAICompatibleReviewClient:
    def __init__(self, provider):
        self.provider = provider

    async def generate(self, messages, *, lora_name, temperature, max_tokens, stream, enable_thinking):
        if lora_name is not None or stream or enable_thinking:
            raise ValueError("reviewers require a deterministic non-streaming base-model request")
        copied = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("invalid reviewer message")
            copied.append({"role": message["role"], "content": message["content"]})
        if not copied:
            raise ValueError("reviewer messages cannot be empty")
        content, _cost = await self.provider.async_complete(copied, temperature=temperature, max_tokens=max_tokens)
        return content


async def get_context_review_client():
    from inference.model_manager import get_model_manager

    manager = get_model_manager()
    selected = manager._current_provider.value
    if selected == "vllm":
        from inference.vllm_client import get_vllm_client

        return await get_vllm_client()
    if selected == "openai_compat":
        return OpenAICompatibleReviewClient(manager.get_current_provider())
    raise RuntimeError("unsupported selected context-review provider: " + selected)
