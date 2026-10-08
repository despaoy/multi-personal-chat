"""Read-only readiness of the selected chat provider; never generate or load weights."""

from inference.openai_protocol import chat_completions_endpoint


def check_model_readiness():
    """Resolve configuration in the probe's bounded executor, then await network I/O."""
    from inference.model_manager import get_model_manager

    provider = get_model_manager().get_current_provider()
    if provider.name == "mock":
        return False  # The chat endpoint rejects mock inference in every environment.
    if provider.name == "transformers_peft":
        return provider.get_status()["loaded"] is True
    if provider.name == "openai_compat":
        provider._refresh_db_config()
        if not provider.api_key or not provider.model:
            raise ValueError("Configure the selected provider's API key and model")
    return _check_remote_model(provider)


def _has_model(payload, model: str) -> bool:
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]
        for item in models
    ):
        raise ValueError("Invalid model-list response")
    return any(item["id"] == model for item in models)


async def _check_remote_model(provider) -> bool:
    if provider.name == "vllm":
        # Use the same initialization gate and shared client as chat generation.
        from api.generate import get_vllm_client

        client = await get_vllm_client()
        if client is None:
            return False
        health = await client.health_check()
        return health["summary"]["healthy"] > 0

    async with provider._acquire_http_client(timeout=3.0) as client:
        if provider.name == "openai_compat":
            model = provider.model
            endpoint = chat_completions_endpoint(provider.base_url).removesuffix("chat/completions") + "models"
            response = await client.get(
                endpoint, headers={"Authorization": f"Bearer {provider.api_key}"}, timeout=3.0,
            )
            response.raise_for_status()
            return _has_model(response.json(), model)
        if provider.name == "ollama":
            response = await client.post(
                provider.base_url.rstrip("/") + "/api/show", json={"model": provider.model}, timeout=3.0,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("details"), dict):
                raise ValueError("Invalid Ollama model response")
            return True
        if provider.name == "llama_cpp":
            response = await client.get(provider.base_url.rstrip("/") + "/health", timeout=3.0)
            response.raise_for_status()
            return response.json()["status"] == "ok"
        raise ValueError("Readiness is not implemented for the selected provider")
