from __future__ import annotations

from bot.async_inference import AsyncInferenceService


def test_vllm_config_uses_served_model_and_normalized_v1(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8001")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "qwen3-8b-instruct-awq")
    monkeypatch.setenv("MODEL_INFERENCE_TIMEOUT", "180")

    service = AsyncInferenceService(backend="vllm")

    assert service.vllm_url == "http://localhost:8001/v1"
    assert service.model_name == "qwen3-8b-instruct-awq"
    assert service.timeout == 180.0


def test_openai_base_does_not_duplicate_v1(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "judge-model")

    service = AsyncInferenceService(backend="openai")

    assert service.openai_url == "https://api.example.test/v1"
    assert service.model_name == "judge-model"
