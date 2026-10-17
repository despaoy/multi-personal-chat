"""Local context configuration must be exact, shared and never silently expanded."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.context_budget import ReviewContextBudget
from inference.generation_request import GenerationRequest
from inference.provider_context import get_provider_context_budget


def selected(name="vllm"):
    return SimpleNamespace(_current_provider=SimpleNamespace(value=name))


def read(entry, env):
    if entry == "request":
        return GenerationRequest(message="三张票，每张两元，请计算总额。").context_window_tokens
    return get_provider_context_budget(selected(), env=env).window_tokens


@pytest.mark.parametrize("entry", ["request", "provider"])
@pytest.mark.parametrize("value", ["-1", "0", "1", "512", "1023", "", "synthetic-private-invalid", "1024.5"])
def test_invalid_local_window_is_named_error_in_both_entrypoints(monkeypatch, entry, value):
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", value)
    with pytest.raises(ValueError) as caught:
        read(entry, {"VLLM_MAX_MODEL_LEN": value})
    assert "VLLM_MAX_MODEL_LEN" in str(caught.value) and "1024" in str(caught.value)
    assert "synthetic-private" not in str(caught.value)


@pytest.mark.parametrize("entry", ["request", "provider"])
@pytest.mark.parametrize("window", [1024, 4096, 8192])
def test_valid_serving_window_is_returned_exactly(monkeypatch, entry, window):
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", str(window))
    assert read(entry, {"VLLM_MAX_MODEL_LEN": str(window)}) == window


@pytest.mark.parametrize("entry", ["request", "provider"])
def test_absent_local_window_keeps_8192_default(monkeypatch, entry):
    monkeypatch.delenv("VLLM_MAX_MODEL_LEN", raising=False)
    assert read(entry, {}) == 8192


def test_cloud_does_not_read_or_change_invalid_unused_local_configuration():
    env = {"VLLM_MAX_MODEL_LEN": "0", "OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536"}
    before = dict(env)
    result = get_provider_context_budget(selected("openai_compat"), env=env)
    assert result.window_tokens == result.review.window_tokens == 65536 and env == before


def test_bad_local_window_stops_context_build_before_semantic_runtime(monkeypatch):
    from character import semantic_review_adapter
    from inference import model_manager
    from services.character_context import build_character_context_service

    monkeypatch.setattr(model_manager, "get_model_manager", selected)
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "0")
    initialize = Mock(side_effect=AssertionError("Invalid budget reached reviewer initialization"))
    monkeypatch.setattr(semantic_review_adapter, "create_default_semantic_review_runtime", initialize)
    with pytest.raises(ValueError, match="VLLM_MAX_MODEL_LEN"):
        build_character_context_service(object())
    initialize.assert_not_called()


@pytest.mark.parametrize("window", [1023, 1024])
def test_internal_review_minimum_keeps_existing_boundary(window):
    if window < 1024:
        with pytest.raises(ValueError):
            ReviewContextBudget(window)
    else:
        assert ReviewContextBudget(window).window_tokens == window
