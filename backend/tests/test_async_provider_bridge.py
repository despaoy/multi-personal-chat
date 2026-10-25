"""Native async providers share their synchronous bridge without fallback."""

from unittest.mock import AsyncMock

import pytest

from inference import model_manager


@pytest.mark.parametrize("name", ["OpenAICompatProvider", "OllamaProvider", "LlamaCppProvider", "VLLMProvider"])
@pytest.mark.parametrize("fails", [False, True])
def test_sync_bridge_forwards_exact_inputs_and_propagates_errors(name, fails):
    provider = getattr(model_manager, name).__new__(getattr(model_manager, name))
    history = [{"role": "user", "content": "Earlier full message"}]
    error = RuntimeError("controlled generation failure")
    provider.async_generate = AsyncMock(side_effect=error if fails else None, return_value=("complete answer", 0.5))
    if fails:
        with pytest.raises(RuntimeError) as caught:
            provider.generate("current input", history, 123)
        assert caught.value is error
    else:
        assert provider.generate("current input", history, 123) == ("complete answer", 0.5)
    provider.async_generate.assert_awaited_once_with("current input", history, 123)
