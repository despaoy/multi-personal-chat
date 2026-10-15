"""Explicit complete cloud connections for orchestration-only unit tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock


def cloud_connection(model, *, completion=None):
    provider = SimpleNamespace(
        base_url="https://fixture.invalid/v1", api_key="synthetic-key", model=model,
        async_complete=completion if completion is not None else AsyncMock(
            side_effect=AssertionError("This orchestration test must not invoke a cloud model")),
    )
    provider.snapshot = lambda: provider
    provider.get_status = lambda: {"modelName": model}
    return provider
