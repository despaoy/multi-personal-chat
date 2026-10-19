"""One resolved inference connection and counter for a generation request."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Any

from inference.token_counting import text_token_counter


@dataclass(frozen=True)
class InferenceRequestContext:
    provider_name: str
    provider: Any = field(repr=False)
    count_tokens: Callable[[str], int] = field(repr=False)

    @property
    def connection_id(self) -> str:
        if self.provider_name != "openai_compat":
            return self.provider_name
        connection = (self.provider.base_url, self.provider.model, self.provider.api_key)
        digest = hashlib.sha256(json.dumps(connection).encode()).hexdigest()
        return f"{self.provider_name}:{digest}"


_current: ContextVar[InferenceRequestContext | None] = ContextVar("inference_request", default=None)


def get_request_context() -> InferenceRequestContext | None:
    return _current.get()


async def resolve_request_context(manager) -> InferenceRequestContext:
    selected = manager.get_current_provider()
    name = selected.name
    provider = await asyncio.to_thread(selected.snapshot) if name == "openai_compat" else selected
    counter = (text_token_counter(name, provider.base_url, provider.model)
               if name == "openai_compat" else text_token_counter(name, "", ""))
    return InferenceRequestContext(name, provider, counter)


@contextmanager
def use_request_context(context: InferenceRequestContext):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def detached_request_context():
    """Long-lived owners keep logging context, without retaining this request."""
    context = copy_context()
    context.run(_current.set, None)
    return context
