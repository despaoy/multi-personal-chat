"""Request-owned provider usage; unavailable responses are never estimated."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class CompletionUsage:
    tokens: tuple[int, int, int] | None = None

    def read(self, payload):
        usage = payload["usage"]
        values = tuple(usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Provider token usage must contain nonnegative integers")
        if values[0] + values[1] != values[2]:
            raise ValueError("Provider token usage total is inconsistent")
        self.tokens = values


_current: ContextVar[list[CompletionUsage] | None] = ContextVar("completion_usage", default=None)


@contextmanager
def collect_usage():
    attempts = []
    token = _current.set(attempts)
    try:
        yield attempts
    finally:
        _current.reset(token)


def begin_completion() -> CompletionUsage | None:
    attempts = _current.get()
    if attempts is None:
        return None
    usage = CompletionUsage()
    attempts.append(usage)
    return usage


def invocation_usage() -> dict:
    attempts = _current.get() or ()
    measured = [attempt.tokens for attempt in attempts if attempt.tokens is not None]
    values = tuple(map(sum, zip(*measured))) if measured else (None, None, None)
    source = "provider" if measured and len(measured) == len(attempts) else "partial" if measured else "unavailable"
    return dict(zip(("promptTokens", "completionTokens", "totalTokens", "usageSource"), (*values, source)))


def clear_usage_context():
    _current.set(None)
