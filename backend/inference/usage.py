"""Request-owned provider usage; unavailable responses are never estimated."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


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


@dataclass
class UsageLedger:
    attempts: list[CompletionUsage] = field(default_factory=list)
    record_started: bool = False


_current: ContextVar[UsageLedger | None] = ContextVar("completion_usage", default=None)


@contextmanager
def collect_usage():
    ledger = UsageLedger()
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


def begin_completion() -> CompletionUsage | None:
    ledger = _current.get()
    if ledger is None:
        return None
    usage = CompletionUsage()
    ledger.attempts.append(usage)
    return usage


def has_pending_usage() -> bool:
    ledger = _current.get()
    return ledger is not None and bool(ledger.attempts) and not ledger.record_started


def usage_record_started() -> bool:
    """A write attempt started; its database commit may still be unconfirmed."""
    ledger = _current.get()
    return ledger is not None and ledger.record_started


def invocation_usage(*, record_started=False) -> dict:
    ledger = _current.get()
    if ledger is not None and record_started:
        ledger.record_started = True
    attempts = ledger.attempts if ledger is not None else ()
    measured = [attempt.tokens for attempt in attempts if attempt.tokens is not None]
    values = tuple(map(sum, zip(*measured))) if measured else (None, None, None)
    source = "provider" if measured and len(measured) == len(attempts) else "partial" if measured else "unavailable"
    return dict(zip(("promptTokens", "completionTokens", "totalTokens", "usageSource"), (*values, source)))


def clear_usage_context():
    _current.set(None)
