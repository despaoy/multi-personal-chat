from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException

from infra.bounded_executor import BlockingWorkRejected, BlockingWorkTimeout


@pytest.mark.asyncio
async def test_password_work_uses_dedicated_executor_thread():
    from api import auth

    thread_name = await auth._run_password_work(
        lambda: threading.current_thread().name
    )

    assert thread_name.startswith("auth-password")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        BlockingWorkRejected("full"),
        BlockingWorkTimeout("slow"),
    ],
)
async def test_password_backpressure_returns_retryable_503(monkeypatch, failure):
    from api import auth

    class FailingExecutor:
        async def run(self, func, *args):
            raise failure

    monkeypatch.setattr(auth, "_password_executor", FailingExecutor())

    with pytest.raises(HTTPException) as exc_info:
        await auth._run_password_work(lambda: True)

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "1"}
