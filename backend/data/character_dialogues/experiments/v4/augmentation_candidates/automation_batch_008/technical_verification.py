from __future__ import annotations

import asyncio
import json
from pathlib import Path


def load_candidate_namespace() -> dict[str, object]:
    sessions = json.loads(
        (Path(__file__).parent / "candidates.json").read_text(encoding="utf-8")
    )
    session = next(
        item for item in sessions if item["session_id"] == "async_retry_full_jitter"
    )
    content = session["messages"][5]["content"]
    fence = "`" * 3
    code = content.split(f"{fence}python\n", 1)[1].split(f"\n{fence}", 1)[0]
    namespace: dict[str, object] = {}
    exec(compile(code, "candidate_retry.py", "exec"), namespace)
    return namespace


async def verify() -> None:
    namespace = load_candidate_namespace()
    retry = namespace["retry"]
    retryable_error = namespace["RetryableError"]

    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise retryable_error("temporary")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    result = await retry(
        operation,
        sleep=fake_sleep,
        random_value=lambda: 0.5,
    )
    assert result == "ok"
    assert calls == 3
    assert delays == [0.1, 0.2]

    direct_calls = 0

    async def bad_input() -> None:
        nonlocal direct_calls
        direct_calls += 1
        raise ValueError("bad input")

    try:
        await retry(bad_input, sleep=fake_sleep)
    except ValueError:
        pass
    else:
        raise AssertionError("non-retryable ValueError was swallowed")
    assert direct_calls == 1

    exhausted_calls = 0

    async def always_temporary() -> None:
        nonlocal exhausted_calls
        exhausted_calls += 1
        raise retryable_error("still failing")

    try:
        await retry(
            always_temporary,
            attempts=2,
            sleep=fake_sleep,
            random_value=lambda: 0.0,
        )
    except retryable_error:
        pass
    else:
        raise AssertionError("retry exhaustion did not preserve the final error")
    assert exhausted_calls == 2


if __name__ == "__main__":
    asyncio.run(verify())
    print("ASYNC_RETRY_OK")
