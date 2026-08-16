from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path


BATCH_DIR = Path(__file__).resolve().parent


def load_writer():
    sessions = json.loads((BATCH_DIR / "candidates.json").read_text(encoding="utf-8"))
    session = next(item for item in sessions if item["session_id"] == "asyncio_cancellation_atomic_file")
    answer = session["messages"][-1]["content"]
    match = re.search(r"```python\n(.*?)\n```", answer, flags=re.DOTALL)
    if match is None:
        raise AssertionError("embedded Python implementation is missing")
    namespace: dict[str, object] = {}
    exec(compile(match.group(1), "<embedded-atomic-writer>", "exec"), namespace)
    return namespace["write_stream_atomically"]


async def verify_async() -> None:
    writer = load_writer()

    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)
        target = directory / "result.bin"
        target.write_bytes(b"old")
        limit = asyncio.Semaphore(1)

        async def successful_chunks():
            yield b"new-"
            await asyncio.sleep(0)
            yield b"content"

        await writer(target, successful_chunks(), limit)
        assert target.read_bytes() == b"new-content"
        assert not list(directory.glob("*.part"))
        assert not limit.locked()

        target.write_bytes(b"stable")
        started = asyncio.Event()
        hold = asyncio.Event()

        async def cancelled_chunks():
            yield b"partial"
            started.set()
            await hold.wait()

        task = asyncio.create_task(writer(target, cancelled_chunks(), limit))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation must propagate")
        assert target.read_bytes() == b"stable"
        assert not list(directory.glob("*.part"))
        assert not limit.locked()

        async def failing_chunks():
            yield b"partial"
            raise RuntimeError("source failed")

        try:
            await writer(target, failing_chunks(), limit)
        except RuntimeError as exc:
            assert str(exc) == "source failed"
        else:
            raise AssertionError("source failure must propagate")
        assert target.read_bytes() == b"stable"
        assert not list(directory.glob("*.part"))
        assert not limit.locked()

        async def invalid_chunks():
            yield bytearray(b"not-bytes")

        try:
            await writer(target, invalid_chunks(), limit)
        except TypeError as exc:
            assert "must yield bytes" in str(exc)
        else:
            raise AssertionError("non-bytes chunks must fail")
        assert target.read_bytes() == b"stable"
        assert not list(directory.glob("*.part"))
        assert not limit.locked()


if __name__ == "__main__":
    asyncio.run(verify_async())
    print("technical verification passed")
