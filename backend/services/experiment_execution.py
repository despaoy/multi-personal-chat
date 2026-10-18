"""Keep experiment work and its terminal record owned until cleanup completes."""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

T = TypeVar("T")


async def run_owned_thread(
    func: Callable[..., T], /, *args: Any, on_cancel: Callable[[], None] | None = None,
) -> T:
    worker = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        await asyncio.wait({worker})
        return worker.result()
    except asyncio.CancelledError as cancellation:
        if on_cancel is not None:
            on_cancel()
        # Repeated cancellation must not abandon the same running worker.
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                continue
        if not worker.cancelled():
            error = worker.exception()
            if error is not None:
                raise cancellation from error
        raise


async def cancel_experiment(database, experiment_id: str, *, mock: bool) -> None:
    await run_owned_thread(
        database.execute_sql,
        "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r "
        "WHERE id=:id AND status='running'",
        {"id": experiment_id, "ts": datetime.now(timezone.utc).isoformat(),
         "r": json.dumps({"mock": mock, "error": "cancelled"})},
    )
