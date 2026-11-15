"""Reconnectable SSE views of authorized, persisted generation state."""

import asyncio
import json
import time

from repositories.character_library import CharacterNotFoundError


async def generation_events(repository, owner, session_id, request_id, *, interval=0.25, missing_timeout=30):
    previous = None
    started = time.monotonic()
    heartbeat = started
    while True:
        try:
            status = await asyncio.to_thread(repository.get, owner, session_id, request_id)
        except CharacterNotFoundError:
            yield 'event: revoked\ndata: {}\n\n'
            return
        if status is None:
            if time.monotonic() - started >= missing_timeout:
                yield 'event: unavailable\ndata: {}\n\n'
                return
        else:
            # Full snapshots make reconnect and guard regeneration deterministic;
            # never append a replayed token delta twice on the client.
            view = {key: status[key] for key in (
                "request_id", "status", "attempt", "draft_revision", "draft_text", "turn_id", "error_code"
            ) if key in status}
            encoded = json.dumps(view, ensure_ascii=False)
            if encoded != previous:
                yield f"event: generation\ndata: {encoded}\n\n"
                previous = encoded
                heartbeat = time.monotonic()
            if status["status"] != "running":
                return
        if time.monotonic() - heartbeat >= 10:
            yield ": heartbeat\n\n"
            heartbeat = time.monotonic()
        await asyncio.sleep(interval)
