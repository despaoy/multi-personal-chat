"""Persist delivered-turn completion independently of physical send status.

One serial worker per app database reserves existing completion capacity before
claiming a durable receipt. Its lease is renewed while the same task is alive;
only abandoned work is eligible for another process. Recovery is at least once,
with existing source identity and erasure fences authoritative on every retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime

from character.context_builder import build_user_scope
from character.models import RelationshipState, UserScope
from services.turn_completion import CompletionUnavailable, get_turn_completion_runtime

logger = logging.getLogger(__name__)
LEASE_SECONDS = 300
POLL_SECONDS = 1


@dataclass(frozen=True)
class CompletionFeedback:
    used_memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompletionSnapshot:
    """Only the generation fields consumed by complete_turn; no new inference."""

    character_id: str
    user_scope: UserScope
    history: tuple[dict[str, str], ...]
    relationship: RelationshipState
    received_at: datetime
    compiled: CompletionFeedback
    memory_operation_receipt: dict | None


def freeze_completion(prepared) -> dict:
    return dict(
        version=1,
        character_id=prepared.character_id,
        user_scope=asdict(prepared.user_scope),
        history=list(prepared.history),
        relationship=asdict(prepared.relationship),
        received_at=prepared.received_at.isoformat(),
        used_memory_ids=list(prepared.compiled.used_memory_ids),
        memory_operation_receipt=prepared.memory_operation_receipt,
    )


def thaw_completion(context: dict) -> tuple[CompletionSnapshot, object]:
    from db.schemas import MessageRequest
    from services.character_context import TurnInput

    frozen = context["completion_snapshot"]
    if frozen["version"] != 1:
        raise ValueError("Unknown completion snapshot")
    request = MessageRequest(**context["request"])
    scope = build_user_scope(
        platform=request.platform,
        adapter=request.adapter,
        sender_id=request.senderId or request.userId,
        conversation_id=request.conversationId or request.sessionId,
        conversation_type=request.conversationType or request.sessionType,
    )
    if asdict(scope) != frozen["user_scope"] or frozen["character_id"] != context["character_id"]:
        raise ValueError("Completion scope mismatch")
    received_at = datetime.fromisoformat(frozen["received_at"])
    if received_at.tzinfo is None or received_at.utcoffset() is None:
        raise ValueError("Missing trusted completion clock")
    snapshot = CompletionSnapshot(
        character_id=frozen["character_id"],
        user_scope=scope,
        history=tuple(dict(row) for row in frozen["history"]),
        relationship=RelationshipState(**frozen["relationship"]),
        received_at=received_at,
        compiled=CompletionFeedback(used_memory_ids=tuple(frozen["used_memory_ids"])),
        memory_operation_receipt=frozen.get("memory_operation_receipt"),
    )
    turn = TurnInput(
        message=request.message,
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_id=scope.conversation_id,
        conversation_type=scope.conversation_type,
        received_at=received_at,
    )
    return snapshot, turn


def encode(stored: dict, state: str) -> str:
    # A server-owned prefix makes portable SQL selection independent of text
    # inside requests. It is never accepted from a delivery callback.
    body = dict(stored)
    body.pop("memory_completion_state", None)
    return json.dumps({"memory_completion_state": state, **body}, ensure_ascii=False)


def delivery_envelope(stored: dict) -> tuple[str, str]:
    context = stored.get("context", {})
    state = "not_applicable"
    if context.get("message_saved") and context.get("character_id"):
        if context.get("completion_snapshot"):
            try:
                thaw_completion(context)
                state = "pending"
            except (KeyError, TypeError, ValueError):
                state = "blocked"
        else:
            # Older receipts have no trusted clock. Never fabricate a fresh
            # one or sweep old delivered messages into memory during startup.
            state = "legacy_unknown"
    context["memory_completion"] = dict(version=1, state=state)
    return encode(stored, state), state


class DeliveryMemoryWorker:
    """Own one actual durable job at a time, including its semantic receipt."""

    def __init__(self, database, service=None):
        self.database = database
        self.service = service
        self.closed = False
        self.wakeup = asyncio.Event()
        self.active_key: str | None = None
        self.task = asyncio.create_task(self._loop(), name="delivery-memory-worker")

    async def operation(self, operation, **params):
        # SyncPg owns its cross-thread futures. Do not use a response timeout
        # to abandon a DB mutation and start the same operation again.
        return await asyncio.to_thread(self.database.integration_receipt, operation, **params)

    async def _loop(self):
        while not self.closed:
            try:
                runtime = get_turn_completion_runtime()
                slot = runtime.reserve()
            except CompletionUnavailable:
                await self._pause()
                continue
            try:
                rows = await self.operation("memory_pending", now=time.time(), limit=1)
                if rows:
                    await self._process(rows[0], runtime, slot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("交付记忆任务未完成: %s", type(exc).__name__)
            finally:
                slot.release()
                self.active_key = None
            await self._pause()

    async def _pause(self):
        with suppress(TimeoutError):
            await asyncio.wait_for(self.wakeup.wait(), POLL_SECONDS)
        self.wakeup.clear()

    async def _process(self, record, runtime, slot):
        stored = json.loads(record["response"])
        context = stored["context"]
        marker = context["memory_completion"]
        marker.update(state="running", attempt_id=uuid.uuid4().hex)
        response = encode(stored, "running")
        key, owner = record["receipt_key"], record["owner"]
        if not await self.operation(
            "memory_claim",
            key=key,
            owner=owner,
            expected=record["response"],
            response=response,
            now=time.time(),
            expires_at=time.time() + LEASE_SECONDS,
        ):
            return
        self.active_key = key
        lost = asyncio.Event()

        async def persist(state, **details):
            nonlocal response
            marker.update(state=state, **details)
            updated = encode(stored, state)
            accepted = await self.operation(
                "memory_update",
                key=key,
                owner=owner,
                expected=response,
                response=updated,
                expires_at=time.time() + (5 if state == "pending" else LEASE_SECONDS),
            )
            if not accepted:
                lost.set()
                raise RuntimeError("Completion lease superseded")
            response = updated

        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                if not await self.operation(
                    "memory_renew",
                    key=key,
                    owner=owner,
                    expected=response,
                    expires_at=time.time() + LEASE_SECONDS,
                ):
                    lost.set()
                    return

        renewal = asyncio.create_task(heartbeat(), name="delivery-memory-lease")
        try:
            try:
                prepared, turn = thaw_completion(context)
            except (KeyError, TypeError, ValueError):
                await persist("blocked", reason="invalid_snapshot")
                return
            if self.service is None:
                from services.character_context import build_character_context_service

                self.service = build_character_context_service(self.database)
            receipt = asyncio.get_running_loop().create_future()

            async def complete():
                return await self.service.complete_turn(
                    prepared,
                    turn,
                    stored["reply"]["replyText"],
                    source_message_id=context["request"]["sourceMessageId"],
                    memory_receipt=receipt,
                    memory_retry_only=bool(marker.get("turn_finished")),
                )

            # Wait on the same runtime-owned task. No observation deadline
            # creates another attempt while its real work is still alive.
            outcome = await runtime.run(complete, timeout=None, reservation=slot)
            await persist("running", turn_finished=True, completion_outcome=asdict(outcome))
            result = (
                await asyncio.shield(receipt)
                if outcome.memory_enrichment_scheduled
                else dict(
                    status=outcome.memory_enrichment_status,
                    source_capture=outcome.source_capture,
                )
            )
            if lost.is_set():
                raise RuntimeError("Completion lease superseded")
            capture = result.get("source_capture") or outcome.source_capture
            status = result.get("status", "unknown")
            if capture in {"stale", "revoked", "conflict"}:
                state = "blocked"
            elif capture == "failed" or status in {
                "failed",
                "partial",
                "cancelled",
                "capacity",
                "queue_full",
                "closed",
                "unknown",
            }:
                state = "pending"
            else:
                state = "completed"
            await persist(state, semantic_receipt=result)
        except asyncio.CancelledError:
            # Persisted running lease survives shutdown; never invent rollback
            # for a live executor thread or a scheduler-owned semantic job.
            raise
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)

    async def wait_idle(self, timeout=90):
        async with asyncio.timeout(timeout):
            while self.active_key or await self.operation("memory_pending", now=float("inf"), limit=1):
                self.wakeup.set()
                await asyncio.sleep(0.1)

    async def shutdown(self, timeout=35):
        self.closed = True
        self.wakeup.set()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


_workers: dict[int, DeliveryMemoryWorker] = {}


def ensure_delivery_memory_worker(database, service=None):
    worker = _workers.get(id(database))
    if worker is None:
        worker = DeliveryMemoryWorker(database, service)
        _workers[id(database)] = worker
    worker.wakeup.set()
    return worker


async def shutdown_delivery_memory():
    workers = tuple(_workers.values())
    for worker in workers:
        worker.closed = True
        worker.wakeup.set()
    await asyncio.gather(*(worker.shutdown() for worker in workers))
    _workers.clear()
