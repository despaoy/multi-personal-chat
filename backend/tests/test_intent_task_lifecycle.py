"""Concurrency and lifecycle tests for RAG intent background jobs."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import knowledge
from db.schemas import IntentSampleGenerateRequest, IntentTrainRequest
from knowledge import intent_trainer


def test_intent_generation_and_training_share_one_task_slot(monkeypatch):
    async def scenario():
        started = asyncio.Event()

        async def generate(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        async def train(*args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(intent_trainer, "generate_samples", generate)
        monkeypatch.setattr(intent_trainer, "train_intent_classifier", train)
        monkeypatch.setattr(
            intent_trainer,
            "get_generation_status",
            lambda: {"running": False},
        )
        monkeypatch.setattr(
            intent_trainer,
            "get_training_status",
            lambda: {"running": False},
        )
        monkeypatch.setattr(intent_trainer, "cancel_training", lambda: True)
        monkeypatch.setattr(knowledge, "_intent_task", None)
        monkeypatch.setattr(knowledge, "_intent_task_lock", None)
        monkeypatch.setattr(knowledge, "_intent_task_lock_loop", None)

        result = await knowledge.generate_intent_samples(
            IntentSampleGenerateRequest(),
            current_user={"role": "admin"},
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        assert result["success"] is True
        with pytest.raises(HTTPException) as exc_info:
            await knowledge.train_intent_classifier(
                IntentTrainRequest(),
                current_user={"role": "admin"},
            )
        assert exc_info.value.status_code == 409

        await knowledge.shutdown_intent_tasks(timeout=0)
        assert knowledge._intent_task is None

    asyncio.run(scenario())


def test_intent_task_lock_restarts_on_a_fresh_event_loop(monkeypatch):
    monkeypatch.setattr(knowledge, "_intent_task", None)
    monkeypatch.setattr(knowledge, "_intent_task_lock", None)
    monkeypatch.setattr(knowledge, "_intent_task_lock_loop", None)

    async def use_lock():
        lock = knowledge._get_intent_task_lock()
        async with lock:
            pass
        return lock

    first = asyncio.run(use_lock())
    second = asyncio.run(use_lock())

    assert second is not first


def test_intent_request_sizes_are_bounded():
    with pytest.raises(ValidationError):
        IntentSampleGenerateRequest(samples_per_kb=501)
    with pytest.raises(ValidationError):
        IntentSampleGenerateRequest(negative_count=1001)
    with pytest.raises(ValidationError):
        IntentSampleGenerateRequest(kb_ids=list(range(9)))
    with pytest.raises(ValidationError):
        IntentTrainRequest(kb_ids=list(range(9)))