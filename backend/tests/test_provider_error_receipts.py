""" safe provider rejection details preserve chat failure receipts.

Chat accounting/archive and erasure below use actual temporary SQLite. The
interactive operation executor and provider responses are explicit controls;
no model semantics, real credentials or paid HTTP calls are involved. The RAG
case uses the actual planner, reviewer adapter and API exception boundaries.

"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from provider_fixtures import cloud_connection
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import ContextAfterMemoryOperationError
from test_circuit_strict_generation import archives, records, turn
from test_circuit_strict_generation import circuit_rig as circuit_rig
from test_interactive_memory_operation import setup_service
from tests.test_public_fact_coverage import QUERY as PUBLIC_QUERY

from api import generate
from character.memory_operation import operation_receipt_context
from character.models import MemoryItem, UserScope
from db.schemas import MessageRequest
from inference import review_client
from inference.context_budget import ReviewContextBudget
from inference.openai_protocol import ProviderResponseError
from knowledge.retrieval_query_plan import QueryPlanReviewError

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("no_live_context_review")]


async def test_direct_chat_401_is_502_with_failed_accounting_and_no_saved_answer(circuit_rig):
    rig = circuit_rig
    rejection = ProviderResponseError(401)
    rig.fault = rejection
    with pytest.raises(HTTPException) as caught:
        await turn(rig, "provider-auth-rejected")
    error = caught.value
    assert error.status_code == 502 and error.__cause__ is rejection
    assert error.detail == rejection.public_detail()
    assert error.detail["kind"] == "provider_auth_error" and error.detail["upstreamStatus"] == 401
    assert len(rig.calls) == 1 and rig.calls[0]["provider_fee"] == 0
    invocation = records(rig)
    assert len(invocation) == 1
    assert invocation[0]["errorType"] == "ProviderResponseError"
    assert invocation[0]["completionTokens"] is None and archives(rig) == []
    assert rig.active == 0 and rig.semaphore._value == 4


async def test_401_and_record_failure_keep_primary_503_receipt_fields(circuit_rig, monkeypatch):
    rig = circuit_rig
    rejection = ProviderResponseError(401)
    rig.fault = rejection
    attempted_records = []

    def failed_record(payload):
        attempted_records.append(dict(payload))
        raise OSError("synthetic-private-database-detail")

    monkeypatch.setattr(rig.database, "add_model_invocation", failed_record)
    with pytest.raises(HTTPException) as caught:
        await turn(rig, "provider-auth-and-record-failed")
    error = caught.value
    assert error.status_code == 503 and error.__cause__ is rejection
    primary = {
        "stage": "model_invocation_record",
        "generationErrorType": "ProviderResponseError",
        "recordErrorType": "OSError",
        "modelInvoked": None,
        "invocationRecorded": None,
        "messageSaved": False,
    }
    for field, value in primary.items():
        assert error.detail[field] == value
    assert set(error.detail) == set(primary) | {"message", "providerError"}
    assert error.detail["message"] == "生成失败，且调用记录保存失败或未确认，请检查数据库和模型服务后重试"
    assert error.detail["providerError"] == rejection.public_detail()
    assert "synthetic-private" not in json.dumps(error.detail, ensure_ascii=False)
    assert len(rig.calls) == len(attempted_records) == 1
    assert attempted_records[0]["errorType"] == "ProviderResponseError"
    assert attempted_records[0]["completionTokens"] is None
    assert records(rig) == archives(rig) == []
    assert rig.active == 0 and rig.semaphore._value == 4


async def test_executed_memory_erasure_keeps_receipt_when_following_review_gets_401(circuit_rig, monkeypatch):
    rig = circuit_rig
    repository = DatabaseCharacterMemoryRepository(rig.database)
    scope = UserScope("web", "test", "u", "u", "private")
    body = "完整虚构资料：我在2026年10月住星禾路20号；这是测试住址，不代表现实授权，10月后不再适用。"
    stamp = datetime.now(timezone.utc)
    source_id = "fiction-erasure-source"
    assert await repository.capture_source(
        "role", scope, source_message_id=source_id, body=body, observed_at=stamp,
    ) == "recorded"
    stored = await repository.append_claim(
        "role", scope, MemoryItem("", "user_fact", body, .9),
        memory_key="fiction_october_address", confidence=.99,
        evidence=(body,), source_message_id=source_id, observed_at=stamp.isoformat(),
    )
    identity = int(stored["id"])
    receipt = {"status": "erased", "persisted": 1, "source_erased": True,
               "error": "synthetic-private-operation-detail"}
    service, worker = setup_service(monkeypatch, receipt)
    service._memory_repo = repository
    rejection = ProviderResponseError(401)

    async def execute_erasure(**kwargs):
        # The test controls scope interpretation, but the authorized storage
        # mutation is real and must finish before snapshot/review preparation.
        assert kwargs["repository"] is repository
        assert kwargs["character_id"] == "role" and kwargs["user_scope"] == scope
        assert kwargs["source_message_id"] == "operation-typed401"
        assert await repository.erase_memory("role", scope, memory_id=identity) == 1
        return receipt

    async def failed_following_preparation(*args):
        assert await repository.get_memory_record(identity, "role", scope) is None
        raise rejection

    worker.schedule_and_wait.side_effect = execute_erasure
    service.prepare_turn.side_effect = failed_following_preparation
    request = MessageRequest(
        message="请从记忆里删除我的住址。", platform="web", adapter="test",
        senderId="u", userId="u", sessionId="c", conversationId="c",
        sourceMessageId="operation-typed401",
    )
    with pytest.raises(HTTPException) as caught:
        await generate._prepare_character_turn(
            request, "role", character_service=service, execute_memory_operations=True,
        )
    error = caught.value
    assert error.status_code == 503
    assert error.detail["code"] == "context_failed_after_memory_operation"
    assert error.detail["memoryOperation"] == operation_receipt_context(receipt)
    assert "已确认完成的操作数：1" in error.detail["memoryOperation"]
    assert error.detail["message"] == (
        "角色上下文准备失败，回复未生成。请先核对本轮记忆操作回执，再检查审核模型和上下文配置。"
    )
    assert error.detail["providerError"] == rejection.public_detail()
    assert isinstance(error.__cause__, ContextAfterMemoryOperationError)
    assert error.__cause__.__cause__ is rejection
    assert "synthetic-private" not in json.dumps(error.detail, ensure_ascii=False)
    assert await repository.get_memory_record(identity, "role", scope) is None
    worker.schedule_and_wait.assert_awaited_once()
    service.prepare_turn.assert_awaited_once()
    assert not rig.calls and records(rig) == archives(rig) == []


async def test_actual_query_planner_cause_is_attached_to_api_review_failure(monkeypatch):
    rejection = ProviderResponseError(401)
    exchanges, answers = [], []

    async def denied_completion(messages, **params):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == PUBLIC_QUERY and payload["segments"]
        assert params == {"temperature": 0.0, "max_tokens": 768}
        exchanges.append({"messages": messages, "params": params})
        raise rejection

    # Isolate only the external model boundary. The actual planner's default
    # _review, OpenAICompatibleReviewClient and API wrapper remain in control.
    client = review_client.OpenAICompatibleReviewClient(
        cloud_connection("fiction-planning-auth", completion=denied_completion),
    )
    monkeypatch.setattr(review_client, "get_context_review_client", AsyncMock(return_value=client))
    monkeypatch.setenv("RAG_TASK_PLANNER_TIMEOUT_SECONDS", "1")
    budget = SimpleNamespace(review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: budget)

    async def answer(**kwargs):
        answers.append(kwargs)
        pytest.fail("A failed planner exchange must not continue to answer generation")

    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=PUBLIC_QUERY), None,
            runtime_config={"maxTokens": 2048, "useKnowledgeBase": True}, model_generate=answer,
        )
    error = caught.value
    assert error.status_code == 503 and len(exchanges) == 1 and not answers
    assert isinstance(error.__cause__, QueryPlanReviewError)
    assert error.__cause__.status == "provider_error" and error.__cause__.__cause__ is rejection
    assert error.detail["message"] == "检索或证据审核失败，请检查检索服务及审核模型状态后重试"
    assert error.detail["providerError"] == rejection.public_detail()
