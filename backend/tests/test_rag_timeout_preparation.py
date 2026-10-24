"""Cold-start budget errors stop before constructing retrieval work."""

from services import chat_answer

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from db.schemas import MessageRequest
from inference.context_budget import ReviewContextBudget
from knowledge.multiscale_rag import runtime

QUERY = "核对那个未说明出处的结论；缺少出处则说明无法确定。"


@pytest.mark.parametrize("warm,steady,cold,expected", [(True, 8, 60, 8), (False, 8, 60, 60), (False, 80, 60, 80)])
def test_timeout_preserves_cold_start_and_steady_limits(monkeypatch, warm, steady, cold, expected):
    monkeypatch.setattr(chat_answer, '_RAG_TIMEOUT', steady)
    monkeypatch.setattr(chat_answer, '_RAG_COLD_START_TIMEOUT', cold)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(is_warm=lambda: warm))
    assert chat_answer._rag_retrieval_timeout() == expected


@pytest.mark.parametrize("stage", ["initialize", "state"])
def test_timeout_preparation_propagates_its_original_failure(monkeypatch, stage):
    failure = OSError("fiction-private-runtime-failure")
    faulty = Mock(side_effect=failure)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", faulty if stage == "initialize" else
                        lambda: SimpleNamespace(is_warm=faulty))
    with pytest.raises(OSError) as error:
        chat_answer._rag_retrieval_timeout()
    assert error.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["initialize", "state"])
async def test_failed_timeout_preparation_stops_before_retrieval_or_answer(monkeypatch, semantic_plan, stage):
    planned = semantic_plan(QUERY, {"unresolved_source": [0], "control": [1]})
    failure = OSError("fiction-private-runtime-failure")
    faulty = Mock(side_effect=failure)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", faulty if stage == "initialize" else
                        lambda: SimpleNamespace(is_warm=faulty))
    retrieve = Mock(side_effect=AssertionError("No retrieval coroutine may be created after preparation failure"))
    answer = AsyncMock(side_effect=AssertionError("No model call may follow preparation failure"))
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_provider_context_budget', lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    with pytest.raises(chat_answer.ChatRetrievalError) as error:
        await chat_answer.generate_chat_answer(MessageRequest(message=QUERY), None,
            runtime_config={"useKnowledgeBase": True, "maxTokens": 256}, model_generate=answer)
    assert error.value.http_status == 503 and error.value.__cause__ is failure
    assert planned and "fiction-private" not in str(error.value.detail)
    retrieve.assert_not_called()
    answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_preparation_preserves_legitimate_empty_retrieval(monkeypatch, semantic_plan):
    planned = semantic_plan(QUERY, {"unresolved_source": [0], "control": [1]})
    events = []

    def timeout():
        events.append("timeout")
        return 8

    def retrieve(*args, **kwargs):
        events.append("create_retrieval")

        async def run():
            events.append("retrieve")
            return {"results": [], "citations": [], "abstained": True}

        return run()

    async def answer(**kwargs):
        events.append("answer")
        return "单元控制：没有提供出处，暂时无法核对该结论。"

    monkeypatch.setattr(chat_answer, '_rag_retrieval_timeout', timeout)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "只根据已提供的完整来源回答，缺失则明确说明。")
    monkeypatch.setattr(chat_answer, 'get_provider_context_budget', lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    reply, used, metadata = await chat_answer.generate_chat_answer(MessageRequest(message=QUERY), None,
        runtime_config={"useKnowledgeBase": True, "maxTokens": 256}, model_generate=answer)
    assert planned and reply and used and metadata["abstained"] and metadata["modelInvoked"]
    assert events == ["timeout", "create_retrieval", "retrieve", "answer"]


@pytest.mark.asyncio
async def test_explicit_retrieval_off_does_not_prepare_a_runtime(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Explicitly disabled retrieval must not initialize an index"))
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", forbidden)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "根据完整题设计算。")
    monkeypatch.setattr(chat_answer, 'get_provider_context_budget', lambda: SimpleNamespace(window_tokens=65536, evidence_max_chars=0))
    answer = AsyncMock(return_value="单元控制：3张票，每张2.25元，总计6.75元。")
    _, used, _ = await chat_answer.generate_chat_answer(
        MessageRequest(message="共3张票，每张2.25元，没有其他费用；请计算总额。"), None,
        runtime_config={"useKnowledgeBase": False, "maxTokens": 256}, model_generate=answer)
    assert not used
    forbidden.assert_not_called()
    answer.assert_awaited_once()
