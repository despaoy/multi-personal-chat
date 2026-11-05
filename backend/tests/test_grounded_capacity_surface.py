"""Independent ask capacity errors are explicit; the same complete source recovers."""

import pytest
from test_ask_request_context import ANSWER, QUESTION, SOURCE, request
from test_ask_request_context import ask_rig as ask_rig
from test_provider_error_surface import events
from test_provider_error_surface import provider_api as provider_api

from inference.context_budget import ReviewContextBudget
from inference.request_context import get_request_context
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder

pytestmark = pytest.mark.usefixtures("no_live_context_review", "literal_grounded_query")
BODY = (
    SOURCE
    + "这是完整的虚构背景说明，不增加额外开放规则。" * 100
    + "末尾限定仍为：暴雨暂停开放，周二固定时段不能豁免暴雨限制。"
)


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "sse"])
@pytest.mark.parametrize("limit", ["packet", "prompt"])
async def test_ask_capacity_failure_and_budget_recovery_keep_complete_source(provider_api, tmp_path, streaming, limit):
    api = provider_api
    rig = api.state.rig
    document = rig.database.save_knowledge_document({"title": "完整虚构档案馆规程", "content": BODY}, chunks=[BODY])
    identity = str(document["id"])
    source = tmp_path / "grounded-capacity-original.txt"
    source.write_text(BODY, encoding="utf-8")

    def retrieve(query, *, domain_id, top_k, filters):
        assert query == QUESTION and domain_id == "story" and top_k == 1 and filters is None
        rig.retrievals += 1
        try:
            row = rig.database.get_knowledge_document(int(identity))
            assert row["content"] == source.read_text(encoding="utf-8") == BODY
            return {
                "domains": ["story"],
                "confidence": 0.95,
                "abstained": False,
                "query_analysis": {"entities": ["虚构档案馆"], "matched_domains": ["story"]},
                "results": [
                    {
                        "id": identity,
                        "document_type": "fact",
                        "title": row["title"],
                        "summary": "",
                        "content": row["content"],
                        "domain_id": "story",
                        "reality_status": "objective",
                        "temporal_scope": "current",
                        "content_scope": "main_story",
                        "index_version": "complete-fiction-v1",
                        "rerank_score": 0.95,
                        "fused_score": 0.95,
                        "source": {"source_path": str(source), "line_start": 1, "line_end": 1},
                        "metadata": {"story_unit_id": "complete-fiction"},
                    }
                ],
                "citations": [
                    {
                        "source_id": identity,
                        "source_title": row["title"],
                        "source_path": str(source),
                        "source_line": 1,
                        "source_line_end": 1,
                        "domain_id": "story",
                        "index_version": "complete-fiction-v1",
                    }
                ],
            }
        finally:
            rig.database.close_connection()

    rig.service.retriever = retrieve
    rig.service.packet_builder = EvidencePacketBuilder(600 if limit == "packet" else None)
    rig.service.prompt_builder = GroundedPromptBuilder(
        600 if limit == "prompt" else None, context_budget=ReviewContextBudget(8192)
    )
    endpoint = "/api/ask/stream" if streaming else "/api/ask"
    failed = await api.client.post(endpoint, json=request().model_dump())
    if streaming:
        assert failed.status_code == 200
        sequence = events(failed)
        assert sequence[-1]["type"] == "error"
        detail = sequence[-1]
        assert not any(event["type"] in {"delta", "citations", "done"} for event in sequence)
    else:
        assert failed.status_code == 503
        detail = failed.json()
        assert detail["success"] is False and "answer" not in detail
    assert detail["kind"] == "context_budget_exceeded" and detail["reason"] == "capacity_exceeded"
    assert detail["stage"] == "context_budget"
    assert rig.calls == [] and rig.retrievals == 1
    assert rig.service.cache.stats()["entries"] == 0
    assert rig.database.execute_sql("SELECT * FROM model_invocations") == []
    assert get_request_context() is None

    # Recover by supplying adequate explicit working limits, with unchanged
    # facts, source identity, question and model. No missing input is invented.
    rig.service.packet_builder = EvidencePacketBuilder(None)
    rig.service.prompt_builder = GroundedPromptBuilder(None, context_budget=ReviewContextBudget(8192))
    recovered = await api.client.post(endpoint, json=request().model_dump())
    assert recovered.status_code == 200
    if streaming:
        sequence = events(recovered)
        assert [event["type"] for event in sequence] == ["meta", "delta", "citations", "done"]
        assert sequence[1]["text"] == ANSWER
        assert sequence[-1]["model_invoked"] is True and sequence[-1]["abstained"] is False
        citations = sequence[2]["citations"]
        assert rig.service.cache.stats()["entries"] == 0
    else:
        payload = recovered.json()
        assert payload["answer"] == ANSWER and payload["modelInvoked"] is True and payload["cacheHit"] is False
        citations = payload["citations"]
        assert rig.service.cache.stats()["entries"] == 1
    assert len(citations) == 1 and citations[0]["document_id"] == identity
    assert len(rig.calls) == 1 and rig.retrievals == 2
    wire = rig.calls[0]["body"]
    assert wire["model"] == "deepseek-flash" and BODY in str(wire["messages"]) and QUESTION in str(wire["messages"])
    (invocation,) = rig.database.execute_sql("SELECT * FROM model_invocations")
    assert (
        invocation["totalTokens"] == 120 and invocation["errorType"] == "" and invocation["usageSource"] == "provider"
    )
    assert source.read_text(encoding="utf-8") == BODY
    assert rig.database.get_knowledge_document(int(identity))["content"] == BODY
    api.state.forbidden_fallback.assert_not_awaited()
    assert get_request_context() is None


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "sse"])
async def test_large_low_confidence_is_normal_abstention_without_model(streaming):
    from copy import deepcopy
    from unittest.mock import AsyncMock, Mock

    from test_grounded_single_retrieval import QUERY as RULE_QUERY
    from test_grounded_single_retrieval import SOURCE as RULE_SOURCE
    from test_grounded_single_retrieval import source_bundle

    from knowledge.grounded_answer.cache import AnswerCache
    from knowledge.grounded_answer.service import GroundedAnswerService

    data = source_bundle()
    data.update(confidence=0.2, abstained=True)
    data["results"][0]["content"] = "完整背景资料。" * 1400 + RULE_SOURCE
    original = deepcopy(data)
    retrieve = Mock(return_value=data)
    model = AsyncMock(side_effect=AssertionError("Low confidence must not invoke a model"))
    service = GroundedAnswerService(
        retriever=retrieve,
        cache=AnswerCache(ttl_seconds=0),
        packet_builder=EvidencePacketBuilder(600),
        prompt_builder=GroundedPromptBuilder(600),
    )
    if streaming:
        sequence = [event async for event in service.answer_stream(RULE_QUERY, generate=model, domain_id="story")]
        assert [event.type for event in sequence] == ["meta", "delta", "citations", "done"]
        result = sequence[-1].data
        assert result["abstained"] and not result["model_invoked"]
        assert result["retrieval_metadata"]["abstention_reason"] == "low_confidence"
    else:
        result = await service.answer(RULE_QUERY, generate=model, domain_id="story")
        assert result.abstained and not result.model_invoked
        assert result.retrieval_metadata["abstention_reason"] == "low_confidence"
    retrieve.assert_called_once_with(RULE_QUERY, top_k=3, filters=None, domain_id="story")
    model.assert_not_awaited()
    assert data == original
