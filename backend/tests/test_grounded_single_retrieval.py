"""One semantic query and one evidence result, with no candidate-word retries."""

from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from test_grounded_dependency_errors import _bundle

from knowledge.grounded_answer.cache import AnswerCache
from knowledge.grounded_answer.service import GroundedAnswerService

pytestmark = pytest.mark.usefixtures("literal_grounded_query")

QUERY = "请核对虚构远岚通行证所需材料，预约是否免原件，周五能否受理。"
SOURCE = "虚构远岚通行证需本人身份证原件与受理号；预约不豁免原件；周五不受理。"
FILTERS = {"knowledge_base_id": 77, "user_id": "synthetic-owner"}


def source_bundle():
    data = _bundle()
    data["query_analysis"]["entities"] = ["虚构远岚通行证"]
    data["results"][0].update(title="虚构远岚通行证", summary=SOURCE, content=SOURCE)
    data["citations"][0]["source_title"] = "虚构远岚通行证"
    return data


@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize("outcome", ["success", "low", "no_domain", "no_results", "budget"])
async def test_one_query_one_result_preserves_evidence_and_normal_business_states(transport, outcome):
    data = None if outcome == "no_domain" else source_bundle()
    if outcome == "low":
        data.update(confidence=0.2, abstained=True)
        data["results"][0].update(title="另一机构的候选提词不应改变原请求", rerank_score=0.2)
    if outcome == "no_results":
        data.update(results=[], citations=[], confidence=0.0, abstained=True)
    if outcome == "budget":
        data["results"][0]["content"] = "完整资料背景。" * 1400 + SOURCE
    reason = {
        "low": "low_confidence",
        "no_domain": "no_domain",
        "no_results": "no_retrieval",
        "budget": "context_budget_exceeded",
    }.get(outcome, "")
    original = deepcopy(data)
    retrieve = Mock(side_effect=[data, AssertionError("No second retrieval may be attempted")])

    async def generate(**kwargs):
        assert SOURCE in kwargs["messages"][-1]["content"]
        assert QUERY in kwargs["messages"][-1]["content"]
        return "需本人身份证原件与受理号；预约不免原件，周五不受理。[S1]"

    model = AsyncMock(side_effect=generate)
    service = GroundedAnswerService(
        retriever=retrieve, cache=AnswerCache(ttl_seconds=60), index_version_resolver=lambda _: "v1"
    )
    if transport == "normal":
        result = await service.answer(QUERY, generate=model, domain_id="story", filters=FILTERS)
        assert result.model_invoked is (outcome == "success")
        assert result.abstained is (outcome != "success")
        assert "corrective" not in result.retrieval_metadata
        if outcome == "success":
            assert result.citations[0]["document_id"] == "doc-1"
            cached = await service.answer(QUERY, generate=model, domain_id="story", filters=FILTERS)
            assert cached.cache_hit and not cached.model_invoked and cached.answer == result.answer
        else:
            assert result.failure_kind == ("no_domain" if outcome == "no_domain" else "")
            assert result.retrieval_metadata["abstention_reason"] == reason
    else:
        events = [
            event async for event in service.answer_stream(QUERY, generate=model, domain_id="story", filters=FILTERS)
        ]
        assert [event.type for event in events] == ["meta", "delta", "citations", "done"]
        assert events[-1].data["model_invoked"] is (outcome == "success")
        assert events[-1].data["abstained"] is (outcome != "success")
        assert (
            events[0].data["answer_mode"]
            == events[-1].data["answer_mode"]
            == ("direct_answer" if outcome == "success" else "abstention")
        )
        assert events[-1].data["retrieval_metadata"]["abstention_reason"] == reason
        assert "corrective" not in events[-1].data["retrieval_metadata"]
    retrieve.assert_called_once_with(QUERY, top_k=3, filters=FILTERS, domain_id="story")
    assert model.await_count == (1 if outcome == "success" else 0)
    assert data == original and FILTERS == {"knowledge_base_id": 77, "user_id": "synthetic-owner"}


@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize(
    "error", [OSError("index unavailable"), TimeoutError("retrieval timeout"), ValueError("invalid retrieval evidence")]
)
async def test_real_failures_never_retry_or_publish_an_abstention_as_success(transport, error):
    retrieve, model = Mock(side_effect=error), AsyncMock()
    service = GroundedAnswerService(retriever=retrieve)
    emitted = []
    with pytest.raises(type(error)) as caught:
        if transport == "normal":
            await service.answer(QUERY, generate=model)
        else:
            async for event in service.answer_stream(QUERY, generate=model):
                emitted.append(event)
    assert type(caught.value) is type(error) and caught.value.args == error.args
    assert emitted == []
    retrieve.assert_called_once()
    model.assert_not_awaited()
    assert service.cache.stats()["entries"] == 0


@pytest.mark.parametrize("option", [dict(corrective_enabled=True), dict(corrective_max_retries=1)])
def test_removed_options_cannot_be_silently_accepted(option):
    with pytest.raises(TypeError, match=next(iter(option))):
        GroundedAnswerService(**option)
