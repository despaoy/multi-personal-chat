"""Configured dependency failures must stop grounded answers before model calls."""
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from knowledge.grounded_answer.service import GroundedAnswerService


def _bundle() -> dict[str, Any]:
    return {
        "domains": ["story"],
        "confidence": 0.9,
        "abstained": False,
        "query_analysis": {"entities": ["甲"], "matched_domains": ["story"]},
        "results": [
            {
                "id": "doc-1",
                "document_type": "fact",
                "title": "甲的身份",
                "summary": "甲是学生。",
                "content": "甲是学生。\n证据：甲说自己仍在上学。",
                "domain_id": "story",
                "reality_status": "objective",
                "temporal_scope": "current",
                "content_scope": "main_story",
                "index_version": "v1",
                "rerank_score": 0.9,
                "fused_score": 0.8,
                "vector_score": 0.8,
                "bm25_score": 0.8,
                "source": {
                    "source_path": "story.txt",
                    "line_start": 10,
                    "line_end": 11,
                },
                "metadata": {"story_unit_id": "unit-1"},
            }
        ],
        "citations": [
            {
                "source_id": "doc-1",
                "source_title": "甲的身份",
                "document_type": "fact",
                "source_path": "story.txt",
                "source_line": 10,
                "source_line_end": 11,
                "domain_id": "story",
                "index_version": "v1",
            }
        ],
    }


@pytest.mark.parametrize("stage", ["version", "supplement", "stream_supplement"])
async def test_dependency_exception_propagates_before_generation(stage):
    error = OSError("synthetic dependency unavailable")
    fail = Mock(side_effect=error)
    retrieve = Mock(return_value=_bundle())
    generate = AsyncMock(return_value="甲是学生。[S1]")
    options = {"index_version_resolver": fail} if stage == "version" else {"domain_supplements": fail}
    service = GroundedAnswerService(retriever=retrieve,  **options)
    with pytest.raises(OSError) as caught:
        if stage == "stream_supplement":
            async for _event in service.answer_stream("甲是谁", generate=generate, domain_id="story"):
                pass
        else:
            await service.answer("甲是谁", generate=generate, domain_id="story")
    assert caught.value is error
    fail.assert_called_once_with("story")
    generate.assert_not_called()
    if stage == "version":
        retrieve.assert_not_called()


@pytest.mark.parametrize("configured", [False, True])
async def test_success_and_absent_optional_callbacks_remain_valid(configured):
    async def generate(**kwargs):
        text = "\n".join(m["content"] for m in kwargs["messages"])
        assert ("synthetic-domain-rule" in text) is configured
        return "甲是学生。[S1]"
    options = dict(index_version_resolver=lambda domain: "v1", domain_supplements=lambda domain: "synthetic-domain-rule") if configured else {}
    result = await GroundedAnswerService(retriever=lambda *a, **kw: _bundle(),  **options).answer("甲是谁", generate=generate, domain_id="story")
    assert not result.abstained and not result.failure_kind
    assert result.citations[0]["document_id"] == "doc-1"
