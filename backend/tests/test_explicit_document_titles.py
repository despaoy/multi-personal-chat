"""Explicit document reads preserve requested scope without guessed ownership."""

from threading import RLock

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.source_expansion import expand_source_context, requested_document_titles
from knowledge.vector_db import VectorDatabase


@pytest.mark.parametrize(
    "query,expected",
    [
        ("请查知识库，逐项比较《A》《B》《C》《D》这四份说明中的选项。", ("A", "B", "C", "D")),
        ("请逐项比较《A》与《B》。", ("A", "B")),
        ("分别读取《A》《B》这2份资料。", ("A", "B")),
        ("查询知识库：逐项核对《A》《B》这二份文档。", ("A", "B")),
        ("逐一比较《 A 版本》《B》。", (" A 版本", "B")),
        ("分别列出《A》《A》《B》的编号。", ("A", "B")),
        (
            "逐项比较" + "".join("《文" + str(i) + "》" for i in range(12)) + "这十二份说明。",
            tuple("文" + str(i) for i in range(12)),
        ),
    ],
)
def test_closed_requested_titles(query, expected):
    assert requested_document_titles(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "昨天读了《A》《B》，我的名字是什么？",
        "不要逐项比较《A》《B》。",
        "逐项比较《A》《B》这四份说明。",
        "逐项比较《A》《B》，不要读取第一份。",
        "逐项比较《A》《B》，只比较A。",
        "逐项比较《A》《B》，另外不要看《C》。",
        "逐项比较《A》《B》，再读取《C》。",
        "分别读取A和B。",
        "比较《A》《B》。",
    ],
)
def test_unknown_or_excluded_whole_task_defers(query):
    assert requested_document_titles(query) == ()


def row(parent, title, content="完整正文", kb=7, index=0):
    return dict(
        id=f"doc_{parent}_chunk_{index}",
        document_id=parent,
        chunk_index=index,
        knowledge_base_id=kb,
        title=title,
        category="目录",
        content=content,
    )


def expand(records, anchors, query, budget=65536, filters=None, abstained=False):
    v = VectorDatabase.__new__(VectorDatabase)
    v._lock = RLock()
    v._cache_generation = 11
    v.snapshot_validated = True
    v.metadata = records
    return expand_source_context(
        dict(results=anchors, confidence=0.8, abstained=abstained),
        v,
        expected_generation=11,
        source_budget_tokens=budget,
        filters=filters,
        query=query,
    )


def compile_result(result, window=65536):
    packets = document_evidence_packets(result["results"])
    return build_generation_request(
        GenerationRequest(
            message="逐项比较《A》《B》《C》《D》",
            context_window_tokens=window,
            evidence_max_chars=0,
            retrieval=RetrievalResult(
                status="ok",
                evidence="\n".join(p["text"] for p in packets),
                evidence_packets=packets,
                source_coverage=result["source_coverage"],
            ),
        )
    )


def test_four_independent_requested_docs_survive_original_top_three():
    records = [row(i + 1, title, f"{title}完整参数与末尾限制") for i, title in enumerate("ABCD")]
    anchors = records[1:]
    result = expand(records, anchors, "逐项比较《A》《B》《C》《D》")
    assert result["results"][:3] == anchors and len(result["results"]) == 4 and result["confidence"] == 0.8
    extra = result["results"][3]
    assert extra["id"] == "doc_1_chunk_0" and extra["retrieval_role"] == "requested_source" and extra["score"] == 0
    assert "supporting_document_ids" not in extra
    plan = compile_result(result)
    assert all(r["content"] in plan.messages[-1]["content"] for r in records)
    assert len(plan.retrieval.source_coverage) == 4


@pytest.mark.parametrize("scope", ["other_kb", "filter"])
def test_explicit_titles_respect_anchor_knowledge_base_and_original_filter(scope):
    anchor = row(1, "A")
    target = row(2, "B", kb=8 if scope == "other_kb" else 7)
    target["category"] = "其他"
    result = expand(
        [anchor, target], [anchor], "逐项比较《A》《B》", filters={"category": "目录"} if scope == "filter" else None
    )
    assert result["results"] == [anchor] and result["unresolved_requested_titles"] == ["B"]


def test_ambiguous_exact_title_fails_instead_of_picking_a_version():
    with pytest.raises(RuntimeError, match="Ambiguous requested document title"):
        expand([row(1, "A"), row(2, "B"), row(3, "B")], [row(1, "A")], "逐项比较《A》《B》")


def test_requested_multi_chunk_budget_and_missing_title_are_truthful():
    anchor = row(1, "A")
    large = row(2, "B", "超长" * 5000)
    tail = row(2, "B", "完整尾部限制", index=1)
    result = expand([anchor, large, tail], [anchor], "分别读取《A》《B》《缺失》", budget=100)
    assert len(result["results"]) == 2 and result["results"][1]["content"] == tail["content"]
    assert result["unresolved_requested_titles"] == ["缺失"]
    scope = next(r for r in compile_result(result).retrieval.source_coverage if r["source_title"] == "B")
    assert scope["status"] == "partial" and scope["admitted_chunk_count"] == 1 and scope["indexed_chunk_count"] == 2


def test_explicit_requested_doc_does_not_depend_on_unrelated_rejected_rank_anchor():
    huge = row(1, "A", "整份巨文" * 3000)
    short = row(2, "B", "本轮明确要求的B完整条件")
    result = expand([huge, short], [huge], "分别读取《A》《B》")
    plan = compile_result(result, 8192)
    assert short["content"] in plan.messages[-1]["content"] and huge["content"] not in plan.messages[-1]["content"]


def test_no_explicit_query_keeps_original_reference_dependency():
    anchor = row(1, "A", "请结合《B》。")
    target = row(2, "B")
    result = expand([anchor, target], [anchor], "这句话引用了《B》，并非读取请求")
    assert result["results"][1]["retrieval_role"] == "source_context" and result["results"][1][
        "supporting_document_ids"
    ] == [anchor["id"]]


def test_abstention_is_not_overridden_by_explicit_titles():
    anchor = row(1, "A")
    result = expand([anchor, row(2, "B")], [anchor], "分别读取《A》《B》", abstained=True)
    assert result["results"] == [anchor] and result["abstained"]
