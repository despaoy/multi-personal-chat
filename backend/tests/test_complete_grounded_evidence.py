"""Complete native evidence and dialogue context must reach the answer intact."""

import json
from html import unescape
from pathlib import Path

import pytest

from knowledge.grounded_answer.cache import AnswerCache
from knowledge.grounded_answer.models import AnswerMode
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder
from knowledge.grounded_answer.service import GroundedAnswerService
from knowledge.grounded_answer.validator import public_citation_view

ROOT = Path(__file__).parent / "fixtures"
FIXTURE = json.loads((ROOT / "deepseek_grounded_evidence_cases.json").read_text(encoding="utf-8"))


def bundle():
    items = []
    packets = []
    for source in FIXTURE["sources"]:
        text = (ROOT / "grounded_sources" / source["source_file"]).read_text(encoding="utf-8").strip()
        items.append(
            dict(
                id=source["id"],
                title=source["title"],
                summary=source["summary"],
                content=source["summary"] + "\n证据：" + text,
                domain_id="fixture",
                document_type="fact",
                reality_status="objective",
                temporal_scope="current",
                content_scope="main_story",
                index_version="fixture-v1",
                metadata=dict(story_unit_id="stage8-permit-control"),
                source=dict(source_path=source["source_file"], line_start=1, line_end=len(text.splitlines())),
            )
        )
        packets.append(dict(text=text, document_ids=[source["id"]], kind="evidence"))
    return dict(
        domains=["fixture"],
        results=items,
        citations=[dict(id=x["id"]) for x in items],
        evidence_packets=packets,
        confidence=0.8,
        abstained=False,
    )


def test_complete_source_tail_survives_packet_and_unequal_prompt_allocation():
    data = bundle()
    packet = EvidencePacketBuilder(evidence_budget_chars=2400).build("许可是否批准？", data, AnswerMode.GROUNDED_ANSWER)
    assert len(packet.documents) == 3
    assert FIXTURE["late_qualifier"] in packet.documents[0].evidence_text
    assert len(packet.context_text) <= 2400
    assert len(packet.documents[0].to_block()) > 800
    prompt = GroundedPromptBuilder(evidence_max_chars=2400).build_evidence_section(packet)
    assert FIXTURE["late_qualifier"] in unescape(prompt)
    assert "琉璃" in unescape(prompt)


def test_admitted_source_packet_is_not_replaced_by_card_abstract():
    data = bundle()
    data["results"][0]["content"] = "这里只保留卡片摘要，原文在检索证据包中。"
    packet = EvidencePacketBuilder().build("许可是否批准？", data, AnswerMode.GROUNDED_ANSWER)
    assert FIXTURE["late_qualifier"] in packet.documents[0].evidence_text


def test_public_citation_scope_and_excerpt_come_from_matching_evidence():
    data = bundle()
    data["citations"][0].update(
        reality_status="forged",
        temporal_scope="forged",
        content_scope="forged",
        story_unit_id="forged",
        evidence_excerpt="forged",
    )
    packet = EvidencePacketBuilder().build("许可是否批准？", data, AnswerMode.GROUNDED_ANSWER)
    citation = public_citation_view(packet.citations[0])
    assert citation["document_id"] == "stage8-kisaki-permit"
    assert citation["reality_status"] == "objective"
    assert citation["temporal_scope"] == "current"
    assert citation["content_scope"] == "main_story"
    assert citation["story_unit_id"] == "stage8-permit-control"
    assert FIXTURE["late_qualifier"] in citation["evidence_excerpt"]


def test_over_budget_document_does_not_hide_complete_later_document():
    data = bundle()
    data["results"][0]["summary"] = "长" * 800
    packet = EvidencePacketBuilder(evidence_budget_chars=600).build("许可是否批准？", data, AnswerMode.GROUNDED_ANSWER)
    assert packet.truncated and "evidence_budget_truncated" in packet.warnings
    assert packet.documents and all(item.document_id != "stage8-kisaki-permit" for item in packet.documents)
    assert len(packet.context_text) <= 600


def test_packet_budget_includes_actual_separators():
    data = bundle()
    for item in data["results"]:
        item["summary"] = "摘要" * 110
        item["content"] = ""
    data.pop("evidence_packets")
    full = EvidencePacketBuilder(evidence_budget_chars=10000).build("测试", data, AnswerMode.GROUNDED_ANSWER)
    budget = len(full.context_text) - 1
    assert budget >= 600
    bounded = EvidencePacketBuilder(evidence_budget_chars=budget).build("测试", data, AnswerMode.GROUNDED_ANSWER)
    assert len(bounded.context_text) <= budget
    assert len(bounded.documents) < len(full.documents)


@pytest.mark.asyncio
async def test_identical_followup_with_different_history_cannot_share_cached_answer(monkeypatch):
    data = bundle()
    data["query_analysis"] = {"entities": ["月社妃", "琉璃"]}
    calls = []

    async def generate(**kwargs):
        topics = [m["content"] for m in kwargs["messages"] if m["role"] == "user" and m["content"].startswith("接下来")]
        calls.append(topics)
        return "琉璃已取得书面许可。[S3]" if "琉璃" in topics[0] else "月社妃尚未获批。[S1]"

    service = GroundedAnswerService(
        retriever=lambda *args, **kwargs: data,
        index_version_resolver=lambda domain: "fixture-v1",
        cache=AnswerCache(ttl_seconds=60),

    )
    from unittest.mock import AsyncMock

    # Cache isolation is tested with explicitly controlled semantic query preparation.
    monkeypatch.setattr("knowledge.grounded_answer.service.contextual_retrieval_query",
                        AsyncMock(side_effect=lambda query, history: query))
    first = await service.answer(
        FIXTURE["cases"][2]["message"], history=FIXTURE["cases"][2]["history"], domain_id="fixture", generate=generate
    )
    second = await service.answer(
        FIXTURE["cases"][3]["message"], history=FIXTURE["cases"][3]["history"], domain_id="fixture", generate=generate
    )
    assert len(calls) == 2
    assert not first.abstained and not second.abstained
    assert first.answer == "月社妃尚未获批。"
    assert second.answer == "琉璃已取得书面许可。"
    assert first.citations[0]["document_id"] == "stage8-kisaki-permit"
    assert second.citations[0]["document_id"] == "stage8-ruri-permit"


@pytest.mark.parametrize("failure", ["conflict", "missing_id", "missing_document"])
async def test_broken_citation_aborts_before_model_generation(failure):
    from unittest.mock import AsyncMock

    data = bundle()
    data["query_analysis"] = {"entities": ["月社妃", "琉璃"]}
    if failure == "conflict":
        data["citations"][0]["source_id"] = "another-document"
    elif failure == "missing_id":
        data["citations"][0].pop("id")
    else:
        data["results"].pop(0)
    generate = AsyncMock(side_effect=AssertionError("broken evidence must not reach the model"))
    service = GroundedAnswerService(retriever=lambda *a, **kw: data)
    with pytest.raises(KeyError if failure == "missing_document" else ValueError):
        await service.answer("月社妃是否获批许可？", generate=generate, domain_id="fixture")
    generate.assert_not_awaited()


@pytest.mark.parametrize("failure", ["metadata", "document_id"])
def test_citation_binding_requires_packet_metadata(failure):
    from knowledge.grounded_answer.validator import CitationValidator

    packet = EvidencePacketBuilder().build("许可是否批准？", bundle(), AnswerMode.GROUNDED_ANSWER)
    validator = CitationValidator()
    validation = validator.validate("月社妃尚未获批。[S1]", packet)
    if failure == "metadata":
        packet.citations = packet.citations[1:]
    else:
        packet.citations[0].pop("document_id")
    with pytest.raises(KeyError):
        validator.bind(validation, packet)








@pytest.mark.parametrize("failure", ["null_list", "not_packet", "kind", "missing_text", "empty_text", "ids_type", "empty_ids", "invalid_id"])
async def test_admitted_evidence_contract_cannot_silently_use_card_content(failure):
    from unittest.mock import AsyncMock

    data = bundle()
    data["query_analysis"] = {"entities": ["月社妃"]}
    if failure == "null_list":
        data["evidence_packets"] = None
    elif failure == "not_packet":
        data["evidence_packets"][0] = "invalid packet"
    else:
        packet = data["evidence_packets"][0]
        if failure == "kind":
            packet["kind"] = "unknown"
        elif failure == "missing_text":
            del packet["text"]
        elif failure == "empty_text":
            packet["text"] = " "
        elif failure == "ids_type":
            packet["document_ids"] = "document"
        elif failure == "empty_ids":
            packet["document_ids"] = []
        else:
            packet["document_ids"] = [None]
    generate = AsyncMock()
    service = GroundedAnswerService(retriever=lambda *a, **kw: data)
    with pytest.raises((ValueError, TypeError, KeyError)):
        await service.answer("许可是否批准？", generate=generate, domain_id="fixture")
    generate.assert_not_awaited()


def test_admitted_evidence_contract_keeps_background_separate_and_deduplicates_text():
    from copy import deepcopy

    data = bundle()
    data["evidence_packets"].append(deepcopy(data["evidence_packets"][0]))
    data["evidence_packets"].append(dict(kind="background", text="背景不应替代原文", document_ids=[]))
    packet = EvidencePacketBuilder().build("许可是否批准？", data, AnswerMode.GROUNDED_ANSWER)
    assert packet.documents[0].evidence_text == data["evidence_packets"][0]["text"]
    assert "背景不应替代原文" not in packet.context_text


@pytest.mark.parametrize("component", ["packet", "prompt", "citation"])
@pytest.mark.parametrize("value", [0, True, 600.5, "600"])
def test_grounded_budget_config_rejects_implicit_corrections(component, value):
    from knowledge.grounded_answer.validator import CitationValidator

    constructor = {"packet": EvidencePacketBuilder, "prompt": GroundedPromptBuilder, "citation": CitationValidator}[component]
    with pytest.raises(ValueError, match="must be an integer"):
        constructor(value)


@pytest.mark.parametrize("constructor", [EvidencePacketBuilder, GroundedPromptBuilder])
def test_grounded_budget_config_rejects_too_small_explicit_evidence_limit(constructor):
    with pytest.raises(ValueError, match=">= 600"):
        constructor(599)


def test_grounded_budget_config_accepts_boundary_and_keeps_complete_evidence():
    from knowledge.grounded_answer.validator import CitationValidator

    packet = EvidencePacketBuilder(600).build("许可是否批准？", bundle(), AnswerMode.GROUNDED_ANSWER)
    prompt = GroundedPromptBuilder(600).build_evidence_section(packet)
    assert packet.documents and len(packet.context_text) <= 600
    assert packet.documents[0].evidence_text in unescape(prompt)
    validation = CitationValidator(1).validate("依据原文回答。[S1]", packet)
    assert validation.valid_keys == ["S1"] and not validation.invalid_keys
