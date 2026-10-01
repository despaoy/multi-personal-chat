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
async def test_identical_followup_with_different_history_cannot_share_cached_answer():
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
        corrective_enabled=False,
    )
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
