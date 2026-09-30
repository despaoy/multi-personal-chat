"""The native multi-scale citation ID must bind to the actual retrieved card."""

import pytest

from knowledge.grounded_answer.models import AnswerMode
from knowledge.grounded_answer.packet import EvidencePacketBuilder


def bundle(citation):
    return dict(
        domains=["workshop"],
        confidence=0.8,
        results=[
            dict(
                id="card-7",
                title="木刻课程",
                summary="课程十点半开始。",
                content="课程十点半开始。\n证据：周日上午十点半开课。",
                domain_id="workshop",
                document_type="fact",
                source=dict(source_path="workshop.txt", line_start=11, line_end=12),
                metadata={},
            )
        ],
        citations=[citation],
    )


@pytest.mark.parametrize(
    "citation", [dict(id="card-7"), dict(source_id="card-7"), dict(id="card-7", source_id="card-7")]
)
def test_native_and_legacy_citations_bind_to_same_authoritative_document(citation):
    packet = EvidencePacketBuilder().build("课程几点开始？", bundle(citation), AnswerMode.GROUNDED_ANSWER)
    assert len(packet.documents) == len(packet.citations) == 1
    assert packet.documents[0].evidence_text == "周日上午十点半开课。"
    assert packet.citations[0]["source_id"] == "card-7"
    assert packet.citations[0]["source_title"] == "木刻课程"
    assert packet.citations[0]["source_path"] == "workshop.txt"
    assert packet.citations[0]["source_line"] == 11
    assert packet.citations[0]["source_line_end"] == 12


def test_conflicting_native_and_legacy_ids_never_bind_another_document():
    packet = EvidencePacketBuilder().build(
        "课程几点开始？", bundle(dict(id="other-card", source_id="card-7")), AnswerMode.GROUNDED_ANSWER
    )
    assert not packet.documents and not packet.citations
    assert "citation_conflicting_document_id" in packet.warnings


def test_unmatched_native_id_never_creates_an_evidence_packet():
    packet = EvidencePacketBuilder().build(
        "课程几点开始？", bundle(dict(id="missing-card")), AnswerMode.GROUNDED_ANSWER
    )
    assert not packet.documents and not packet.citations
    assert "citation_missing_document" in packet.warnings
