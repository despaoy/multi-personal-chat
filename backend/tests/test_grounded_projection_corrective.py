"""Two corrective rounds keep one complete curated source authority.

Only the synchronous retrieval boundary is controlled. Source files, extraction,
receipts, the two-round adapter, packet projection and prompt builder are real.
No model is called.
"""

from copy import deepcopy
from html import unescape
from types import SimpleNamespace

import pytest

from knowledge import curated_sources
from knowledge.curated_sources import collect_curated_sources, validate_curated_source
from knowledge.grounded_answer.corrective import CorrectiveRetrievalAdapter
from knowledge.grounded_answer.models import AnswerMode
from knowledge.grounded_answer.packet import EvidencePacketBuilder
from knowledge.grounded_answer.prompt import GroundedPromptBuilder
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference

DOMAIN = "fiction-corrective-archive"
QUERY = "compare original archive sources"
BODY_PRIMARY = "虚构西岭档案馆周三只办理归还，不办理续借。\n雨天暂停，预约不能豁免雨天限制。"
BODY_SECONDARY = "虚构南汀档案馆周五只办理续借，不办理归还。\n大风暂停，预约不能豁免大风限制。"
BODY_REPLACED = "虚构西岭档案馆周六只办理登记，不办理归还。\n暴雪暂停，预约不能豁免暴雪限制。"


@pytest.fixture
def source_rounds(tmp_path, monkeypatch):
    extractor = OriginalTextExtractor(tmp_path)
    monkeypatch.setattr(curated_sources, "_extractor", lambda: extractor)
    cards, evidence, paths = {}, {}, {}
    bodies = {"primary": BODY_PRIMARY, "secondary": BODY_SECONDARY}
    for label in ("primary", "primary-unavailable", "secondary", "secondary-unavailable"):
        reference = SourceReference(source_path=f"{label}.txt", line_start=1, line_end=2)
        indexed = f"INDEXED_ONLY_{label}: this old retrieval annotation is not current original evidence."
        card = KnowledgeIndexDocument(
            id=f"fiction:card:{label}", domain_id=DOMAIN, document_type="fact",
            title=f"retrieval{label.replace('-', '')}", summary=indexed,
            content=indexed, embedding_text=indexed, source=reference,
            metadata={"volume_number": 9, "story_unit_id": f"old-unit-{label}"},
            reality_status="objective", temporal_scope="current", content_scope="main_story",
            index_version="fiction-loaded-index-v1",
        )
        cards[label] = card
        if label in bodies:
            path = tmp_path / reference.source_path
            path.write_bytes(bodies[label].encode("utf-8"))
            paths[label] = path
            evidence[card.id] = KnowledgeIndexDocument(
                id=f"fiction:evidence:{label}", domain_id=DOMAIN, document_type="evidence",
                title=f"declared source {label}", summary=indexed, content=indexed,
                embedding_text=indexed, source=reference, metadata={"parent_id": card.id},
            )

    def make_round(labels, *, abstained):
        selected = [SimpleNamespace(document=cards[label]) for label in labels]
        catalog = collect_curated_sources(selected, evidence, extractor, None)
        results = []
        for label in labels:
            row = cards[label].to_dict()
            row.update(fused_score=0.9 if label.startswith("secondary") else 0.7, rerank_score=0.9)
            results.append(row)
        return dict(
            domains=[DOMAIN], results=results,
            citations=[dict(id=row["id"], **row["source"]) for row in results],
            confidence=0.2 if abstained else 0.9, abstained=abstained,
            query_analysis={"entities": ["虚构档案馆"], "matched_domains": [DOMAIN]},
            curated_source_catalog=catalog,
            evidence_packets=[dict(kind="evidence", document_ids=[row["id"]], text=row["content"]) for row in results],
        )

    return SimpleNamespace(cards=cards, paths=paths, make_round=make_round)


def two_round_adapter(first, second):
    calls = []

    def retrieve(query, *, top_k, filters):
        index = len(calls)
        calls.append(dict(query=query, top_k=top_k, filters=filters))
        assert index < 2, "Correction must stay within the configured two retrievals"
        return (first, second)[index]

    return CorrectiveRetrievalAdapter(retrieve, max_retries=1), calls


def assert_two_rounds(info, calls):
    assert len(calls) == len(info["rounds"]) == 2
    assert calls[0]["query"] == QUERY
    assert calls[1]["query"].startswith(QUERY + " ")
    assert info["reformulated"] and info["reformulated_query"] == calls[1]["query"]
    assert all(call["top_k"] == 4 and call["filters"] is None for call in calls)


def test_distinct_rounds_preserve_complete_receipts_and_explicit_unavailable(source_rounds):
    first = source_rounds.make_round(["primary", "primary-unavailable"], abstained=True)
    second = source_rounds.make_round(["secondary", "secondary-unavailable"], abstained=False)
    originals = deepcopy([first, second])
    adapter, calls = two_round_adapter(first, second)
    merged, info = adapter.retrieve_with_correction(QUERY, top_k=4, filters=None)
    assert_two_rounds(info, calls)
    assert [first, second] == originals
    catalog = merged["curated_source_catalog"]
    assert len(catalog["sources"]) == len(catalog["unavailable"]) == 2
    assert {source["parent_id"]: source for source in catalog["sources"]} == {
        source["parent_id"]: source for original in originals for source in original["curated_source_catalog"]["sources"]
    }
    assert {row["parent_id"]: row for row in catalog["unavailable"]} == {
        row["parent_id"]: row for original in originals for row in original["curated_source_catalog"]["unavailable"]
    }
    for source in catalog["sources"]:
        validate_curated_source(source)
    covered = {row["parent_id"] for rows in catalog.values() for row in rows}
    assert covered == {row["id"] for row in merged["results"]}
    packet = EvidencePacketBuilder().build(QUERY, merged, AnswerMode.GROUNDED_ANSWER)
    assert {item.document_id for item in packet.documents} == {
        source_rounds.cards[label].id for label in ("primary", "secondary")
    }
    wire = unescape("\n".join(message["content"] for message in GroundedPromptBuilder().build_messages(packet)))
    assert BODY_PRIMARY in wire and BODY_SECONDARY in wire
    assert "INDEXED_ONLY_" not in wire


def test_repeated_identical_source_and_unavailable_deduplicate_across_rounds(source_rounds):
    first = source_rounds.make_round(["primary", "primary-unavailable"], abstained=True)
    second = source_rounds.make_round(["primary", "primary-unavailable"], abstained=False)
    second["results"][0]["fused_score"] = 0.95  # Scores may differ; source authority may not.
    originals = deepcopy([first, second])
    adapter, calls = two_round_adapter(first, second)
    merged, info = adapter.retrieve_with_correction(QUERY, top_k=4, filters=None)
    assert_two_rounds(info, calls)
    assert [first, second] == originals
    assert merged["curated_source_catalog"] == first["curated_source_catalog"]
    assert len(merged["results"]) == 2
    packet = EvidencePacketBuilder().build(QUERY, merged, AnswerMode.GROUNDED_ANSWER)
    assert len(packet.documents) == 1 and BODY_PRIMARY in packet.documents[0].evidence_text


def test_same_parent_new_snapshot_is_a_conflict_even_when_indexed_document_is_unchanged(source_rounds):
    first = source_rounds.make_round(["primary"], abstained=True)
    source_rounds.paths["primary"].write_bytes(BODY_REPLACED.encode("utf-8"))
    second = source_rounds.make_round(["primary"], abstained=False)
    assert first["results"] == second["results"]  # Existing document-authority conflict checks cannot catch this.
    old = first["curated_source_catalog"]["sources"][0]
    new = second["curated_source_catalog"]["sources"][0]
    assert old["parent_id"] == new["parent_id"] and old["source_id"] == new["source_id"]
    assert old["source_file_sha256"] != new["source_file_sha256"]
    assert old["excerpt"]["text"] == BODY_PRIMARY and new["excerpt"]["text"] == BODY_REPLACED
    validate_curated_source(old, verify_original=False)
    validate_curated_source(new)
    originals = deepcopy([first, second])
    adapter, calls = two_round_adapter(first, second)
    with pytest.raises(ValueError):
        adapter.retrieve_with_correction(QUERY, top_k=4, filters=None)
    assert len(calls) == 2
    assert [first, second] == originals


def test_document_only_correction_keeps_existing_merge_and_prompt_contract(source_rounds):
    first = source_rounds.make_round(["primary"], abstained=True)
    second = source_rounds.make_round(["secondary"], abstained=False)
    for bundle, body in ((first, BODY_PRIMARY), (second, BODY_SECONDARY)):
        bundle.pop("curated_source_catalog")
        bundle["evidence_packets"][0]["text"] = body
    originals = deepcopy([first, second])
    adapter, calls = two_round_adapter(first, second)
    merged, info = adapter.retrieve_with_correction(QUERY, top_k=4, filters=None)
    assert_two_rounds(info, calls)
    assert [first, second] == originals
    assert "curated_source_catalog" not in merged
    expected_ids = [source_rounds.cards[label].id for label in ("secondary", "primary")]
    assert [row["id"] for row in merged["results"]] == expected_ids
    assert [row["id"] for row in merged["citations"]] == expected_ids
    assert merged["evidence_packets"] == first["evidence_packets"] + second["evidence_packets"]
    assert merged["confidence"] == second["confidence"] and merged["abstained"] is False
    packet = EvidencePacketBuilder().build(QUERY, merged, AnswerMode.GROUNDED_ANSWER)
    assert [item.document_id for item in packet.documents] == expected_ids
    assert [item.citation_key for item in packet.documents] == ["S1", "S2"]
    assert {item.evidence_text for item in packet.documents} == {BODY_PRIMARY, BODY_SECONDARY}
    assert all(item.document_type == "fact" and item.reality_status == "objective" for item in packet.documents)
    assert [item.title for item in packet.documents] == [source_rounds.cards[label].title for label in ("secondary", "primary")]
    assert all(item.summary.startswith("INDEXED_ONLY_") for item in packet.documents)
