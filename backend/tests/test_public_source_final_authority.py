"""Normal persisted SQLite lifecycle; declared zero-cloud component models.

Fixtures include full scope, negation, exceptions, independent source and query.
No model selection receipt, external authentication or real answer is fabricated.
"""

from dataclasses import replace
from datetime import datetime, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response
from inference.public_context_authority import make_public_context_revalidator
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources

BODY_A = "虚构甲库2031年度成年申请规则：鹭岸核验每周三办理，费用0元，不收费，不要求原件。持甲库当日回执且已窗口核验身份者可免复印件；仅预约不能免除。"
BODY_B = "虚构独立乙库2031年度成年申请规则：杉港登记每周五办理，费用12元，必须复印件，不要求原件。未规定2032年度规则，也不担保现实办理成功。"
QUERY = "分别核对甲库鹭岸核验、乙库杉港登记2031年度成年申请的办理日、费用、原件和复印件条件与例外；核对2032年度未知材料要求，并读取本人已保存的茶饮偏好原话。来源改变或缺失时保留未知，不证明现实办理成功。"
PRIVATE = "虚构本人完整自述：我只在周一晚间独自读书时选桂花茶，不在工作日早晨饮用；出差时例外改白水。此为偏好，不代表已完成任何审批。"


def fixture(tmp_path):
    db = SQLiteDB(tmp_path / "knowledge.sqlite")
    bases = [db.create_knowledge_base(name, "完整虚构单元资料") for name in ("甲库", "乙库")]
    documents = [
        db.add_knowledge_document(dict(title=title, content=body, category="规则", knowledge_base_id=base["id"]))
        for title, body, base in zip(("鹭岸规则", "杉港规则"), (BODY_A, BODY_B), bases)
    ]
    results = [
        dict(doc, id=f"doc_{doc['id']}_chunk_0", document_id=doc["id"], chunk_id=f"doc_{doc['id']}_chunk_0")
        for doc in documents
    ]
    coverage = [
        dict(
            source_id=f"doc_{doc['id']}",
            source_title=doc["title"],
            indexed_document_ids=[row["id"]],
            retrieved_document_ids=[row["id"]],
        )
        for doc, row in zip(documents, results)
    ]
    bundle = attach_original_sources(
        dict(results=results, source_coverage=coverage),
        db.get_knowledge_document,
        source_budget_tokens=10000,
        authority_revision=0,
    )
    packets = document_evidence_packets(results) + bundle["original_source_packets"]
    retrieval = RetrievalResult(
        status="ok",
        evidence="\n\n".join(p["text"] for p in packets),
        documents=tuple(results),
        citations=tuple(dict(id=row["id"], source_id=row["id"]) for row in results),
        evidence_packets=packets,
        source_coverage=bundle["source_coverage"],
        requested_sources=(
            dict(title="鹭岸规则", lookup_status="matched_in_index_scope", source_ids=[f"doc_{documents[0]['id']}"]),
        ),
    )
    request = GenerationRequest(
        message=QUERY,
        persona_prompt="按对应原文和完整条件回答，未知保持未知。",
        retrieval=retrieval,
        context_window_tokens=65536,
        public_context_revalidator=make_public_context_revalidator(db.get_knowledge_document),
    )
    return db, documents, request


def wire(messages):
    return unescape("\n".join(row["content"] for row in messages))


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "content", "title", "category", "knowledge_base_id", "read_error"])
async def test_authority_change_removes_whole_source_and_keeps_independent(tmp_path, change):
    db, docs, request = fixture(tmp_path)
    identity = docs[0]["id"]
    if change == "delete":
        assert db.delete_knowledge_document(identity)
    elif change == "read_error":

        def read(doc_id):
            if doc_id == identity:
                raise OSError("declared isolated source read failure")
            return db.get_knowledge_document(doc_id)

        request = replace(request, public_context_revalidator=make_public_context_revalidator(read))
    else:
        value = docs[1]["knowledge_base_id"] if change == "knowledge_base_id" else "修改后的版本不继承旧审核"
        assert db.update_knowledge_document(identity, {change: value})
    updated = await request.public_context_revalidator(request)
    assert BODY_A not in updated.retrieval.evidence and BODY_B in updated.retrieval.evidence
    assert len(updated.retrieval.documents) == len(updated.retrieval.citations) == 1
    assert updated.retrieval.reason == "public_source_authority_changed"
    assert updated.retrieval.requested_sources[0]["lookup_status"] == "not_resolved"
    assert updated.retrieval.requested_sources[0]["source_ids"] == []
    calls = []

    async def model(**kwargs):
        calls.append(kwargs["messages"])
        return "明确单元模型占位：仅保留独立资料，其他未知。"

    result = await generate_character_response(
        replace(updated, retrieval=replace(updated.retrieval, citations=())), model
    )
    assert len(calls) == 1 and BODY_A not in wire(calls[0]) and BODY_B in wire(calls[0]) and QUERY in wire(calls[0])
    assert result.plan.retrieval.has_evidence


@pytest.mark.asyncio
async def test_same_source_and_irrelevant_metadata_preserve_exact_request(tmp_path):
    db, docs, request = fixture(tmp_path)
    assert await request.public_context_revalidator(request) is request
    db.update_knowledge_document(docs[0]["id"], dict(sourceUrl="https://example.invalid/fiction"))
    assert await request.public_context_revalidator(request) is request
    assert "source_authority_snapshot" not in wire(
        __import__("inference.generation_request", fromlist=["build_generation_request"])
        .build_generation_request(request)
        .messages
    )


@pytest.mark.asyncio
async def test_removed_registry_revokes_dependent_background_transitively(tmp_path):
    db, docs, request = fixture(tmp_path)
    first, second = [f"doc_{doc['id']}_chunk_0" for doc in docs]
    dependent = dict(
        kind="background",
        document_ids=["dependent-alias-rule"],
        text="虚构别名登记所依赖的规则片段，不提供独立身份。",
        supporting_document_ids=[first],
    )
    descendant = dict(
        kind="background",
        document_ids=["dependent-exception"],
        text="虚构依赖规则的例外片段。",
        supporting_document_ids=["dependent-alias-rule"],
    )
    retrieval = replace(
        request.retrieval,
        evidence_packets=(*request.retrieval.evidence_packets, dependent, descendant),
        admitted_evidence_packets=(*request.retrieval.evidence_packets, dependent, descendant),
    )
    db.delete_knowledge_document(docs[0]["id"])
    updated = await request.public_context_revalidator(replace(request, retrieval=retrieval))
    ids = {identity for packet in updated.retrieval.evidence_packets for identity in packet["document_ids"]}
    assert first not in ids and second in ids
    assert "dependent-alias-rule" not in ids and "dependent-exception" not in ids


@pytest.mark.asyncio
async def test_all_public_revoked_preserves_normally_captured_private_source(tmp_path):
    db, docs, request = fixture(tmp_path)
    scope = UserScope("web", "unit-authority", "fiction-owner", "fiction-owner", "private")
    fields = dict(
        character_id="fiction-role",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_type=scope.conversation_type,
        conversation_id=scope.conversation_id,
    )
    assert (
        db.capture_memory_source(
            **fields, source_message_id="fiction-private-original", body=PRIVATE, observed_at=datetime.now(timezone.utc)
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    recalled = await SourceMemoryService(repo, defer_budget=True).recall("fiction-role", scope, QUERY)
    context = attach_sources(CompiledCharacterContext("虚构评测画像", "", "", memory_status="no_match"), recalled)
    assert PRIVATE in context.episodic_reference_context or PRIVATE in context.source_candidate_context
    for doc in docs:
        db.delete_knowledge_document(doc["id"])
    updated = await request.public_context_revalidator(replace(request, character_context=context))
    assert updated.retrieval.status == "character_abstention" and not updated.retrieval.evidence
    assert updated.character_context is context
    assert len(db.list_memory_sources(**fields)) == 1


@pytest.mark.asyncio
async def test_second_boundary_rechecks_after_initial_compilation(tmp_path):
    db, docs, request = fixture(tmp_path)
    native = request.public_context_revalidator
    boundaries = []

    async def revalidate(current):
        boundaries.append(1)
        if len(boundaries) == 2:
            db.delete_knowledge_document(docs[0]["id"])
        return await native(current)

    calls = []

    async def model(**kwargs):
        calls.append(kwargs["messages"])
        return "明确单元模型占位。"

    await generate_character_response(
        replace(request, retrieval=replace(request.retrieval, citations=()), public_context_revalidator=revalidate),
        model,
    )
    assert len(calls) == 1 and BODY_A not in wire(calls[0]) and BODY_B in wire(calls[0])
    assert len(boundaries) >= 2


@pytest.mark.asyncio
async def test_guard_retry_rechecks_public_source_after_first_model(tmp_path):
    from character.output_guard import ReplyGuard

    db, docs, request = fixture(tmp_path)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            db.delete_knowledge_document(docs[0]["id"])
            return "哈哈哈哈哈哈！"
        return "本轮仅依据当前可核验的独立来源回答。"

    # Explicit style guard fault exercises the existing strict retry, without
    # pretending either response is a real DeepSeek semantic acceptance.
    guard = ReplyGuard(forbid_laughter=True)
    result = await generate_character_response(
        replace(
            request, retrieval=replace(request.retrieval, citations=()), reply_guard=guard, reply_guard_mode="strict"
        ),
        model,
    )
    assert result.guard_retried and len(calls) == 2
    assert BODY_A in wire(calls[0]) and BODY_A not in wire(calls[1]) and BODY_B in wire(calls[1])


@pytest.mark.asyncio
async def test_source_revoked_during_main_cannot_reach_citation_repair(tmp_path):
    db, docs, request = fixture(tmp_path)
    from inference.citation_recovery import ANNOTATION_POLICY

    calls = []
    annotations = []

    async def model(**kwargs):
        messages = kwargs["messages"]
        if messages[0]["content"] == ANNOTATION_POLICY:
            import json

            payload = json.loads(messages[-1]["content"])
            annotations.append(payload)
            assert BODY_A not in json.dumps(payload, ensure_ascii=False)
            return '{"citations":[]}'
        calls.append(messages)
        if len(calls) == 1:
            db.delete_knowledge_document(docs[0]["id"])
            return BODY_A
        assert len(calls) == 2 and BODY_A not in wire(messages) and BODY_B in wire(messages)
        return BODY_B

    result = await generate_character_response(request, model)
    assert len(calls) == 2 and len(annotations) == 1 and not result.response_citations
    assert result.authority_refreshed and not result.authority_fallback
    assert result.reply == BODY_B and BODY_A not in result.reply
    assert BODY_A not in result.plan.retrieval.evidence and BODY_B in result.plan.retrieval.evidence
