"""Complete persisted fixtures; explicit zero-cloud models and lifecycle faults."""

import json
from dataclasses import replace

import pytest
from test_private_source_final_authority import BODY, PUBLIC, SIBLING, complete_request
from test_public_source_final_authority import BODY_A, BODY_B, QUERY, fixture, wire

from inference.citation_recovery import ANNOTATION_POLICY
from inference.generation_request import generate_character_response
from inference.public_context_authority import make_public_context_revalidator


@pytest.mark.asyncio
async def test_deleted_public_source_regenerates_from_only_remaining_full_sources(tmp_path):
    db, docs, request = fixture(tmp_path)
    calls = []

    async def model(**kwargs):
        text = wire(kwargs["messages"])
        calls.append(text)
        assert QUERY in text
        if len(calls) == 1:
            assert BODY_A in text and BODY_B in text
            db.delete_knowledge_document(docs[0]["id"])
            return BODY_A
        assert len(calls) == 2 and BODY_A not in text and BODY_B in text
        return BODY_B

    result = await generate_character_response(
        replace(request, retrieval=replace(request.retrieval, citations=())), model
    )
    assert len(calls) == 2 and result.reply == BODY_B and BODY_A not in result.reply
    assert result.authority_refreshed and not result.authority_fallback
    assert BODY_B in result.plan.retrieval.evidence and BODY_A not in result.plan.retrieval.evidence


@pytest.mark.asyncio
async def test_second_withdrawal_cannot_return_obsolete_answer_or_start_third_main(tmp_path):
    db, docs, request = fixture(tmp_path)
    calls = []

    async def model(**kwargs):
        calls.append(wire(kwargs["messages"]))
        assert QUERY in calls[-1] and len(calls) <= 2
        if len(calls) == 1:
            db.delete_knowledge_document(docs[0]["id"])
            return BODY_A
        assert BODY_A not in calls[-1] and BODY_B in calls[-1]
        db.delete_knowledge_document(docs[1]["id"])
        return BODY_B

    result = await generate_character_response(
        replace(request, retrieval=replace(request.retrieval, citations=())), model
    )
    assert len(calls) == 2 and result.authority_refreshed and result.authority_fallback
    assert BODY_A not in result.reply and BODY_B not in result.reply
    assert "暂时无法可靠核对" in result.reply
    assert result.plan.retrieval.status == "character_abstention" and not result.response_citations


@pytest.mark.asyncio
async def test_changed_source_during_actual_annotation_boundary_regenerates_without_old_answer(tmp_path):
    db, docs, request = fixture(tmp_path)
    mains, annotations = [], []

    async def model(**kwargs):
        messages = kwargs["messages"]
        if messages[0]["content"] == ANNOTATION_POLICY:
            payload = json.loads(messages[-1]["content"])
            annotations.append(payload)
            if len(annotations) == 1:
                assert payload["answer"] == BODY_A
                db.delete_knowledge_document(docs[0]["id"])
            else:
                assert len(annotations) == 2 and payload["answer"] == BODY_B
                assert BODY_A not in json.dumps(payload, ensure_ascii=False)
            return '{"citations":[]}'  # Declared unit response, not a citation grant.
        mains.append(wire(messages))
        assert QUERY in mains[-1]
        if len(mains) == 1:
            return BODY_A
        assert len(mains) == 2 and BODY_A not in mains[-1] and BODY_B in mains[-1]
        return BODY_B

    result = await generate_character_response(request, model)
    assert len(mains) == len(annotations) == 2 and result.reply == BODY_B
    assert result.authority_refreshed and not result.authority_fallback and not result.response_citations


@pytest.mark.asyncio
async def test_fresh_authority_read_failure_after_main_removes_old_answer_only_for_affected_source(tmp_path):
    db, _docs, request = fixture(tmp_path)
    reads_a = []

    def read(identity):
        if identity == 1:
            reads_a.append(1)
            if len(reads_a) == 3:
                raise OSError("declared one source authority read failure after main")
        return db.get_knowledge_document(identity)

    calls = []

    async def model(**kwargs):
        calls.append(wire(kwargs["messages"]))
        if len(calls) == 1:
            assert BODY_A in calls[-1]
            return BODY_A
        assert len(calls) == 2 and BODY_A not in calls[-1] and BODY_B in calls[-1]
        return BODY_B

    result = await generate_character_response(
        replace(
            request,
            retrieval=replace(request.retrieval, citations=()),
            public_context_revalidator=make_public_context_revalidator(read),
        ),
        model,
    )
    assert len(calls) == 2 and result.reply == BODY_B and result.authority_refreshed
    assert db.get_knowledge_document(1) is not None


@pytest.mark.asyncio
async def test_private_erasure_during_main_cannot_return_erased_body_and_preserves_public(tmp_path):
    request, _prepared, _repo, db, fields = await complete_request(tmp_path)
    calls = []

    async def model(**kwargs):
        calls.append(wire(kwargs["messages"]))
        assert request.message in calls[-1]
        if len(calls) == 1:
            assert BODY in calls[-1] and SIBLING in calls[-1] and PUBLIC in calls[-1]
            db.clear_character_memories(**fields)
            return BODY
        assert len(calls) == 2 and BODY not in calls[-1] and SIBLING not in calls[-1] and PUBLIC in calls[-1]
        return PUBLIC

    result = await generate_character_response(request, model)
    assert len(calls) == 2 and result.reply == PUBLIC and BODY not in result.reply
    assert result.authority_refreshed and not result.authority_fallback
    assert not result.plan.character_context.episodic_reference_context
