"""Complete question bindings precede and constrain domain routing."""

import copy
import json
from types import SimpleNamespace

import pytest

from knowledge.public_question_binding import (
    QuestionBindingCapacityError,
    all_bound_objects_outside_character_domain,
    resolve_question_binding,
    validate_question_binding,
)
from knowledge.turn_dependencies import parse_dependencies

QUERY = "只读已保存的完整本人偏好。请查清青妃集市展期费用、原件条件和末尾否定限制。"


def dependencies(query=QUERY):
    return parse_dependencies(
        dict(private_memory=[0], public_knowledge=[1], current_input=[], control=[], unresolved_source=[]), query
    )


async def binding(name="青妃集市展期", query=QUERY):
    seen = []

    async def reviewer(messages):
        seen.append(messages)
        payload = json.loads(messages[-1]["content"])
        assert payload == dict(query=query, public_tasks=[dict(id=1, text=dependencies(query).segments[1])])
        assert set(payload) == {"query", "public_tasks"}
        return json.dumps(dict(scopes=[dict(task_id=1, objects=[name] if name else [])]))

    receipt = await resolve_question_binding(dependencies(query), query, window_tokens=65536, reviewer=reviewer)
    assert len(seen) == 1
    return receipt


def config():
    from knowledge.retrieval_core.domains import tsukiyashiro_kisaki_domain

    return tsukiyashiro_kisaki_domain()


async def test_complete_business_object_rejects_embedded_character_alias_and_reuses_exact_payload():
    receipt = await binding()
    assert validate_question_binding(receipt, receipt["input"]) == receipt["scopes"]
    assert all_bound_objects_outside_character_domain(receipt, config())
    assert receipt["input"]["query"] == QUERY


async def test_explicit_registered_alias_keeps_strict_character_route():
    receipt = await binding("妃", "只读已保存的完整本人偏好。请核对妃的家庭关系和末尾否定限制。")
    assert not all_bound_objects_outside_character_domain(receipt, config())


async def test_unresolved_scope_does_not_grant_generic_override():
    assert not all_bound_objects_outside_character_domain(await binding(None), config())


@pytest.mark.parametrize("change", ["query", "task_text", "task_id", "raw", "scope", "digest"])
async def test_changed_query_task_identity_or_actual_response_cannot_reuse_binding(change):
    receipt = await binding()
    expected = copy.deepcopy(receipt["input"])
    if change == "query":
        expected["query"] += "新限定"
    elif change == "task_text":
        expected["public_tasks"][0]["text"] += "另一个来源"
    elif change == "task_id":
        expected["public_tasks"][0]["id"] = True
    elif change == "raw":
        receipt["raw"] = json.dumps(dict(scopes=[dict(task_id=1, objects=["妃"])]))
    elif change == "scope":
        receipt["scopes"]["objects"][0]["query_text"] = "妃"
    else:
        receipt["input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_question_binding(receipt, expected)


async def test_complete_over_capacity_input_stops_before_call_without_truncating():
    async def forbidden(messages):
        pytest.fail("No clipped question or paid request is allowed")

    with pytest.raises(QuestionBindingCapacityError):
        await resolve_question_binding(dependencies(), QUERY, window_tokens=128, reviewer=forbidden)


async def test_cancellation_propagates_without_invented_binding():
    import asyncio

    async def cancelled(messages):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await resolve_question_binding(dependencies(), QUERY, window_tokens=65536, reviewer=cancelled)


async def test_generic_route_reaches_fresh_authority_and_full_body(monkeypatch):
    from threading import RLock

    from api import generate, knowledge
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    body = "青妃集市展期须付18元并带签署表原件；假日不办理，预约不能免交原件。"
    row = dict(
        id="doc_1_chunk_0",
        document_id=1,
        chunk_index=0,
        title="完整合成规程",
        category="演练",
        knowledge_base_id=7,
        content=body,
    )
    calls = []

    def no_character(*args, **kwargs):
        pytest.fail("An embedded character alias cannot replace a complete business object")

    monkeypatch.setattr(
        runtime,
        "get_multiscale_rag_service",
        lambda: SimpleNamespace(config=config(), retrieve_with_citations=no_character),
    )
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: calls.append("fresh") or True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 77)
    monkeypatch.setattr(
        vector_db,
        "get_vector_db",
        lambda: SimpleNamespace(_lock=RLock(), cache_generation=11, snapshot_validated=True, metadata=[row]),
    )
    monkeypatch.setattr(
        generate,
        "db",
        SimpleNamespace(
            get_knowledge_document=lambda i: dict(
                id=i, title=row["title"], category=row["category"], knowledge_base_id=7, content=body
            )
        ),
    )
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")

    def retrieve(query, **kwargs):
        assert query == QUERY and kwargs["filters"] is None
        assert kwargs["additional_queries"] == ("青妃集市展期",)
        calls.append("generic")
        return dict(results=[row], confidence=0.8, abstained=False)

    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(retrieve_with_citations=retrieve, build_citations=lambda records: []),
    )
    bundle = await generate._retrieve_rag_bundle(QUERY, 3, None, question_binding=await binding())
    assert calls == ["fresh", "generic"]
    assert bundle["original_source_packets"][0]["original_body"] == body
    assert bundle["source_coverage"][0]["original_source_receipt"]["authority_revision"] == 77


async def test_explicit_character_unavailable_is_still_strict(monkeypatch):
    from api import generate
    from knowledge.multiscale_rag import runtime

    calls = []
    monkeypatch.setattr(
        runtime,
        "get_multiscale_rag_service",
        lambda: SimpleNamespace(
            config=config(), retrieve_with_citations=lambda *a, **kw: calls.append("curated") or None
        ),
    )
    with pytest.raises(RuntimeError, match="Requested character knowledge domain is unavailable"):
        await generate._retrieve_rag_bundle(
            "只读已保存的完整本人偏好。请核对妃的家庭关系和末尾否定限制。",
            3,
            None,
            question_binding=await binding("妃", "只读已保存的完整本人偏好。请核对妃的家庭关系和末尾否定限制。"),
        )
    assert calls == ["curated"]


async def test_evidence_review_reuses_exact_pre_retrieval_binding_without_second_scope_call():
    from test_public_object_scope import QUERY as q
    from test_public_object_scope import bundle, obligations, source_reply
    from test_public_object_scope import dependencies as dep

    from knowledge.public_task_evidence import review_public_candidates

    calls = []

    async def scope(messages):
        calls.append("scope")
        from test_public_object_scope import scopes

        return json.dumps(scopes())

    receipt = await resolve_question_binding(
        dep(), q, window_tokens=65536, public_obligations=obligations(), reviewer=scope
    )

    async def source(messages):
        calls.append("source")
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == q and payload["object_scopes"] == receipt["scopes"]
        assert payload["sources"][0]["original_body"].endswith("本人携带完整登记材料才能办理。")
        return json.dumps(source_reply())

    async def forbidden(messages):
        pytest.fail("Already resolved exact question binding must not be paid for again")

    result = await review_public_candidates(
        bundle(),
        dep(),
        q,
        window_tokens=65536,
        public_obligations=obligations(),
        reviewer=source,
        scope_reviewer=forbidden,
        question_binding=receipt,
    )
    assert calls == ["scope", "source"]
    assert result["public_task_review"]["review_status"] == "reviewed"


async def test_complete_object_views_cannot_be_trimmed_or_rebound():
    from knowledge.public_question_binding import bound_object_search_views

    receipt = await binding()
    assert bound_object_search_views(receipt, QUERY) == ("青妃集市展期",)
    with pytest.raises(ValueError, match="not literal"):
        bound_object_search_views(receipt, "查询另一对象的费用。")
    names = ["松湾续期", "树桥延期", "南庭租赁", "北岭换签", "西港复核"]
    query = "核对" + "、".join(names) + "的费用和末尾例外。"
    deps = parse_dependencies(
        dict(private_memory=[], public_knowledge=[0], current_input=[], control=[], unresolved_source=[]), query
    )

    async def scope(messages):
        assert json.loads(messages[-1]["content"])["query"] == query
        return json.dumps(dict(scopes=[dict(task_id=0, objects=names)]))

    complete = await resolve_question_binding(deps, query, window_tokens=65536, reviewer=scope)
    with pytest.raises(QuestionBindingCapacityError, match="Complete object view set"):
        bound_object_search_views(complete, query)
