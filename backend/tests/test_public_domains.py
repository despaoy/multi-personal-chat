"""Fictional source authorities exercise real review and final packet admission."""

from knowledge import index_lifecycle

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from scope_fixtures import open_scope_response

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge import public_domains
from knowledge.public_domains import assemble_domains, domain_plan, render_domains, retrieve_domains
from knowledge.public_question_binding import resolve_question_binding
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

pytestmark = pytest.mark.usefixtures("no_live_context_review")

QUERY = "核对叶澄岚与杜木汐的原作关系，保留非血缘限定。核对澄岚展期费用、否定和末尾例外。"


@pytest.fixture
def config(monkeypatch):
    aliases = {"叶澄岚": "澄岚", "澄岚": "澄岚", "杜木汐": "木汐"}
    value = SimpleNamespace(domain_id="fictional_curated", story_titles=[], canonical_entity=aliases.get)
    original = public_domains.validate_domain_plan
    monkeypatch.setattr(public_domains, "validate_domain_plan", lambda plan, **_: original(plan, config=value))
    return value


async def make_plan(config, *, unresolved=False):
    deps = parse_dependencies(
        dict(private_memory=[], public_knowledge=[0, 1], current_input=[], control=[], unresolved_source=[]), QUERY
    )

    async def scope(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return open_scope_response(messages, [
                    dict(task_id=0, objects=[] if unresolved else ["叶澄岚", "杜木汐"]),
                    dict(task_id=1, objects=["澄岚展期"]),
                ])

    binding = await resolve_question_binding(deps, QUERY, window_tokens=65536, reviewer=scope, with_retrieval_scope=True)
    return domain_plan(binding, config), deps


def curated(query, *, top_k, object_names, domain_id, query_scope):
    assert query_scope.query == query and query_scope.domain_id == domain_id
    assert query == QUERY and object_names == ("叶澄岚", "杜木汐")
    assert domain_id == "fictional_curated"
    return dict(
        abstained=False,
        domains=["fictional_curated"],
        evidence_packets=[
            dict(
                kind="evidence",
                document_ids=["fictional-relation:7"],
                text="叶澄岚与杜木汐为同门，并非血缘亲属。出处：独立虚构卷第二节。",
            )
        ],
    )


def generic():
    return dict(
        retrieval_strategy="generic",
        abstained=False,
        results=[
            dict(
                id="doc_1_chunk_0",
                document_id=1,
                knowledge_base_id=2,
                title="独立虚构业务资料",
                content="澄岚展期费用42元，并非免费；末尾例外为停办日不计。叶澄岚与杜木汐是业务样例姓名，不是原作证明。",
            )
        ],
    )


async def test_full_binding_retains_registered_pair_business_whole_name_and_all_tasks(config):
    plan, _ = await make_plan(config)
    assert plan["mixed"]
    assert [o["authority"] for o in plan["objects"]] == ["curated_character", "curated_character", "generic_knowledge"]
    assert [t["task_id"] for t in plan["tasks"]] == [0, 1]
    assert plan["binding"]["input"]["query"] == QUERY


async def test_both_branches_called_with_complete_question_and_native_ids_kept(config):
    plan, _ = await make_plan(config)
    result = retrieve_domains(plan, QUERY, 3, curated=curated, generic=generic)
    bundle = assemble_domains(result["independent_domains"], generic())
    assert bundle["results"][0]["id"] == "doc_1_chunk_0"
    assert bundle["evidence_packets"][1]["document_ids"] == ["fictional-relation:7"]
    assert (
        "并非血缘" in bundle["context_text"]
        and "并非免费" in bundle["context_text"]
        and "停办日不计" in bundle["context_text"]
    )


@pytest.mark.parametrize("failed", ["curated_character", "generic_knowledge"])
@pytest.mark.parametrize("defect", ["provider", "timeout", "invalid"])
async def test_branch_failure_propagates_without_returning_partial_result(config, failed, defect):
    plan, _ = await make_plan(config)
    calls = []
    error = TimeoutError("Declared branch timeout") if defect == "timeout" else RuntimeError("Declared branch failure")

    def fetch(authority, *args, **kwargs):
        calls.append(authority)
        if authority == failed:
            if defect == "invalid":
                return None
            raise error
        return curated(*args, **kwargs) if authority == "curated_character" else generic()

    expected = (RuntimeError if failed == "curated_character" else TypeError) if defect == "invalid" else type(error)
    with pytest.raises(expected) as caught:
        retrieve_domains(plan, QUERY, 3,
            curated=lambda *args, **kwargs: fetch("curated_character", *args, **kwargs),
            generic=lambda: fetch("generic_knowledge"))
    assert calls == (["curated_character"] if failed == "curated_character" else ["curated_character", "generic_knowledge"])
    if defect == "invalid":
        assert failed in str(caught.value) and ("unavailable" if failed == "curated_character" else "expected a dictionary") in str(caught.value)
    else:
        assert caught.value is error


async def test_curated_abstention_does_not_erase_business(config):
    plan, _ = await make_plan(config)

    def abstention(*args, **kwargs):
        return {**curated(*args, **kwargs), "abstained": True}

    result = retrieve_domains(plan, QUERY, 3, curated=abstention, generic=generic)
    bundle = assemble_domains(result["independent_domains"], generic())
    assert not bundle["abstained"] and len(bundle["evidence_packets"]) == 1
    assert not bundle["public_domain_branches"]["curated_packet_manifest"]


async def test_business_same_name_cannot_certify_curated_object_and_final_prompt_keeps_both(config):
    plan, deps = await make_plan(config)
    binding = plan["binding"]
    ids = {o["query_text"]: o["object_id"] for o in plan["objects"]}

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["public_task_ids"] == [0, 1]
        return json.dumps(
            dict(
                decisions=[
                    dict(
                        source_id="doc_1",
                        task_ids=[0, 1],
                        object_evidence=[
                            dict(object_id=ids["叶澄岚"], source_quote="叶澄岚与杜木汐是业务样例姓名，不是原作证明。"),
                            dict(
                                object_id=ids["澄岚展期"],
                                source_quote="澄岚展期费用42元，并非免费；末尾例外为停办日不计。",
                            ),
                        ],
                    )
                ]
            )
        )

    result = retrieve_domains(plan, QUERY, 3, curated=curated, generic=generic)
    bundle = await review_public_candidates(
        result, deps, QUERY, window_tokens=65536, reviewer=reviewer, question_binding=binding
    )
    assert bundle["public_task_review"]["decisions"] == [dict(source_id="doc_1", task_ids=[1])]
    retrieval = RetrievalResult(
        status="ok",
        evidence=bundle["context_text"],
        evidence_packets=bundle["evidence_packets"],
        public_task_review=bundle["public_task_review"],
        public_task_query=QUERY,
        public_dependency_indices=(0, 1),
        public_domain_branches=bundle["public_domain_branches"],
    )
    built = build_generation_request(
        GenerationRequest(message=QUERY, retrieval=retrieval, context_window_tokens=65536, evidence_max_chars=0)
    )
    wire = "\n".join(m["content"] for m in built.messages)
    assert "并非血缘亲属" in wire and "停办日不计" in wire and "public_domains" in wire
    assert render_domains(built.retrieval)[0]["status"] == "selected_packets_admitted"
    assert render_public_tasks(built.retrieval)[0]["status"] == "handled_by_independent_curated_domain"
    assert render_public_tasks(built.retrieval)[0]["source_authority"] == "generic_knowledge"
    assert render_public_tasks(built.retrieval)[0]["applicable_object_ids"] == []


async def test_final_admission_cannot_be_forged_with_same_quote_from_other_source(config):
    plan, _ = await make_plan(config)
    result = retrieve_domains(plan, QUERY, 3, curated=curated, generic=generic)
    bundle = assemble_domains(result["independent_domains"], generic())
    original = bundle["evidence_packets"][1]
    retrieval = RetrievalResult(
        status="ok",
        public_task_query=QUERY,
        public_domain_branches=bundle["public_domain_branches"],
        admitted_evidence_packets=({**original, "document_ids": ["doc_2_chunk_0"]},),
    )
    assert render_domains(retrieval)[0]["status"] == "candidate_packets_not_admitted"
    changed = {**original, "text": original["text"].replace("并非血缘亲属", "血缘亲属")}
    assert (
        render_domains(replace(retrieval, admitted_evidence_packets=(changed,)))[0]["status"]
        == "candidate_packets_not_admitted"
    )


async def test_changed_question_ownership_cannot_reuse_actual_scope(config):
    plan, _ = await make_plan(config)
    tampered = copy.deepcopy(plan)
    tampered["objects"][-1]["authority"] = "curated_character"
    with pytest.raises(ValueError):
        public_domains.validate_domain_plan(tampered)
    with pytest.raises(ValueError):
        retrieve_domains(plan, "只问业务费用", 3, curated=curated, generic=generic)


async def test_unresolved_scope_is_retained_without_fabricating_character_binding(config):
    plan, _ = await make_plan(config, unresolved=True)
    assert not plan["mixed"] and plan["tasks"][0]["object_ids"] == []


async def test_branch_id_collision_rejected_instead_of_renaming_curated_sources(config):
    plan, _ = await make_plan(config)

    def collision(*args, **kwargs):
        value = curated(*args, **kwargs)
        value["evidence_packets"][0]["document_ids"] = ["doc_1_chunk_0"]
        return value

    result = retrieve_domains(plan, QUERY, 3, curated=collision, generic=generic)
    with pytest.raises(ValueError):
        assemble_domains(result["independent_domains"], generic())


@pytest.mark.parametrize("filters", [None, {"knowledge_base_id": 2}])
async def test_actual_api_mixed_retrieval_reads_fresh_business_after_curated_success(config, monkeypatch, filters):
    from api import generate

    installed = install_api_retrieval(config, monkeypatch, filters)
    plan, _ = await make_plan(config)
    result = await generate._retrieve_rag_bundle(QUERY, 3, filters, question_binding=plan["binding"])
    assert installed.calls == ["curated", "fresh", "generic"]
    container = result["independent_domains"]
    assert (
        container["branches"]["generic_knowledge"]["bundle"]["original_source_packets"][0]["original_body"]
        == installed.row["content"]
    )


def install_api_retrieval(config, monkeypatch, filters=None):
    from threading import RLock

    from api import generate
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    row = generic()["results"][0]
    row.update(chunk_index=0, category="独立虚构规程")
    calls = []

    def actual_curated(query, **kwargs):
        calls.append("curated")
        return curated(query, **kwargs)

    def business(query, **kwargs):
        assert query == QUERY and kwargs["filters"] == filters
        calls.append("generic")
        return dict(results=[row], abstained=False)

    character = SimpleNamespace(config=config, retrieve_with_citations=actual_curated, is_warm=lambda: True)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: character)
    monkeypatch.setattr(index_lifecycle, "_ensure_vector_index", lambda: calls.append("fresh") or True)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", 79)
    monkeypatch.setattr(index_lifecycle, "_get_rebuild_revision", lambda: 79)
    monkeypatch.setattr(
        vector_db,
        "get_vector_db",
        lambda: SimpleNamespace(
            _lock=RLock(),
            cache_generation=11,
            snapshot_validated=True,
            metadata=[row],
            _match_filters=lambda record, selected: all(record.get(key) == value for key, value in selected.items()),
        ),
    )
    monkeypatch.setattr(
        generate,
        "db",
        SimpleNamespace(
            get_knowledge_document=lambda identity: dict(
                id=identity, title=row["title"], category=row["category"], knowledge_base_id=2, content=row["content"]
            )
        ),
    )
    helper = SimpleNamespace(retrieve_with_citations=business, build_citations=lambda rows: [])
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: helper)
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    return SimpleNamespace(calls=calls, row=row, character=character, generic=helper)


def test_service_scope_retains_full_question_and_excludes_incidental_entity(config, monkeypatch):
    from knowledge.multiscale_rag import service
    from knowledge.retrieval_core.query import QueryAnalysis

    analysis = QueryAnalysis(
        original_query=QUERY,
        normalized_query=QUERY,
        entities=["澄岚", "木汐", "误匹配"],
        story_hits=[],
    )
    target = service.RoutedMultiScaleService.__new__(service.RoutedMultiScaleService)
    target.analyzer = SimpleNamespace(analyze=lambda *args, **_scope: analysis)
    target.config = config
    observed = []

    class ReachedRecall(Exception):
        pass

    def search(scoped, **kwargs):
        observed.append(scoped)
        raise ReachedRecall()

    target.retriever = SimpleNamespace(search=search)
    with pytest.raises(ReachedRecall):
        target.retrieve(QUERY, object_names=("叶澄岚", "杜木汐"))
    assert observed[0].original_query == QUERY
    assert observed[0].entities == ["澄岚", "木汐"]
    with pytest.raises(ValueError):
        target.retrieve(QUERY, object_names=("澄岚展期",))


@pytest.mark.parametrize("failed", ["curated", "generic"])
@pytest.mark.parametrize("defect", ["provider", "timeout"])
async def test_actual_chat_branch_failure_stops_review_and_answer(config, monkeypatch, semantic_plan, failed, defect):
    from fastapi import HTTPException

    from api import generate
    from db.schemas import MessageRequest
    from inference.context_budget import ReviewContextBudget
    from knowledge import public_task_evidence

    installed = install_api_retrieval(config, monkeypatch)
    planning = semantic_plan(QUERY, dict(public_knowledge=[0, 1]))
    scopes, reviews, answers = [], [], []
    error = TimeoutError("Declared branch timeout") if defect == "timeout" else RuntimeError("Declared branch failure")

    async def scope(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY
        tasks = payload["public_tasks"]
        assert len(tasks) == 2 and tasks[0]["text"] == "核对叶澄岚与杜木汐的原作关系，保留非血缘限定。"
        scopes.append(payload)
        return open_scope_response(messages, [
            dict(task_id=tasks[0]["id"], objects=["叶澄岚", "杜木汐"]),
            dict(task_id=tasks[1]["id"], objects=["澄岚展期"]),
        ])

    def unavailable(*args, **kwargs):
        assert args[0] == QUERY
        installed.calls.append(failed)
        raise error

    async def forbidden_review(*args, **kwargs):
        reviews.append(True)
        pytest.fail("No evidence review is allowed after retrieval failure")

    async def forbidden_answer(**kwargs):
        answers.append(True)
        pytest.fail("No answer is allowed after retrieval failure")

    monkeypatch.setattr(installed.character if failed == "curated" else installed.generic, "retrieve_with_citations", unavailable)
    monkeypatch.setattr(public_task_evidence, "_review", scope)
    monkeypatch.setattr(public_task_evidence, "review_public_candidates", forbidden_review)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(
        review=ReviewContextBudget(65536), window_tokens=65536, evidence_max_chars=0,
    ))
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY), None,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True), model_generate=forbidden_answer,
        )
    assert caught.value.status_code == 503
    if defect == "timeout":
        # asyncio converts the thread future's TimeoutError to a new instance.
        assert type(caught.value.__cause__) is TimeoutError and caught.value.__cause__.args == error.args
    else:
        assert caught.value.__cause__ is error
    assert len(planning) == len(scopes) == 1 and not reviews and not answers
    assert installed.calls == (["curated"] if failed == "curated" else ["curated", "fresh", "generic"])
