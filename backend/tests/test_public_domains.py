"""Fictional source authorities exercise real review and final packet admission."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge import public_domains
from knowledge.public_domains import assemble_domains, domain_plan, render_domains, retrieve_domains
from knowledge.public_question_binding import resolve_question_binding
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

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
        return json.dumps(
            dict(
                scopes=[
                    dict(task_id=0, objects=[] if unresolved else ["叶澄岚", "杜木汐"]),
                    dict(task_id=1, objects=["澄岚展期"]),
                ]
            )
        )

    binding = await resolve_question_binding(deps, QUERY, window_tokens=65536, reviewer=scope)
    return domain_plan(binding, config), deps


def curated(query, *, top_k, object_names):
    assert query == QUERY and object_names == ("叶澄岚", "杜木汐")
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
async def test_one_branch_failure_preserves_other_without_claiming_full_coverage(config, failed):
    plan, _ = await make_plan(config)

    def unavailable(*args, **kwargs):
        raise RuntimeError("fictional branch unavailable")

    result = retrieve_domains(
        plan,
        QUERY,
        3,
        curated=unavailable if failed == "curated_character" else curated,
        generic=unavailable if failed == "generic_knowledge" else generic,
    )
    container = result["independent_domains"]
    bundle = assemble_domains(container, container["branches"]["generic_knowledge"]["bundle"])
    assert not bundle["abstained"]
    assert container["branches"][failed]["status"] == "unavailable"
    assert len(bundle["evidence_packets"]) == 1


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
    from threading import RLock

    from api import generate, knowledge
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    plan, _ = await make_plan(config)
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

    monkeypatch.setattr(
        runtime,
        "get_multiscale_rag_service",
        lambda: SimpleNamespace(config=config, retrieve_with_citations=actual_curated),
    )
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: calls.append("fresh") or True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 79)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 79)
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
    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(retrieve_with_citations=business, build_citations=lambda rows: []),
    )
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    result = await generate._retrieve_rag_bundle(QUERY, 3, filters, question_binding=plan["binding"])
    assert calls == ["curated", "fresh", "generic"]
    container = result["independent_domains"]
    assert (
        container["branches"]["generic_knowledge"]["bundle"]["original_source_packets"][0]["original_body"]
        == row["content"]
    )


def test_service_scope_retains_full_question_and_excludes_incidental_entity(config, monkeypatch):
    from knowledge.multiscale_rag import service
    from knowledge.retrieval_core.query import QueryAnalysis

    analysis = QueryAnalysis(
        original_query=QUERY,
        normalized_query=QUERY,
        entities=["澄岚", "木汐", "误匹配"],
        expanded_keywords=["澄岚", "木汐", "误匹配", "关系"],
        story_hits=[],
    )
    monkeypatch.setattr(service, "analyze_explicit_domain", lambda *args: analysis)
    target = service.RoutedMultiScaleService.__new__(service.RoutedMultiScaleService)
    target.analyzer = object()
    target.config = config
    observed = []

    class ReachedRecall(Exception):
        pass

    def search(scoped, **kwargs):
        observed.append(scoped)
        raise ReachedRecall()

    monkeypatch.setattr(service, "choose_card_types", lambda *args: service.CARD_TYPES)
    target.retrievers = {service.CARD_TYPES: SimpleNamespace(search=search)}
    with pytest.raises(ReachedRecall):
        target.retrieve(QUERY, object_names=("叶澄岚", "杜木汐"))
    assert observed[0].original_query == QUERY
    assert observed[0].entities == ["澄岚", "木汐"] and "误匹配" not in observed[0].expanded_keywords
    with pytest.raises(ValueError):
        target.retrieve(QUERY, object_names=("澄岚展期",))
