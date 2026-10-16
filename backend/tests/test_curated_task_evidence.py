"""Independent fictional originals exercise scope and final source admission."""

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge import curated_sources, public_domains
from knowledge.curated_sources import collect_curated_sources, curated_source_packet
from knowledge.curated_task_evidence import (
    CuratedTaskCapacityError,
    curated_payload,
    parse_curated_review,
    render_curated_tasks,
    review_curated_bundle,
)
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.multiscale_rag.visibility import KnowledgeBoundary
from knowledge.public_question_binding import resolve_question_binding
from knowledge.public_task_evidence import review_public_candidates
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.turn_dependencies import parse_dependencies

QUERY = "请保留本人已存的限制。核对林澄澈与白露川的关系和引用，区分人物自述与客观事实。另核对晨露街续借费用、否定和末尾例外。"
BODY = (
    "澄澈对露川说：‘我当你是朋友。’\n这只是澄澈的自述，并非已核实的现实关系。\n本章没有说明其他关系；假设不能改变原作。"
)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    path = tmp_path / "fictional-original.txt"
    path.write_text(BODY, encoding="utf8")
    extractor = OriginalTextExtractor(tmp_path)
    monkeypatch.setattr(curated_sources, "_extractor", lambda: extractor)
    aliases = {"林澄澈": "澄澈", "白露川": "露川"}
    config = SimpleNamespace(domain_id="independent-fiction", story_titles=[], canonical_entity=aliases.get)
    original = public_domains.validate_domain_plan
    monkeypatch.setattr(public_domains, "validate_domain_plan", lambda plan, **_: original(plan, config=config))
    source = SourceReference(source_path=path.name, line_start=1, line_end=3)

    def document(identity, kind, content=BODY):
        return KnowledgeIndexDocument(
            id=identity,
            domain_id=config.domain_id,
            document_type=kind,
            title="独立虚构的叙事片段",
            summary="人物自述，不核验客观关系",
            content=content,
            embedding_text=content,
            source=source,
            entities=["澄澈", "露川"],
            metadata=dict(subject="澄澈", relation="朋友", target="露川", viewpoint="人物自述"),
            reality_status="subjective",
        )

    card = document("fiction:card:1", "relation", "独立索引声明，不能单独证明关系。\n" + BODY)
    evidence = document("fiction:evidence:1", "evidence")
    catalog = collect_curated_sources([SimpleNamespace(document=card)], {card.id: evidence}, extractor, None)
    return SimpleNamespace(path=path, extractor=extractor, config=config, card=card, evidence=evidence, catalog=catalog)


async def plan_for(corpus, query=QUERY):
    dependencies = parse_dependencies(
        dict(private_memory=[0], public_knowledge=[1, 2], current_input=[], control=[], unresolved_source=[]), query
    )

    async def resolver(messages):
        assert json.loads(messages[-1]["content"])["query"] == query
        return json.dumps(
            dict(scopes=[dict(task_id=1, objects=["林澄澈", "白露川"]), dict(task_id=2, objects=["晨露街续借"])])
        )

    binding = await resolve_question_binding(dependencies, query, window_tokens=65536, reviewer=resolver)
    return public_domains.domain_plan(binding, corpus.config), dependencies


def bundle(corpus):
    return dict(
        abstained=False,
        retrieval_strategy="multi_scale_character",
        results=[corpus.card.to_dict()],
        curated_source_catalog=corpus.catalog,
        evidence_packets=[dict(kind="evidence", document_ids=[corpus.card.id], text="索引摘要不得单独证明客观关系。")],
    )


def selection(payload, scope="attributed_statement"):
    return dict(
        tasks=[
            dict(
                task_id=task["id"],
                scope=scope,
                evidence=[]
                if scope == "not_supported"
                else [dict(source_span_id=payload["source_spans"][0]["span_id"], object_ids=task["object_ids"])],
                limitations="只保留人物自述，不能确认客观现实关系，也不补齐其他关系。",
            )
            for task in payload["tasks"]
        ]
    )


async def reviewed(corpus):
    plan, _ = await plan_for(corpus)

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["sources"][0]["excerpt"]["text"] == BODY
        assert payload["sources"][0]["indexed_claim"] == corpus.card.content
        assert (
            "并非已核实" in payload["sources"][0]["excerpt"]["text"]
            and "假设不能改变原作" in payload["sources"][0]["excerpt"]["text"]
        )
        return json.dumps(selection(payload))

    return await review_curated_bundle(bundle(corpus), plan, window_tokens=65536, reviewer=reviewer)


def retrieval(result, *, packets=None):
    return RetrievalResult(
        status="ok",
        evidence=result.get("context_text", ""),
        evidence_packets=tuple(result.get("evidence_packets", ())),
        admitted_evidence_packets=tuple(packets) if packets is not None else (),
        public_task_query=QUERY,
        public_curated_review=result["public_curated_review"],
    )


def test_fresh_original_keeps_native_identity_complete_declared_span_and_unverified_claim(corpus):
    source = corpus.catalog["sources"][0]
    assert source["source_id"] == corpus.evidence.id and source["parent_id"] == corpus.card.id
    assert source["excerpt"]["text"] == BODY and not source["excerpt"]["truncated"]
    assert (
        source["indexed_claim"] == corpus.card.content
        and source["indexed_claim_status"] == "not_independently_verified"
    )
    assert curated_source_packet(source)["document_ids"] == [corpus.evidence.id]


def test_hidden_or_missing_original_cannot_become_curated_source(corpus):
    hidden = collect_curated_sources(
        [SimpleNamespace(document=corpus.card)],
        {corpus.card.id: corpus.evidence},
        corpus.extractor,
        KnowledgeBoundary("fictional-role", 2),
    )
    assert not hidden["sources"] and hidden["unavailable"][0]["status"] == "no_visible_source"
    corpus.path.unlink()
    with pytest.raises(FileNotFoundError):
        collect_curated_sources(
            [SimpleNamespace(document=corpus.card)], {corpus.card.id: corpus.evidence}, corpus.extractor, None
        )


async def test_complete_question_and_exact_registered_aliases_bind_literal_curated_tasks(corpus):
    plan, _ = await plan_for(corpus)
    payload = curated_payload(bundle(corpus), plan)
    assert payload["query"] == QUERY and [task["id"] for task in payload["tasks"]] == [1]
    assert [obj["canonical_entity"] for obj in payload["objects"]] == ["澄澈", "露川"]
    assert [span["source_quote"] for span in payload["source_spans"]] == BODY.splitlines()


async def test_attributed_source_scope_quote_and_reference_are_program_generated(corpus):
    result = await reviewed(corpus)
    receipt = result["public_curated_review"]
    assert receipt["status"] == "reviewed"
    row = render_curated_tasks(retrieval(result))[0]
    assert row["status"] == "related_candidate_admitted" and row["scope"] == "attributed_statement"
    assert row["scope_truth"] == "model_judgement_not_program_truth" and row["semantic_coverage"] == "unverified"
    assert row["evidence"][0]["source_quote"] == BODY.splitlines()[0]
    assert row["evidence"][0]["source_reference"] == "fictional-original.txt L1"


async def test_actual_generation_keeps_whole_original_and_limits_with_admitted_scope(corpus):
    result = await reviewed(corpus)
    request = GenerationRequest(
        message=QUERY, retrieval=retrieval(result), context_window_tokens=65536, evidence_max_chars=0
    )
    built = build_generation_request(request)
    wire = "\n".join(message["content"] for message in built.messages)
    assert QUERY in wire and BODY in wire and "curated_tasks" in wire
    assert "attributed_statement" in wire and "人物自述" in wire and "假设不能改变原作" in wire
    assert render_curated_tasks(built.retrieval)[0]["evidence"][0]["status"] == "literal_source_evidence_admitted"


async def test_same_quote_from_business_id_does_not_admit_curated_source(corpus):
    result = await reviewed(corpus)
    packet = curated_source_packet(corpus.catalog["sources"][0])
    forged = {**packet, "document_ids": ["doc_9_chunk_0"], "curated_source_id": "doc_9"}
    assert render_curated_tasks(retrieval(result, packets=[forged]))[0]["status"] == "related_candidate_not_admitted"
    changed = {**packet, "text": packet["text"].replace("并非已核实", "已经核实")}
    assert render_curated_tasks(retrieval(result, packets=[changed]))[0]["status"] == "related_candidate_not_admitted"


async def test_request_budget_missing_source_packet_does_not_claim_admission(corpus):
    result = await reviewed(corpus)
    other = dict(kind="evidence", document_ids=["doc_9_chunk_0"], text="晨露街续借42元，假日不办理。")
    assert render_curated_tasks(retrieval(result, packets=[other]))[0]["status"] == "related_candidate_not_admitted"
    proof = render_curated_tasks(retrieval(result, packets=[other]))[0]["evidence"][0]
    assert "source_quote" not in proof and "source_reference" not in proof
    assert render_curated_tasks(retrieval(result, packets=[other]))[0]["limitations"] == ""


async def test_original_changed_after_review_is_unknown_only_for_that_source(corpus):
    result = await reviewed(corpus)
    corpus.path.write_text(BODY + "\n后来版本另有补充。", encoding="utf8")
    state = retrieval(result)
    row = render_curated_tasks(state)[0]
    assert (
        row["status"] == "source_changed_since_review" and row["evidence"][0]["status"] == "source_changed_since_review"
    )
    business = dict(
        kind="evidence", document_ids=["doc_9_chunk_0"], text="晨露街续借42元，假日不办理，预约不能豁免原件。"
    )
    built = build_generation_request(
        GenerationRequest(
            message=QUERY,
            retrieval=replace(state, evidence_packets=(*state.evidence_packets, business)),
            context_window_tokens=65536,
            evidence_max_chars=0,
        )
    )
    assert business["text"] in "\n".join(message["content"] for message in built.messages)
    assert render_curated_tasks(built.retrieval)[0]["status"] == "source_changed_since_review"
    assert BODY not in "\n".join(message["content"] for message in built.messages)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_task",
        "generic_task",
        "unknown_span",
        "quote_rewritten",
        "generic_object",
        "unknown_scope",
        "empty_supported",
        "nonempty_unsupported",
        "duplicate_key",
    ],
)
async def test_unverifiable_or_cross_authority_review_is_rejected(corpus, defect):
    plan, _ = await plan_for(corpus)
    payload = curated_payload(bundle(corpus), plan)
    value = selection(payload)
    row = value["tasks"][0]
    if defect == "missing_task":
        value["tasks"] = []
    elif defect == "generic_task":
        row["task_id"] = 2
    elif defect == "unknown_span":
        row["evidence"][0]["source_span_id"] = "curated-span:999"
    elif defect == "quote_rewritten":
        row["evidence"][0]["source_quote"] = "伪造的关系原文"
    elif defect == "generic_object":
        row["evidence"][0]["object_ids"] = [plan["objects"][-1]["object_id"]]
    elif defect == "unknown_scope":
        row["scope"] = "objective_verified"
    elif defect == "empty_supported":
        row["evidence"] = []
    elif defect == "nonempty_unsupported":
        row["scope"] = "not_supported"
    raw = '{"tasks":[],"tasks":[]}' if defect == "duplicate_key" else json.dumps(value)
    with pytest.raises(ValueError):
        parse_curated_review(raw, payload)


async def test_changed_payload_span_or_actual_response_cannot_reuse_receipt(corpus):
    result = await reviewed(corpus)
    for field in ["span", "raw", "decision", "query"]:
        altered = copy.deepcopy(result)
        receipt = altered["public_curated_review"]
        if field == "span":
            receipt["payload"]["source_spans"][0]["source_quote"] = "虚构改写"
        elif field == "raw":
            receipt["raw"] = json.dumps(dict(tasks=[]))
        elif field == "decision":
            receipt["decisions"][0]["scope"] = "direct_statement"
        else:
            receipt["query"] += "另一个问题"
        with pytest.raises(ValueError):
            render_curated_tasks(retrieval(altered))


async def test_no_support_does_not_become_a_negative_fact(corpus):
    plan, _ = await plan_for(corpus)

    async def reviewer(messages):
        return json.dumps(selection(json.loads(messages[-1]["content"]), scope="not_supported"))

    result = await review_curated_bundle(bundle(corpus), plan, window_tokens=65536, reviewer=reviewer)
    row = render_curated_tasks(retrieval(result))[0]
    assert row["status"] == "no_related_evidence" and row["semantic_coverage"] == "unverified" and not row["evidence"]
    assert BODY in result["context_text"]


async def test_review_capacity_never_crops_complete_question_or_sources(corpus):
    plan, _ = await plan_for(corpus)

    async def forbidden(messages):
        pytest.fail("No clipped paid request is allowed")

    original = bundle(corpus)
    before = copy.deepcopy(original)
    with pytest.raises(CuratedTaskCapacityError, match="exceeds context budget"):
        await review_curated_bundle(original, plan, window_tokens=128, reviewer=forbidden)
    assert original == before and original["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODY


async def test_complete_catalogue_overflow_is_explicit_without_sending(corpus):
    corpus.path.write_text("\n".join(["澄澈和露川的独立叙事行。"] * 300), encoding="utf8")
    corpus.evidence.source.line_end = 300
    catalog = collect_curated_sources(
        [SimpleNamespace(document=corpus.card)], {corpus.card.id: corpus.evidence}, corpus.extractor, None
    )
    original = bundle(corpus)
    original["curated_source_catalog"] = catalog
    plan, _ = await plan_for(corpus)

    async def forbidden(messages):
        pytest.fail("Catalogue must not omit lines to call a model")

    before = copy.deepcopy(original)
    with pytest.raises(CuratedTaskCapacityError):
        await review_curated_bundle(original, plan, window_tokens=65536, reviewer=forbidden)
    assert original == before
    assert len(original["curated_source_catalog"]["sources"][0]["excerpt"]["text"].splitlines()) == 300


async def test_cancelled_curated_review_propagates(corpus):
    plan, _ = await plan_for(corpus)

    async def cancelled(messages):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await review_curated_bundle(bundle(corpus), plan, window_tokens=65536, reviewer=cancelled)


async def test_known_unowned_empty_review_cannot_erase_curated_proof_or_assert_business(corpus):
    plan, _ = await plan_for(corpus)
    saved_raw = None

    async def reviewer(messages):
        nonlocal saved_raw
        payload = json.loads(messages[-1]["content"])
        value = selection(payload)
        value["tasks"].append(
            dict(task_id=2, scope="not_supported", evidence=[], limitations="这不是业务资料审核结论。")
        )
        saved_raw = json.dumps(value)
        return saved_raw

    result = await review_curated_bundle(bundle(corpus), plan, window_tokens=65536, reviewer=reviewer)
    receipt = result["public_curated_review"]
    assert receipt["status"] == "reviewed" and receipt["raw"] == saved_raw
    assert receipt["unowned_empty_reviews_ignored"] == 1 and len(receipt["decisions"]) == 1
    rows = render_curated_tasks(retrieval(result))
    assert [row["task_id"] for row in rows] == [1] and rows[0]["status"] == "related_candidate_admitted"
    assert "这不是业务资料审核结论" not in json.dumps(rows, ensure_ascii=False)
    altered = copy.deepcopy(result)
    altered["public_curated_review"]["unowned_empty_reviews_ignored"] = 0
    with pytest.raises(ValueError):
        render_curated_tasks(retrieval(altered))


@pytest.mark.parametrize("defect", ["supported", "evidence", "unknown", "duplicate", "missing_owned", "boolean_id"])
async def test_unowned_review_cannot_borrow_curated_proof_or_hide_missing_tasks(corpus, defect):
    plan, _ = await plan_for(corpus)
    payload = curated_payload(bundle(corpus), plan)
    value = selection(payload)
    extra = dict(task_id=2, scope="not_supported", evidence=[], limitations="仅为空结果，不证明业务规则。")
    if defect == "supported":
        extra.update(scope="limited_context", evidence=copy.deepcopy(value["tasks"][0]["evidence"]))
    elif defect == "evidence":
        extra["evidence"] = copy.deepcopy(value["tasks"][0]["evidence"])
    elif defect == "unknown":
        extra["task_id"] = 99
    elif defect == "duplicate":
        value["tasks"].append(copy.deepcopy(extra))
    elif defect == "missing_owned":
        value["tasks"] = []
    elif defect == "boolean_id":
        extra["task_id"] = True
    value["tasks"].append(extra)
    with pytest.raises(ValueError):
        parse_curated_review(json.dumps(value), payload)


async def test_failed_review_raises_without_returning_partial_receipt(corpus):
    plan, _ = await plan_for(corpus)
    raw = '{"tasks":[]}'

    async def reviewer(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return raw

    original = bundle(corpus)
    before = copy.deepcopy(original)
    with pytest.raises(ValueError):
        await review_curated_bundle(original, plan, window_tokens=65536, reviewer=reviewer)
    assert original == before and "public_curated_review" not in original


async def test_real_mixed_review_entry_keeps_business_and_curated_authorities(corpus):
    plan, deps = await plan_for(corpus)
    business = dict(
        results=[
            dict(
                id="doc_9_chunk_0",
                document_id=9,
                knowledge_base_id=4,
                title="独立虚构续借规则",
                content="晨露街续借42元，假日不办理，预约不能豁免原件。",
            )
        ],
        abstained=False,
    )
    container = dict(
        plan=plan,
        branches=dict(
            curated_character=dict(status="retrieved", bundle=bundle(corpus)),
            generic_knowledge=dict(status="retrieved", bundle=business),
        ),
    )

    async def generic_review(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        obj = next(o["object_id"] for o in plan["objects"] if o["authority"] == "generic_knowledge")
        return json.dumps(
            dict(
                decisions=[
                    dict(
                        source_id="doc_9",
                        task_ids=[2],
                        object_evidence=[dict(object_id=obj, source_quote=business["results"][0]["content"])],
                    )
                ]
            )
        )

    async def curated_review(messages):
        return json.dumps(selection(json.loads(messages[-1]["content"])))

    result = await review_public_candidates(
        dict(independent_domains=container),
        deps,
        QUERY,
        window_tokens=65536,
        reviewer=generic_review,
        question_binding=plan["binding"],
        curated_reviewer=curated_review,
    )
    assert result["results"][0]["id"] == "doc_9_chunk_0" and result["public_curated_review"]["status"] == "reviewed"
    assert result["public_task_review"]["decisions"] == [dict(source_id="doc_9", task_ids=[2])]
    state = retrieval(result)
    state = replace(
        state,
        public_domain_branches=result["public_domain_branches"],
        public_task_review=result["public_task_review"],
        public_dependency_indices=(1, 2),
    )
    built = build_generation_request(
        GenerationRequest(message=QUERY, retrieval=state, context_window_tokens=65536, evidence_max_chars=0)
    )
    wire = "\n".join(m["content"] for m in built.messages)
    assert BODY in wire and business["results"][0]["content"] in wire
    assert render_curated_tasks(built.retrieval)[0]["status"] == "related_candidate_admitted"


@pytest.mark.parametrize("has_binding", [True, False])
async def test_pure_curated_entry_uses_actual_config_binding_without_generic_authority(
    corpus, monkeypatch, has_binding
):
    from knowledge.multiscale_rag import runtime

    query = "请保留本人已存的限制。核对林澄澈与白露川的关系和引用，区分人物自述与客观事实。"
    deps = parse_dependencies(
        dict(private_memory=[0], public_knowledge=[1], current_input=[], control=[], unresolved_source=[]), query
    )

    async def resolver(messages):
        assert json.loads(messages[-1]["content"])["query"] == query
        return json.dumps(dict(scopes=[dict(task_id=1, objects=["林澄澈", "白露川"])]))

    binding = await resolve_question_binding(deps, query, window_tokens=65536, reviewer=resolver)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(config=corpus.config))
    reviewed_payloads = []

    async def source_review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and payload["sources"][0]["excerpt"]["text"] == BODY
        assert [task["id"] for task in payload["tasks"]] == [1]
        assert all(obj["authority"] == "curated_character" for obj in payload["objects"])
        reviewed_payloads.append(payload)
        return json.dumps(selection(payload))

    original = bundle(corpus)
    result = await review_public_candidates(
        original,
        deps,
        query,
        window_tokens=65536,
        question_binding=binding if has_binding else None,
        curated_reviewer=source_review,
    )
    if has_binding:
        assert len(reviewed_payloads) == 1 and result["public_curated_review"]["status"] == "reviewed"
        assert "public_task_review" not in result and "public_domain_branches" not in result
        state = replace(retrieval(result), public_task_query=query)
        built = build_generation_request(
            GenerationRequest(message=query, retrieval=state, context_window_tokens=65536, evidence_max_chars=0)
        )
        assert render_curated_tasks(built.retrieval)[0]["status"] == "related_candidate_admitted"
    else:
        assert result is original and not reviewed_payloads and "public_curated_review" not in result


@pytest.mark.parametrize("failure", ["decode", "over_limit", "changed_during_read"])
async def test_catalogue_read_failure_is_not_partial_success_or_model_input(corpus, monkeypatch, failure):
    from knowledge.grounded_answer.service import GroundedAnswerService

    error_type, message = ValueError, "changed during original read"
    if failure == "decode":
        corpus.path.write_bytes(b"\xff")
        error_type, message = UnicodeError, "decode"
    elif failure == "over_limit":
        corpus.extractor = OriginalTextExtractor(corpus.path.parent, max_lines=1)
        message = "max_lines"
    else:
        original = corpus.extractor.extract

        def changing(source, **kwargs):
            result = original(source, **kwargs)
            corpus.path.write_text(BODY + "\n新增条件：原来源已改变。", encoding="utf-8")
            return result

        monkeypatch.setattr(corpus.extractor, "extract", changing)

    def retrieve(query, **kwargs):
        return collect_curated_sources(
            [SimpleNamespace(document=corpus.card)], {corpus.card.id: corpus.evidence}, corpus.extractor, None
        )

    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "must not generate after a source read failure"

    service = GroundedAnswerService(retriever=retrieve)
    with pytest.raises(error_type, match=message):
        await service.answer("核对来源中的人物自述和全部限制。", domain_id=corpus.config.domain_id, generate=generate)
    assert not calls


def test_catalogue_snapshot_keeps_excerpt_and_file_hash_on_same_original(corpus, monkeypatch):
    import hashlib

    from knowledge.curated_sources import validate_curated_source

    original = corpus.extractor.extract

    def transient_change(source, **kwargs):
        corpus.path.write_text("中间版本：人物说法已改变。\n限制也改变。\n不可混入原快照。", encoding="utf-8")
        try:
            return original(source, **kwargs)
        finally:
            corpus.path.write_text(BODY, encoding="utf-8")

    monkeypatch.setattr(corpus.extractor, "extract", transient_change)
    catalog = collect_curated_sources(
        [SimpleNamespace(document=corpus.card)], {corpus.card.id: corpus.evidence}, corpus.extractor, None
    )
    source = catalog["sources"][0]
    expected = hashlib.sha256(BODY.encode()).hexdigest()
    assert source["excerpt"]["text"] == BODY
    assert source["body_sha256"] == source["source_file_sha256"] == expected
    assert validate_curated_source(source) is source
    assert corpus.path.read_text(encoding="utf-8") == BODY


@pytest.mark.parametrize("failure", ["provider", "timeout", "invalid", "capacity", "source_changed", "source_deleted"])
async def test_curated_review_failure_stops_chat_before_answer(corpus, monkeypatch, failure):
    from fastapi import HTTPException

    from api import generate
    from db.schemas import MessageRequest
    from inference.context_budget import ReviewContextBudget
    from knowledge import curated_task_evidence, public_question_binding, retrieval_query_plan
    from knowledge.curated_task_evidence import CuratedTaskCapacityError
    from knowledge.multiscale_rag import runtime

    plan, dependencies = await plan_for(corpus)
    seen = dict(review=0, answer=0)
    error = asyncio.TimeoutError("Declared isolated timeout") if failure == "timeout" else RuntimeError("Declared isolated provider failure")
    budget = 1024 if failure == "capacity" else 65536

    async def planner(query, **kwargs):
        assert query == QUERY
        return retrieval_query_plan.RetrievalQueryPlan(status="applied", dependencies=dependencies)

    async def binding(deps, query, **kwargs):
        assert deps == dependencies and query == QUERY
        return plan["binding"]

    async def retrieve(query, top_k, filters, **kwargs):
        assert query == QUERY and kwargs["question_binding"] == plan["binding"]
        return bundle(corpus)

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["sources"][0]["excerpt"]["text"] == BODY
        seen["review"] += 1
        if failure in {"provider", "timeout"}:
            raise error
        if failure == "invalid":
            return '{"tasks":[]}'
        if failure == "source_changed":
            corpus.path.write_text(BODY + "\n本次测试原文版本已改变。", encoding="utf8")
        elif failure == "source_deleted":
            corpus.path.unlink()
        return json.dumps(selection(payload))

    async def answer(**kwargs):
        seen["answer"] += 1
        pytest.fail("A failed original-source review cannot generate an answer")

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(config=corpus.config, is_warm=lambda: True))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(curated_task_evidence, "_review", reviewer)
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: SimpleNamespace(
        review=ReviewContextBudget(budget), window_tokens=budget, evidence_max_chars=0,
    ))
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(
            MessageRequest(message=QUERY), None,
            runtime_config=dict(useKnowledgeBase=True, maxTokens=256), model_generate=answer,
        )
    expected = {
        "provider": RuntimeError, "timeout": asyncio.TimeoutError, "invalid": ValueError,
        "capacity": CuratedTaskCapacityError, "source_changed": curated_sources.CuratedSourceChangedError,
        "source_deleted": FileNotFoundError,
    }
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, expected[failure])
    if failure in {"provider", "timeout"}:
        assert caught.value.__cause__ is error
    assert seen == dict(review=0 if failure == "capacity" else 1, answer=0)



async def test_registered_curated_names_reach_final_prompt_with_their_actual_task(corpus):
    from html import unescape

    from inference.evidence_coverage import render_coverage
    result = await reviewed(corpus)
    state = retrieval(result)
    rows = render_curated_tasks(state)
    objects = rows[0]["objects"]
    assert [(obj["query_text"], obj["canonical_entity"]) for obj in objects] == [("林澄澈", "澄澈"), ("白露川", "露川")]
    assert all(set(obj) == {"object_id", "query_text", "canonical_entity"} for obj in objects)
    assert all(obj["query_text"] != "晨露街续借" for obj in objects)
    assert {obj["object_id"] for obj in objects} == set(rows[0]["evidence"][0]["object_ids"])
    built = build_generation_request(GenerationRequest(message=QUERY, retrieval=state, context_window_tokens=65536, evidence_max_chars=0))
    payload = json.loads(render_coverage(built.retrieval))
    assert payload["curated_tasks"][0]["objects"] == objects
    assert "不要求原文重复用户称呼" in built.messages[0]["content"]
    wire = unescape(built.messages[-1]["content"])
    coverage = wire.split('<retrieval_coverage source="application" scope="retrieved_material">', 1)[1].split("</retrieval_coverage>", 1)[0]
    assert json.loads(coverage)["curated_tasks"][0]["objects"] == objects
    assert BODY in built.messages[-1]["content"] and QUERY in built.messages[-1]["content"]
    assert rows[0]["status"] == "related_candidate_admitted"


@pytest.mark.parametrize("defect", ["canonical", "object_id", "authority"])
async def test_changed_registration_cannot_gain_a_final_name_projection(corpus, defect):
    result = await reviewed(corpus)
    altered = copy.deepcopy(result)
    obj = altered["public_curated_review"]["plan"]["objects"][0]
    obj[{"canonical": "canonical_entity", "object_id": "object_id", "authority": "authority"}[defect]] = "synthetic-unregistered"
    with pytest.raises(ValueError, match="binding"):
        render_curated_tasks(retrieval(altered))


async def test_name_registration_survives_missing_admission_without_becoming_factual_proof(corpus):
    result = await reviewed(corpus)
    state = replace(retrieval(result), evidence="", evidence_packets=(), admitted_evidence_packets=())
    row = render_curated_tasks(state)[0]
    assert [(obj["query_text"], obj["canonical_entity"]) for obj in row["objects"]] == [("林澄澈", "澄澈"), ("白露川", "露川")]
    assert row["status"] == "related_candidate_not_admitted" and row["scope"] == "not_admitted"
    assert row["semantic_coverage"] == "unverified" and row["limitations"] == ""
    assert all("source_quotes" not in proof for proof in row["evidence"])
