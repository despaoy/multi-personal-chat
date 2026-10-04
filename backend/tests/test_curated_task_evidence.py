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
    missing = collect_curated_sources(
        [SimpleNamespace(document=corpus.card)], {corpus.card.id: corpus.evidence}, corpus.extractor, None
    )
    assert not missing["sources"] and missing["unavailable"][0]["status"] == "source_unavailable"


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

    result = await review_curated_bundle(bundle(corpus), plan, window_tokens=128, reviewer=forbidden)
    receipt = result["public_curated_review"]
    assert receipt["reason"] == "complete_curated_input_capacity_exceeded"
    assert receipt["payload"]["query"] == QUERY and receipt["payload"]["sources"][0]["excerpt"]["text"] == BODY
    assert BODY in result["context_text"]


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

    result = await review_curated_bundle(original, plan, window_tokens=65536, reviewer=forbidden)
    assert result["public_curated_review"]["reason"] == "complete_curated_input_capacity_exceeded"
    assert len(result["curated_source_catalog"]["sources"][0]["excerpt"]["text"].splitlines()) == 300


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


async def test_failed_review_retains_actual_raw_without_promoting_decisions(corpus):
    plan, _ = await plan_for(corpus)
    raw = '{"tasks":[]}'

    async def reviewer(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return raw

    result = await review_curated_bundle(bundle(corpus), plan, window_tokens=65536, reviewer=reviewer)
    receipt = result["public_curated_review"]
    assert receipt["status"] == "unavailable" and receipt["raw"] == raw and "decisions" not in receipt


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
