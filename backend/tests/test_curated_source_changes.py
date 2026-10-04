"""Complete fictional sources cover final source changes and shared context."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge import curated_sources, public_domains
from knowledge.curated_sources import collect_curated_sources
from knowledge.curated_task_evidence import render_curated_tasks, review_curated_bundle
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.public_question_binding import resolve_question_binding
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对林澄澈与白露川的两份约定，分别保留地点、时间、否定及例外，注明哪些内容当前能核对。"
STALE_DETAIL = "周日紫桐亭"
FIRST = "澄澈对露川说：‘我只约你周日紫桐亭见面。’\n这是澄澈的自述，不能证明已经实际会面，雨天不适用。"
SECOND = "露川对澄澈说：‘每周三黄昏只能在杏溪桥问候，雨天不见面。’\n仅限这一自述场景，不表示双方每天会面，预约不能豁免雨天限制。"
OLD_LIMITS = "旧约定只涉及周日紫桐亭，不能推出每天会面；另有周三黄昏杏溪桥问候，雨天不见面且预约不能豁免。"
BUSINESS = "晨露街续借42元，公众假期暂停，预约不能豁免盖章凭据原件。"


@pytest.fixture
def sources(tmp_path, monkeypatch):
    extractor = OriginalTextExtractor(tmp_path)
    monkeypatch.setattr(curated_sources, "_extractor", lambda: extractor)
    aliases = {"林澄澈": "澄澈", "白露川": "露川"}
    config = SimpleNamespace(domain_id="independent-change-fiction", story_titles=[], canonical_entity=aliases.get)
    validator = public_domains.validate_domain_plan
    monkeypatch.setattr(public_domains, "validate_domain_plan", lambda plan, **_: validator(plan, config=config))
    cards, evidence, paths = [], {}, []
    for number, body in enumerate([FIRST, SECOND]):
        path = tmp_path / f"original-{number}.txt"
        path.write_text(body, encoding="utf8")
        paths.append(path)
        reference = SourceReference(source_path=path.name, line_start=1, line_end=2)

        def document(kind, number=number, body=body, reference=reference):
            return KnowledgeIndexDocument(
                id=f"fiction:{kind}:{number}",
                domain_id=config.domain_id,
                document_type=kind,
                title="独立虚构人物自述",
                summary=body,
                content=body,
                embedding_text=body,
                source=reference,
                entities=["澄澈", "露川"],
                metadata=dict(viewpoint="人物自述"),
                reality_status="subjective",
            )

        card = document("relation")
        cards.append(card)
        evidence[card.id] = document("evidence")
    catalog = collect_curated_sources([SimpleNamespace(document=card) for card in cards], evidence, extractor, None)
    shared = dict(
        kind="background",
        document_ids=[],
        supporting_document_ids=[card.id for card in cards],
        text="【同场景摘要】\n" + FIRST + "\n" + SECOND,
    )
    independent = dict(
        kind="background", document_ids=[], supporting_document_ids=[cards[1].id], text="【未变化场景摘要】\n" + SECOND
    )
    packets = [dict(kind="evidence", document_ids=[card.id], text=card.content) for card in cards] + [
        shared,
        independent,
    ]
    return SimpleNamespace(
        paths=paths,
        cards=cards,
        catalog=catalog,
        config=config,
        bundle=dict(abstained=False, curated_source_catalog=catalog, evidence_packets=packets),
        shared=shared,
        independent=independent,
    )


async def reviewed(sources):
    deps = parse_dependencies(
        dict(private_memory=[], public_knowledge=[0], current_input=[], control=[], unresolved_source=[]), QUERY
    )

    async def resolver(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(dict(scopes=[dict(task_id=0, objects=["林澄澈", "白露川"])]))

    binding = await resolve_question_binding(deps, QUERY, window_tokens=65536, reviewer=resolver)
    plan = public_domains.domain_plan(binding, sources.config)

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and [source["excerpt"]["text"] for source in payload["sources"]] == [
            FIRST,
            SECOND,
        ]
        selected = [span for span in payload["source_spans"] if span["line_number"] == 1]
        assert len(selected) == 2
        return json.dumps(
            dict(
                tasks=[
                    dict(
                        task_id=0,
                        scope="attributed_statement",
                        evidence=[
                            dict(source_span_id=span["span_id"], object_ids=payload["tasks"][0]["object_ids"])
                            for span in selected
                        ],
                        limitations=OLD_LIMITS,
                    )
                ]
            )
        )

    result = await review_curated_bundle(sources.bundle, plan, window_tokens=65536, reviewer=reviewer)
    assert result["public_curated_review"]["status"] == "reviewed"
    return RetrievalResult(
        status="ok",
        evidence=result["context_text"],
        evidence_packets=tuple(result["evidence_packets"]),
        public_task_query=QUERY,
        public_curated_review=result["public_curated_review"],
    )


async def test_one_changed_original_cannot_survive_shared_scene_while_other_facts_remain(sources):
    state = await reviewed(sources)
    sources.paths[0].write_text("澄澈和露川的旧约定已修订，本段没有新地点或时间。", encoding="utf8")
    business = dict(kind="evidence", document_ids=["doc_9_chunk_0"], text=BUSINESS)
    built = build_generation_request(
        GenerationRequest(
            message=QUERY,
            retrieval=replace(state, evidence_packets=(*state.evidence_packets, business)),
            context_window_tokens=65536,
            evidence_max_chars=0,
        )
    )
    wire = "\n".join(message["content"] for message in built.messages)
    assert sources.shared not in built.retrieval.evidence_packets
    assert sources.independent in built.retrieval.evidence_packets
    assert SECOND in wire and BUSINESS in wire
    assert FIRST not in wire
    assert STALE_DETAIL not in wire


@pytest.mark.parametrize("loss", ["changed", "deleted", "not_admitted"])
async def test_partial_selected_evidence_cannot_reuse_whole_selection_scope_or_limits(sources, loss):
    state = await reviewed(sources)
    if loss == "changed":
        sources.paths[0].write_text("澄澈和露川的旧约定已修订，本段没有新地点或时间。", encoding="utf8")
    elif loss == "deleted":
        sources.paths[0].unlink()
    else:
        state = replace(
            state,
            admitted_evidence_packets=tuple(
                packet
                for packet in state.evidence_packets
                if packet.get("curated_source_id") == sources.catalog["sources"][1]["source_id"]
            ),
        )
    row = render_curated_tasks(state)[0]
    assert row["status"] == "partial_source_evidence"
    assert row["scope"] == "not_revalidated_for_partial_evidence" and row["limitations"] == ""
    assert STALE_DETAIL not in json.dumps(row, ensure_ascii=False)
    visible = [proof for proof in row["evidence"] if proof["status"] == "literal_source_evidence_admitted"]
    assert len(visible) == 1 and visible[0]["source_quote"] == SECOND.splitlines()[0]


async def test_complete_selection_preserves_original_scope_and_negation_limits(sources):
    row = render_curated_tasks(await reviewed(sources))[0]
    assert row["status"] == "related_candidate_admitted" and row["scope"] == "attributed_statement"
    assert row["limitations"] == OLD_LIMITS and len(row["evidence"]) == 2


async def test_changed_task_cannot_clear_independent_task_scope_negation_or_exception(sources):
    query = "核对林澄澈与白露川的周日约定地点时间及雨天限制。核对林澄澈与白露川的周三约定地点时间及预约限制。"
    deps = parse_dependencies(
        dict(private_memory=[], public_knowledge=[0, 1], current_input=[], control=[], unresolved_source=[]), query
    )

    async def resolver(messages):
        assert json.loads(messages[-1]["content"])["query"] == query
        return json.dumps(dict(scopes=[dict(task_id=task, objects=["林澄澈", "白露川"]) for task in [0, 1]]))

    binding = await resolve_question_binding(deps, query, window_tokens=65536, reviewer=resolver)
    plan = public_domains.domain_plan(binding, sources.config)

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and len(payload["sources"]) == 2
        return json.dumps(
            dict(
                tasks=[
                    dict(
                        task_id=task["id"],
                        scope="attributed_statement",
                        evidence=[
                            dict(source_span_id=span["span_id"], object_ids=task["object_ids"])
                            for span in payload["source_spans"]
                            if span["source_id"] == payload["sources"][task["id"]]["source_id"]
                        ],
                        limitations=[FIRST, SECOND][task["id"]],
                    )
                    for task in payload["tasks"]
                ]
            )
        )

    result = await review_curated_bundle(sources.bundle, plan, window_tokens=65536, reviewer=reviewer)
    assert result["public_curated_review"]["status"] == "reviewed"
    state = RetrievalResult(
        status="ok",
        evidence=result["context_text"],
        evidence_packets=tuple(result["evidence_packets"]),
        public_task_query=query,
        public_curated_review=result["public_curated_review"],
    )
    sources.paths[0].unlink()
    built = build_generation_request(
        GenerationRequest(message=query, retrieval=state, context_window_tokens=65536, evidence_max_chars=0)
    )
    lost, good = render_curated_tasks(built.retrieval)
    assert (
        lost["status"] == "source_changed_since_review"
        and lost["scope"] == "not_admitted"
        and lost["limitations"] == ""
    )
    assert (
        good["status"] == "related_candidate_admitted"
        and good["scope"] == "attributed_statement"
        and good["limitations"] == SECOND
    )
    assert all(proof["status"] == "literal_source_evidence_admitted" for proof in good["evidence"])
    wire = "\n".join(message["content"] for message in built.messages)
    assert FIRST not in wire and STALE_DETAIL not in wire and SECOND in wire


@pytest.mark.parametrize("identity", ["0", False])
def test_task_identity_type_difference_cannot_rebind_complete_query(identity):
    from knowledge.public_object_scope import parse_object_scopes

    raw = json.dumps(dict(scopes=[dict(task_id=identity, objects=["林澄澈", "白露川"])]))
    with pytest.raises(ValueError, match="Invalid scoped task identity"):
        parse_object_scopes(raw, QUERY, [0])
