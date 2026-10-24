"""Complete synthetic originals exercise semantic ownership before ranking."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_multiscale_card_recall import ControlledRanking, arrange

from knowledge.multiscale_rag.runtime import _retrieval_index
from knowledge.multiscale_rag.service import RoutedMultiScaleService
from knowledge.public_domains import domain_plan, retrieve_curated_scope
from knowledge.public_question_binding import QuestionBindingReviewError, resolve_question_binding
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.retrieval_scope import scope_for_plan
from knowledge.turn_dependencies import parse_dependencies, query_segments

QUERY = "不要只找林栖和穆远的关系；还要核对林栖与其导师的任命经过。"
MENTOR = "林栖与导师程砚为师生；任命由程砚提出并签字，穆远没有提名权。"


def corpus(root):
    rig = arrange(root)
    rig.config.aliases["程砚"] = "程砚"
    docs = list(rig.documents)
    for identity, kind, subject, target, body in [
        ("mentor", "relation", "林栖", "程砚", MENTOR),
        ("other-pair", "relation", "穆远", "程砚", "穆远与程砚登记为同学，林栖不在这份同学登记中。"),
        ("other-owner", "fact", "程砚", "", "程砚登记为虚构北潭馆的馆长；林栖是他的学生，不担任馆长。"),
    ]:
        path = root / (identity + ".txt")
        path.write_text(body)
        source = SourceReference(path.name, 1, 1)
        docs.append(
            KnowledgeIndexDocument(
                identity,
                rig.config.domain_id,
                kind,
                identity,
                body,
                body,
                body,
                source=source,
                entities=["林栖", "穆远", "程砚"],
                metadata=dict(subject=subject, target=target),
            )
        )
        docs.append(
            KnowledgeIndexDocument(
                "source-" + identity,
                rig.config.domain_id,
                "evidence",
                identity,
                body,
                body,
                body,
                source=source,
                entities=["林栖", "穆远", "程砚"],
                metadata=dict(parent_id=identity),
            )
        )
    rank = ControlledRanking(["mentor", "other-pair", "other-owner", "relation", "fact", "event", "scene"])
    routed = RoutedMultiScaleService(
        rig.config,
        _retrieval_index(rig.config.domain_id, docs, np.ones((len(docs), 384), dtype=np.float32) / np.sqrt(384)),
        rig.embedding,
        all_documents=docs,
        source_extractor=rig.service.extractor,
        reranker=rank,
    )
    return SimpleNamespace(config=rig.config, service=routed, rank=rank, embedding=rig.embedding)


async def bound_plan(rig, query, rows, *, groups=None, with_scope=True):
    ids = list(range(len(query_segments(query))))
    deps = parse_dependencies(
        groups or dict(public_knowledge=ids, private_memory=[], current_input=[], control=[], unresolved_source=[]),
        query,
    )

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        if with_scope:
            assert "scope_policy" in payload
        return json.dumps(dict(scopes=rows))

    binding = await resolve_question_binding(
        deps, query, window_tokens=65536, reviewer=reviewer, with_retrieval_scope=with_scope
    )
    return domain_plan(binding, rig.config)


async def test_multitask_open_scope_keeps_existing_complete_mentor_evidence(tmp_path):
    rig = corpus(tmp_path)
    plan = await bound_plan(
        rig,
        QUERY,
        [
            dict(task_id=0, objects=["林栖", "穆远"], retrieval_scope="direct_relation"),
            dict(task_id=1, objects=["林栖"], retrieval_scope="open"),
        ],
    )
    result = rig.service.retrieve(
        QUERY, object_names=("林栖", "穆远"), query_scope=scope_for_plan(plan, QUERY), top_k=3
    )
    assert "mentor" in rig.rank.pools[0] and result["results"][0]["id"] == "mentor"
    assert any(source["excerpt"]["text"] == MENTOR for source in result["curated_source_catalog"]["sources"])
    assert [rule["mode"] for rule in result["query_scope"]["rules"]] == ["direct_relation", "open"]
    assert result["identity_task"] == {} and result["identity_subtask"] == {}


async def test_direct_pair_uses_bound_scope_with_opaque_wording_and_structured_endpoints(tmp_path):
    rig = corpus(tmp_path)
    query = "林栖、穆远：请对照两者的直接关联，给出原文。"
    plan = await bound_plan(rig, query, [dict(task_id=0, objects=["林栖", "穆远"], retrieval_scope="direct_relation")])
    result = rig.service.retrieve(query, query_scope=scope_for_plan(plan, query), top_k=3)
    assert "mentor" not in rig.rank.pools[0] and "other-pair" not in rig.rank.pools[0]
    assert "relation" in rig.rank.pools[0]
    assert result["query_scope"]["excluded_candidates"] == 2
    assert all(row["id"] not in {"mentor", "other-pair"} for row in result["relation_timeline"])


async def test_two_closed_tasks_preserve_their_union_not_an_incidental_third_pair(tmp_path):
    rig = corpus(tmp_path)
    query = "核对林栖与穆远的直接关联；另核对林栖与程砚的直接关联。"
    rows = [
        dict(task_id=0, objects=["林栖", "穆远"], retrieval_scope="direct_relation"),
        dict(task_id=1, objects=["林栖", "程砚"], retrieval_scope="direct_relation"),
    ]
    plan = await bound_plan(rig, query, rows)
    result = rig.service.retrieve(query, query_scope=scope_for_plan(plan, query), top_k=5)
    assert {"mentor", "relation"} <= set(rig.rank.pools[0]) and "other-pair" not in rig.rank.pools[0]
    assert result["query_scope"]["excluded_candidates"] == 1


@pytest.mark.parametrize("mode", ["identity", "independent_identity"])
async def test_identity_scope_checks_actual_owner_and_preserves_history_policy_contract(tmp_path, mode):
    rig = corpus(tmp_path)
    query = "请给出林栖的登记名片，保留原文边界。"
    plan = await bound_plan(rig, query, [dict(task_id=0, objects=["林栖"], retrieval_scope=mode)])
    result = rig.service.retrieve(query, query_scope=scope_for_plan(plan, query), top_k=3)
    assert "other-owner" not in rig.rank.pools[0] and "fact" in rig.rank.pools[0]
    expected = {"query": query, "subject": "林栖"}
    assert result["identity_task"] == (expected if mode == "independent_identity" else {})
    assert result["identity_subtask"] == (expected if mode == "identity" else {})


@pytest.mark.parametrize("defect", ["missing", "unknown", "identity_count", "relation_count", "private_independence"])
async def test_required_semantic_field_failure_never_becomes_an_open_scope(tmp_path, defect):
    rig = corpus(tmp_path)
    query = "林栖与穆远的登记信息。"
    row = dict(task_id=0, objects=["林栖", "穆远"], retrieval_scope="open")
    groups = None
    if defect == "missing":
        del row["retrieval_scope"]
    elif defect == "unknown":
        row["retrieval_scope"] = "guess"
    elif defect == "identity_count":
        row["retrieval_scope"] = "identity"
    elif defect == "relation_count":
        row.update(objects=["林栖"], retrieval_scope="direct_relation")
    else:
        row.update(objects=["林栖"], retrieval_scope="independent_identity")
        groups = dict(public_knowledge=[0], private_memory=[0], current_input=[], control=[], unresolved_source=[])
    with pytest.raises(QuestionBindingReviewError, match="invalid_or_incomplete_review"):
        await bound_plan(rig, query, [row], groups=groups)
    assert rig.embedding.calls == [] and rig.rank.pools == []


@pytest.mark.parametrize("changed", ["query", "domain"])
async def test_scope_from_another_request_cannot_enter_retrieval(tmp_path, changed):
    rig = corpus(tmp_path)
    query = "林栖的登记信息。"
    plan = await bound_plan(rig, query, [dict(task_id=0, objects=["林栖"], retrieval_scope="identity")])
    scope = scope_for_plan(plan, query)
    scope = replace(scope, query="另一问题。") if changed == "query" else replace(scope, domain_id="other")
    with pytest.raises(ValueError, match="another request or domain"):
        rig.service.retrieve(query, query_scope=scope)
    assert rig.embedding.calls == [] and rig.rank.pools == []


async def test_names_only_binding_cannot_silently_supply_required_character_scope(tmp_path):
    rig = corpus(tmp_path)
    query = "林栖的登记信息。"
    plan = await bound_plan(rig, query, [dict(task_id=0, objects=["林栖"])], with_scope=False)
    retrieve = Mock()
    with pytest.raises(ValueError, match="requires a semantic task scope"):
        retrieve_curated_scope(plan, query, 3, curated=retrieve)
    retrieve.assert_not_called()


@pytest.mark.parametrize("kind,mode,names", [
    ("fact", "identity", ("林栖",)),
    ("relation", "direct_relation", ("林栖", "穆远")),
])
def test_scope_retains_complete_source_beyond_unfiltered_candidate_window(tmp_path, kind, mode, names):
    from knowledge.retrieval_scope import QueryScope

    rig = arrange(tmp_path)
    docs = [doc for doc in rig.documents if doc.id in {kind, "source-" + kind}]
    original = docs[0].content
    query = "核对登记资料，按指定对象范围提供完整依据。"
    for i in range(80):
        body = f"登记资料第{i}项：程砚与顾青登记为同学；此材料不说明其他人的身份或关系。"
        source = SourceReference(f"distractor-{i}.txt", 1, 1)
        (tmp_path / source.source_path).write_text(body, encoding="utf-8")
        identity = f"distractor-{i}"
        docs.append(KnowledgeIndexDocument(identity, rig.config.domain_id, kind, query, body, body, body,
                    entities=["程砚", "顾青"], source=source, metadata=dict(subject="程砚", target="顾青")))
        docs.append(KnowledgeIndexDocument("source-" + identity, rig.config.domain_id, "evidence", "登记原文",
                    body, body, body, source=source, metadata=dict(parent_id=identity)))
    vectors = np.ones((len(docs), 384), dtype=np.float32) / np.sqrt(384)
    vectors[:2] = 0
    vectors[:2, 0] = 1
    ranking = ControlledRanking([kind])
    service = RoutedMultiScaleService(rig.config, _retrieval_index(rig.config.domain_id, docs, vectors),
                rig.embedding, all_documents=docs, source_extractor=rig.service.extractor, reranker=ranking)
    scope = QueryScope(rig.config.domain_id, query, ((0, mode, names),))
    result = service.retrieve(query, query_scope=scope, top_k=1)
    assert ranking.pools == [[kind]]
    assert [row["id"] for row in result["results"]] == [kind]
    assert any(row["excerpt"]["text"] == original for row in result["curated_source_catalog"]["sources"])
    assert rig.embedding.calls == [query]
