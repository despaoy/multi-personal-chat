"""Complete complementary alias rules settle each fact and registry independently."""

import json
import re
from datetime import datetime, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.original_sources import attach_original_sources
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对栖舟续签的费用、受理日、所需原件和预约例外，另核对杉屿登记的费用；读取我档案核对的历史提醒限制。"
REGISTRY = "当前名称登记：栖舟续签与舟渡延期是同一业务。本页仅登记名称，不载办理规则。"
DENIAL = "当前名称登记：栖舟续签不是舟渡延期，二者是不同业务，费用和办理条件不得互换。本页仅登记名称。"
RULE_A = "舟渡延期不收费，每周二受理。本页仅载费用和受理日，不证明材料或预约例外。"
RULE_B = "舟渡延期必须携带签收原件；预约不得免除原件，雨天暂停。本页不载费用和受理日。"
DIRECT = "杉屿登记收费26元，仅适用于杉屿登记，不能作为其他业务的规则。"
PRIVATE = (
    "历史自述：我的档案核对只在每周五20:10后用纯文本提醒，单次不超过16分钟，不要语音；仅为偏好，不能证明我已经执行。"
)
FIELDS = ["费用", "受理日", "所需原件", "预约例外", "杉屿登记的费用"]


def dependencies():
    return parse_dependencies(
        dict(public_knowledge=[0], private_memory=[1], current_input=[], control=[], unresolved_source=[]), QUERY
    )


def original_bundle(mode):
    bodies = [DENIAL if mode == "denial" else REGISTRY, RULE_A, RULE_B, DIRECT]
    long_index = {"registry_long": 0, "first_rule_long": 1, "second_rule_long": 2}.get(mode)
    if long_index is not None:
        bodies[long_index] += (
            "\n" + "Complete independent fictional appendix; no further rules or name relations. " * 120
        )
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立完整资料{i}",
            content=body,
            knowledge_base_id=17,
            score=0.9,
        )
        for i, body in enumerate(bodies, 1)
    ]
    raw = dict(
        results=rows,
        citations=[],
        confidence=0.9,
        abstained=False,
        source_coverage=tuple(
            dict(
                source_id=f"doc_{i}",
                source_title=row["title"],
                indexed_document_ids=[row["id"]],
                retrieved_document_ids=[row["id"]],
            )
            for i, row in enumerate(rows, 1)
        ),
    )
    return attach_original_sources(
        raw, lambda i: dict(rows[i - 1], id=i), source_budget_tokens=65536, authority_revision=5
    ), bodies


async def reviewed_bundle(mode):
    bundle, bodies = original_bundle(mode)

    async def objects(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(dict(scopes=[dict(task_id=0, objects=["栖舟续签", "杉屿登记"])]))

    async def aspects(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(
            dict(
                tasks=[
                    dict(
                        task_id=0,
                        aspects=[
                            dict(object_id="query-object:1" if i == 4 else "query-object:0", query_quote=field)
                            for i, field in enumerate(FIELDS)
                        ],
                    )
                ]
            )
        )

    async def identity(messages):
        data = json.loads(messages[-1]["content"])
        assert [source["original_body"] for source in data["sources"]] == bodies
        return json.dumps(
            dict(
                sources=[
                    dict(source_id=f"doc_{i}", purpose="identity_only" if i == 1 else "rules") for i in range(1, 5)
                ],
                relations=[
                    dict(
                        object_id="query-object:0",
                        alias="舟渡延期",
                        source_id="doc_1",
                        source_quote=DENIAL if mode == "denial" else REGISTRY,
                        relation="different" if mode == "denial" else "same",
                    )
                ],
            )
        )

    async def sources(messages):
        data = json.loads(messages[-1]["content"])
        assert [source["original_body"] for source in data["sources"]] == bodies
        rows = []
        for i in range(1, 5):
            proofs = []
            if i == 4 or i in (2, 3) and mode != "denial":
                span = next(span for span in data["source_span_catalog"] if span["source_id"] == f"doc_{i}")
                proof = dict(object_id="query-object:1" if i == 4 else "query-object:0", source_span_id=span["span_id"])
                if i != 4:
                    proof["identity_binding_id"] = data["identity_review"]["bindings"][0]["binding_id"]
                proofs.append(proof)
            rows.append(dict(source_id=f"doc_{i}", task_ids=[0] if proofs else [], object_evidence=proofs))
        return json.dumps(dict(decisions=rows))

    async def facts(messages):
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY
        quotes = [
            ("doc_2", "不收费", "negative"),
            ("doc_2", "每周二受理", "affirmative"),
            ("doc_3", "必须携带签收原件", "affirmative"),
            ("doc_3", "预约不得免除原件，雨天暂停", "negative"),
            ("doc_4", "收费26元", "affirmative"),
        ]
        return json.dumps(
            dict(
                assessments=[
                    dict(
                        aspect_id=f"fact-aspect:{i}",
                        evidence=[]
                        if mode == "denial" and i < 4
                        else [dict(source_id=sid, source_quote=quote, assertion=kind)],
                    )
                    for i, (sid, quote, kind) in enumerate(quotes)
                ]
            )
        )

    result = await review_public_candidates(
        bundle,
        dependencies(),
        QUERY,
        window_tokens=65536,
        scope_reviewer=objects,
        fact_scope_reviewer=aspects,
        identity_reviewer=identity,
        reviewer=sources,
        fact_reviewer=facts,
        span_references=True,
    )
    assert result["public_task_review"]["review_status"] == "reviewed"
    assert result["public_task_review"]["fact_review"]["review_status"] == "reviewed"
    return result, bodies


async def granted_private_context(tmp_path):
    db = SQLiteDB(tmp_path / "private.sqlite")
    scope = UserScope("web", "alias-matrix", "owner", "owner", "private")
    assert (
        db.capture_memory_source(
            character_id="fiction-role",
            platform=scope.platform,
            adapter=scope.adapter,
            sender_id=scope.sender_id,
            conversation_type=scope.conversation_type,
            conversation_id=scope.conversation_id,
            source_message_id="complete-fiction-private",
            body=PRIVATE,
            observed_at=datetime.now(timezone.utc),
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    recalled = await SourceMemoryService(repo, defer_budget=True).recall("fiction-role", scope, QUERY)
    assert not await repo.list_memory_records("fiction-role", scope, limit=None)
    return attach_sources(CompiledCharacterContext("独立测试画像", "", "", memory_status="no_match"), recalled)


@pytest.mark.parametrize("mode", ["full", "registry_long", "first_rule_long", "second_rule_long", "denial"])
async def test_complete_alias_complementary_rules_settle_each_fact_without_erasing_direct_or_private(tmp_path, mode):
    result, bodies = await reviewed_bundle(mode)
    context = await granted_private_context(tmp_path)
    packets = document_evidence_packets(result["results"]) + result["original_source_packets"]
    retrieval = RetrievalResult(
        status="ok",
        evidence="\n".join(packet["text"] for packet in packets),
        evidence_packets=packets,
        source_coverage=result["source_coverage"],
        public_task_review=result["public_task_review"],
        public_task_query=QUERY,
        public_dependency_indices=(0,),
    )
    plan = build_generation_request(
        GenerationRequest(
            message=QUERY, character_context=context, retrieval=retrieval, context_window_tokens=65536, max_tokens=2048
        )
    )
    wire = unescape("\n".join(message["content"] for message in plan.messages))
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    assert (
        match
        and json.loads(match[1])["records"][0]["text"] == PRIVATE
        and not plan.character_context.memory_packets
        and QUERY in wire
    )
    row = render_public_tasks(plan.retrieval)[0]
    facts = row["fact_coverage"]
    assert len(facts) == 5 and facts[4]["status"] == "fact_evidence_admitted" and DIRECT in wire
    expected = {
        "full": [True] * 5,
        "registry_long": [False] * 4 + [True],
        "first_rule_long": [False, False, True, True, True],
        "second_rule_long": [True, True, False, False, True],
        "denial": [False] * 4 + [True],
    }[mode]
    assert [fact["status"] == "fact_evidence_admitted" for fact in facts] == expected
    for fact, admitted in zip(facts, expected, strict=True):
        assert bool(fact["admitted_evidence"]) is admitted
    if mode == "full":
        assert all(body in wire for body in bodies) and facts[0]["admitted_evidence"][0]["assertion"] == "negative"
    if mode == "registry_long":
        assert REGISTRY not in wire and row["object_scope"] == "partial_source_object_evidence_verified"
    if mode == "denial":
        assert not result["public_task_review"]["object_scope_input"]["identity_review"]["bindings"]
    assert json.dumps(row, ensure_ascii=False) in wire
