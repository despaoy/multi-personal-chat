"""Received invalid review text remains diagnostic, never an accepted proof."""

import asyncio
import json
import re
from copy import deepcopy
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
from knowledge.public_fact_coverage import fact_input, settle_fact_coverage, validate_fact_scope
from knowledge.public_object_scope import scope_input_digest
from knowledge.public_task_evidence import render_public_tasks, review_public_candidates
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对霁浦登记的费用、日期、材料和例外。核对竹原续签的费用。读取我档案核对的历史提醒限制。"
FIRST = "霁浦登记不收费，每周三受理，必须携带核验原件；雨天暂停，预约不能豁免原件要求。"
SECOND = "竹原续签收费29元，仅适用于本业务，不证明霁浦登记的任何规则。"
PRIVATE = "历史自述：我的档案核对仅在每周一19:20后使用纯文本提醒，不要语音，每次不超过14分钟；偏好不证明已经执行。"
MARKER = "DIAGNOSTIC_DO_NOT_AUTHORIZE_142"


def dependency():
    return parse_dependencies(
        dict(public_knowledge=[0, 1], private_memory=[2], current_input=[], control=[], unresolved_source=[]), QUERY
    )


def complete_sources():
    rows = [
        dict(
            id=f"doc_{i}_chunk_0",
            document_id=i,
            title=f"独立完整资料{i}",
            content=body,
            category="fiction",
            knowledge_base_id=23,
            score=0.9,
        )
        for i, body in enumerate([FIRST, SECOND], 1)
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
        raw, lambda i: dict(rows[i - 1], id=i), source_budget_tokens=65536, authority_revision=4
    )


def scope_value():
    return dict(
        tasks=[
            dict(
                task_id=0,
                aspects=[
                    dict(object_id="query-object:0", query_quote=field) for field in ["费用", "日期", "材料", "例外"]
                ],
            ),
            dict(task_id=1, aspects=[dict(object_id="query-object:1", query_quote="费用")]),
        ]
    )


def fact_value():
    evidence = [
        ("doc_1", "不收费", "negative"),
        ("doc_1", "每周三受理", "affirmative"),
        ("doc_1", "必须携带核验原件", "affirmative"),
        ("doc_1", "雨天暂停，预约不能豁免原件要求", "negative"),
        ("doc_2", "收费29元", "affirmative"),
    ]
    return dict(
        assessments=[
            dict(aspect_id=f"fact-aspect:{i}", evidence=[dict(source_id=sid, source_quote=quote, assertion=kind)])
            for i, (sid, quote, kind) in enumerate(evidence)
        ]
    )


async def run_reviews(failure=None):
    observed = {"scope_calls": 0, "fact_calls": 0, "source_calls": 0}
    received = {}

    async def objects(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(dict(scopes=[dict(task_id=0, objects=["霁浦登记"]), dict(task_id=1, objects=["竹原续签"])]))

    async def scope(messages):
        observed["scope_calls"] += 1
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        value = scope_value()
        if failure == "scope_transport":
            raise RuntimeError("transport unavailable")
        if failure == "scope_timeout":
            raise asyncio.TimeoutError
        if failure == "scope_json":
            raw = "{" + MARKER
        elif failure == "scope_type":
            value["tasks"][0]["task_id"] = "0"
            raw = json.dumps(value)
        elif failure == "scope_fence":
            raw = "```json\n" + json.dumps(value) + "\n```\n" + MARKER
        else:
            raw = json.dumps(value)
        received["scope"] = raw
        return raw

    async def sources(messages):
        observed["source_calls"] += 1
        assert MARKER not in json.dumps(messages)
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [source["original_body"] for source in data["sources"]] == [FIRST, SECOND]
        return json.dumps(
            dict(
                decisions=[
                    dict(
                        source_id=f"doc_{i}",
                        task_ids=[i - 1],
                        object_evidence=[
                            dict(
                                object_id=f"query-object:{i - 1}",
                                source_span_id=next(
                                    span["span_id"]
                                    for span in data["source_span_catalog"]
                                    if span["source_id"] == f"doc_{i}"
                                ),
                            )
                        ],
                    )
                    for i in (1, 2)
                ]
            )
        )

    async def facts(messages):
        observed["fact_calls"] += 1
        assert MARKER not in json.dumps(messages)
        data = json.loads(messages[-1]["content"])
        assert data["query"] == QUERY and [source["original_body"] for source in data["sources"]] == [FIRST, SECOND]
        if failure == "fact_transport":
            raise RuntimeError("transport unavailable")
        value = fact_value()
        if failure == "fact_json":
            raw = "{" + MARKER
        elif failure == "fact_quote":
            value["assessments"][0]["evidence"][0]["source_quote"] = MARKER + "费用999元"
            raw = json.dumps(value)
        else:
            raw = json.dumps(value)
        received["facts"] = raw
        return raw

    result = await review_public_candidates(
        complete_sources(),
        dependency(),
        QUERY,
        window_tokens=65536,
        scope_reviewer=objects,
        fact_scope_reviewer=scope,
        reviewer=sources,
        fact_reviewer=facts,
        span_references=True,
    )
    assert result["public_task_review"]["review_status"] == "reviewed" and len(result["results"]) == 2
    return result, received, observed


async def private_context(tmp_path):
    db = SQLiteDB(tmp_path / "private.sqlite")
    scope = UserScope("web", "failed-review", "owner", "owner", "private")
    assert (
        db.capture_memory_source(
            character_id="fiction-role",
            platform=scope.platform,
            adapter=scope.adapter,
            sender_id=scope.sender_id,
            conversation_type=scope.conversation_type,
            conversation_id=scope.conversation_id,
            source_message_id="complete-private",
            body=PRIVATE,
            observed_at=datetime.now(timezone.utc),
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    recall = await SourceMemoryService(repo, defer_budget=True).recall("fiction-role", scope, QUERY)
    assert not await repo.list_memory_records("fiction-role", scope, limit=None)
    return attach_sources(CompiledCharacterContext("独立测试画像", "", "", memory_status="no_match"), recall)


async def plan_from(result, tmp_path):
    packets = document_evidence_packets(result["results"]) + result["original_source_packets"]
    retrieval = RetrievalResult(
        status="ok",
        evidence="\n".join(packet["text"] for packet in packets),
        evidence_packets=packets,
        source_coverage=result["source_coverage"],
        public_task_review=result["public_task_review"],
        public_task_query=QUERY,
        public_dependency_indices=(0, 1),
    )
    plan = build_generation_request(
        GenerationRequest(
            message=QUERY,
            character_context=await private_context(tmp_path),
            retrieval=retrieval,
            context_window_tokens=65536,
            max_tokens=2048,
        )
    )
    wire = unescape("\n".join(message["content"] for message in plan.messages))
    match = re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>", wire, re.S)
    assert match and json.loads(match[1])["records"][0]["text"] == PRIVATE and not plan.character_context.memory_packets
    assert all(body in wire for body in [QUERY, FIRST, SECOND]) and MARKER not in wire
    return plan


async def test_complete_positive_keeps_all_review_raw_and_five_facts(tmp_path):
    result, received, observed = await run_reviews()
    receipt = result["public_task_review"]
    assert receipt["object_scope_input"]["fact_scope_review"]["raw"] == received["scope"]
    assert receipt["fact_review"]["raw"] == received["facts"] and observed == dict(
        scope_calls=1, fact_calls=1, source_calls=1
    )
    rows = render_public_tasks((await plan_from(result, tmp_path)).retrieval)
    assert sum(len(row["fact_coverage"]) for row in rows) == 5 and all(
        fact["status"] == "fact_evidence_admitted" for row in rows for fact in row["fact_coverage"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "scope_json",
        "scope_type",
        "scope_fence",
        "fact_json",
        "fact_quote",
        "scope_transport",
        "scope_timeout",
        "fact_transport",
    ],
)
async def test_failed_review_keeps_received_text_only_as_diagnostic_and_preserves_sources(tmp_path, failure):
    result, received, observed = await run_reviews(failure)
    receipt = result["public_task_review"]
    scope_receipt = receipt["object_scope_input"]["fact_scope_review"]
    fact_receipt = receipt["fact_review"]
    if failure.startswith("scope"):
        assert (
            scope_receipt["review_status"] == "unavailable"
            and scope_receipt["scope"] is None
            and scope_receipt["raw"] is None
        )
        assert receipt.get("fact_scope_failure_raw") == received.get("scope")
        assert (
            fact_receipt["reason"] == "fact_scope_unavailable"
            and fact_receipt["raw"] is None
            and fact_receipt["assessments"] is None
            and observed["fact_calls"] == 0
        )
    else:
        assert fact_receipt["review_status"] == "unavailable" and fact_receipt["assessments"] is None
        assert fact_receipt["raw"] == received.get("facts")
    plan = await plan_from(result, tmp_path)
    rows = render_public_tasks(plan.retrieval)
    assert all(row["status"] == "fact_review_unavailable" and not row["fact_coverage"] for row in rows)
    assert all(row["object_scope"] == "source_object_evidence_verified" for row in rows) and plan.should_generate


@pytest.mark.parametrize("stage", ["scope", "facts"])
async def test_unavailable_received_text_never_promotes_graph_or_assessments(stage):
    result, received, _ = await run_reviews()
    receipt = deepcopy(result["public_task_review"])
    payload = receipt["object_scope_input"]
    scopes = receipt["object_scopes"]
    if stage == "scope":
        payload["fact_scope_review"] = dict(
            review_status="unavailable", reason="invalid_or_incomplete_fact_scope", raw=received["scope"], scope=None
        )
        assert validate_fact_scope(payload, scopes) is None
        payload["fact_scope_review"]["scope"] = json.loads(received["scope"])
        with pytest.raises(ValueError, match="asserted graph"):
            validate_fact_scope(payload, scopes)
    else:
        fact_receipt = dict(
            receipt["fact_review"],
            review_status="unavailable",
            reason="invalid_or_incomplete_fact_evidence",
            raw=received["facts"],
            assessments=None,
        )
        fact_receipt["input_sha256"] = scope_input_digest(
            fact_input(payload, receipt["scoped_decisions"], receipt["decisions"])
        )
        settled = settle_fact_coverage(
            RetrievalResult(status="ok"),
            payload,
            scopes,
            receipt["scoped_decisions"],
            receipt["decisions"],
            fact_receipt,
        )
        assert settled == dict(review_status="unavailable", tasks={})
        fact_receipt["assessments"] = json.loads(received["facts"])
        with pytest.raises(ValueError, match="asserted proof"):
            settle_fact_coverage(
                RetrievalResult(status="ok"),
                payload,
                scopes,
                receipt["scoped_decisions"],
                receipt["decisions"],
                fact_receipt,
            )


@pytest.mark.parametrize("stage", ["scope", "facts"])
async def test_unavailable_diagnostic_rejects_non_text_payload(stage):
    result, _, _ = await run_reviews()
    receipt = deepcopy(result["public_task_review"])
    payload = receipt["object_scope_input"]
    scopes = receipt["object_scopes"]
    if stage == "scope":
        payload["fact_scope_review"] = dict(
            review_status="unavailable", reason="invalid", raw={"fake": "data"}, scope=None
        )
        with pytest.raises(ValueError, match="asserted graph"):
            validate_fact_scope(payload, scopes)
    else:
        review = dict(
            receipt["fact_review"],
            review_status="unavailable",
            reason="invalid",
            raw={"fake": "data"},
            assessments=None,
        )
        with pytest.raises(ValueError, match="asserted proof"):
            settle_fact_coverage(
                RetrievalResult(status="ok"), payload, scopes, receipt["scoped_decisions"], receipt["decisions"], review
            )
