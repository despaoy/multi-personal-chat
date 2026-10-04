"""Final public packet budgets keep complete granted private speech independent."""

import json
import re
from datetime import datetime, timezone
from html import unescape
from types import SimpleNamespace

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.generation_request import (
    GenerationRequest,
    RetrievalResult,
    build_generation_request,
    generate_character_response,
)
from knowledge import curated_sources, public_domains
from knowledge.curated_sources import collect_curated_sources
from knowledge.curated_task_evidence import render_curated_tasks, review_curated_bundle
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.public_question_binding import resolve_question_binding
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.turn_dependencies import parse_dependencies

QUERY = "核对本人档案核对的历史提醒限制，同时核对林澄澈和白露川两份独立约定的地点时间、雨天否定与预约例外，缺依据部分保持未知。"
PRIVATE_BODY = (
    "历史自述：我的档案核对只在每周四19:40后用纯文本提醒，单次不超过18分钟，不要语音；公交延误时仅作参考，偏好不能证明我已执行或完成。\n"
    + "Complete independent synthetic journal row; no additional permission.\n" * 300
    + "末尾私人限制：我没有承诺每天核对，预约不能豁免纯文本要求。"
)
LONG_TAIL = "周日只能在岚浦屋自述问候，雨天不见面，预约不能豁免雨天限制。"
SHORT = (
    "露川对澄澈自述：每周三黄昏只能在杏溪桥问候，雨天不见面。\n仅限这一自述场景，预约不能豁免雨天限制，不证明已经会面。"
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    extractor = OriginalTextExtractor(tmp_path)
    monkeypatch.setattr(curated_sources, "_extractor", lambda: extractor)
    config = SimpleNamespace(
        domain_id="independent-budget-fiction",
        story_titles=[],
        canonical_entity={"林澄澈": "澄澈", "白露川": "露川"}.get,
    )
    validator = public_domains.validate_domain_plan
    monkeypatch.setattr(public_domains, "validate_domain_plan", lambda plan, **_: validator(plan, config=config))
    return SimpleNamespace(root=tmp_path, extractor=extractor, config=config)


async def context_from_granted_source(setup):
    db = SQLiteDB(setup.root / "private.sqlite")
    scope = UserScope("web", "budget-fixture", "owner", "owner", "private")
    assert (
        db.capture_memory_source(
            character_id="role",
            platform=scope.platform,
            adapter=scope.adapter,
            sender_id=scope.sender_id,
            conversation_type=scope.conversation_type,
            conversation_id=scope.conversation_id,
            source_message_id="complete-private-source",
            body=PRIVATE_BODY,
            observed_at=datetime.now(timezone.utc),
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    result = await SourceMemoryService(repo, max_chars=16384, defer_budget=True).recall("role", scope, QUERY)
    assert result.diagnostics["status"] == "budget_omitted" and result.candidate_context and not result.context
    assert json.loads(result.candidate_context)["records"][0]["text"] == PRIVATE_BODY
    assert await repo.list_memory_records("role", scope, limit=None) == []
    return attach_sources(CompiledCharacterContext("独立测试画像", "", "", memory_status="no_match"), result)


async def public_retrieval(setup, lengths):
    sources, cards, by_parent = [], [], {}
    for number, long in enumerate(lengths):
        body = (
            (
                "澄澈向露川的独立虚构原话。\n"
                + "Neutral complete fictional scene record without extra permission. " * 160
                + "\n"
                + LONG_TAIL
            )
            if long
            else SHORT
        )
        path = setup.root / f"public-{number}.txt"
        path.write_text(body, encoding="utf8")
        reference = SourceReference(source_path=path.name, line_start=1, line_end=len(body.splitlines()))
        for kind in ["relation", "evidence"]:
            document = KnowledgeIndexDocument(
                id=f"fiction:{kind}:{number}",
                domain_id=setup.config.domain_id,
                document_type=kind,
                title="独立虚构约定",
                summary=body,
                content=body,
                embedding_text=body,
                source=reference,
                entities=["澄澈", "露川"],
                metadata=dict(viewpoint="人物自述"),
                reality_status="subjective",
            )
            if kind == "relation":
                cards.append(document)
            else:
                by_parent[cards[-1].id] = document
        sources.append(body)
    catalog = collect_curated_sources(
        [SimpleNamespace(document=card) for card in cards], by_parent, setup.extractor, None
    )
    assert (
        len(catalog["sources"]) == 2
        and not catalog["unavailable"]
        and all(not source["excerpt"]["truncated"] for source in catalog["sources"])
    )
    deps = parse_dependencies(
        dict(private_memory=[0], public_knowledge=[0], current_input=[], control=[], unresolved_source=[]), QUERY
    )

    async def resolver(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        return json.dumps(dict(scopes=[dict(task_id=0, objects=["林澄澈", "白露川"])]))

    binding = await resolve_question_binding(deps, QUERY, window_tokens=65536, reviewer=resolver)

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert [source["excerpt"]["text"] for source in payload["sources"]] == sources
        selected = [
            span
            for span in payload["source_spans"]
            if span["line_number"] in [1, len(sources[int(span["source_id"].rsplit(":", 1)[1])].splitlines())]
        ]
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
                        limitations="两份人物自述都不能证明实际会面，预约不能豁免雨天限制。",
                    )
                ]
            )
        )

    result = await review_curated_bundle(
        dict(
            abstained=False,
            curated_source_catalog=catalog,
            evidence_packets=[dict(kind="evidence", document_ids=[card.id], text=card.content) for card in cards],
        ),
        public_domains.domain_plan(binding, setup.config),
        window_tokens=65536,
        reviewer=reviewer,
    )
    assert result["public_curated_review"]["status"] == "reviewed"
    return RetrievalResult(
        status="ok",
        evidence=result["context_text"],
        evidence_packets=tuple(result["evidence_packets"]),
        public_task_query=QUERY,
        public_curated_review=result["public_curated_review"],
    )


def private_packet(plan):
    match = re.search(
        r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>",
        "\n".join(message["content"] for message in plan.messages),
        re.S,
    )
    assert match
    return json.loads(unescape(match[1]))


@pytest.mark.parametrize("lengths", [(True, True), (False, True), (False, False)])
async def test_final_public_admission_keeps_whole_granted_private_speech_and_exact_coverage(setup, lengths):
    context = await context_from_granted_source(setup)
    retrieval = await public_retrieval(setup, lengths)
    request = GenerationRequest(
        message=QUERY, character_context=context, retrieval=retrieval, context_window_tokens=65536, max_tokens=2048
    )
    plan = build_generation_request(request)
    wire = unescape("\n".join(message["content"] for message in plan.messages))
    assert private_packet(plan)["records"][0]["text"] == PRIVATE_BODY
    assert (
        plan.character_context.memory_source_status == "available"
        and plan.character_context.source_candidate_context == ""
    )
    assert not plan.character_context.memory_packets and not context.memory_packets
    assert context.memory_source_status == "budget_omitted" and context.source_candidate_context
    assert QUERY in wire and LONG_TAIL not in wire
    assert (
        sum(estimated_tokens(message["content"]) + 4 for message in plan.messages) + 2048 + CONTEXT_SAFETY_MARGIN_TOKENS
        <= 65536
    )
    row = render_curated_tasks(plan.retrieval)[0]
    if all(lengths):
        assert plan.retrieval.status == "character_abstention" and plan.retrieval.reason == "evidence_budget_exhausted"
        assert not plan.retrieval.evidence_packets and not plan.retrieval.admitted_evidence_packets
        assert row["status"] == "related_candidate_not_admitted" and row["limitations"] == ""
        assert all("source_quote" not in proof and "source_reference" not in proof for proof in row["evidence"])
        assert plan.should_generate
    elif any(lengths):
        assert (
            SHORT in wire
            and row["status"] == "partial_source_evidence"
            and row["scope"] == "not_revalidated_for_partial_evidence"
        )
        assert row["limitations"] == "" and any(
            proof["status"] == "literal_source_evidence_admitted" for proof in row["evidence"]
        )
    else:
        assert (
            SHORT in wire and row["status"] == "related_candidate_admitted" and row["scope"] == "attributed_statement"
        )
        assert row["limitations"] and all(
            proof["status"] == "literal_source_evidence_admitted" for proof in row["evidence"]
        )


async def test_public_empty_final_budget_does_not_skip_generation_with_private_sources(setup):
    context = await context_from_granted_source(setup)
    retrieval = await public_retrieval(setup, (True, True))
    observed = []

    async def generate(**kwargs):
        observed.append(kwargs["messages"])
        assert private_packet(SimpleNamespace(messages=kwargs["messages"]))["records"][0]["text"] == PRIVATE_BODY
        assert LONG_TAIL not in unescape("\n".join(message["content"] for message in kwargs["messages"]))
        return "历史原话的提醒条件仍可核对，公共约定本轮未接纳，不能补造。"

    result = await generate_character_response(
        GenerationRequest(
            message=QUERY, character_context=context, retrieval=retrieval, context_window_tokens=65536, max_tokens=2048
        ),
        generate,
    )
    assert len(observed) == 1 and result.model_invoked and not result.guard_retried and not result.guard_fallback
    assert (
        result.plan.character_context.memory_source_status == "available"
        and result.plan.retrieval.status == "character_abstention"
    )


@pytest.mark.parametrize("identity", [0, "0", False])
async def test_curated_task_identity_type_is_preserved_without_output_coercion(setup, identity):
    from knowledge.curated_task_evidence import parse_curated_review

    retrieval = await public_retrieval(setup, (False, False))
    receipt = retrieval.public_curated_review
    value = json.loads(receipt["raw"])
    value["tasks"][0]["task_id"] = identity
    raw = json.dumps(value)
    if type(identity) is int:
        checked = parse_curated_review(raw, receipt["payload"])
        assert checked == list(receipt["decisions"])
    else:
        with pytest.raises(ValueError, match="Unknown curated task or scope"):
            parse_curated_review(raw, receipt["payload"])
