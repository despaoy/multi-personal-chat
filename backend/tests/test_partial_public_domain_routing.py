"""Complete fictional requests keep known businesses beside unresolved tasks.

Resolver and retrieval components are controlled; these are routing tests,
not native authorization, vector quality, or actual model evaluations.
"""

import json
from threading import RLock
from types import SimpleNamespace

import pytest

from knowledge.public_domains import domain_plan, retrieve_domains
from knowledge.public_question_binding import resolve_question_binding
from knowledge.turn_dependencies import parse_dependencies

QUERY = (
    "核对青妃集市展期的费用、时长、截止、材料及末尾例外。核对晴汀续领的相同五项。另一公共任务缺项不能改变独立业务规则。"
)
BODY = "晴汀续领收费26元，需要2个工作日，每周四15:05截止；登记证原件和收件表原件均必需，暴雨天暂停，预约不能豁免这两项原件。"


def config():
    from knowledge.retrieval_core.domains import tsukiyashiro_kisaki_domain

    return tsukiyashiro_kisaki_domain()


async def binding(*, empty=False):
    deps = parse_dependencies(
        dict(private_memory=[], public_knowledge=[0, 1, 2], current_input=[], control=[], unresolved_source=[]), QUERY
    )

    async def resolver(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload == dict(query=QUERY, public_tasks=[dict(id=i, text=deps.segments[i]) for i in range(3)])
        return json.dumps(
            dict(
                scopes=[
                    dict(task_id=0, objects=[] if empty else ["青妃集市展期"]),
                    dict(task_id=1, objects=[] if empty else ["晴汀续领"]),
                    dict(task_id=2, objects=[]),
                ]
            )
        )

    return await resolve_question_binding(deps, QUERY, window_tokens=65536, reviewer=resolver)


@pytest.mark.parametrize("curated_error", [False, True])
async def test_api_keeps_business_full_original_beside_unresolved_task(monkeypatch, curated_error):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db
    from knowledge.multiscale_rag import runtime

    calls = []
    receipt = await binding()
    row = dict(
        id="doc_1_chunk_0",
        document_id=1,
        chunk_index=0,
        title="独立虚构业务完整规则",
        category="演练",
        knowledge_base_id=7,
        content=BODY,
    )

    def curated(query, **kwargs):
        assert query == QUERY
        calls.append("curated")
        if curated_error:
            raise RuntimeError("Controlled unresolved character branch unavailable")
        return dict(abstained=True, results=[], evidence_packets=[])

    monkeypatch.setattr(
        runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(config=config(), retrieve_with_citations=curated)
    )
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: calls.append("fresh") or True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 88)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: 88)
    monkeypatch.setattr(
        vector_db,
        "get_vector_db",
        lambda: SimpleNamespace(_lock=RLock(), cache_generation=12, snapshot_validated=True, metadata=[row]),
    )
    monkeypatch.setattr(
        generate,
        "db",
        SimpleNamespace(
            get_knowledge_document=lambda i: dict(
                id=i, title=row["title"], category=row["category"], knowledge_base_id=7, content=BODY
            )
        ),
    )
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")

    def generic(query, **kwargs):
        assert query == QUERY and kwargs["filters"] is None
        assert kwargs["additional_queries"] == ("青妃集市展期", "晴汀续领")
        calls.append("generic")
        return dict(results=[row], confidence=0.8, abstained=False)

    monkeypatch.setattr(
        rag_helper,
        "get_rag_helper",
        lambda: SimpleNamespace(retrieve_with_citations=generic, build_citations=lambda records: []),
    )
    bundle = await generate._retrieve_rag_bundle(QUERY, 3, None, question_binding=receipt)
    assert calls == ["curated", "fresh", "generic"]
    branches = bundle["independent_domains"]["branches"]
    assert branches["curated_character"]["status"] == ("unavailable" if curated_error else "retrieved")
    business = branches["generic_knowledge"]["bundle"]
    assert business["original_source_packets"][0]["original_body"] == BODY
    assert business["source_coverage"][0]["original_source_receipt"]["authority_revision"] == 88
    assert bundle["independent_domains"]["plan"]["binding"] == receipt
    assert receipt["scopes"]["task_scopes"][2]["object_ids"] == []


async def test_unresolved_branch_cannot_receive_a_business_object_grant():
    receipt = await binding()
    plan = domain_plan(receipt, config())
    seen = []

    def curated(query, **kwargs):
        assert query == QUERY and kwargs["object_names"] == ()
        seen.append("curated")
        return dict(abstained=True, results=[])

    def generic():
        seen.append("generic")
        return dict(abstained=False, results=[])

    result = retrieve_domains(plan, QUERY, 3, curated=curated, generic=generic)
    assert seen == ["curated", "generic"]
    assert result["independent_domains"]["plan"] == plan
    assert plan["tasks"][2]["object_ids"] == plan["tasks"][2]["authorities"] == []
    assert not plan["mixed"]


async def test_wholly_unresolved_question_cannot_invent_independent_business_route():
    plan = domain_plan(await binding(empty=True), config())

    def forbidden(*args, **kwargs):
        pytest.fail("No identified business object grants an independent route")

    with pytest.raises(ValueError, match="Independent retrieval"):
        retrieve_domains(plan, QUERY, 3, curated=forbidden, generic=forbidden)
