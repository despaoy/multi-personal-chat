"""Independent source authorities for complete, source-blind public questions."""

import hashlib

from knowledge.public_question_binding import validate_question_binding

POLICY = (
    "【独立公共资料来源】public_domains按原始问题对象分别说明角色原作与业务知识库的证据范围。"
    "public_tasks及其fact_coverage只审核generic_knowledge业务来源；对curated_character对象的未知状态不能解释为角色资料缺失。"
    "角色资料是否可见看public_domains中的curated_character分支和实际角色证据。"
    "handled_by_independent_curated_domain表示该原作任务由独立角色来源处理，业务分支不负责核验；这不是没有角色证据，也不是已经验证了全部角色事实。"
    "selected_packets_admitted只证明候选片段进入请求，不证明问题全部充分、原作全文完整或事实客观真实。"
    "逐项依据对应来源回答，保留叙事、否定和条件；不得用业务名称登记确认角色别名，也不得用角色关系担保业务规定。"
    "一个来源不可用或片段未进入请求，只限制该部分，不抹掉其他独立证据或私人记忆。"
)


def domain_plan(binding, config):
    """Exact registered whole names own curated objects; retain every task."""
    scopes = validate_question_binding(binding, binding["input"])
    objects = []
    for obj in scopes["objects"]:
        name = obj["query_text"]
        canonical = config.canonical_entity(name)
        authority = "curated_character" if canonical is not None or name in config.story_titles else "generic_knowledge"
        objects.append(dict(**obj, authority=authority, canonical_entity=canonical))
    by_id = {obj["object_id"]: obj for obj in objects}
    tasks = [
        dict(**task, authorities=list(dict.fromkeys(by_id[identity]["authority"] for identity in task["object_ids"])))
        for task in scopes["task_scopes"]
    ]
    authorities = {obj["authority"] for obj in objects}
    return dict(
        binding=binding,
        domain_id=config.domain_id,
        objects=objects,
        tasks=tasks,
        mixed=authorities == {"curated_character", "generic_knowledge"},
    )


def validate_domain_plan(plan, *, config=None):
    if config is None:
        from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

        config = get_multiscale_rag_service().config
    if not isinstance(plan, dict) or domain_plan(plan["binding"], config) != plan:
        raise ValueError("Public domain ownership changed its source-blind question binding")
    return plan


def requires_independent_domains(plan):
    """Keep known business objects beside scopes that remain unresolved.

    Unresolved tasks retain the curated path, but cannot suppress identified
    generic objects or gain their object bindings. Wholly unresolved requests
    still use the strict existing route.
    """
    return plan["mixed"] or (
        any(obj["authority"] == "generic_knowledge" for obj in plan["objects"])
        and any(not task["object_ids"] for task in plan["tasks"])
    )


def retrieve_domains(plan, query, top_k, *, curated, generic):
    """One failed branch cannot erase another; never crop the original query."""
    if plan["binding"]["input"]["query"] not in query or not requires_independent_domains(plan):
        raise ValueError("Independent retrieval requires the complete mixed question")
    names = tuple(obj["query_text"] for obj in plan["objects"] if obj["authority"] == "curated_character")
    branches = {}
    for authority, retrieve in (
        ("curated_character", lambda: curated(query, top_k=top_k, object_names=names)),
        ("generic_knowledge", generic),
    ):
        try:
            bundle = retrieve()
            if not isinstance(bundle, dict):
                raise RuntimeError("Requested source authority is unavailable")
            branches[authority] = dict(status="retrieved", bundle=bundle)
        except Exception as exc:
            branches[authority] = dict(status="unavailable", error_type=type(exc).__name__, bundle={})
    business = branches["generic_knowledge"]["bundle"]
    return {
        **business,
        "independent_domains": dict(plan=plan, branches=branches),
        "retrieval_strategy": "independent_public_domains",
    }


def constrain_generic_sources(payload, plan):
    """A business source cannot certify a curated object with the same name."""
    validate_domain_plan(plan)
    if plan["binding"]["input"] != dict(query=payload["query"], public_tasks=payload["public_tasks"]):
        raise ValueError("Domain ownership discarded a question obligation")
    allowed = [obj["object_id"] for obj in plan["objects"] if obj["authority"] == "generic_knowledge"]
    payload["domain_plan"] = plan
    for source in payload["sources"]:
        source["permitted_object_ids"] = list(allowed)
    return payload


def validate_generic_sources(payload):
    plan = payload.get("domain_plan")
    if plan is None:
        return
    validate_domain_plan(plan)
    if plan["binding"]["input"] != dict(query=payload["query"], public_tasks=payload["public_tasks"]):
        raise ValueError("Domain proof changed the complete question")
    allowed = [obj["object_id"] for obj in plan["objects"] if obj["authority"] == "generic_knowledge"]
    if any(source.get("permitted_object_ids") != allowed for source in payload["sources"]):
        raise ValueError("Generic evidence escaped its source authority")


def assemble_domains(container, reviewed):
    from knowledge.evidence_packets import document_evidence_packets

    plan, branches = container["plan"], container["branches"]
    character = branches["curated_character"]["bundle"]
    curated_packets = tuple(character.get("evidence_packets") or ()) if not character.get("abstained", True) else ()
    business_packets = document_evidence_packets(reviewed.get("results", ())) + tuple(
        reviewed.get("original_source_packets") or ()
    )
    curated_ids = {identity for packet in curated_packets for identity in packet.get("document_ids", ())}
    generic_ids = {identity for packet in business_packets for identity in packet.get("document_ids", ())}
    if curated_ids & generic_ids:
        raise ValueError("Independent source identities collide")
    packets = business_packets + curated_packets
    manifest = [
        dict(
            document_ids=list(packet.get("document_ids", ())),
            text_sha256=hashlib.sha256(packet["text"].encode()).hexdigest(),
        )
        for packet in curated_packets
    ]
    domains = dict(
        plan=plan,
        curated_status=branches["curated_character"]["status"],
        generic_status=branches["generic_knowledge"]["status"],
        curated_packet_manifest=manifest,
    )
    return {
        **reviewed,
        "retrieval_strategy": "independent_public_domains",
        "abstained": not bool(packets),
        "context_text": "\n\n".join(packet["text"] for packet in packets),
        "evidence_packets": packets,
        "public_domain_branches": domains,
        "public_curated_review": character.get("public_curated_review") or {},
        "domains": character.get("domains", ()),
        "warnings": [*reviewed.get("warnings", ()), *character.get("warnings", ())],
    }


def render_domains(retrieval):
    receipt = retrieval.public_domain_branches
    if not receipt:
        return []
    plan = validate_domain_plan(receipt["plan"])
    if plan["binding"]["input"]["query"] != retrieval.public_task_query:
        raise ValueError("Domain evidence is bound to another complete question")
    packets = retrieval.admitted_evidence_packets or retrieval.evidence_packets
    manifest = receipt["curated_packet_manifest"]
    admitted = (
        sum(
            any(
                list(packet.get("document_ids", ())) == item["document_ids"]
                and hashlib.sha256(packet["text"].encode()).hexdigest() == item["text_sha256"]
                for packet in packets
            )
            for item in manifest
        )
        if retrieval.status == "ok"
        else 0
    )
    if receipt["curated_status"] not in {"retrieved", "unavailable"} or receipt["generic_status"] not in {
        "retrieved",
        "unavailable",
    }:
        raise ValueError("Invalid independent source state")
    status = (
        "unavailable"
        if receipt["curated_status"] == "unavailable"
        else "no_reliable_candidates"
        if not manifest
        else "selected_packets_admitted"
        if admitted == len(manifest)
        else "partial"
        if admitted
        else "candidate_packets_not_admitted"
    )
    return [
        dict(
            authority="curated_character",
            status=status,
            semantic_coverage="unverified",
            original_full_text="unverified",
            objects=[
                dict(query_text=o["query_text"], canonical_entity=o["canonical_entity"])
                for o in plan["objects"]
                if o["authority"] == "curated_character"
            ],
            task_ids=[t["task_id"] for t in plan["tasks"] if "curated_character" in t["authorities"]],
        ),
        dict(
            authority="generic_knowledge",
            status=receipt["generic_status"],
            coverage_details="public_tasks",
            objects=[
                dict(query_text=o["query_text"]) for o in plan["objects"] if o["authority"] == "generic_knowledge"
            ],
            task_ids=[t["task_id"] for t in plan["tasks"] if "generic_knowledge" in t["authorities"]],
        ),
    ]
