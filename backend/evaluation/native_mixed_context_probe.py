"""Complete private constraints with long native history and knowledge retrieval."""

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
from dataclasses import asdict
from html import unescape
from pathlib import Path


def audit_mixed_wire(proof, calls, fixture):
    source = fixture["cases"][0]["message"]
    last = proof["generation"][-1]
    answer = [c for c in calls[slice(*last["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
    if len(answer) != 1:
        return {"one_actual_mixed_answer": False}
    messages = answer[0]["request"]["messages"]
    wire = unescape(messages[-1]["content"])
    all_wire = "\n".join(unescape(m["content"]) for m in messages)
    memory = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    knowledge = re.search(r"<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>", wire, re.S)
    packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith("- {")] if memory else []
    related = [p for p in packets if source in p.get("evidence", [])]
    reply = last["response"].get("reply", "")
    doc = fixture["documents"][0]
    diag = proof["prepared_diagnostics"][-1]
    return dict(
        one_actual_mixed_answer=True,
        complete_private_source_persisted=any(json.loads(row.get("evidence_json") or "[]") == [source] for row in proof["claims"]),
        complete_private_source_in_selected_memory=bool(related),
        complete_private_source_in_actual_history=any(m.get("role") == "user" and m.get("content") == source for m in messages),
        complete_knowledge_source_in_final_evidence=bool(knowledge) and doc["content"] in knowledge[1],
        private_source_outside_knowledge=bool(knowledge) and source not in knowledge[1],
        knowledge_source_outside_private_memory=bool(memory) and doc["content"] not in memory[1],
        materials_not_system_rules=all(source not in m["content"] and doc["content"] not in m["content"] for m in messages if m["role"] == "system"),
        all_twelve_complete_history_sources_in_actual_wire=all(turn["message"] in all_wire for turn in fixture["history_turns"]),
        selection_succeeded=diag["selection_status"] == "selected",
        semantic_review_succeeded=diag["semantic_status"] == "applied",
        policy_succeeded=diag["policy_status"] == "applied",
        current_rain_and_missing_confirmation_explained=bool(re.search(r"雨", reply)) and bool(re.search(r"(?:没|未|不).{0,12}(?:书面|确认)|(?:书面|确认).{0,12}(?:没|未|不)", reply)),
        personal_trip_not_authorized=bool(re.search(r"不.{0,8}(?:出门|出行|参加)|取消.{0,8}(?:出行|计划)|(?:不能|无法|不满足).{0,12}(?:条件|出门|出行)", reply)),
        public_class_still_runs=bool(re.search(r"(?:课程|工坊|开课|纸雕课).{0,30}(?:照常|不停|不取消|正常)|(?:照常|正常).{0,12}(?:开课|举办|上课)", reply)),
        no_private_attendance_invented=bool(re.search(r"(?:尚未|没有|还没|未曾).{0,10}(?:预约|报名|参加|上课)", reply)),
        course_citation_provenance=any(c.get("source_id") == "doc_1_chunk_0" and c.get("source_title") == doc["title"] for c in last["response"].get("citations", [])),
    )


def audit_window_boundary_wire(proof, calls, fixture):
    """At the serving boundary, full memory must survive whole-turn pruning."""
    from inference.context_budget import ReviewContextBudget

    checks = audit_mixed_wire(proof, calls, fixture)
    # This case deliberately reaches the budget boundary: older history may
    # yield, but the full authorized private source must remain in memory.
    checks.pop("complete_private_source_in_actual_history", None)
    checks.pop("all_twelve_complete_history_sources_in_actual_wire", None)
    source = fixture["cases"][0]["message"]
    final = [c for c in calls[slice(*proof["generation"][-1]["cloud_call_range"])]
             if c["request"].get("max_tokens") == 1024]
    checks["actual_answer_fits_serving_budget"] = len(final) == 1 and ReviewContextBudget(65536).fits(final[0]["request"]["messages"], 1024)
    checks["history_really_trimmed_at_boundary"] = len(final) == 1 and not any(m.get("content") == source for m in final[0]["request"]["messages"])
    reply = proof["generation"][-1]["response"].get("reply", "")
    checks["private_no_rain_condition_recovered"] = bool(re.search(r"不下雨|无雨|没有下雨|未下雨", reply))
    checks["historical_no_appointment_recovered"] = bool(re.search(r"(?:尚未|没有|还没|未曾|未).{0,10}(?:提交预约|预约|报名)", reply))
    checks["historical_no_attendance_recovered"] = bool(re.search(r"(?:尚未|没有|还没|未曾|未).{0,10}(?:参加|上课)", reply))
    return checks


def audit_citation_namespace(messages):
    """Read the application example, not marker-like user/source text."""
    found = [key for m in messages if m.get('role') == 'system'
             for key in re.findall(r'本轮来源标记示例：\[\[cite:([0-9a-f]{12}):S1\]\]', m['content'])]
    return found[-1] if found else ''


def audit_source_keys(raw, namespace=''):
    if namespace:
        from inference.answer_citations import citation_keys

        return [key.removeprefix('S') for key in citation_keys(raw, namespace)]
    # Historical saved requests used legacy application markers. Current
    # request namespaces never accept these user-controlled literal strings.
    return re.findall(r'\[S(\d{1,2})\]', raw)


def audit_literal_citation_wire(proof, calls, fixture):
    last = proof['generation'][-1]
    answer = [c for c in calls[slice(*last['cloud_call_range'])] if c['request'].get('max_tokens') == 1024]
    if len(answer) != 1:
        return {'one_actual_literal_answer': False}
    messages = answer[0]['request']['messages']
    wire = unescape(messages[-1]['content'])
    namespace = audit_citation_namespace(messages)
    raw = answer[0]['response']['choices'][0]['message']['content']
    reply = last['response'].get('reply', '')
    citations = last['response'].get('citations') or []
    keys = audit_source_keys(raw, namespace)
    return dict(
        one_actual_literal_answer=True,
        complete_current_query_reached_actual_model=fixture['cases'][0]['message'] in wire,
        all_complete_source_documents_reached_model=all(d['content'] in wire for d in fixture['documents']),
        request_specific_namespace_present=bool(namespace),
        raw_model_kept_literal_first_line=bool(raw) and raw.splitlines()[0].strip() == '[S1]',
        final_api_kept_literal_first_line=bool(reply) and reply.splitlines()[0].strip() == '[S1]',
        literal_first_line_cannot_bind_source=bool(namespace) and not audit_source_keys(raw.splitlines()[0], namespace),
        public_number_recovered='YY-573-R' in reply,
        only_course_source_returned=[c.get('source_id') for c in citations] == ['doc_1_chunk_0'],
        course_title_from_authoritative_document=bool(citations) and citations[0].get('source_title') == fixture['documents'][0]['title'],
        citations_bound_from_actual_owned_markers=bool(citations) and all(c.get('key', '').removeprefix('S') in keys for c in citations),
        owned_transport_removed_from_visible_reply=bool(namespace) and ('[[cite:' + namespace + ':') not in reply,
        source_material_not_system_instruction=all(all(d['content'] not in m['content'] for d in fixture['documents']) for m in messages if m['role'] == 'system'),
    )


def audit_citation_precision_wire(proof, calls, fixture):
    """Every returned source must be bound from a real answer marker."""
    checks = {}
    for case, generation in zip(fixture["cases"], proof["generation"]):
        citations = generation["response"].get("citations") or []
        actual = [c.get("source_id") for c in citations]
        checks[case["id"] + "_only_answer_used_sources"] = set(actual) == set(case["expected_citation_ids"]) and len(actual) == len(set(actual))
        answer = [c for c in calls[slice(*generation["cloud_call_range"])] if c["request"].get("max_tokens") == 1024]
        raw = answer[-1]["response"]["choices"][0]["message"]["content"] if answer else ""
        keys = audit_source_keys(raw, audit_citation_namespace(answer[-1]["request"]["messages"]) if answer else "")
        checks[case["id"] + "_returned_keys_from_actual_model"] = bool(citations) and all(c.get("key", "")[1:] in keys for c in citations)
        wire = unescape(answer[-1]["request"]["messages"][-1]["content"]) if answer else ""
        checks[case["id"] + "_all_complete_sources_still_offered"] = all(d["content"] in wire for d in fixture["documents"])
    return checks


def private_public_marker_checks(raw, private_code, public_code, public_keys, namespace=""):
    """Separate explicit clauses; a semicolon can change the source subject."""
    clauses = re.split(r"[。！？\n；;]", raw)
    private = [c for c in clauses if private_code in c or
               ("你" in c and re.search(r"收到|参加|出发|私人", c))]
    public = [c for c in clauses if public_code in c]
    def markers(clause):
        return set(audit_source_keys(clause, namespace))
    allowed = {key.removeprefix('S') for key in public_keys}
    return dict(
        private_receipt_present=any(private_code in c for c in private),
        private_claims_not_cited_as_public=bool(private) and all(not markers(c) for c in private),
        public_code_has_own_authorized_citation=bool(public) and any(markers(c) & allowed for c in public),
    )


def audit_private_public_citation_wire(proof, calls, fixture):
    source = fixture['cases'][0]['message']
    last = proof['generation'][-1]
    answer = [c for c in calls[slice(*last['cloud_call_range'])] if c['request'].get('max_tokens') == 1024]
    if len(answer) != 1:
        return {'one_actual_mixed_answer': False}
    messages = answer[0]['request']['messages']
    wire = unescape(messages[-1]['content'])
    memory = re.search(r'<character_memory[^>]*>\n(.*?)\n</character_memory>', wire, re.S)
    knowledge = re.search(r'<retrieved_evidence[^>]*>\n(.*?)\n</retrieved_evidence>', wire, re.S)
    packets = [json.loads(line[2:]) for line in memory[1].splitlines() if line.startswith('- {')] if memory else []
    private_code, public_code = fixture['cases'][-1]['expected']
    doc = fixture['documents'][0]
    citations = last['response'].get('citations') or []
    course = [c for c in citations if c.get('source_id') == 'doc_1_chunk_0' and c.get('source_title') == doc['title']]
    raw = answer[0]['response']['choices'][0]['message']['content']
    diag = proof['prepared_diagnostics'][-1]
    reply = last['response'].get('reply', '')
    checks = dict(
        one_actual_mixed_answer=True,
        complete_private_source_persisted=any(source in json.loads(row.get('evidence_json') or '[]') for row in proof['claims']),
        complete_private_source_in_selected_memory=any(source in p.get('evidence', []) for p in packets),
        complete_private_source_in_actual_history=any(m.get('role') == 'user' and m.get('content') == source for m in messages),
        complete_public_course_in_actual_evidence=bool(knowledge) and doc['content'] in knowledge[1],
        private_receipt_outside_public_evidence=bool(knowledge) and private_code not in knowledge[1] and source not in knowledge[1],
        public_course_outside_private_memory=bool(memory) and public_code not in memory[1] and doc['content'] not in memory[1],
        materials_not_system_rules=all(source not in m['content'] and doc['content'] not in m['content'] for m in messages if m['role'] == 'system'),
        selected_actual_private_memories=diag['selection_status'] == 'selected' and bool(diag['used_memory_ids']),
        semantic_review_applied=diag['semantic_status'] == 'applied',
        decision_policy_applied=diag['policy_status'] == 'applied',
        both_distinct_codes_answered=private_code in reply and public_code in reply,
        received_confirmation_preserved=bool(re.search(r'你.{0,20}(?:已经|已|刚).{0,6}收到.{0,6}(?:书面|确认)', reply)),
        no_attendance_promoted_from_confirmation=bool(re.search(r'(?:尚未|还没|没有|未曾|未).{0,10}(?:参加|上课)', reply)),
        only_authoritative_course_cited=len(citations) == len(course) == 1,
        visible_reply_has_no_internal_citation_marker=('[[cite:' + audit_citation_namespace(messages) + ':') not in reply if audit_citation_namespace(messages) else not re.search(r'\[S\d{1,2}\]', reply),
    )
    checks.update(private_public_marker_checks(raw, private_code, public_code, {c.get('key', '') for c in course}, audit_citation_namespace(messages)))
    return checks


def expand_history(fixture):
    """Expand fully specified synthetic history without storing repeated text."""
    import hashlib

    template = fixture["history_template"]
    turns = []
    for turn in range(1, template["turn_count"] + 1):
        message = template["heading"].format(turn=turn) + "\n" + "\n".join(
            template["line"].format(turn=turn, record=record)
            for record in range(template["records_per_turn"]))
        turns.append(dict(message=message, reply=template["reply"].format(turn=turn)))
    encoded = json.dumps(turns, ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != template["expanded_sha256"]:
        raise ValueError("Synthetic history differs from the complete source manifest")
    return turns


def capture_storage_proof(proof, output):
    """Snapshot adapter-owned facts while application resources remain open."""
    from api.knowledge import _get_expected_chunk_count
    from knowledge.vector_db import VectorDatabase

    proof["persisted_vector_stats"] = VectorDatabase(db_path=str(output / "vectors")).get_stats()
    proof["persisted_valid_chunk_count"] = _get_expected_chunk_count()
    (output / "checkpoint.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))


def isolated_probe_paths(root, run_label, api_key_file):
    root = Path(root).resolve()
    allowed = Path("/home/boot/lhm/multipersonal-runtime/evaluations").resolve()
    if root.parent != allowed or not re.fullmatch(r"r148pg\.[a-zA-Z0-9_.-]+", root.name):
        raise ValueError("This probe only accepts a stage-three disposable cluster")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", run_label):
        raise ValueError("Invalid isolated run label")
    key_path = Path(api_key_file).resolve()
    if key_path.parent != Path("/home/boot/lhm/multipersonal-runtime/config").resolve():
        raise ValueError("API key must remain in the private runtime config directory")
    if not (root / "data").is_dir() or not (root / "socket").is_dir():
        raise ValueError("Create the disposable cluster before running this probe")
    return root, root / run_label, key_path


async def main(args):
    from evaluation.conversation_source_probe import verify_cluster

    if sum((args.window_boundary, args.citation_precision, args.private_public_citations, args.literal_citations, args.memory_correction, args.memory_history, args.memory_owner, args.memory_friend, args.memory_erasure, args.memory_scope, args.memory_collision, args.memory_replay, args.memory_race, args.memory_capacity, args.memory_capture_failure, args.memory_history_feedback, args.memory_late_capture)) > 1:
        raise ValueError("Choose one specific probe scenario")

    if args.memory_replay or args.memory_race or args.memory_capacity or args.memory_capture_failure or args.memory_history_feedback or args.memory_late_capture:
        args.memory_collision = True

    if args.late_cold_query and not args.memory_late_capture:
        raise ValueError("The new cold recall requires the isolated late capture probe")

    if args.capture_retry and not args.memory_capture_failure:
        raise ValueError("Capture retry requires the isolated capture failure probe")

    if args.capacity_cold_query and not args.memory_capacity:
        raise ValueError("The new cold recall is scoped to the capacity probe")

    if args.enable_source_recall and not (args.memory_friend or args.memory_erasure):
        raise ValueError("The explicit source-recall variant requires the friend-only scenario")
    source_recall_enabled = args.memory_owner or args.memory_erasure or args.memory_scope or args.memory_collision or (args.memory_friend and args.enable_source_recall)
    reuse_native_fixture = args.memory_history or args.memory_owner or args.memory_friend or args.memory_erasure or args.memory_scope or args.memory_collision
    cold_memory = args.memory_correction or reuse_native_fixture
    history_ablation = cold_memory and not (args.memory_scope or args.memory_collision)

    ROOT, OUT, key_path = isolated_probe_paths(args.root, args.run_label, args.api_key_file)
    bootstrap_url = "postgresql+asyncpg://boot@/postgres?host=" + str(ROOT / "socket") + "&port=25433"
    await verify_cluster(bootstrap_url, ROOT / "data")
    database_name = "stage3_" + args.run_label.replace("-", "_")
    source_label = "stage24-source-erasure" if args.memory_replay else "stage22-owner-only" if args.memory_friend or args.memory_erasure or args.memory_scope or args.memory_collision else "stage20-cold-fixed"
    source_account_username = "stage20-cold-fixed"
    source_database = "stage3_" + source_label.replace("-", "_")
    if reuse_native_fixture:
        source_proof = json.loads((ROOT / source_label / "result.json").read_text())
        if args.memory_replay:
            erasure_gate = json.loads(Path("/home/boot/lhm/multipersonal-runtime/backups/backend-chain-20261001/stage24/native-audit.json").read_text())
            if not (erasure_gate["source_erasure_gate_passed"] == erasure_gate["source_erasure_gate_total"] == 43 and all(erasure_gate["source_erasure_gate_checks"].values())):
                raise ValueError("Replay reuse requires independently verified native source erasure state")
        elif not all(source_proof["checks"].values()):
            raise ValueError("Historical reuse requires the verified native source fixture")
    OUT.mkdir(exist_ok=False)
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures" / ("deepseek_memory_late_capture_cases.json" if args.memory_late_capture else "deepseek_memory_history_feedback_cases.json" if args.memory_history_feedback else "deepseek_memory_capture_failure_cases.json" if args.memory_capture_failure else "deepseek_memory_capacity_cases.json" if args.memory_capacity else "deepseek_memory_race_cases.json" if args.memory_race else "deepseek_memory_replay_cases.json" if args.memory_replay else "deepseek_memory_collision_cases.json" if args.memory_collision else "deepseek_memory_scope_cases.json" if args.memory_scope else "deepseek_memory_erasure_cases.json" if args.memory_erasure else "deepseek_memory_friend_cases.json" if args.memory_friend else "deepseek_memory_owner_cases.json" if args.memory_owner else "deepseek_memory_history_cases.json" if args.memory_history else "deepseek_memory_correction_cases.json" if args.memory_correction else "deepseek_literal_citation_cases.json" if args.literal_citations else "deepseek_private_public_citation_cases.json" if args.private_public_citations else "deepseek_citation_precision_cases.json" if args.citation_precision else "deepseek_mixed_window_boundary.json" if args.window_boundary else "deepseek_mixed_long_context_cases.json")).read_text()
    )
    fixture["history_turns"] = expand_history(fixture)
    cases = fixture["cases"]
    executed_cases = cases[-(2 if args.memory_owner or args.memory_erasure else 1):] if reuse_native_fixture else cases
    prerequisite_count = 3 if args.memory_friend or args.memory_erasure or args.memory_scope or args.memory_collision else 2
    if reuse_native_fixture and source_proof["cases"][:prerequisite_count] != cases[:prerequisite_count]:
        raise ValueError("Historical reuse requires the same complete original source statements")
    if (args.memory_friend or args.memory_erasure or args.memory_scope or args.memory_collision) and not any(
            row["body"] == cases[2]["message"] for row in source_proof.get("sources_before_erasure" if args.memory_replay else "sources_before_question", [])):
        raise ValueError("Friend reuse requires the complete native persisted source")
    key = key_path.read_text().strip()
    os.environ.update(
        DATABASE_URL="postgresql+asyncpg://boot@/" + database_name + "?host=" + str(ROOT / "socket") + "&port=25433",
        USE_POSTGRESQL="true",
        ENVIRONMENT="production",
        JWT_SECRET=secrets.token_urlsafe(48),
        ENCRYPTION_KEY=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        ASTRBOT_INTEGRATION_TOKEN=secrets.token_urlsafe(48),
        MULTIPERSONAL_BACKEND_URL="https://stage3-evaluation.invalid",
        ALLOWED_ORIGINS="https://stage3-evaluation.invalid",
        ALLOW_PUBLIC_REGISTRATION="true" if args.memory_scope else "false",
        SECURITY_MIDDLEWARE_ENABLED="true",
        LOG_LEVEL="INFO",
        BACKEND_WORKERS="1",
        AUDIT_LOG_DIR=str(OUT / "audit"),
        BACKUP_DIR=str(OUT / "backups"),
        MODEL_PROVIDER="openai_compat",
        VLLM_MAX_MODEL_LEN="8192",
        OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS="65536",
        VLLM_ENABLED="false",
        VLLM_BASE_URL="http://127.0.0.1:1",
        VLLM_BASE_URLS="http://127.0.0.1:1",
        OPENAI_COMPAT_BASE_URL="https://api.deepseek.com",
        OPENAI_COMPAT_API_KEY=key,
        OPENAI_COMPAT_MODEL="deepseek-v4-pro",
        MEMORY_LLM_ENABLED="true",
        **(dict(MEMORY_LLM_QUEUE_SIZE="1", MEMORY_LLM_IDLE_SECONDS="0") if args.memory_capacity else {}),
        MEMORY_LLM_BASE_URL="https://api.deepseek.com",
        MEMORY_LLM_MODEL="deepseek-v4-pro",
        MEMORY_LLM_API_KEY=key,
        MEMORY_LLM_CONTEXT_WINDOW_TOKENS="65536",
        MEMORY_SOURCE_RECALL_ENABLED="true" if source_recall_enabled else "false",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED="true",
        DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS="30",
        CONTEXTUAL_MEMORY_SELECTION_ENABLED="true",
        CONTEXTUAL_DECISION_POLICY_ENABLED="true",
        REDIS_URL="redis://127.0.0.1:1/0",
        EMBEDDING_MODEL_PATH="/home/boot/lhm/multipersonal-runtime/models/paraphrase-multilingual-MiniLM-L12-v2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        VECTOR_DB_PATH=str(OUT / "vectors"),
        INTENT_MODEL_PATH=str(OUT / "no-intent-model"),
        CHARACTER_RAG_INDEX_ROOT=str(OUT / "no-character-index"),
        CORRECTIVE_RAG_ENABLED="false",
        RERANKER_ENABLED="false",
        CHAT_CONVERSATION_BURST="10",
        CHAT_SENDER_BURST="10",
    )

    await verify_cluster(bootstrap_url, ROOT / "data")
    import asyncpg

    connection = await asyncpg.connect(user="boot", database="postgres", host=str(ROOT / "socket"), port=25433)
    try:
        assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
        if reuse_native_fixture:
            assert await connection.fetchval("SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)", source_database)
            await connection.execute("CREATE DATABASE " + database_name + " TEMPLATE " + source_database)
        else:
            await connection.execute("CREATE DATABASE " + database_name)
    finally:
        await connection.close()
    await verify_cluster(os.environ["DATABASE_URL"], ROOT / "data")
    if reuse_native_fixture:
        import hashlib
        import shutil

        source_vectors = ROOT / source_label / "vectors"
        assert source_vectors.resolve().parent == (ROOT / source_label).resolve()
        source_hashes = {str(p.relative_to(source_vectors)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source_vectors.rglob("*") if p.is_file()}
        assert source_hashes and all(not p.is_symlink() for p in source_vectors.rglob("*"))
        shutil.copytree(source_vectors, OUT / "vectors")
        copied_hashes = {str(p.relative_to(OUT / "vectors")): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (OUT / "vectors").rglob("*") if p.is_file()}
        assert copied_hashes == source_hashes

    import httpx

    cloud_calls = []
    local_review_calls = []
    original_send = httpx.AsyncClient.send

    async def observed_send(client, request, **kwargs):
        if request.url.host != "api.deepseek.com":
            if request.url.host == "127.0.0.1" and request.url.path.endswith("/chat/completions"):
                local_body = json.loads(request.content)
                local_review_calls.append(
                    dict(url=str(request.url), model=local_body.get("model"), max_tokens=local_body.get("max_tokens"))
                )
                (OUT / "local-review-calls.json").write_text(json.dumps(local_review_calls, indent=2))
            return await original_send(client, request, **kwargs)
        body = json.loads(request.content)
        response = await original_send(client, request, **kwargs)
        await response.aread()
        data = response.json()
        cloud_calls.append(dict(url=str(request.url), request=body, http_status=response.status_code, response=data))
        (OUT / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
        return response

    httpx.AsyncClient.send = observed_send
    from app.main import create_app
    from character.memory_llm import get_memory_enrichment_scheduler, shutdown_memory_enrichment
    from db.adapter import db, is_pg_mode
    from infra.concurrency_control import inference_runtime
    from services.character_context import CharacterContextService

    cold_history_diagnostics = []
    candidate_diagnostics = []
    original_history = CharacterContextService._load_history
    original_candidates = CharacterContextService._load_memory_candidates

    async def cold_history(service, turn, user_scope, character_id):
        history = await original_history(service, turn, user_scope, character_id)
        if turn.message == cases[-1]["message"]:
            cold_history_diagnostics.append(dict(query=turn.message, original_history=history, returned_history=[]))
            return []
        return history

    async def observed_candidates(service, character_id, user_scope, query, **kwargs):
        result = await original_candidates(service, character_id, user_scope, query, **kwargs)
        candidate_diagnostics.append(dict(query=query, items=[asdict(item) for item in result[0]],
                                         candidate_count=result[1], recall=result[2]))
        return result

    if cold_memory:
        if history_ablation:
            CharacterContextService._load_history = cold_history
        CharacterContextService._load_memory_candidates = observed_candidates

    prepared_diagnostics = []
    generation_diagnostics = []
    original_prepare = CharacterContextService.prepare_turn
    operation_diagnostics = []
    original_interactive = CharacterContextService.prepare_interactive_turn

    async def observed_interactive(service, turn, character_id, **kwargs):
        result = await original_interactive(service, turn, character_id, **kwargs)
        operation_diagnostics.append(dict(query=turn.message, receipt=result.memory_operation_receipt,
                                          compiled_receipt=result.compiled.memory_operation_receipt))
        return result

    if args.memory_erasure:
        CharacterContextService.prepare_interactive_turn = observed_interactive

    from api import generate as generation_api

    retrieval_diagnostics = []
    original_retrieval = generation_api._retrieve_rag_bundle

    async def observed_retrieval(query, top_k, filters):
        bundle = await original_retrieval(query, top_k, filters)
        retrieval_diagnostics.append(dict(query=query, filters=filters, bundle=bundle))
        return bundle

    generation_api._retrieve_rag_bundle = observed_retrieval
    original_generation = generation_api.generate_character_response

    async def observed_prepare(service, turn, character_id):
        prepared = await original_prepare(service, turn, character_id)
        prepared_diagnostics.append(
            dict(
                query=turn.message,
                history=list(prepared.history),
                selection_status=prepared.memory_selection_status,
                selection_reason=prepared.memory_selection_reason,
                semantic_status=prepared.semantic_review_status,
                semantic_reason=prepared.semantic_review_fallback_reason,
                policy_status=prepared.contextual_policy_status,
                policy_reason=prepared.contextual_policy_reason,
                used_memory_ids=list(prepared.compiled.used_memory_ids),
                raw_source_status=prepared.compiled.memory_source_status,
                raw_source_diagnostics=prepared.memory_recall.get("sources", {}),
                episodic_context=prepared.compiled.episodic_reference_context,
                **(dict(user_scope=asdict(prepared.user_scope), memory_scope_key=list(prepared.user_scope.memory_scope_key)) if args.memory_scope or args.memory_collision else {}),
            )
        )
        return prepared

    async def observed_generation(request, generate):
        row = dict(
            context_window_tokens=request.context_window_tokens,
            input_chars=len(request.message),
            history=list(request.history),
        )
        generation_diagnostics.append(row)
        try:
            result = await original_generation(request, generate)
            row["model_messages"] = list(result.plan.messages)
            row["citation_namespace"] = result.plan.retrieval.citation_namespace
            return result
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            raise

    CharacterContextService.prepare_turn = observed_prepare
    generation_api.generate_character_response = observed_generation
    assert is_pg_mode()
    db.update_config(dict(useKnowledgeBase=True, temperature=0.2, maxTokens=1024, topP=0.9))
    app = create_app()
    proof = dict(
        cases=cases,
        generation=[],
        prepared_diagnostics=prepared_diagnostics,
        generation_diagnostics=generation_diagnostics,
        operation_diagnostics=operation_diagnostics,
        provider="native_openai_compat",
        transport="authenticated_ASGI",
        source_ablation=False,
        source_mode=dict(test_setting=source_recall_enabled, controlled_variant=source_recall_enabled, production_setting_not_changed=True),
        rag_enabled=True,
        documents=[],
        searches=[],
        retrieval_diagnostics=retrieval_diagnostics,
        cold_history_diagnostics=cold_history_diagnostics,
        candidate_diagnostics=candidate_diagnostics,
        controlled_ablation=dict(history_ablation_enabled=history_ablation, raw_source_recall_enabled=source_recall_enabled),
    )
    password = secrets.token_urlsafe(24)
    if reuse_native_fixture:
        # Reset only the disposable cloned fixture account to a new random
        # credential; authenticate through the real login endpoint afterwards.
        from api.auth import _hash_password

        connection = await asyncpg.connect(user="boot", database=database_name, host=str(ROOT / "socket"), port=25433)
        try:
            assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
            row = await connection.fetchrow("SELECT id, username FROM users WHERE id=1")
            assert row["username"] == source_account_username
            password_hash = await asyncio.to_thread(_hash_password, password)
            changed = await connection.fetchval("UPDATE users SET password_hash=$1 WHERE id=1 AND username=$2 RETURNING id", password_hash, source_account_username)
            assert changed == 1
        finally:
            await connection.close()
        proof["reused_native_fixture"] = dict(source_run=source_label, source_database=source_database,
            cloned_database=database_name, source_vectors_sha256=source_hashes,
            source_writing_generations_replayed=0, prior_answer_generations_replayed=0,
            cloned_account_password_reset=True, native_login_required=True)
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="https://stage3-evaluation.invalid",
                timeout=180,
            ) as client,
        ):
            if reuse_native_fixture:
                login = await client.post("/api/auth/login", json=dict(username=source_account_username, password=password))
                me = await client.get("/api/auth/me")
                proof["auth_statuses"] = [login.status_code, me.status_code]
                assert proof["auth_statuses"] == [200, 200]
            else:
                register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=password))
                client.cookies.clear()
                login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=password))
                me = await client.get("/api/auth/me")
                proof["auth_statuses"] = [register.status_code, login.status_code, me.status_code]
                assert proof["auth_statuses"] == [200, 200, 200]
            identity = str(me.json()["user"]["id"])
            if args.memory_scope or args.memory_collision:
                proof["scope_owner_identity"] = identity
                if args.memory_replay:
                    old = [row for row in source_proof["sources_before_erasure"] if row["body"] == cases[2]["message"]]
                    assert len(old) == 1
                    proof["reused_erased_source"] = dict(source_message_id=old[0]["source_message_id"])
                    proof["reused_native_fixture"]["verified_erasure_state"] = True

                proof["scope_owner_sources_before"] = db.list_memory_sources(
                    "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity, limit=100)
                proof["scope_owner_claims_before"] = db.list_character_memory_claims(
                    "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity, limit=None, include_inactive=True)
                if args.memory_scope:
                    client.cookies.clear()
                    second_password = secrets.token_urlsafe(24)
                    register = await client.post("/api/auth/register", json=dict(username=args.run_label, password=second_password))
                    client.cookies.clear()
                    second_login = await client.post("/api/auth/login", json=dict(username=args.run_label, password=second_password))
                    second_me = await client.get("/api/auth/me")
                    proof["scope_secondary_auth_statuses"] = [register.status_code, second_login.status_code, second_me.status_code]
                    assert proof["scope_secondary_auth_statuses"] == [200, 200, 200]
                    identity = str(second_me.json()["user"]["id"])
                    proof["scope_current_identity"] = identity
                    proof["scope_secondary_role"] = second_me.json()["user"]["role"]
                    assert identity != proof["scope_owner_identity"] and proof["scope_secondary_role"] == "user"
                else:
                    proof["scope_current_identity"] = identity
                    from character.memory_llm import is_memory_erasure_request
                    proof["collision_query_is_erasure"] = is_memory_erasure_request(cases[-1]["message"])
                    assert not proof["collision_query_is_erasure"]
                connection = await asyncpg.connect(user="boot", database=database_name, host=str(ROOT / "socket"), port=25433)
                try:
                    assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
                    rows = await connection.fetch('SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id', "web", "web-character", proof["scope_owner_identity"], "tsukiyashiro_kisaki")
                    proof["scope_sql"] = dict(owner_message_count_before=len(rows), owner_message_max_id_before=max(row["id"] for row in rows), owner_messages_sha256_before=hashlib.sha256(json.dumps([dict(row) for row in rows],sort_keys=True,default=str).encode()).hexdigest())
                    if args.memory_replay:
                        from db.memory_source import source_identity, source_scope
                        key = source_identity(source_scope("tsukiyashiro_kisaki","web","web-character",identity,"private",identity),proof["reused_erased_source"]["source_message_id"])["source_key"]
                        row = await connection.fetchrow("SELECT state, body IS NULL AS body_is_null, observed_at IS NULL AS observed_at_is_null FROM memory_sources WHERE source_key=$1",key)
                        before = dict(row)
                        before["terms"] = await connection.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1",key)
                        before["links"] = await connection.fetchval("SELECT count(*) FROM memory_source_links WHERE source_key=$1",key)
                        assert before == dict(state="revoked",body_is_null=True,observed_at_is_null=True,terms=0,links=0)
                        proof["scope_sql"]["replay_anchor_before"] = before

                finally:
                    await connection.close()
                source_connection = await asyncpg.connect(user="boot", database=source_database, host=str(ROOT / "socket"), port=25433)
                try:
                    assert await source_connection.fetchval("SHOW data_directory") == str(ROOT / "data")
                    snapshots = {}
                    for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
                        rows = await source_connection.fetch("SELECT * FROM " + table)
                        snapshots[table] = hashlib.sha256(json.dumps(sorted([dict(row) for row in rows],key=lambda row:json.dumps(row,sort_keys=True,default=str)),sort_keys=True,default=str).encode()).hexdigest()
                    proof["scope_sql"]["source_database_snapshot_before"] = snapshots
                finally:
                    await source_connection.close()
            from knowledge.vector_db import get_vector_db

            if reuse_native_fixture:
                proof["documents"] = json.loads((ROOT / source_label / "result.json").read_text())["documents"]
                proof["document_imports_replayed"] = 0
            else:
                base = await client.post(
                    "/api/knowledge/bases",
                    json=dict(name=fixture["knowledge_base"], description="Complete synthetic chain fixtures"),
                )
                assert base.status_code == 200, base.text
                proof["base"] = base.json()
                base_id = base.json()["base"]["id"]
                for document in fixture["documents"]:
                    response = await client.post(
                        "/api/knowledge/documents", json=dict(**document, knowledge_base_id=base_id)
                    )
                    proof["documents"].append(dict(http_status=response.status_code, response=response.json()))
                    assert response.status_code == 200, response.text
            vector_db = get_vector_db()
            proof["indexed_count"] = len(vector_db.metadata)
            for query in fixture["search_queries"]:
                response = await client.post(
                    "/api/knowledge/search", json=dict(query=query, topK=3, knowledgeBaseName=fixture["knowledge_base"])
                )
                proof["searches"].append(dict(query=query, http_status=response.status_code, response=response.json()))
            if args.memory_erasure:
                proof["sources_before_erasure"] = db.list_memory_sources(
                    "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity, limit=100)
                proof["claims_before_erasure"] = db.list_character_memory_claims(
                    "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity,
                    limit=None, include_inactive=True)
            if args.memory_late_capture:
                from evaluation.memory_late_capture_probe import run_late_capture_probe

                await run_late_capture_probe(client,args,proof,fixture,cloud_calls,
                    database_name=database_name,source_database=source_database,root=ROOT,
                    output=OUT,identity=identity,database=db,runtime=inference_runtime,
                    capture_storage_proof=capture_storage_proof)
                return
            if args.memory_history_feedback:
                from evaluation.memory_history_feedback_probe import run_history_feedback_probe

                await run_history_feedback_probe(client,args,proof,fixture,cloud_calls,
                    database_name=database_name,source_database=source_database,root=ROOT,
                    output=OUT,identity=identity,database=db,runtime=inference_runtime,
                    capture_storage_proof=capture_storage_proof)
                return
            if args.memory_capture_failure:
                from evaluation.memory_capture_failure_probe import run_capture_failure_probe

                await run_capture_failure_probe(client,args,proof,fixture,cloud_calls,
                    database_name=database_name,source_database=source_database,root=ROOT,
                    output=OUT,identity=identity,database=db,runtime=inference_runtime,
                    capture_storage_proof=capture_storage_proof)
                return
            if args.memory_capacity:
                from evaluation.memory_capacity_probe import run_capacity_probe

                await run_capacity_probe(client,args,proof,fixture,cloud_calls,
                    database_name=database_name,source_database=source_database,root=ROOT,
                    output=OUT,identity=identity,database=db,runtime=inference_runtime,
                    capture_storage_proof=capture_storage_proof)
                return
            if args.memory_race:
                from evaluation.memory_race_probe import run_memory_race

                await run_memory_race(client, args, proof, fixture, cloud_calls,
                    database_name=database_name, source_database=source_database,
                    root=ROOT, output=OUT, identity=identity, database=db,
                    runtime=inference_runtime, capture_storage_proof=capture_storage_proof)
                return
            for case in executed_cases:
                if (cold_memory) and case["id"] == cases[-1]["id"]:
                    proof["claims_before_question"] = db.list_character_memory_claims(
                        "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity,
                        limit=None, include_inactive=True)
                    if args.memory_owner or args.memory_friend or args.memory_erasure:
                        proof["sources_before_question"] = db.list_memory_sources(
                            "tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity, limit=100)
                    (OUT / "pre-question.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
                extra_request = {}
                if args.memory_scope or args.memory_collision:
                    owner = proof["scope_owner_identity"]
                    friend = ([dict(source_message_id=proof["reused_erased_source"]["source_message_id"])] if args.memory_replay
                              else [row for row in proof["scope_owner_sources_before"] if row["body"] == cases[2]["message"]])
                    assert len(friend) == 1
                    extra_request = dict(senderId=owner, userId=owner, conversationId=owner, sourceMessageId=friend[0]["source_message_id"])
                    proof["scope_forged_request"] = dict(**extra_request, sessionId=owner)
                if args.memory_collision:
                    proof["collision_submitted_message"] = case["message"]
                before = len(cloud_calls)
                response = await client.post(
                    "/api/generate",
                    json=dict(
                        message=case["message"],
                        characterId="tsukiyashiro_kisaki",
                        loraId="default",
                        sessionId=proof["scope_owner_identity"] if args.memory_scope or args.memory_collision else args.run_label,
                        sessionType="private",
                        **extra_request,
                    ),
                )
                assert await get_memory_enrichment_scheduler().flush_memory(timeout=90)
                proof["generation"].append(
                    dict(
                        id=case["id"],
                        http_status=response.status_code,
                        response=response.json(),
                        cloud_call_range=[before, len(cloud_calls)],
                    )
                )
                if case["id"] == "personal_schedule":
                    for index, turn in enumerate(fixture["history_turns"]):
                        await asyncio.to_thread(db.add_message, dict(
                            platform="web", adapter="web-character", senderId=identity, userId=identity,
                            conversationType="private", sessionType="private", conversationId=identity,
                            sessionId=args.run_label, characterId="tsukiyashiro_kisaki", loraName="default",
                            sourceMessageId=args.run_label + "-history-" + str(index),
                            message=turn["message"], reply=turn["reply"], modelName="synthetic-fixture-history"))
                (OUT / "partial.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            proof["claims"] = db.list_character_memory_claims(
                "tsukiyashiro_kisaki",
                "web",
                "web-character",
                identity,
                "private",
                identity,
                limit=None,
                include_inactive=True,
            )
            if args.memory_erasure:
                from db.memory_source import source_identity, source_scope

                friends = [row for row in proof["sources_before_erasure"] if row["body"] == cases[2]["message"]]
                assert len(friends) == 1
                scoped_source = source_identity(source_scope("tsukiyashiro_kisaki", "web", "web-character",
                                                            identity, "private", identity), friends[0]["source_message_id"])
                connection = await asyncpg.connect(user="boot", database=database_name, host=str(ROOT / "socket"), port=25433)
                try:
                    assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
                    row = await connection.fetchrow("SELECT state, body IS NULL AS body_is_null, observed_at IS NULL AS observed_at_is_null FROM memory_sources WHERE source_key=$1", scoped_source["source_key"])
                    proof["source_erasure_sql"] = dict(row) if row else {}
                    proof["source_erasure_sql"].update(
                        term_count=await connection.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1", scoped_source["source_key"]),
                        link_count=await connection.fetchval("SELECT count(*) FROM memory_source_links WHERE source_key=$1", scoped_source["source_key"]),
                        archive_quote_row_count=await connection.fetchval('SELECT count(*) FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 AND "sourceMessageId"=$5 AND message=$6',
                            "web", "web-character", identity, "tsukiyashiro_kisaki", friends[0]["source_message_id"], cases[2]["message"]))
                finally:
                    await connection.close()
            if args.memory_scope or args.memory_collision:
                owner = proof["scope_owner_identity"]
                proof["scope_owner_sources_after"] = db.list_memory_sources("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner, limit=100)
                proof["scope_owner_claims_after"] = db.list_character_memory_claims("tsukiyashiro_kisaki", "web", "web-character", owner, "private", owner, limit=None, include_inactive=True)
                proof["scope_current_sources_after"] = db.list_memory_sources("tsukiyashiro_kisaki", "web", "web-character", identity, "private", identity, limit=100)
                connection = await asyncpg.connect(user="boot", database=database_name, host=str(ROOT / "socket"), port=25433)
                try:
                    assert await connection.fetchval("SHOW data_directory") == str(ROOT / "data")
                    rows = await connection.fetch('SELECT * FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 ORDER BY id', "web", "web-character", owner, "tsukiyashiro_kisaki")
                    proof["scope_sql"]["owner_messages_sha256_after"] = hashlib.sha256(json.dumps([dict(row) for row in rows],sort_keys=True,default=str).encode()).hexdigest()
                    proof["scope_sql"]["owner_message_count_after"] = len(rows)
                    old_rows = [dict(row) for row in rows if row["id"] <= proof["scope_sql"]["owner_message_max_id_before"]]
                    proof["scope_sql"]["owner_original_messages_sha256_after"] = hashlib.sha256(json.dumps(old_rows,sort_keys=True,default=str).encode()).hexdigest()
                    source_id = proof["scope_forged_request"]["sourceMessageId"]
                    from db.memory_source import source_identity, source_scope
                    keys = [source_identity(source_scope("tsukiyashiro_kisaki", "web", "web-character", account, "private", account),source_id)["source_key"] for account in [owner, identity]]
                    row = await connection.fetchrow("SELECT state, body FROM memory_sources WHERE source_key=$1",keys[0])
                    proof["scope_sql"].update(owner_friend_state=row["state"],owner_friend_body_preserved=row["body"]==cases[2]["message"],source_keys_distinct=keys[0]!=keys[1])
                    if args.memory_replay:
                        tombstone = await connection.fetchrow("SELECT state, body IS NULL AS body_is_null, observed_at IS NULL AS observed_at_is_null FROM memory_sources WHERE source_key=$1",keys[0])
                        after = dict(tombstone)
                        after["terms"] = await connection.fetchval("SELECT count(*) FROM memory_source_terms WHERE source_key=$1",keys[0])
                        after["links"] = await connection.fetchval("SELECT count(*) FROM memory_source_links WHERE source_key=$1",keys[0])
                        proof["scope_sql"]["replay_anchor_after"] = after

                    for label,account,text in [("owner_friend",owner,cases[2]["message"]),("current_request",identity,cases[-1]["message"])]:
                        relation = "<=" if label == "owner_friend" else ">"
                        proof["scope_sql"][label+"_archive_count"] = await connection.fetchval('SELECT count(*) FROM messages WHERE platform=$1 AND adapter=$2 AND "senderId"=$3 AND "characterId"=$4 AND "sourceMessageId"=$5 AND message=$6 AND id '+relation+' $7',"web","web-character",account,"tsukiyashiro_kisaki",source_id,text,proof["scope_sql"]["owner_message_max_id_before"])
                finally:
                    await connection.close()
                source_connection = await asyncpg.connect(user="boot", database=source_database, host=str(ROOT / "socket"), port=25433)
                try:
                    assert await source_connection.fetchval("SHOW data_directory") == str(ROOT / "data")
                    snapshots = {}
                    for table in ["memory_sources", "memory_source_terms", "character_memories", "messages"]:
                        rows = await source_connection.fetch("SELECT * FROM " + table)
                        snapshots[table] = hashlib.sha256(json.dumps(sorted([dict(row) for row in rows],key=lambda row:json.dumps(row,sort_keys=True,default=str)),sort_keys=True,default=str).encode()).hexdigest()
                    proof["scope_sql"]["source_database_snapshot_after"] = snapshots
                finally:
                    await source_connection.close()
            proof["queue_stats"] = inference_runtime.stats()
            proof["memory_status"] = asdict(get_memory_enrichment_scheduler().status)
            capture_storage_proof(proof, OUT)
        if args.memory_collision and proof["generation"][-1]["http_status"] == 409:
            if args.memory_replay:
                from evaluation.memory_replay_audit import (
                    audit_memory_replay_rejected as audit_memory_collision_rejected,
                )
            else:
                from evaluation.memory_scope_audit import audit_memory_collision_rejected

            proof["configured_model"] = os.environ["OPENAI_COMPAT_MODEL"]
            proof["configured_context_window"] = int(os.environ["OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS"])
            proof["checks"] = audit_memory_collision_rejected(proof, cloud_calls, fixture)
            proof["cloud_summary"] = []
            (OUT / "cloud-calls.json").write_text(json.dumps(cloud_calls, ensure_ascii=False, indent=2))
            (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
            print(json.dumps(dict(result=str(OUT / "result.json"), checks=proof["checks"], cloud_calls=len(cloud_calls)), ensure_ascii=False))
            if args.require_success:
                assert all(proof["checks"].values()), proof["checks"]
            return
        checks = dict(
            all_generations_success=all(g["http_status"] == 200 for g in proof["generation"]),
            all_cloud_calls_completed=bool(cloud_calls)
            and all(
                c["http_status"] == 200
                and c["response"].get("model") == "deepseek-v4-pro"
                and all(x.get("finish_reason") == "stop" for x in c["response"].get("choices", []))
                for c in cloud_calls
            ),
            native_cloud_window_used=bool(generation_diagnostics)
            and all(g["context_window_tokens"] == 65536 for g in generation_diagnostics),
            documents_really_indexed=proof["indexed_count"] >= len(fixture["documents"]),
            native_searches_success=all(
                s["http_status"] == 200
                and s["response"].get("retrievalMode") == "evidence"
                and s["response"].get("results")
                for s in proof["searches"]
            ),
            no_wrong_local_completions=not local_review_calls,
            memory_writer_no_errors=proof["memory_status"]["failed"] == 0,
            long_history_loaded=len(prepared_diagnostics[-1]["history"]) >= 26
            and sum(len(h["content"]) for h in prepared_diagnostics[-1]["history"]) >= 18000,
        )
        if args.memory_scope or args.memory_collision:
            checks.pop("native_searches_success")
            checks["no_native_searches_replayed"] = proof["searches"] == []
        if args.citation_precision or args.private_public_citations or args.literal_citations or cold_memory:
            checks.pop("long_history_loaded")
            if args.memory_erasure or args.memory_collision:
                checks["no_synthetic_long_history_injected"] = not fixture["history_turns"]
            else:
                checks["no_long_history_replay"] = all(len(d["history"]) < 10 for d in prepared_diagnostics)
        for case, generation in zip(executed_cases, proof["generation"]):
            if not case.get("knowledge_expected"):
                continue
            reply = generation["response"].get("reply", "")
            checks[case["id"] + "_answer_fields"] = all(v in reply for v in case["expected"])
            checks[case["id"] + "_citations"] = bool(generation["response"].get("citations"))
            selected = cloud_calls[slice(*generation["cloud_call_range"])]
            wires = [
                unescape(c["request"]["messages"][-1]["content"])
                for c in selected
                if c["request"].get("max_tokens") == 1024
            ]
            # These source fields do not occur in the user question. Check the
            # actual last-turn compiled evidence, not earlier answers/history.
            checks[case["id"] + "_knowledge_on_wire"] = bool(wires) and any(
                all(v in wire for v in case["knowledge_expected"]) for wire in wires
            )
        if cold_memory:
            if args.memory_collision:
                if args.memory_replay:
                    from evaluation.memory_replay_audit import audit_memory_replay_wire as audit_wire
                else:
                    from evaluation.memory_scope_audit import audit_memory_collision_wire as audit_wire
            elif args.memory_scope:
                from evaluation.memory_scope_audit import audit_memory_scope_wire as audit_wire
            elif args.memory_erasure:
                from evaluation.memory_erasure_audit import audit_memory_erasure_wire as audit_wire
            elif args.memory_friend:
                from evaluation.memory_friend_audit import audit_memory_friend_wire as audit_wire
            elif args.memory_owner:
                from evaluation.memory_owner_audit import audit_memory_owner_wire as audit_wire
            elif args.memory_history:
                from evaluation.memory_history_audit import audit_memory_history_wire as audit_wire
            else:
                from evaluation.memory_correction_audit import audit_memory_correction_wire as audit_wire

            if history_ablation:
                checks["final_question_cold_history"] = prepared_diagnostics[-1]["history"] == []
            checks.update(audit_wire(proof, cloud_calls, fixture))
        else:
            checks.update(audit_literal_citation_wire(proof, cloud_calls, fixture) if args.literal_citations
                          else audit_private_public_citation_wire(proof, cloud_calls, fixture) if args.private_public_citations
                          else audit_citation_precision_wire(proof, cloud_calls, fixture) if args.citation_precision
                          else audit_window_boundary_wire(proof, cloud_calls, fixture) if args.window_boundary
                          else audit_mixed_wire(proof, cloud_calls, fixture))

        reopened = proof["persisted_vector_stats"]
        checks.update(
            persisted_index_reopens_with_all_three_documents=reopened["total_documents"]
            == reopened["index_size"]
            == reopened["bm25_corpus_size"]
            == 3,
            real_postgresql_valid_chunk_count_matches=proof["persisted_valid_chunk_count"] == 3,
        )
        proof["persisted_vector_stats"] = reopened
        proof["checks"] = checks
        proof["controlled_ablation"] = dict(
            history_ablation_enabled=history_ablation,
            raw_source_recall_enabled=source_recall_enabled,
            loaded_history_count=len(prepared_diagnostics[-1]["history"]),
            scope="only " + cases[-1]["id"] + "; stored history observed unchanged; no model messages reconstructed" if history_ablation else "none",
        )
        proof["cloud_summary"] = [
            dict(
                http_status=c["http_status"],
                model=c["response"].get("model"),
                finish_reasons=[x.get("finish_reason") for x in c["response"].get("choices", [])],
                output_budget=c["request"].get("max_tokens"),
            )
            for c in cloud_calls
        ]
        (OUT / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                dict(result=str(OUT / "result.json"), checks=proof["checks"], cloud_calls=len(cloud_calls)),
                ensure_ascii=False,
            )
        )
        if args.require_success:
            assert all(proof["checks"].values()), proof["checks"]
    finally:
        from services.turn_completion import shutdown_turn_completions

        await shutdown_turn_completions()
        await shutdown_memory_enrichment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--window-boundary", action="store_true")
    parser.add_argument("--citation-precision", action="store_true")
    parser.add_argument("--private-public-citations", action="store_true")
    parser.add_argument("--literal-citations", action="store_true")
    parser.add_argument("--memory-correction", action="store_true")
    parser.add_argument("--memory-history", action="store_true")
    parser.add_argument("--memory-owner", action="store_true")
    parser.add_argument("--memory-friend", action="store_true")
    parser.add_argument("--memory-erasure", action="store_true")
    parser.add_argument("--memory-scope", action="store_true")
    parser.add_argument("--memory-collision", action="store_true")
    parser.add_argument("--memory-replay", action="store_true")
    parser.add_argument("--memory-race", action="store_true")
    parser.add_argument("--memory-capacity", action="store_true")
    parser.add_argument("--memory-capture-failure", action="store_true")
    parser.add_argument("--memory-history-feedback", action="store_true")
    parser.add_argument("--memory-late-capture", action="store_true")
    parser.add_argument("--late-cold-query", action="store_true")
    parser.add_argument("--capture-retry", action="store_true")
    parser.add_argument("--capacity-cold-query", action="store_true")
    parser.add_argument("--enable-source-recall", action="store_true")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
