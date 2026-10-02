"""Saved evidence of current memory recall with a complete dated background."""

import re
from html import unescape

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens


def actual_primary(call):
    return call.get("request", {}).get("max_tokens") == 2048 and any(
        "<user_query>" in m.get("content", "") for m in call.get("request", {}).get("messages", [])
    )


def audit_query_time_role(proof, fixture, calls, storage):
    primary = [call for call in calls if actual_primary(call)]
    messages = primary[0]["request"]["messages"] if len(primary) == 1 else []
    wire = unescape(messages[-1]["content"]) if messages else ""
    memory = re.search(r"<character_memory[^>]*>\n(.*?)\n</character_memory>", wire, re.S)
    expected = fixture["expected"]
    compound = expected.get("query_time_mode") == "compound"
    prepared = (proof.get("prepared") or [{}])[-1]
    recall = prepared.get("recall", {})
    current = [
        row
        for row in proof.get("seed_records", [])
        if row.get("status") == "active" and row.get("memory_key") == "user_residence"
    ]
    old = [
        row
        for row in proof.get("seed_records", [])
        if row.get("status") == "superseded" and row.get("memory_key") == "user_residence"
    ]
    current_id = str(current[0]["id"]) if len(current) == 1 else ""
    old_id = str(old[0]["id"]) if len(old) == 1 else ""
    packets = prepared.get("memory_packets", [])
    active = [p for p in packets if p.get("memory_id") == current_id]
    response = proof.get("response", {})
    reply = response.get("reply", "")
    chain = {
        "complete_synthetic_input": fixture.get("synthetic") is True and proof.get("synthetic_only") is True,
        "authenticated_postgresql": proof.get("transport") == "authenticated_ASGI"
        and proof.get("database_mode") == "PostgreSQL",
        "fresh_ordinary_user_authenticated": proof.get("chat_auth_statuses") == [200, 200]
        and proof.get("seed_scope", {}).get("owner") not in {None, "1"},
        "real_capture_and_rule_setup": proof.get("seed_method") == "actual_source_capture_and_rule_write",
        "complete_sources_unchanged": len(proof.get("seed_sources", [])) == 2
        and [row["body"] for row in proof["seed_sources"]]
        == [fixture["initial"]["body"], fixture["correction"]["body"]],
        "seed_durable_before_question": proof.get("durable_seed_verified_before_question") is True,
        "one_current_residence_and_superseded_old": len(current) == len(old) == 1
        and expected["current_residence"] in current[0]["content"]
        and expected["superseded_residence"] in old[0]["content"],
        "stable_private_owner_scope": proof.get("seed_scope", {}).get("conversation")
        == proof.get("seed_scope", {}).get("owner")
        and proof.get("seed_scope", {}).get("character") == "tsukiyashiro_kisaki",
        "actual_generate_http200": proof.get("http_status") == 200,
        "one_actual_primary": len(primary) == 1 and proof.get("primary_calls") == 1,
        "all_calls_actual_pro_200_stop": bool(calls)
        and all(
            c["request"].get("model") == "deepseek-v4-pro"
            and c.get("http_status") == 200
            and c["response"].get("model") == "deepseek-v4-pro"
            and c["response"]["choices"][0].get("finish_reason") == "stop"
            for c in calls
        ),
        "real_selector_call_separate_from_primary": any(
            c["request"].get("max_tokens") == 2048
            and not actual_primary(c)
            and "candidates" in c["request"]["messages"][-1]["content"]
            for c in calls
        ),
        "current_recall_not_filtered_by_background_year": recall.get("status") == "selected"
        and recall.get("selected_count") == (2 if compound else 1)
        and recall.get("usable_records") == (2 if compound else 1),
        "current_field_presence_known": recall.get("field_presence", {}).get("residence") is True,
        "semantic_memory_selection_success": prepared.get("selection_status") == "selected",
        "current_memory_id_used": bool(current_id)
        and current_id in prepared.get("used_memory_ids", [])
        and (
            (old_id in prepared.get("used_memory_ids", []))
            if compound
            else (old_id not in prepared.get("used_memory_ids", []))
        ),
        "active_current_packet_preserved": len(active) == 1
        and active[0].get("status") == "active"
        and not active[0].get("historical")
        and expected["current_residence"] in active[0].get("content", ""),
        "current_packet_has_exact_correction_source": len(active) == 1
        and fixture["correction"]["id"] in active[0].get("source_message_ids", []),
        "current_residence_in_actual_private_wire": bool(memory) and expected["current_residence"] in memory[1],
        "background_and_complete_question_still_in_wire": fixture["question"] in wire,
        "private_data_not_system_instruction": bool(messages)
        and all(
            fixture["correction"]["body"] not in m["content"]
            and "用户说自己居住在" + expected["current_residence"] not in m["content"]
            for m in messages
            if m["role"] == "system"
        ),
        "fresh_owner_history_empty": prepared.get("history") == [],
        "dynamic_policy_applied": prepared.get("policy_status") == "applied",
        "working_budget_fits": bool(messages)
        and sum(estimated_tokens(m["content"]) + 4 for m in messages) + 2048 + CONTEXT_SAFETY_MARGIN_TOKENS <= 65536,
        "no_knowledge_or_citation_dependency": proof.get("retrieval") == [] and not response.get("citations"),
        "residence_version_chain_unchanged": proof.get("seed_records") == proof.get("residence_records_after"),
        "all_sync_writes_finished": proof.get("sync_pending_final") == 0,
        "independent_exact_readonly_storage": storage.get("read_only_transaction") is True
        and storage.get("exact_isolated_database") is True
        and storage.get("exact_owner_scope") is True,
        "current_and_old_states_still_correct": storage.get("current_residence_count") == 1
        and storage.get("superseded_residence_count") == 1,
        "both_complete_seed_receipts_unchanged": storage.get("complete_seed_sources_unchanged") is True,
        "question_saved_once_in_same_scope": storage.get("question_source_count") == 1,
    }
    answer = {
        "current_residence_correct": expected["current_residence"] in reply,
        "not_superseded_residence_as_current": not re.search(
            (
                r"(?:目前|现在|当前|现居地)[^，,；;。！？\n]{0,15}"
                if compound
                else r"(?:目前|现在|当前|现居地|住址)[^。！？\n]{0,15}"
            )
            + expected["superseded_residence"],
            reply,
        ),
        "answers_user_not_character": bool(re.search(r"你[^。！？\n]{0,20}" + expected["current_residence"], reply)),
        "no_background_year_residence_claim": not re.search(
            str(expected.get("background_event_year", expected.get("birth_year", "")))
            + r"年[^。！？\n]{0,15}(?:住|居住|现居)",
            reply,
        ),
        "does_not_falsely_claim_missing_residence": not re.search(
            r"(?:不知道|不清楚|没有记录|无法确认)[^。！？\n]{0,15}(?:住|居住|现居)", reply
        ),
    }
    if compound:
        del answer["no_background_year_residence_claim"]
        answer["current_not_reported_as_september_residence"] = not re.search(
            r"(?:9月|九月)[^，,；;。！？\n]{0,20}" + expected["current_residence"], reply
        )
        import json

        selector_calls = [c for c in calls if c["request"].get("max_tokens") == 2048 and not actual_primary(c)]
        selector = (
            json.loads(selector_calls[0]["request"]["messages"][-1]["content"]) if len(selector_calls) == 1 else {}
        )
        transmitted = selector.get("query_tasks", [])
        historical = [p for p in packets if p.get("memory_id") == old_id]
        tasks = recall.get("personal_time_tasks", [])
        chain.update(
            separate_task_windows_preserved=len(tasks) == 2
            and tasks[0].get("historical") is True
            and tasks[1].get("historical") is False
            and tasks[0].get("start", "").startswith("2026-09-01"),
            selector_receives_both_original_time_tasks=len(transmitted) == 2
            and [t.get("time_mode") for t in transmitted] == ["historical", "current"]
            and all(t.get("query") in fixture["question"] for t in transmitted),
            selector_receives_exact_typed_candidates=len(selector.get("candidates", [])) == 2
            and {r.get("id") for r in selector["candidates"]} == {old_id, current_id}
            and all(r.get("memory_key") == "user_residence" for r in selector["candidates"]),
            both_time_tasks_covered=len(tasks) == 2 and all(t.get("covered_fields") == ["residence"] for t in tasks),
            historical_packet_preserved=len(historical) == 1
            and historical[0].get("historical") is True
            and historical[0].get("status") == "superseded"
            and expected["superseded_residence"] in historical[0].get("content", ""),
            historical_packet_exact_source=len(historical) == 1
            and fixture["initial"]["id"] in historical[0].get("source_message_ids", []),
            historical_value_in_actual_wire=bool(memory) and expected["superseded_residence"] in memory[1],
            actual_declared_intervals_durable=proof.get("durable_declared_intervals_verified") is True
            and storage.get("declared_source_intervals_unchanged") is True,
        )
        answer.update(
            historical_residence_answered=expected["superseded_residence"] in reply
            and bool(
                re.search(
                    r"(?:9月|九月|当时|那时|以前|之前)[^。！？\n]{0,30}" + expected["superseded_residence"], reply
                )
            )
        )
    return {"chain": chain, "answer": answer}
