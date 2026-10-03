"""Read-only evidence audit for a real favorite authoring turn and later lookup."""

import argparse
import hashlib
import json
import re
from html import unescape
from pathlib import Path


def run(phase):
    runtime = Path("/home/boot/lhm/multipersonal-runtime")
    assert phase.parent == runtime / "backups/backend-chain-20261001"
    assert re.fullmatch(r"stage[1-9]\d*", phase.name)

    def read(name):
        return json.loads((phase / name).read_text())

    case = read("fixture.json")
    authoring_this_phase = case.get("writer_or_imports_required") is not False
    author_phase = phase
    if not authoring_this_phase:
        assert re.fullmatch(r"stage[1-9]\d*", case["seed_parent_phase"])
        author_phase = phase.parent / case["seed_parent_phase"]

    def author_read(name):
        return json.loads((author_phase / name).read_text())

    author = author_phase / "native-pg-report"
    lookup = phase / "native-pg-read"
    seed = read("native-pg-favorite-seed/before-question.json")
    old = author_read("native-pg-seed/before-question.json")
    result = read("native-pg-read/result.json")
    author_calls = author_read("native-pg-report/cloud-calls.json")
    read_calls = read("native-pg-read/cloud-calls.json")
    calls = author_calls + read_calls if authoring_this_phase else read_calls
    diagnostic = author_read("additional-authoring-diagnostic.json")
    gate = read("native-pg-read/primary-input-observed.json")
    review = read("semantic-review.json")
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check("complete_new_source_not_copied_into_question", case["synthetic"]
          and len(case["documents"]) == 5 and len(case["bridges"]) == 9
          and not case["input_information_reduced"]
          and case["additional_source_message"] not in case["question"]
          and (case.get("previous_action_report_not_provided_in_this_question")
               or case["current_user_action_report"] in case["question"]))
    check("new_authoring_receipt_and_old_claims_intact", seed["additional_source_status"] == 200
          and seed["additional_source_written_by_real_native_turn"]
          and len(seed["seed_records"]) == 4
          and all(row in seed["seed_records"] for row in old["seed_records"]))
    check("favorite_strength_and_limit_preserved_by_actual_writer",
          any("最喜欢寄件受理" in row["content"]
              and row["attributed_to"] == "user"
              and row["qualifiers"].get("context") == "绿泽合成办理场景"
              and "不表示我已实际使用或选择寄件受理" in row["qualifiers"].get("certainty", "")
              and all(e in case["additional_source_message"] for e in row["evidence"])
              for row in seed["seed_records"]))
    check("whole_new_private_source_durable_before_lookup",
          any(row["body"] == case["additional_source_message"] for row in seed["durable_seed_sources"])
          and diagnostic["full_source_present"] and diagnostic["production_writes"] == 0)
    check("immutable_snapshot_from_existing_authoring_without_replay",
          hashlib.sha256((phase / "native-pg-favorite-seed/before-question.dump").read_bytes()).hexdigest()
          == seed["before_question_backup"]["database_sha256"]
          == diagnostic["after_authoring_sha256"]
          and seed["before_question_backup"]["prior_same_task_answers"] == 0
          and diagnostic["no_authoring_replay"])
    check("authoring_harness_failure_retained_and_correctly_classified",
          seed["authoring_harness_assertion_failed"]
          and "Actual favorite authoring produced no source-grounded stored claim" in
          (author_phase / "native-report.log").read_text()
          and not (author / "result.json").exists()
          and seed["additional_writer_actual_calls"] == len(author_calls) == 4)
    check("real_authenticated_postgresql_lookup", result["transport"] == "authenticated_ASGI"
          and result["database_mode"] == "PostgreSQL" and result["http_status"] == 200
          and not result["response"]["abstained"]
          and result["auth_statuses"] == result["chat_auth_statuses"] == [200, 200])
    check("successful_old_tasks_and_new_authoring_not_replayed_on_lookup",
          result["knowledge_imports_replayed"] == result["seed_model_calls_replayed"]
          == result["successful_bridge_turns_replayed"] == 0
          and result["bridge_turns_completed"] == 9
          and result["seed_records"] == seed["seed_records"])
    check("actual_official_pro_successful_provider_calls", len(calls) == (8 if authoring_this_phase else 4)
          and all(c["http_status"] == 200
                  and c["request"]["model"] == c["response"]["model"] == "deepseek-v4-pro"
                  and c["response"]["choices"][0]["finish_reason"] == "stop" for c in calls))
    primary = [c for c in read_calls if any("<user_query>" in m["content"]
               and case["question"] in unescape(m["content"]) for m in c["request"]["messages"])]
    check("one_actual_lookup_main_without_retry", len(primary) == result["primary_calls"] == 1
          and len(result["generation"]) == 1
          and not result["generation"][0]["guard_retried"]
          and not result["generation"][0]["guard_fallback"])
    raw = primary[0]["response"]["choices"][0]["message"]["content"]
    wire = primary[0]["request"]["messages"]
    whole = "\n".join(unescape(m["content"]) for m in wire)
    check("actual_raw_reply_never_rewritten", raw == result["response"]["reply"]
          and hashlib.sha256(raw.encode()).hexdigest() == review["raw_reply_sha256"])
    check("all_requested_whole_public_sources_and_both_private_sources_on_wire",
          all(gate["document_bodies_present"][str(i) + ":" + case["documents"][i]["title"]]
              and case["documents"][i]["content"] in whole for i in case["required_gate_document_indices"])
          and gate["private_sources_present"] == [True, True]
          and all(source in whole for source in [case["source_message"], case["additional_source_message"]]))
    prepared = result["prepared"][-1]
    check("all_four_real_user_observations_selected",
          set(prepared["used_memory_ids"]) == {str(row["id"]) for row in seed["seed_records"]}
          and prepared["recall"]["selected_count"] == 4
          and prepared["recall"]["temporal_views"] == {"fact": 0, "asserted_state": 0, "observation": 4})
    records = {str(row["id"]): row for row in seed["seed_records"]}
    observation_integrity = []
    for packet in prepared["memory_packets"]:
        row = records[packet["memory_id"]]
        observed_source = (case["additional_source_message"]
                           if "最喜欢寄件受理" in row["content"] else case["source_message"])
        match = re.fullmatch(r"用户原话记录（时效未核实，不代表当前状态；记录于(.+?)）：(.+)",
                             packet["content"], re.S)
        originals = json.loads(match[2]) if match else []
        observation_integrity.append(bool(
            match and match[1] == row["observed_at"] and observed_source in originals
            and all(evidence in originals for evidence in row["evidence"])
            and dict(packet["qualifiers"]) == row["qualifiers"]
            and packet["temporal_mode"] == "observation"))
    check("observation_render_keeps_whole_sources_excerpts_qualifiers_and_clock",
          len(observation_integrity) == 4 and all(observation_integrity))
    check("old_and_necessary_condition_not_lost",
          any(dict(packet["qualifiers"]).get("condition") == "完整表格已提交且身份核验通过"
              for packet in prepared["memory_packets"]))
    speech = json.loads(re.search(r"<dialogue_evidence[^>]*>\n(.*?)\n</dialogue_evidence>",
                                 unescape(wire[-1]["content"]), re.S)[1])
    check("all_nine_archives_and_both_full_user_sources_remain",
          all(any(row["text"] == bridge["message"].strip() for row in speech["records"])
              for bridge in case["bridges"])
          and all(any(row["text"] == source for row in speech["records"])
                  for source in [case["source_message"], case["additional_source_message"]]))
    check("scope_contracts_still_present",
          all(text in wire[0]["content"] for text in ["【用户历史依据范围】", "【本轮召回的历史原话】",
                                                     "长期记忆参考中的‘用户’始终指当前对话者"]))
    check("all_actual_provider_requests_within_unchanged_context_budget",
          all(c["budget"]["configured_window"] == 65536
                  and c["budget"]["input_bound"] + c["budget"]["output_reserved"]
                  + c["budget"]["safety_margin"] <= 65536 for c in calls))
    check("assistant_review_truthfully_labelled_and_quotes_bound",
          review["assessor"] == "assistant" and review["human_review_performed"] is False
          and review["case_question_sha256"] == hashlib.sha256(case["question"].encode()).hexdigest()
          and all(q in raw for item in review["criteria"] for q in item["quotes"]))
    from tokenizers import Tokenizer

    from inference.token_counting import _TOKENIZER_PATH

    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
    bound = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) + 4 for m in wire)
    check("bound_independently_recomputed_from_exact_wire", bound == primary[0]["budget"]["input_bound"])
    pre = read("preflight.json")
    check("production_sources_and_configuration_unchanged",
          all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
              for name, sha in {**pre["original_hashes"], **pre["config_hashes"]}.items()
              if name != "backend/evaluation/mixed_subject_history_probe.py"))
    outcome = dict(checks=checks, passed=sum(checks.values()), total=len(checks),
                   actual_calls=len(calls), actual_http200=sum(c["http_status"] == 200 for c in calls),
                   actual_native_api_replies=(2 if authoring_this_phase else 1),
                   actual_primary_calls=(2 if authoring_this_phase else 1),
                   historical_authoring_calls_verified=(0 if authoring_this_phase else len(author_calls)),
                   authoring_this_phase=authoring_this_phase,
                   public_source_reads_requested=len(case["required_gate_document_indices"]),
                   main_prompt_tokens=primary[0]["response"]["usage"]["prompt_tokens"],
                   main_input_bound=bound, complete_private_originals=len(speech["records"]),
                   semantic_criteria_passed=sum(item["passed"] for item in review["criteria"]),
                   semantic_criteria_total=len(review["criteria"]),
                   full_native_case_qualified=int(all(checks.values()) and all(item["passed"] for item in review["criteria"])),
                   runtime_code_changed=False, production_restarted=False,
                   previous_cases_retested=False, audit_provider_calls=0,
                   lookup_variant=lookup.name, semantic_review_assessor="assistant")
    (phase / "native-audit.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(outcome, ensure_ascii=False))
    assert all(checks.values()), "Inspect exact failing evidence; do not replay provider calls"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    run(Path(parser.parse_args().phase).resolve())
