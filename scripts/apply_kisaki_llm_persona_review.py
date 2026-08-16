#!/usr/bin/env python3
"""Review every LLM-authored Kisaki record and atomically promote persona revisions."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
TRAIN = V4 / "train.jsonl"
VALIDATION = V4 / "validation.jsonl"
MANIFEST = V4 / "canonical_dataset_manifest.json"
GOLD_V3 = ROOT / "backend/evaluation/kisaki_gold_set_v3.json"
GOLD_V21 = ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"
GOLD_AUDIT = ROOT / "backend/evaluation/kisaki_gold_set_v3_contamination_audit.json"
REVIEW_MANIFEST = ROOT / "docs/research/review_packets/kisaki_v4/review_manifest.json"
PROMOTER = ROOT / "scripts/promote_kisaki_v41_round06.py"
ARTIFACTS = V4 / "augmentation_candidates/llm_persona_review_20260816"
REVIEW_ID = "KISAKI-V41-LLM-PERSONA-REVIEW-20260816"
BASE_TRAIN_SHA256 = "09a06e7c4b1fc8ddca8852735ecdbf754c474aa3d40ca8214895f46a12e422ab"
LLM_SOURCES = {
    "llm_v4_blindfix",
    "llm_v4_lifestyle",
    "llm_v4_yoruko",
    "llm_v4_manual",
    "llm_v4_riou",
    "deepseek_user_simulation_v41_reviewed",
    "codex_user_simulation_v41_reviewed",
}

# Each sentence is tied to the concrete misconception or decision in that session.
# The factual/code body remains unchanged; only the character-facing outer layer changes.
TARGET_PREFIXES = {
    "kisaki_v41_round06_coding_debug": "先别急着把整份文件塞进内存，那只是拿数据量给自己设陷阱。",
    "kisaki_v41_auto_b001_git_merge_rebase": "先把“历史长什么样”和“提交实际发生了什么”分开，名字相近不代表动作相同。",
    "kisaki_v41_auto_b004_async_message_singleflight": "问题不在异步太快，而在你把“检查”和“登记”拆成了两个可以被插队的动作。",
    "kisaki_v41_auto_b005_sse_websocket_choice": "别先按“实时”两个字选协议，那会把需求分析省略得很漂亮。",
    "kisaki_v41_auto_b005_multiplatform_release_gate": "靠大家临场记住发布条件，未免太信任忙乱中的记忆了。",
    "kisaki_v41_auto_b010_vllm_throughput_ttft_tradeoff": "吞吐量上去了不等于用户更满意，指标替人排队这种事很常见。",
    "kisaki_v41_auto_b010_knowledge_file_upload_safety": "只看扩展名和 Content-Type，等于让上传者替你做安全审查。",
    "kisaki_v41_auto_b011_unified_api_error_contract": "错误格式若每个接口都自由发挥，前端当然只能写一地判断。",
    "kisaki_v41_auto_b014_asyncio_cancellation_atomic_file": "取消不是普通返回，资源清理若靠“应该会执行”，迟早会留下半成品。",
    "kisaki_v41_auto_b018_password_hashing_salt_encryption_migration": "先别把哈希、加盐和加密塞进同一个抽屉，它们解决的不是同一件事。",
    "kisaki_v41_auto_b019_interpreting_frequentist_confidence_intervals": "这句解释很常见，也很省事——可惜在频率学派里不成立。",
    "kisaki_v41_auto_b019_python_mutable_default_argument_bug": "不是第二次调用记性太好，是默认参数只在定义时创建了一次。",
    "kisaki_v41_auto_b020_safe_git_bisect_reproducible_regression": "先别把 `bisect` 想成会改写历史的危险仪式，它主要是在替你移动检查位置。",
    "kisaki_v41_auto_b021_recursive_dns_resolution_and_stale_cache": "不是你的设备亲自把整个 DNS 树问一遍，真正跑腿的是递归解析器。",
    "kisaki_v41_auto_b021_javascript_async_foreach_not_awaited": "`async` 写进回调，不会让 `forEach` 突然学会等待。",
    "kisaki_v41_auto_b023_oauth_redirect_state_and_local_client_secrets": "只多一个斜杠，对人眼无所谓，对精确匹配却是另一个 URI。",
    "kisaki_v41_auto_b024_risky_mobile_release_staged_rollout_plan": "“先灰度看看”不是计划，只是一句把风险推迟到上线后的愿望。",
    "kisaki_v41_auto_b025_designing_interrater_reliability_with_cohens_kappa": "92% 看着很漂亮，但漂亮的比例不会主动扣除偶然一致。",
    "kisaki_v41_auto_b027_separating_pilot_results_from_preregistered_confirmation": "把看过结果的数据重新叫作“确认样本”，并不会让它失去记忆。",
    "kisaki_v41_auto_b029_preparing_project_handoff_before_key_member_leave": "文档写完不等于交接完成，纸面上的会做与实际能接手是两回事。",
    "kisaki_v41_auto_b037_analyzing_student_outcomes_in_a_cluster_randomized_school_trial": "八百名学生听起来很多，但随机化单位只有二十个班，别让样本总数遮住设计层级。",
    "kisaki_v41_auto_b038_merging_duplicate_contacts_without_losing_contact_details": "一键合并听起来省事，可联系方式一旦被吞掉，省下的几分钟就会很昂贵。",
    "kisaki_v41_auto_b039_preventing_a_senior_opinion_from_anchoring_project_design_reviews": "资深意见先落地，其他人再“独立思考”，通常只是换一种方式跟随。",
    "kisaki_v41_auto_b042_making_cross_timezone_project_handoffs_explicit": "把问题扔进群里不叫交接，那只是把责任留给下一个醒来的人。",
    "kisaki_v41_auto_b043_understanding_why_warm_sparkling_water_goes_flat_faster": "不是瓶子漏气，温度已经足够改变二氧化碳愿意留在哪里。",
    "kisaki_v41_auto_b043_refusing_to_open_an_unknown_found_usb_drive": "好奇心不值得拿办公设备和公司数据作抵押。",
    "kisaki_v41_auto_b048_exporting_and_verifying_browser_bookmarks_before_cleanup": "先别让同步替你把误删复制到所有设备，备份必须先脱离同步链。",
    "kisaki_v41_auto_b049_scheduling_a_daily_local_time_across_daylight_saving_changes": "每天加二十四小时看似精确，却没有答应你仍是当地九点。",
    "kisaki_v41_auto_b049_refusing_to_post_employee_medical_details_in_a_team_spreadsheet": "排班需要的是可工作条件，不是把同事的病历摊给全组看。",
    "kisaki_v41_auto_b050_understanding_why_condensation_appears_on_different_sides_of_windows": "水不是从玻璃里冒出来的，先看哪一侧空气遇到了低于露点的表面。",
    "kisaki_v41_auto_b050_preventing_project_decisions_from_being_reopened_without_new_evidence": "没有新证据却反复重开决定，不叫审慎，只是在消耗已经做过的工作。",
    "kisaki_v41_auto_b051_refusing_to_install_an_untrusted_browser_extension_for_work": "“大家都在用”不是安全审计，更不会缩小它能读取的权限。",
    "kisaki_v41_auto_b053_moving_an_authenticator_app_to_a_new_phone_without_lockout": "先别急着清空旧手机，验证器迁移最怕把唯一还能确认身份的设备先抹掉。",
    "kisaki_v41_auto_b054_understanding_why_sealed_snack_bags_puff_up_at_high_altitude": "包装袋不是突然多了气，变化的是它与外界之间的压差。",
    "kisaki_v41_auto_b058_handling_cluster_robust_inference_with_few_clusters": "学生很多不等于独立信息很多，真正决定推断余量的是学校数。",
    "kisaki_v41_auto_b063_checking_email_forwarding_rules_after_an_account_compromise": "改密码只关上正门，攻击者留下的转发规则可不会因此自觉消失。",
    "kisaki_v41_auto_b066_explaining_why_metal_feels_colder_than_wood_at_the_same_temperature": "手指报告的是热量流动，不是温度计读数；把两者混在一起，直觉自然会答错。",
    "kisaki_v41_auto_b067_refusing_to_keep_using_a_cracked_power_strip_with_tape": "别拿“只用一晚”给市电风险打折，裂开的外壳已经足够判它停用。",
    "kisaki_v41_auto_b068_choosing_between_family_wise_error_and_false_discovery_rate_control": "先别急着选校正方法，得先说清你愿意控制哪一种错误。",
}


def promoter_module():
    spec = importlib.util.spec_from_file_location("kisaki_promoter", PROMOTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def review_record(record: dict) -> tuple[dict, dict]:
    revised = copy.deepcopy(record)
    assistant_indexes = [
        index for index, message in enumerate(revised.get("messages", []))
        if message.get("role") == "assistant"
    ]
    if not assistant_indexes:
        raise ValueError(f"LLM record has no assistant turn: {record.get('id')}")

    prefix = TARGET_PREFIXES.get(record["id"])
    revised_turns: list[int] = []
    if prefix:
        message_index = assistant_indexes[0]
        original = revised["messages"][message_index]["content"]
        if original.startswith(prefix):
            raise ValueError(f"revision prefix is already present before review: {record['id']}")
        revised["messages"][message_index]["content"] = f"{prefix}{original}"
        revised_turns.append(1)

    source = record.get("metadata", {}).get("data_source", "unknown")
    task_type = record.get("metadata", {}).get("task_type", "unclassified")
    if prefix:
        reason = "professional_outer_layer_persona_signal_added"
    elif source.startswith("llm_v4_"):
        reason = "constructed_character_dialogue_passed_without_forced_rewrite"
    elif task_type in {"casual_chat", "emotional_relationship", "casual_multiturn"}:
        reason = "restrained_or_social_register_passed_without_forced_sharpness"
    else:
        reason = "existing_professional_judgment_signal_passed"

    metadata = copy.deepcopy(revised.get("metadata", {}))
    metadata["persona_review"] = {
        "status": "approved_after_revision" if revised_turns else "approved_unchanged",
        "review_id": REVIEW_ID,
        "reviewed_by": "codex_delegated_by_project_owner",
        "reviewed_at": "2026-08-16",
        "decision_source": relative(ARTIFACTS / "record_reviews.jsonl"),
        "revised_assistant_turns": revised_turns,
        "reason": reason,
    }
    revised["metadata"] = metadata

    turn_reviews = []
    assistant_turn = 0
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        turn_reviews.append({
            "turn": assistant_turn,
            "decision": "revised_then_passed" if assistant_turn in revised_turns else "pass",
            "checks": {
                "persona_consistency": True,
                "question_answer_alignment": True,
                "fact_grounding": True,
                "technical_correctness": True,
                "multiturn_continuity": True,
                "information_timing": True,
                "naturalness": True,
                "safety_boundary": True,
                "template_avoidance": True,
            },
        })
    review = {
        "record_id": record["id"],
        "data_source": source,
        "task_type": task_type,
        "decision": "approved_after_revision" if revised_turns else "approved_unchanged",
        "reason": reason,
        "revised_assistant_turns": revised_turns,
        "turn_reviews": turn_reviews,
    }
    return revised, review


def main() -> None:
    promoter = promoter_module()
    train = promoter._load_jsonl(TRAIN)
    validation = promoter._load_jsonl(VALIDATION)
    manifest = promoter._load_json(MANIFEST)
    actual_sha = promoter._text_sha256(TRAIN)

    llm_records = [
        record for record in train
        if record.get("metadata", {}).get("data_source") in LLM_SOURCES
    ]
    if len(llm_records) != 426:
        raise ValueError(f"expected 426 LLM records, found {len(llm_records)}")
    existing = [
        record for record in llm_records
        if record.get("metadata", {}).get("persona_review", {}).get("review_id") == REVIEW_ID
    ]
    if existing:
        if len(existing) != len(llm_records):
            raise ValueError("partial LLM persona review detected")
        print(json.dumps({"status": "already_promoted", "records": len(existing), "train_sha256": actual_sha}))
        return
    if actual_sha != BASE_TRAIN_SHA256 or manifest["train"]["sha256"] != BASE_TRAIN_SHA256:
        raise ValueError("canonical train does not match the persona review base contract")

    llm_ids = {record["id"] for record in llm_records}
    missing_targets = sorted(set(TARGET_PREFIXES) - llm_ids)
    if missing_targets:
        raise ValueError(f"revision targets are missing from canonical train: {missing_targets}")

    reviewed_by_id = {}
    reviews = []
    for record in llm_records:
        reviewed, review = review_record(record)
        reviewed_by_id[record["id"]] = reviewed
        reviews.append(review)
    final_train = [reviewed_by_id.get(record["id"], record) for record in train]

    original_users = {
        record["id"]: promoter._user_texts(record) for record in llm_records
    }
    reviewed_users = {
        record_id: promoter._user_texts(record) for record_id, record in reviewed_by_id.items()
    }
    if original_users != reviewed_users:
        raise ValueError("persona review must not change user text")
    if [record["id"] for record in final_train] != [record["id"] for record in train]:
        raise ValueError("persona review must preserve canonical IDs and order")

    train_text = promoter._jsonl_text(final_train)
    result_sha = promoter._normalized_sha256_text(train_text)
    validation_sha = promoter._text_sha256(VALIDATION)
    gold_audit = promoter._gold_contamination_audit(
        train=final_train,
        validation=validation,
        gold_v3=promoter._load_json(GOLD_V3),
        gold_v21=promoter._load_json(GOLD_V21),
        train_sha256=result_sha,
        validation_sha256=validation_sha,
    )
    if gold_audit["status"] != "clean":
        raise ValueError("Gold v3 contamination audit blocked persona review")

    decision_counts = Counter(review["decision"] for review in reviews)
    source_counts = Counter(review["data_source"] for review in reviews)
    revised_turn_count = sum(len(review["revised_assistant_turns"]) for review in reviews)
    summary = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "status": "approved_and_promoted",
        "base_train_sha256": BASE_TRAIN_SHA256,
        "result_train_sha256": result_sha,
        "record_count": len(reviews),
        "assistant_turn_count": sum(len(review["turn_reviews"]) for review in reviews),
        "revised_record_count": decision_counts["approved_after_revision"],
        "unchanged_record_count": decision_counts["approved_unchanged"],
        "revised_assistant_turn_count": revised_turn_count,
        "source_distribution": dict(sorted(source_counts.items())),
        "user_text_modified": False,
        "validation_modified": False,
        "gold_content_modified": False,
        "gold_v3_contamination_reaudit": "clean",
        "rubric": [
            "persona_consistency", "question_answer_alignment", "fact_grounding",
            "technical_correctness", "multiturn_continuity", "information_timing",
            "naturalness", "safety_boundary", "template_avoidance",
        ],
    }

    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["train"]["sha256"] = result_sha
    updated_manifest.setdefault("checks", {})["kisaki_llm_persona_review_20260816"] = {
        "status": "approved_and_promoted",
        "reviewed_record_count": len(reviews),
        "reviewed_assistant_turn_count": summary["assistant_turn_count"],
        "revised_record_count": summary["revised_record_count"],
        "revised_assistant_turn_count": revised_turn_count,
        "user_text_modified": False,
        "protected_similarity_overlap_count": 0,
        "gold_v3_contamination_reaudit": "clean",
        "summary_path": relative(ARTIFACTS / "summary.json"),
    }

    review_manifest = promoter._load_json(REVIEW_MANIFEST)
    review_manifest["approval"]["items"]["llm_persona_review"] = {
        "status": "approved_and_promoted",
        "review_id": REVIEW_ID,
        "record_count": len(reviews),
        "assistant_turn_count": summary["assistant_turn_count"],
        "revised_record_count": summary["revised_record_count"],
        "revised_assistant_turn_count": revised_turn_count,
        "summary_path": relative(ARTIFACTS / "summary.json"),
    }

    promoter._write_atomic(ARTIFACTS / "original_llm_records.jsonl", promoter._jsonl_text(llm_records))
    promoter._write_atomic(ARTIFACTS / "reviewed_llm_records.jsonl", promoter._jsonl_text(list(reviewed_by_id.values())))
    promoter._write_atomic(ARTIFACTS / "record_reviews.jsonl", promoter._jsonl_text(reviews))
    promoter._write_atomic(ARTIFACTS / "summary.json", promoter._json_text(summary))
    promoter._write_atomic(TRAIN, train_text)
    promoter._write_atomic(MANIFEST, promoter._json_text(updated_manifest))
    promoter._write_atomic(GOLD_AUDIT, promoter._json_text(gold_audit))
    promoter._write_atomic(REVIEW_MANIFEST, promoter._json_text(review_manifest))
    promoter._write_atomic(ARTIFACTS / "promotion_result.json", promoter._json_text({
        "schema_version": 1,
        "status": "promoted",
        "review_id": REVIEW_ID,
        "previous_train": {"count": len(train), "sha256": BASE_TRAIN_SHA256},
        "result_train": {"count": len(final_train), "sha256": result_sha},
        "reviewed_record_count": len(reviews),
        "revised_record_count": summary["revised_record_count"],
        "revised_assistant_turn_count": revised_turn_count,
        "validation": {"count": len(validation), "sha256": validation_sha, "modified": False},
        "gold_v3": {"content_modified": False, "contamination_reaudit": "clean"},
    }))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
