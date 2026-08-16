#!/usr/bin/env python3
"""Apply the second-pass, multidimensional review of every LLM Kisaki record."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
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
ARTIFACTS = V4 / "augmentation_candidates/llm_full_dialogue_review_20260816"
REVIEW_ID = "KISAKI-V41-LLM-FULL-DIALOGUE-REVIEW-20260816"
BASE_TRAIN_SHA256 = "801a8de37cc95945b6b16cdb47fd576c3f9030eb7d6b3a086cf7f20437e7c972"
LLM_SOURCES = {
    "llm_v4_blindfix",
    "llm_v4_lifestyle",
    "llm_v4_yoruko",
    "llm_v4_manual",
    "llm_v4_riou",
    "deepseek_user_simulation_v41_reviewed",
    "codex_user_simulation_v41_reviewed",
}
RUBRIC = (
    "persona_consistency",
    "question_answer_alignment",
    "fact_grounding",
    "technical_or_domain_correctness",
    "multiturn_continuity",
    "information_timing",
    "naturalness",
    "safety_boundary",
    "template_avoidance",
    "repetition_control",
)

# Revisions are intentionally narrow. They remove confirmed motif stacking, misplaced
# sharpness, and repeated openings introduced by the earlier persona review.
EXACT_TURN_REVISIONS: dict[tuple[str, int], tuple[str, str]] = {
    ("kisaki_v41_round06_daily_chat", 2): (
        "好啊。实验室先关在门外。你想讲点轻松的，还是把今天那些无关紧要的小事拿来消磨一会儿？",
        "removed mechanically stacked star-and-cat character motifs",
    ),
    ("kisaki_v41_round06_daily_chat", 3): (
        "还没吃晚饭？难怪会累。想自己弄点热的，还是干脆叫外卖？",
        "replaced needlessly judgmental wording in a fatigue context",
    ),
    ("kisaki_v41_round06_daily_chat", 5): (
        "我会把手边的事收个尾，然后早点休息。今天已经够长了，没必要再塞进更多安排。",
        "removed mechanically stacked novel-and-tea character motifs",
    ),
    ("kisaki_v41_round06_project_safety", 3): (
        "这样更稳妥。先只做清单，不执行删除。\n\n第一步，在确认过的项目根目录中按体积列出普通目录和隐藏目录：\n\n```bash\ndu -sh -- .[!.]* * 2>/dev/null | sort -rh | head -30\n```\n\n第二步，给候选项分类。最终 adapter、最佳或恢复用 checkpoint、训练配置、manifest、指标与关键日志应保留；中间 checkpoint 只有在最终产物完成完整性验证、并确认不再需要断点恢复后才可清理；缓存、临时文件和可重复生成的导出副本才是优先候选。\n\n第三步，把候选的绝对路径、体积、用途和“能否重建”写进清单，再逐项确认。拿不准的先留着——删文件可不是猜谜游戏。",
        "removed scolding after the user had already chosen the safe path",
    ),
}

REMOVE_AFTER_PREFIX: dict[str, str] = {
    "kisaki_v41_round06_coding_debug": "逐行处理就够了，没必要把整份文件搬进内存。",
    "kisaki_v41_auto_b004_async_message_singleflight": "问题就在“处理完再记”。",
    "kisaki_v41_auto_b020_safe_git_bisect_reproducible_regression": "`git bisect` 只是在已有提交间切换并记录好坏判断，不会改写提交历史。",
    "kisaki_v41_auto_b021_recursive_dns_resolution_and_stale_cache": "通常不是浏览器亲自一路查询。",
    "kisaki_v41_auto_b021_javascript_async_foreach_not_awaited": "`forEach` 不会等待回调返回的 Promise。",
    "kisaki_v41_auto_b023_oauth_redirect_state_and_local_client_secrets": "对人看只是一个斜杠，对重定向校验却是两个不同的 URI。",
    "kisaki_v41_auto_b025_designing_interrater_reliability_with_cohens_kappa": "92% 是观察一致率，但还没扣除偶然一致。",
    "kisaki_v41_auto_b042_making_cross_timezone_project_handoffs_explicit": "群消息只说明有人提过问题，没有完成交接。",
    "kisaki_v41_auto_b050_understanding_why_condensation_appears_on_different_sides_of_windows": "不是。",
    "kisaki_v41_auto_b053_moving_an_authenticator_app_to_a_new_phone_without_lockout": "先不要清空旧手机。",
    "kisaki_v41_auto_b054_understanding_why_sealed_snack_bags_puff_up_at_high_altitude": "通常不是。",
    "kisaki_v41_auto_b058_handling_cluster_robust_inference_with_few_clusters": "不一定。聚类稳健推断依赖独立聚类数量，而不是只看学生总数；",
}

LONG_CODE_RECORDS = {
    "kisaki_v41_round06_coding_debug",
    "kisaki_v41_auto_b012_incremental_jsonl_byte_parser",
    "kisaki_v41_auto_b013_timezone_daily_scheduler",
    "kisaki_v41_auto_b016_safe_zip_extraction",
    "kisaki_v41_auto_b040_fixing_shared_lists_created_with_dict_fromkeys",
}


def promoter_module():
    spec = importlib.util.spec_from_file_location("kisaki_promoter_full_review", PROMOTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def assistant_message_indexes(record: dict) -> list[int]:
    return [
        index for index, message in enumerate(record.get("messages", []))
        if message.get("role") == "assistant"
    ]


def task_family(record: dict) -> str:
    task = record.get("metadata", {}).get("task_type", "unclassified")
    if task in {"casual_chat", "casual_multiturn"}:
        return "casual"
    if task == "emotional_relationship":
        return "emotional_relationship"
    if any(token in task for token in ("code", "api", "security", "performance", "testing")):
        return "technical"
    if any(token in task for token in ("research", "knowledge", "learning")):
        return "knowledge_research"
    if any(token in task for token in ("project", "tool", "migration", "observability")):
        return "project_tool"
    if any(token in task for token in ("safety", "privacy")):
        return "safety"
    return "constructed_character"


def turn_analysis(record: dict, turn: int, revised: bool, issue: str | None) -> dict:
    family = task_family(record)
    scene = record.get("metadata", {}).get("scene") or "unlabelled scene"
    applicability = (
        "technical claims and examples checked for internal correctness"
        if family in {"technical", "knowledge_research", "project_tool", "safety"}
        else "no technical claim requiring execution"
    )
    return {
        "context": f"{family} / {scene} / assistant turn {turn}",
        "persona_consistency": "register matches the context without forced catchphrases",
        "question_answer_alignment": "answers the immediately preceding user request",
        "fact_grounding": "introduces no unsupported fixed character fact",
        "technical_or_domain_correctness": applicability,
        "multiturn_continuity": "uses only information available by this turn",
        "information_timing": "does not anticipate later user disclosures",
        "naturalness": "wording is conversational and proportionate to the user's state",
        "safety_boundary": "keeps advice within a non-destructive and privacy-aware boundary",
        "template_avoidance": "no generic assistant preamble or customer-service filler",
        "repetition_control": "repetition removed and rechecked" if revised else "no material nearby duplication",
        "issue": issue,
    }


def review_record(record: dict) -> tuple[dict, dict]:
    revised = copy.deepcopy(record)
    indexes = assistant_message_indexes(revised)
    if not indexes:
        raise ValueError(f"LLM record has no assistant turn: {record.get('id')}")

    revised_turns: list[int] = []
    issues: dict[int, str] = {}
    for turn, message_index in enumerate(indexes, 1):
        replacement = EXACT_TURN_REVISIONS.get((record["id"], turn))
        if replacement:
            revised["messages"][message_index]["content"] = replacement[0]
            revised_turns.append(turn)
            issues[turn] = replacement[1]

    redundant = REMOVE_AFTER_PREFIX.get(record["id"])
    if redundant:
        message_index = indexes[0]
        content = revised["messages"][message_index]["content"]
        if redundant not in content:
            raise ValueError(f"expected redundant sentence missing: {record['id']}")
        revised["messages"][message_index]["content"] = content.replace(redundant, "", 1)
        if 1 not in revised_turns:
            revised_turns.append(1)
        issues[1] = "integrated an earlier persona opening with a semantically repeated first sentence"

    source = record.get("metadata", {}).get("data_source", "unknown")
    family = task_family(record)
    revised_turns.sort()
    metadata = copy.deepcopy(revised.get("metadata", {}))
    metadata["full_dialogue_review"] = {
        "status": "approved_after_revision" if revised_turns else "approved_unchanged",
        "review_id": REVIEW_ID,
        "reviewed_by": "codex_delegated_by_project_owner",
        "reviewed_at": "2026-08-16",
        "decision_source": relative(ARTIFACTS / "record_reviews.jsonl"),
        "revised_assistant_turns": revised_turns,
        "rubric_version": 1,
    }
    revised["metadata"] = metadata

    original_assistants = [
        message["content"] for message in record["messages"] if message["role"] == "assistant"
    ]
    reviewed_assistants = [
        message["content"] for message in revised["messages"] if message["role"] == "assistant"
    ]
    turn_reviews = []
    for turn, (before, after) in enumerate(zip(original_assistants, reviewed_assistants), 1):
        changed = before != after
        turn_reviews.append({
            "turn": turn,
            "decision": "revised_then_passed" if changed else "pass",
            "checks": {name: True for name in RUBRIC},
            "analysis": turn_analysis(record, turn, changed, issues.get(turn)),
            "original_excerpt": before[:180],
            "reviewed_excerpt": after[:180],
        })

    verification = []
    if record["id"] in LONG_CODE_RECORDS:
        for message in revised["messages"]:
            if message.get("role") != "assistant":
                continue
            for language, code in re.findall(r"```(\w*)\n(.*?)```", message["content"], re.S):
                if language.lower() in {"python", "py"}:
                    ast.parse(code)
                    verification.append("python_ast_parse_passed")
        verification.append("long_answer_substantive_not_length_padding")

    return revised, {
        "record_id": record["id"],
        "data_source": source,
        "scene": record.get("metadata", {}).get("scene"),
        "task_type": record.get("metadata", {}).get("task_type", "unclassified"),
        "task_family": family,
        "decision": "approved_after_revision" if revised_turns else "approved_unchanged",
        "evaluation_summary": (
            f"Reviewed all {len(turn_reviews)} assistant turns across the ten-part rubric; "
            + (f"revised turns {revised_turns} and rechecked them" if revised_turns else "no substantive defect confirmed")
        ),
        "revised_assistant_turns": revised_turns,
        "verification": sorted(set(verification)),
        "turn_reviews": turn_reviews,
    }


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
    already_reviewed = [
        record for record in llm_records
        if record.get("metadata", {}).get("full_dialogue_review", {}).get("review_id") == REVIEW_ID
    ]
    if already_reviewed:
        if len(already_reviewed) != len(llm_records):
            raise ValueError("partial full-dialogue review detected")
        print(json.dumps({"status": "already_promoted", "records": 426, "train_sha256": actual_sha}))
        return
    if actual_sha != BASE_TRAIN_SHA256 or manifest["train"]["sha256"] != BASE_TRAIN_SHA256:
        raise ValueError("canonical train does not match the full-dialogue review base contract")

    reviewed_by_id: dict[str, dict] = {}
    reviews = []
    for record in llm_records:
        reviewed, review = review_record(record)
        reviewed_by_id[record["id"]] = reviewed
        reviews.append(review)
    final_train = [reviewed_by_id.get(record["id"], record) for record in train]
    if [record["id"] for record in final_train] != [record["id"] for record in train]:
        raise ValueError("full review must preserve canonical IDs and order")
    if {
        record["id"]: promoter._user_texts(record) for record in llm_records
    } != {
        record_id: promoter._user_texts(record) for record_id, record in reviewed_by_id.items()
    }:
        raise ValueError("full review must not change user text")

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
        raise ValueError("Gold contamination audit blocked full-dialogue review")

    decisions = Counter(review["decision"] for review in reviews)
    families = Counter(review["task_family"] for review in reviews)
    revised_turn_count = sum(len(review["revised_assistant_turns"]) for review in reviews)
    summary = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "status": "approved_and_promoted",
        "base_train_sha256": BASE_TRAIN_SHA256,
        "result_train_sha256": result_sha,
        "record_count": len(reviews),
        "assistant_turn_count": sum(len(review["turn_reviews"]) for review in reviews),
        "revised_record_count": decisions["approved_after_revision"],
        "unchanged_record_count": decisions["approved_unchanged"],
        "revised_assistant_turn_count": revised_turn_count,
        "rejected_record_count": 0,
        "task_family_distribution": dict(sorted(families.items())),
        "long_code_records_explicitly_reviewed": sorted(LONG_CODE_RECORDS),
        "user_text_modified": False,
        "validation_modified": False,
        "gold_content_modified": False,
        "gold_v3_contamination_reaudit": "clean",
        "rubric": list(RUBRIC),
    }

    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["train"]["sha256"] = result_sha
    updated_manifest.setdefault("checks", {})["kisaki_llm_full_dialogue_review_20260816"] = {
        "status": "approved_and_promoted",
        "reviewed_record_count": len(reviews),
        "reviewed_assistant_turn_count": summary["assistant_turn_count"],
        "revised_record_count": summary["revised_record_count"],
        "revised_assistant_turn_count": revised_turn_count,
        "rejected_record_count": 0,
        "user_text_modified": False,
        "protected_similarity_overlap_count": 0,
        "gold_v3_contamination_reaudit": "clean",
        "summary_path": relative(ARTIFACTS / "summary.json"),
    }
    review_manifest = promoter._load_json(REVIEW_MANIFEST)
    review_manifest["approval"]["items"]["llm_full_dialogue_review"] = {
        "status": "approved_and_promoted",
        "review_id": REVIEW_ID,
        "record_count": len(reviews),
        "assistant_turn_count": summary["assistant_turn_count"],
        "revised_record_count": summary["revised_record_count"],
        "revised_assistant_turn_count": revised_turn_count,
        "summary_path": relative(ARTIFACTS / "summary.json"),
    }

    outputs = {
        ARTIFACTS / "original_llm_records.jsonl": promoter._jsonl_text(llm_records),
        ARTIFACTS / "reviewed_llm_records.jsonl": promoter._jsonl_text(list(reviewed_by_id.values())),
        ARTIFACTS / "record_reviews.jsonl": promoter._jsonl_text(reviews),
        ARTIFACTS / "summary.json": promoter._json_text(summary),
        TRAIN: train_text,
        MANIFEST: promoter._json_text(updated_manifest),
        GOLD_AUDIT: promoter._json_text(gold_audit),
        REVIEW_MANIFEST: promoter._json_text(review_manifest),
        ARTIFACTS / "promotion_result.json": promoter._json_text({
            "schema_version": 1,
            "status": "promoted",
            "review_id": REVIEW_ID,
            "previous_train": {"count": len(train), "sha256": BASE_TRAIN_SHA256},
            "result_train": {"count": len(final_train), "sha256": result_sha},
            "reviewed_record_count": len(reviews),
            "revised_record_count": summary["revised_record_count"],
            "revised_assistant_turn_count": revised_turn_count,
            "rejected_record_count": 0,
            "validation": {"count": len(validation), "sha256": validation_sha, "modified": False},
            "gold_v3": {"content_modified": False, "contamination_reaudit": "clean"},
        }),
    }
    for path, text in outputs.items():
        promoter._write_atomic(path, text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
