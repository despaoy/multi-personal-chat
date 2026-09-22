"""Generate locally blinded reply comparisons through the real prompt/guard path.

Synthetic dialogue only; no database, LoRA, RAG, memory writes or remote calls.
The three arms isolate all-non-safety state review and then contextual policy.
No automated character-quality score or human approval is manufactured.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from character.contextual_policy import ContextualDecisionPolicy  # noqa: E402
from character.models import RelationshipStage  # noqa: E402
from character.profile_registry import CharacterProfileRegistry  # noqa: E402
from character.semantic_state_estimator import SemanticStateEstimator  # noqa: E402
from inference.generation_request import GenerationRequest, generate_character_response  # noqa: E402
from inference.lora_registry import get_lora_system_prompt  # noqa: E402
from services.character_context import CharacterContextService, TurnInput  # noqa: E402

ARMS = ("rules", "semantic_all", "semantic_all_policy")
CHARACTER_ID = "tsukiyashiro_kisaki"
RELATIONSHIP_STAGES = get_args(RelationshipStage)


class EmptyRepository:
    def __init__(self, stage="stranger"):
        if stage not in RELATIONSHIP_STAGES:
            raise ValueError("unknown fixture relationship stage")
        self.stage = stage

    async def get_relationship_record(self, *args):
        return {"relationship_stage": self.stage, "interaction_count": 12}


class EmptyMemories:
    async def load_relevant_memories(self, *args, **kwargs):
        return (), 0


class EmptyMessages:
    async def list_recent_conversation_history(self, *args, **kwargs):
        return ()


def check_environment():
    # None means environment-controlled in the production constructor. Refuse
    # ambient opt-ins rather than silently contaminating the baseline arm.
    for key in ("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "CONTEXTUAL_DECISION_POLICY_ENABLED"):
        if os.getenv(key, "false").lower().strip() in {"true", "1", "yes", "on"}:
            raise ValueError(f"offline isolation requires {key} disabled")


def validate_cases(cases):
    if not cases:
        raise ValueError("empty reply fixture")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("case needs an ID")
        if case["id"] in seen:
            raise ValueError("duplicate case ID")
        seen.add(case["id"])
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("case needs a query")
        history = case.get("history", [])
        if not isinstance(history, list) or any(
            not isinstance(row, dict)
            or row.get("role") not in {"user", "assistant"}
            or not isinstance(row.get("content"), str)
            for row in history
        ):
            raise ValueError("invalid dialogue history")


async def evaluate(
    cases,
    reviewer,
    *,
    progress=None,
    profile_registry=None,
    persona_prompt=None,
    character_id=CHARACTER_ID,
    arms=ARMS,
    relationship_stage="stranger",
    capture_reply_attempts=False,
):
    validate_cases(cases)
    check_environment()
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("unknown or duplicate experimental arms")
    registry = CharacterProfileRegistry() if profile_registry is None else profile_registry
    persona = get_lora_system_prompt("kisaki") if persona_prompt is None else persona_prompt
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("a nonempty persona prompt is required")
    rows = []
    for case in cases:
        for arm in arms:
            estimator = (
                None
                if arm == "rules"
                else SemanticStateEstimator(reviewer, timeout_seconds=120, review_mode="all_non_safety")
            )
            policy = ContextualDecisionPolicy(reviewer, timeout_seconds=120) if arm.endswith("_policy") else None
            service = CharacterContextService(
                registry,
                EmptyRepository(relationship_stage),
                EmptyMessages(),
                memory_service=EmptyMemories(),
                semantic_estimator=estimator,
                contextual_policy=policy,
            )
            row = {"id": case["id"], "arm": arm, "status": "error", "reply": ""}
            row["generation_attempt_count"] = 0
            if capture_reply_attempts:
                row["model_reply_attempts"] = []
            error_stage = "prepare_turn"
            try:
                prepared = await service.prepare_turn(
                    TurnInput(
                        case["query"],
                        "evaluation",
                        "offline",
                        case["id"],
                        case["id"],
                        "private",
                        history=tuple(case.get("history", [])),
                    ),
                    character_id,
                )
                if prepared.relationship.stage != relationship_stage:
                    raise RuntimeError("fixture relationship stage was not applied")
                row["diagnostics"] = {
                    "relationship": asdict(prepared.relationship),
                    "reply_guard": asdict(prepared.reply_guard),
                    "interaction": asdict(prepared.interaction),
                    "decision": asdict(prepared.decision),
                    "state_status": prepared.semantic_review_status,
                    "state_reasons": list(prepared.semantic_review_reasons),
                    "state_fallback": prepared.semantic_review_fallback_reason,
                    "policy_status": prepared.contextual_policy_status,
                    "policy_reason": prepared.contextual_policy_reason,
                    "used_memory_ids": list(prepared.compiled.used_memory_ids),
                }
                request = GenerationRequest(
                    # Preserve this frozen research protocol, not runtime defaults.
                    reply_guard_mode="strict",
                    message=case["query"],
                    history=tuple(case.get("history", [])),
                    persona_prompt=persona,
                    character_context=prepared.compiled,
                    reply_guard=prepared.reply_guard,
                    lora_name=None,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=768,
                    context_window_tokens=8192,
                    enable_thinking=False,
                )

                async def adapter(*, messages, case_row=row, **settings):
                    # This CLI intentionally uses the same greedy recipe as its
                    # reviewer. It does not pretend to implement other settings.
                    assert settings["temperature"] == 0.0 and settings["max_tokens"] == 768
                    assert settings["lora_name"] is None and settings["enable_thinking"] is False
                    case_row["generation_attempt_count"] += 1
                    attempt = {"index": case_row["generation_attempt_count"]}
                    if capture_reply_attempts:
                        attempt["prompt_sha256"] = hashlib.sha256(
                            json.dumps(messages, ensure_ascii=False).encode()
                        ).hexdigest()
                        case_row["model_reply_attempts"].append(attempt)
                    try:
                        reply = await reviewer(messages)
                    except Exception as error:
                        attempt.update(status="error", error_type=type(error).__name__)
                        raise
                    attempt.update(status="generated", reply=reply)
                    return reply

                error_stage = "generate_response"
                result = await generate_character_response(request, adapter)
                row.update(
                    status="generated",
                    reply=result.reply,
                    prompt_sha256=hashlib.sha256(
                        json.dumps(result.plan.messages, ensure_ascii=False).encode()
                    ).hexdigest(),
                    prompt_policy_version=result.plan.prompt_policy_version,
                    guard_retried=result.guard_retried,
                    guard_violations=list(result.guard_violations),
                    guard_post_retry_violations=list(result.guard_post_retry_violations),
                    guard_fallback=result.guard_fallback,
                )
            except Exception as error:
                # Record failures, never replace missing model replies with an
                # authored answer or expose provider error text as conversation.
                row["error_type"] = type(error).__name__
                row["error_stage"] = error_stage
                # Fixed local contracts only. Never persist arbitrary provider
                # error messages, which may contain credentials or input text.
                contracts = {
                    "deterministic closed guard fallback is missing": "guard_fallback_missing",
                    "deterministic closed guard fallback did not close the violation": "guard_fallback_invalid",
                    "fixture relationship stage was not applied": "fixture_relationship_mismatch",
                }
                if isinstance(error, RuntimeError) and str(error) in contracts:
                    row["local_contract_error"] = contracts[str(error)]
            rows.append(row)
            if progress is not None:
                progress(row)
            print(f"{case['id']}/{arm}: {row['status']}", flush=True)
    return {
        "benchmark_status": "synthetic_reply_review_packet_not_scored",
        "reply_guard_mode": "strict",
        "quality_claim": "No human ratings yet. Guard success is not character or factual quality.",
        "character_id": character_id,
        "persona_sha256": hashlib.sha256(persona.encode()).hexdigest(),
        "persona_lora_loaded": False,
        "relationship_stage": relationship_stage,
        "reply_attempts_recorded": capture_reply_attempts,
        "arms": list(arms),
        "cases": rows,
        "summary": {
            "distinct_queries": len(cases),
            "replies": len(rows),
            "generation_failures": sum(row["status"] != "generated" for row in rows),
            "guard_fallbacks": sum(bool(row.get("guard_fallback")) for row in rows),
            "human_reviews_completed": 0,
        },
    }


def blind_packet(cases, report, *, seed=20260919):
    arms = tuple(report.get("arms", ARMS))
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("unknown or duplicate experimental arms")
    lookup = {(row["id"], row["arm"]): row for row in report["cases"]}
    expected = {(case["id"], arm) for case in cases for arm in arms}
    if set(lookup) != expected or len(lookup) != len(report["cases"]):
        raise ValueError("report must contain each case and arm exactly once")
    packet = []
    key = []
    for case in cases:
        order = list(arms)
        local_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{case['id']}".encode()).digest()[:8], "big")
        random.Random(local_seed).shuffle(order)
        outputs = {}
        for label, arm in zip("ABC"[: len(order)], order, strict=True):
            row = lookup[(case["id"], arm)]
            outputs[label] = row["reply"] if row["status"] == "generated" else "（本候选生成失败，不能视为正确回答）"
            key.append({"id": case["id"], "label": label, "arm": arm})
        packet.append(
            {"id": case["id"], "query": case["query"], "history": case.get("history", []), "outputs": outputs}
        )
    return packet, key


def markdown_packet(packet):
    lines = [
        "# 最终回复匿名对照审阅",
        "",
        "人物：月社妃；均使用同一底座与人物提示，没有加载 LoRA、RAG 或长期记忆。每题 A/B/C 顺序独立打乱。",
        "这些是合成开发题，尚无人评。请先独立评价，再打开另存的映射文件；相同回答可判平局。",
        "",
        "评价维度：任务完成、用户边界、事实与主体归属、人物一致性、表达自然度。每项可记 1–5 分；证据不足可记无法判断。",
        "人物一致性应结合项目原作资料审查；不了解原作时不要猜评分。生成失败不是拒答成功。",
        "",
    ]
    for case in packet:
        lines += [f"## {case['id']}", ""]
        for message in case["history"]:
            lines += [f"{message['role']}：{message['content']}", ""]
        lines += [f"用户：{case['query']}", ""]
        for label, reply in case["outputs"].items():
            lines += [f"### 候选 {label}", "", reply, ""]
        lines += ["判断：□ A 更好　□ B 更好　□ C 更好　□ 平局　□ 均不合格　□ 无法判断", "", "理由及各项评分：", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "backend/evaluation/fixtures/contextual_policy_smoke.jsonl"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--relationship-stage", choices=RELATIONSHIP_STAGES, default="stranger")
    parser.add_argument("--capture-reply-attempts", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    source = args.input.read_text(encoding="utf-8-sig")
    cases = [json.loads(line) for line in source.splitlines() if line.strip()]
    validate_cases(cases)
    check_environment()
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "scripts/evaluate_contextual_replies.py",
            "backend/character/contextual_policy.py",
            "backend/character/semantic_state_estimator.py",
            "backend/character/context_builder.py",
            "backend/character/situation_analyzer.py",
            "backend/character/decision_policy.py",
            "backend/services/character_context.py",
            "backend/repositories/character_memory.py",
            "backend/character/models.py",
            "backend/inference/generation_request.py",
            "backend/character/output_guard.py",
            "backend/inference/prompt_policy.py",
            "backend/evaluation/offline_reviewer.py",
            "backend/data/character_profiles/tsukiyashiro_kisaki.json",
        )
    }
    from evaluation.offline_reviewer import OfflineTransformersReviewer

    reviewer = OfflineTransformersReviewer(args.model_path, max_input_tokens=8192, max_new_tokens=768, max_seconds=120)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "progress.jsonl").open("x", encoding="utf-8") as progress:

        def write_progress(row):
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()

        report = asyncio.run(
            evaluate(
                cases,
                reviewer,
                progress=write_progress,
                relationship_stage=args.relationship_stage,
                capture_reply_attempts=args.capture_reply_attempts,
            )
        )
    report.update(
        inference=reviewer.metadata,
        inference_calls=reviewer.calls,
        input_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_sha256=source_hashes,
    )
    packet, key = blind_packet(cases, report)
    for name, value in (
        ("report.json", report),
        ("blinded.json", packet),
        ("mapping_do_not_open_before_review.json", key),
    ):
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    with (args.output_dir / "review.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown_packet(packet))
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["summary"]["generation_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
