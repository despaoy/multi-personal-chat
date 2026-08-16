#!/usr/bin/env python3
"""Generate review-only, real-user Kisaki augmentation candidates with DeepSeek.

The script never mutates frozen V4 data. It sends the approved character
profile, system prompt, and all 1,598 direct Kisaki lines as a stable prefix,
then asks for small batches of realistic task conversations. API keys are
loaded from the environment/.env and are never written to artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl"
TRAIN_PATH = ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl"
VALIDATION_PATH = ROOT / "backend/data/character_dialogues/experiments/v4/validation.jsonl"
PROFILE_PATH = ROOT / "docs/research/review_packets/kisaki_v4/01_PROFILE_PROMPT/01_character_profile.md"
PROMPT_PATH = ROOT / "docs/research/review_packets/kisaki_v4/01_PROFILE_PROMPT/02_system_prompt_v3.md"
DEFAULT_OUTPUT = ROOT / "backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_round01"
DEFAULT_REVIEW = ROOT / "docs/research/review_packets/kisaki_v4/11_AUGMENTATION_DEEPSEEK_ROUND01"


PILOT_SPECS: list[dict[str, str]] = [
    {"id": "casual_001", "task_type": "casual_chat", "brief": "用户刚结束一天实验，觉得疲惫，只想随意聊两句。"},
    {"id": "casual_002", "task_type": "casual_chat", "brief": "用户纠结晚饭吃什么，希望得到简单建议。"},
    {"id": "casual_003", "task_type": "casual_chat", "brief": "天气突然转凉，用户问角色今天冷不冷。"},
    {"id": "casual_004", "task_type": "casual_chat", "brief": "用户想找一部周末看的电影，但没有说明类型。"},
    {"id": "emotion_001", "task_type": "emotional_support", "brief": "用户的实验连续失败，开始怀疑自己是否适合做研究。"},
    {"id": "emotion_002", "task_type": "emotional_support", "brief": "用户拖延了重要任务，既自责又不知道怎样开始。"},
    {"id": "emotion_003", "task_type": "emotional_support", "brief": "用户和朋友发生争执，希望判断是否应该立刻道歉。"},
    {"id": "codegen_001", "task_type": "code_generation", "brief": "用户要一段 Python，将 JSONL 逐行读取并统计 data_source 数量；代码需处理空行和坏 JSON。"},
    {"id": "codegen_002", "task_type": "code_generation", "brief": "用户要一个 FastAPI 健康检查端点，返回 status 与模型是否就绪。"},
    {"id": "codegen_003", "task_type": "code_generation", "brief": "用户要 JavaScript fetch 示例，包含超时、非 2xx 检查和 JSON 解析。"},
    {"id": "debug_001", "task_type": "code_debugging", "brief": "用户遇到 Python AttributeError: 'NoneType' object has no attribute 'get'，只给出 value = cache.get(key); value.get('name')。"},
    {"id": "debug_002", "task_type": "code_debugging", "brief": "用户训练 LoRA 时 CUDA out of memory，询问应优先检查哪些参数，不要求凭空保证修复。"},
    {"id": "debug_003", "task_type": "code_debugging", "brief": "用户 git pull 后发生 merge conflict，不知道应先保留现场还是直接覆盖。"},
    {"id": "knowledge_001", "task_type": "technical_explanation", "brief": "初学者询问 RAG 与把知识直接训练进 LoRA 有什么区别。"},
    {"id": "knowledge_002", "task_type": "summarization", "brief": "用户提供三句项目进度：数据冻结、Gold 待审、训练门禁阻塞；要求总结成两点。"},
    {"id": "knowledge_003", "task_type": "translation", "brief": "用户要求把『实验结果尚不足以支持最终结论』翻译成自然英文。"},
    {"id": "planning_001", "task_type": "planning", "brief": "用户每晚只有两小时，希望制定一周的 LoRA 学习计划。"},
    {"id": "planning_002", "task_type": "technical_explanation", "brief": "用户问 LoRA rank 8、16、32 应如何做控制变量实验。"},
    {"id": "safety_001", "task_type": "safe_clarification", "brief": "用户只说『把旧文件删掉』，没有给路径、所有权和清单。"},
    {"id": "safety_002", "task_type": "safe_refusal", "brief": "用户要求把日志中的真实 API 密钥完整贴出来。"},
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.casefold())


def extract_approved_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:text)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[0].strip() if blocks else text.strip()


def build_source_context() -> tuple[str, dict[str, list[str]], list[dict[str, Any]]]:
    raw = load_jsonl(RAW_PATH)
    if len(raw) != 1598:
        raise ValueError(f"expected 1598 direct lines, found {len(raw)}")
    quote_index: dict[str, list[str]] = {}
    lines = []
    for row in raw:
        quote_index.setdefault(row["text"].strip(), []).append(row["id"])
        lines.append(f"[{row['id']}] {row['text'].strip()}")
    profile = PROFILE_PATH.read_text(encoding="utf-8").strip()
    system_prompt = extract_approved_text(PROMPT_PATH)
    context = (
        "以下资料是唯一人物依据。先理解整体分布，不要机械复制台词，也不要把剧情词强塞进普通任务。\n\n"
        "=== 已审核人物画像 ===\n" + profile + "\n\n"
        "=== 已审核 System Prompt ===\n" + system_prompt + "\n\n"
        "=== 月社妃全部 1,598 条原作直接台词 ===\n" + "\n".join(lines)
    )
    return context, quote_index, raw


def generator_system(source_context: str) -> str:
    return source_context + """

=== 本次任务 ===
你是训练数据设计者，需要模拟真实用户与月社妃之间的可用聊天，而不是续写游戏剧本。

每个样本必须满足：
1. 用户话语像真实聊天、开发或学习场景，不出现“请体现月社妃风格”等元指令。
2. assistant 完整解决用户任务，同时自然体现月社妃的聪慧、简洁、判断力、适度反问或轻微戏谑。
3. 非原作话题不得强行提及琉璃、夜子、理央、彼方、魔法之书、作者、故事或命运。
4. 编程任务中，角色风格只放在解释与判断里；代码、命令、JSON、变量名保持专业中性。
5. 不以“当然可以”“以下是”“作为 AI”开头，不使用固定模板，不为了角色化牺牲正确性。
6. 信息不足时先澄清，不虚构文件、运行结果、实时信息或用户情绪。
7. 每条提供 2 条逐字引用的原作风格证据。引用只证明语气或思维方式，不能作为回答事实。
8. 单轮样本 messages 为 user/assistant 两条；多轮样本为 4 或 6 条并严格交替。
9. 除代码任务外，assistant 通常不超过 180 个汉字。代码任务需给可运行的最小代码和必要说明。
10. 只输出 JSON，不输出分析过程。
"""


def batch_prompt(specs: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            "instruction": "为每个 spec 生成一个候选对话，保持 spec_id 和 task_type 不变。",
            "output_schema": {
                "candidates": [
                    {
                        "spec_id": "string",
                        "task_type": "string",
                        "messages": [{"role": "user|assistant", "content": "string"}],
                        "style_evidence_quotes": ["原作逐字台词 1", "原作逐字台词 2"],
                        "style_rationale": "一句话说明人物风格如何体现在回答中",
                        "verification_notes": ["需要人工或程序核验的事实/代码点"],
                    }
                ]
            },
            "specs": specs,
        },
        ensure_ascii=False,
        indent=2,
    )


def call_deepseek(*, system: str, prompt: str, model: str, temperature: float, timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing; set it in .env or the environment")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            usage = body.get("usage", {})
            metadata = {"request_id": response.headers.get("x-request-id"), "usage": usage}
            return parsed, metadata
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")


def validate_messages(messages: Any) -> list[str]:
    errors = []
    if not isinstance(messages, list) or len(messages) not in {2, 4, 6}:
        return ["messages must contain 2, 4, or 6 entries"]
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            errors.append(f"message {index + 1} must have role={expected}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"message {index + 1} content is empty")
    return errors


def validate_candidate(
    candidate: dict[str, Any],
    spec: dict[str, str],
    quote_index: dict[str, list[str]],
    existing_prompts: set[str],
) -> tuple[list[str], list[str]]:
    errors = validate_messages(candidate.get("messages"))
    warnings = []
    if candidate.get("spec_id") != spec["id"]:
        errors.append("spec_id changed")
    if candidate.get("task_type") != spec["task_type"]:
        errors.append("task_type changed")
    messages = candidate.get("messages") if isinstance(candidate.get("messages"), list) else []
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistants = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    joined_user = "\n".join(users)
    joined_assistant = "\n".join(assistants)
    if any(term in joined_user for term in ("月社妃风格", "扮演月社妃", "训练数据")):
        errors.append("user prompt contains dataset/role meta-instruction")
    if any(term in joined_assistant for term in ("作为AI", "作为 AI", "语言模型", "当然可以", "以下是")):
        errors.append("assistant contains generic AI/template language")
    if spec["task_type"] not in {"casual_chat", "emotional_support"}:
        forced = [name for name in ("琉璃", "夜子", "理央", "彼方", "魔法之书") if name in joined_assistant]
        if forced:
            errors.append("unrelated task forces original lore: " + ", ".join(forced))
    if spec["task_type"] == "code_generation" and "```" not in joined_assistant:
        errors.append("code task has no fenced code")
    for prompt in users:
        if normalize(prompt) in existing_prompts:
            errors.append("user prompt exactly overlaps frozen train/validation")
    quotes = candidate.get("style_evidence_quotes")
    if not isinstance(quotes, list) or len(quotes) != 2:
        errors.append("exactly two style evidence quotes are required")
    else:
        resolved_ids: list[str | None] = []
        for quote in quotes:
            clean = quote.strip()
            exact_ids = quote_index.get(clean, [])
            if exact_ids:
                resolved_ids.append(exact_ids[0])
                continue
            containing = [
                (len(source_text), event_ids[0])
                for source_text, event_ids in quote_index.items()
                if len(clean) >= 4 and clean in source_text
            ]
            resolved_ids.append(min(containing)[1] if containing else None)
        if any(event_id is None for event_id in resolved_ids):
            errors.append("style evidence is not an exact direct-line quote")
        candidate["style_evidence_event_ids"] = resolved_ids
    if len(joined_assistant) > 180 and spec["task_type"] not in {"code_generation", "code_debugging"}:
        warnings.append("non-code response exceeds 180 Chinese characters")
    if spec["task_type"] in {"code_generation", "code_debugging"}:
        warnings.append("code requires independent correctness and safety review")
    return sorted(set(errors)), sorted(set(warnings))


def write_review(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 月社妃 V4.1 DeepSeek 真实用户任务扩写：首轮审核",
        "",
        "> 本文件只包含候选数据，不会自动进入训练集。请审核用户问题是否真实、任务答案是否正确、角色风格是否自然。",
        "",
    ]
    for index, row in enumerate(candidates, 1):
        lines += [f"## {index:02d}. `{row['id']}`", "", f"- task_type：`{row['task_type']}`",
                  f"- status：`{row['status']}`", f"- 自动错误：`{row['automatic_errors']}`",
                  f"- 自动警告：`{row['automatic_warnings']}`", ""]
        turn = 0
        for message in row["messages"]:
            if message["role"] == "user":
                turn += 1
            label = f"用户 {turn}" if message["role"] == "user" else f"妃 {turn}"
            lines += [f"**{label}**", "", message["content"], ""]
        lines += ["**原作风格证据**", ""]
        for quote, event_id in zip(row["style_evidence_quotes"], row.get("style_evidence_event_ids", [])):
            lines.append(f"- `{event_id}`：{quote}")
        lines += ["", f"- 风格说明：{row.get('style_rationale', '')}",
                  f"- 待核验：`{row.get('verification_notes', [])}`", "",
                  "- [ ] 通过", "- [ ] 修改", "- [ ] 排除", "- 审核意见：", "", "---", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_AUGMENT_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--revalidate-existing", action="store_true")
    parser.add_argument("--repair-blocked-evidence", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)
    output_dir = args.output_dir.resolve()
    review_dir = args.review_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    source_context, quote_index, raw = build_source_context()
    frozen = load_jsonl(TRAIN_PATH) + load_jsonl(VALIDATION_PATH)
    existing_prompts = {
        normalize(message["content"])
        for row in frozen
        for message in row.get("messages", [])
        if message.get("role") == "user"
    }
    specs = PILOT_SPECS[: args.limit]
    spec_by_id = {row["id"]: row for row in specs}
    if args.revalidate_existing:
        candidates_path = output_dir / "candidates.jsonl"
        if not candidates_path.exists():
            raise SystemExit(f"existing candidates not found: {candidates_path}")
        all_candidates = load_jsonl(candidates_path)
        for row in all_candidates:
            spec = spec_by_id[row["spec_id"]]
            errors, warnings = validate_candidate(row, spec, quote_index, existing_prompts)
            row["automatic_errors"] = errors
            row["automatic_warnings"] = warnings
            row["status"] = "pending_human_review" if not errors else "blocked_automatic_gate"
        if args.repair_blocked_evidence:
            blocked = [
                row for row in all_candidates
                if row["automatic_errors"] == ["style evidence is not an exact direct-line quote"]
            ]
            if blocked:
                repair_prompt = json.dumps(
                    {
                        "instruction": (
                            "不要修改 messages。只为每个候选重新选择两条风格证据。"
                            "每条证据必须完整复制上方原作直接台词中的一个带 ID 条目正文，"
                            "不得截短、改写或引用人物画像。只输出 JSON。"
                        ),
                        "output_schema": {"repairs": [{"spec_id": "string", "style_evidence_quotes": ["完整原作台词1", "完整原作台词2"]}]},
                        "candidates": [
                            {"spec_id": row["spec_id"], "task_type": row["task_type"], "messages": row["messages"],
                             "invalid_quotes": row["style_evidence_quotes"]}
                            for row in blocked
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                repaired, request_meta = call_deepseek(
                    system=generator_system(source_context), prompt=repair_prompt, model=args.model,
                    temperature=0.2, timeout=args.timeout,
                )
                repair_by_id = {entry.get("spec_id"): entry for entry in repaired.get("repairs", [])}
                for row in blocked:
                    repair = repair_by_id.get(row["spec_id"], {})
                    if isinstance(repair.get("style_evidence_quotes"), list):
                        row["style_evidence_quotes"] = repair["style_evidence_quotes"]
                    spec = spec_by_id[row["spec_id"]]
                    errors, warnings = validate_candidate(row, spec, quote_index, existing_prompts)
                    row["automatic_errors"] = errors
                    row["automatic_warnings"] = warnings
                    row["status"] = "pending_human_review" if not errors else "blocked_automatic_gate"
                raw_path = output_dir / "raw_responses.json"
                raw_responses = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_responses.append({"kind": "evidence_repair", "spec_ids": [row["spec_id"] for row in blocked],
                                      "response": repaired, "request_metadata": request_meta})
                raw_path.write_text(
                    json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
                )
        candidates_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_candidates),
            encoding="utf-8", newline="\n",
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status_counts"] = dict(Counter(row["status"] for row in all_candidates))
        manifest["revalidated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        write_review(all_candidates, review_dir / "review.md")
        print(json.dumps({"revalidated": len(all_candidates), "status_counts": manifest["status_counts"]}, ensure_ascii=False))
        return 0

    system = generator_system(source_context)
    all_candidates: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    for offset in range(0, len(specs), args.batch_size):
        batch = specs[offset : offset + args.batch_size]
        parsed, request_meta = call_deepseek(
            system=system,
            prompt=batch_prompt(batch),
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        returned = parsed.get("candidates", [])
        by_id = {row.get("spec_id"): row for row in returned if isinstance(row, dict)}
        raw_responses.append({"batch": offset // args.batch_size + 1, "spec_ids": [s["id"] for s in batch],
                              "response": parsed, "request_metadata": request_meta})
        for spec in batch:
            candidate = by_id.get(spec["id"])
            if candidate is None:
                candidate = {"spec_id": spec["id"], "task_type": spec["task_type"], "messages": [],
                             "style_evidence_quotes": [], "style_rationale": "", "verification_notes": []}
            errors, warnings = validate_candidate(candidate, spec, quote_index, existing_prompts)
            all_candidates.append({
                "id": f"kisaki_v41_aug_{spec['id']}",
                "spec_id": spec["id"],
                "task_type": spec["task_type"],
                "brief": spec["brief"],
                "messages": candidate.get("messages", []),
                "style_evidence_quotes": candidate.get("style_evidence_quotes", []),
                "style_evidence_event_ids": candidate.get("style_evidence_event_ids", []),
                "style_rationale": candidate.get("style_rationale", ""),
                "verification_notes": candidate.get("verification_notes", []),
                "automatic_errors": errors,
                "automatic_warnings": warnings,
                "status": "pending_human_review" if not errors else "blocked_automatic_gate",
                "metadata": {"data_source": "deepseek_v41_real_user_augmentation", "model": args.model,
                             "temperature": args.temperature, "source_corpus_count": len(raw)},
            })

    (output_dir / "raw_responses.json").write_text(
        json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_candidates),
        encoding="utf-8", newline="\n",
    )
    counts = Counter(row["status"] for row in all_candidates)
    manifest = {
        "schema_version": 1,
        "status": "pending_human_review",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "temperature": args.temperature,
        "candidate_count": len(all_candidates),
        "status_counts": dict(counts),
        "task_counts": dict(Counter(row["task_type"] for row in all_candidates)),
        "source_contract": {
            "direct_line_count": len(raw),
            "raw_sha256": sha256_text(RAW_PATH.read_text(encoding="utf-8")),
            "profile_sha256": sha256_text(PROFILE_PATH.read_text(encoding="utf-8")),
            "system_prompt_sha256": sha256_text(PROMPT_PATH.read_text(encoding="utf-8")),
            "frozen_train_sha256": sha256_text(TRAIN_PATH.read_text(encoding="utf-8")),
            "frozen_validation_sha256": sha256_text(VALIDATION_PATH.read_text(encoding="utf-8")),
        },
        "secrets_persisted": False,
        "formal_training_use_allowed": False,
        "output_files": ["candidates.jsonl", "raw_responses.json", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_review(all_candidates, review_dir / "review.md")
    print(json.dumps({"output_dir": str(output_dir), "review": str(review_dir / 'review.md'),
                      "candidate_count": len(all_candidates), "status_counts": dict(counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
