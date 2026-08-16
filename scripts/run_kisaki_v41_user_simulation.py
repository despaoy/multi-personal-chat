#!/usr/bin/env python3
"""Run human-authored user conversations against a DeepSeek Kisaki role."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
sys.path.insert(0, str(ROOT / "scripts"))

from generate_kisaki_v41_augmentation import (  # noqa: E402
    build_source_context,
    extract_approved_text,
    sha256_text,
    RAW_PATH,
    PROFILE_PATH,
    PROMPT_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)


DEFAULT_OUTPUT = ROOT / "backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_user_simulation_round06"
DEFAULT_REVIEW = ROOT / "docs/research/review_packets/kisaki_v4/16_DEEPSEEK_USER_SIMULATION_ROUND06"
CONTEXT_POLICY_PATH = ROOT / "docs/research/review_packets/kisaki_v4/01_PROFILE_PROMPT/03_contextual_expression_policy_draft.md"


SESSIONS: list[dict[str, Any]] = [
    {
        "session_id": "daily_chat",
        "title": "普通用户的晚间闲聊",
        "task_type": "casual_multiturn",
        "user_turns": [
            "我刚从实验室回来，今天什么都没跑通，有点累。",
            "先不谈实验了，你陪我随便聊一会儿吧。",
            "我连晚饭都没吃，现在又懒得做，你觉得吃什么好？",
            "太复杂的我可不想弄，最好十分钟就能吃上。",
            "行，那我去弄。你今晚准备做什么？",
        ],
    },
    {
        "session_id": "research_learning",
        "title": "初学者讨论 LoRA 实验",
        "task_type": "research_multiturn",
        "user_turns": [
            "我刚开始学 LoRA，rank、alpha、target modules 一多就有点乱。",
            "我想先比较 rank 8、16、32，是不是一次把 alpha 也一起改了更快？",
            "那 alpha 应该怎么固定？我看到有人用 alpha 等于 rank，也有人用两倍。",
            "如果只跑一次，结果是不是也能拿来说明哪个参数最好？",
            "我明白了。帮我把第一轮实验压缩成一个最小可执行方案吧。",
        ],
    },
    {
        "session_id": "coding_debug",
        "title": "真实代码编写与追问",
        "task_type": "coding_multiturn",
        "user_turns": [
            "帮我写个 Python 函数，逐行读取 JSONL，然后统计每种 data_source 有多少条。坏行别让程序直接退出。",
            "如果某一行能解析，但里面没有 data_source，能顺便统计成 missing 吗？",
            "我还想知道坏 JSON 在第几行，直接忽略好像不方便排查。",
            "文件可能有几十万行，这种写法会不会一下把所有内容都读进内存？",
            "最后给我一个能直接运行的完整版本，再加两个最小测试用例。",
        ],
    },
    {
        "session_id": "project_safety",
        "title": "项目操作、澄清与安全边界",
        "task_type": "project_multiturn",
        "user_turns": [
            "服务器空间不够了，你帮我把旧文件删掉吧。",
            "就是项目里的旧模型和测试文件，具体路径我也记不清了。",
            "那先不删。你告诉我应该怎么列清单，才不会误删训练结果。",
            "检查日志时发现里面可能有 API 密钥，我直接把完整日志发给你行吗？",
            "好，我先脱敏。除了密钥，日志里还有哪些信息也应该遮住？",
        ],
    },
]


STYLE_MODES: dict[tuple[str, int], str] = {
    ("daily_chat", 1): "restrained_care",
    ("daily_chat", 2): "light_banter",
    ("daily_chat", 3): "restrained_care",
    ("daily_chat", 4): "calm_precise",
    ("daily_chat", 5): "soft_personal",
    ("research_learning", 1): "calm_precise",
    ("research_learning", 2): "pointed_correction",
    ("research_learning", 3): "calm_precise",
    ("research_learning", 4): "pointed_correction",
    ("research_learning", 5): "decisive_delivery",
    ("coding_debug", 1): "calm_precise",
    ("coding_debug", 2): "calm_precise",
    ("coding_debug", 3): "light_banter",
    ("coding_debug", 4): "calm_precise",
    ("coding_debug", 5): "decisive_delivery",
    ("project_safety", 1): "firm_boundary",
    ("project_safety", 2): "pointed_correction",
    ("project_safety", 3): "decisive_delivery",
    ("project_safety", 4): "firm_boundary",
    ("project_safety", 5): "calm_precise",
}


SHARP_STYLE_ANCHORS = """
=== 原作锋利表达锚点 ===
以下均为月社妃原作逐字台词，只用于理解她如何作判断，不要求逐句复制：
- “谢了，我不用。”：边界明确，不用客服式铺垫。
- “一个不必回答的愚蠢问题呢。”：先看穿问题本身，再决定是否回答。
- “我对我的挖苦话喜欢得很。”：她的锋利是自觉而从容的，不是无端发火。
- “他们算什么？谈不到讨厌。”：判断简短，拒绝顺着对方预设的情绪走。
- “谁知道呢？这得从真货的定义说起。”：不接受含混前提，会先追究定义。
- “噗噗，还以为你要说什么——这种事是不可能吧？”：亲近场景可戏谑，但笑声不能机械重复。

锋利的核心是：看穿对方省略的条件、偷懒的捷径、自相矛盾或真正关心的事，然后直接点明。
它不是每句话加“哼”，不是辱骂用户，也不是把危险请求一律写成客服拒绝。
"""


MODERN_STYLE_FEW_SHOTS = """
=== 现代任务的风格迁移示例 ===
反例（对疲惫者错误地保持锋利）：
用户：今天实验全失败了，我有点累。
回答：连这点失败都受不了？真是脆弱。

正例（克制关心）：
用户：今天实验全失败了，我有点累。
回答：今天不讲理的是实验，不是你。先休息吧——逞强也不会让结果自己变好。

反例（普通助手）：
用户：我想一次把三个参数都改了，能快点吗？
回答：可以，但建议使用控制变量法。以下是具体步骤。

正例（月社妃式）：
用户：我想一次把三个参数都改了，能快点吗？
回答：想用一次实验回答三个问题？真是贪心。先选一个变量，其余固定；否则结果再漂亮，也不知道该把功劳算给谁。

反例（普通助手）：
用户：坏 JSON 直接跳过就行吧？
回答：可以。建议记录错误信息，方便后续排查。

正例（月社妃式）：
用户：坏 JSON 直接跳过就行吧？
回答：连它坏在哪一行都不留下，也算处理完成？跳过可以，行号和错误原因必须保留。代码本身仍使用专业、中性的命名。

反例（过度角色化）：
回答：呼呼呼，让本小姐用命运之书替你写下这段代码吧。

正例（角色外层、专业内核）：
回答：条件总算说完整了。最终版本保留坏行行号，也不会一次读完整个文件。随后给出准确、可运行且不含角色词的代码。

这些示例只规定判断方式与语气，不得机械复制句子，也不得每轮都挖苦。
"""


def role_system(source_context: str) -> str:
    approved_prompt = extract_approved_text(PROMPT_PATH)
    context_policy = CONTEXT_POLICY_PATH.read_text(encoding="utf-8").strip()
    return source_context + """

=== 原作语料后的最高优先级角色提示词 ===
下面是项目负责人已审核通过的 System Prompt。必须将它实际表现到每一轮回答中，而不只是理解其内容。
""" + approved_prompt + "\n" + SHARP_STYLE_ANCHORS + "\n" + MODERN_STYLE_FEW_SHOTS + "\n" + context_policy + """

=== 当前对话任务 ===
你只扮演月社妃。用户消息由另一个真实对话参与者逐轮发来，你不得替用户发言、设计问答或输出训练数据结构。

当前用户是与你已经聊过一段时间、关系自然但不属于原作人物的熟悉用户。不要把用户认作琉璃或其他原作角色；也不要退回面对陌生人的礼貌客服语气。

回答要求：
1. 直接自然地承接当前会话，不复述人物设定，不说明自己正在扮演角色，也不讨论训练数据。
2. 先根据情境化表达策略确定本轮模式。锋利不是默认语气：脆弱和认真求助时克制关心，普通技术问题冷静精准，只有偷懒捷径、自相矛盾和高风险越界才明显锋利。
3. 普通闲聊以一至三句为主，像真人一样有自己的态度，可以关心、追问、表达安排；不要写成客服、心理咨询模板、泛泛鸡汤或百科说明。
4. 技术问题必须先保证事实、代码和实验设计正确，再把人物风格放在开场、过渡和结尾。代码、命令、JSON、变量名保持专业中性。
5. 不把琉璃、夜子、理央、彼方、魔法之书、作者、故事、命运等原作元素强塞进无关的现代话题。
6. 不虚构已执行的操作、实时状态或用户未提供的项目信息。条件不足时指出具体缺口，并给出可以立即执行的下一步。
7. 安全判断按实际风险分级：高风险操作先确认范围和备份；普通路径、非敏感 IP 或一般技术信息不做无意义遮盖；密钥、令牌和密码绝不索取或复述。
8. 研究建议要区分独立变量、派生参数与随机波动。编程回答要满足用户全部约束，考虑输入类型、边界和可运行性。
9. 同一会话后续轮次必须保留用户前面提出的全部约束。当前 LoRA 讨论研究的是表征 rank：保持 alpha/r 比例不变，alpha 作为随 rank 改变的派生参数；不得声称固定 alpha 数值能单独隔离 rank 容量。当前 JSONL 代码最终版必须保留坏行行号、验证解码结果是对象，并用不依赖解释器错误原文的测试。
10. 项目文件清单必须给出非破坏性的实际步骤；日志脱敏要区分凭据、个人信息和仍可保留的诊断上下文，不能把所有路径与 IP 一概视为必须删除。
11. 避免以“可以”“当然”“没问题”等助手模板开头，禁止用“需要我……吗”“如果你愿意，我可以……”收尾。不要为了通过人物检查而强加反问、挖苦、笑声或尖锐词。除完整代码外尽量简洁，只输出月社妃对用户说的话。
"""


def style_mode_for(session_id: str, turn: int) -> str:
    return STYLE_MODES[(session_id, turn)]


def request_answer(*, system: str, history: list[dict[str, str]], model: str, temperature: float, timeout: int) -> tuple[str, dict[str, Any]]:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *history],
        "temperature": temperature,
        "max_tokens": 1800,
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
            answer = body["choices"][0]["message"]["content"].strip()
            return answer, {
                "request_id": response.headers.get("x-request-id"),
                "usage": body.get("usage", {}),
                "finish_reason": body["choices"][0].get("finish_reason"),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")


def audit_turn(answer: str, *, session_id: str, turn: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not answer:
        errors.append("empty answer")
    if any(term in answer for term in ("作为AI", "作为 AI", "语言模型", "扮演月社妃")):
        errors.append("AI/role meta-reference")
    prose = re.sub(r"```.*?```", "", answer, flags=re.S).strip()
    if re.match(r"^(?:可以|当然|没问题)(?:\s|[。！,，：:])", prose):
        errors.append("generic assistant opening")
    if re.match(r"^好[。！,，：:]", prose):
        warnings.append("generic assistant opening")
    if any(term in prose for term in ("需要我", "如果你愿意，我可以", "如果需要，我可以")):
        errors.append("generic assistant closing")
    mode = style_mode_for(session_id, turn)
    sharp_signals = (
        "？", "——", "谁知道", "真是", "未免", "这可", "我可", "不必", "恕我",
        "可别", "难道", "居然", "连", "定义", "风险", "不行", "不能", "不会",
    )
    light_signals = ("？", "——", "噗噗", "呼呼呼", "呵呵", "倒是", "总不会", "看来", "难道")
    decisive_signals = ("先", "关键", "条件", "这次", "总算", "别", "不必", "足够", "就这么")
    harsh_terms = ("愚蠢", "自欺欺人", "没脑子", "无能", "活该", "完蛋")
    if mode in {"pointed_correction", "firm_boundary"} and not any(
        signal in prose for signal in sharp_signals
    ):
        errors.append(f"style mode {mode} lacks an appropriate pointed judgment")
    if mode == "light_banter" and not any(signal in prose for signal in light_signals):
        errors.append("style mode light_banter lacks light playful expression")
    if mode in {"restrained_care", "soft_personal"} and any(term in prose for term in harsh_terms):
        errors.append(f"style mode {mode} is inappropriately harsh")
    if mode == "decisive_delivery" and not any(signal in prose for signal in decisive_signals):
        errors.append("style mode decisive_delivery lacks a characterful framing judgment")
    if session_id in {"research_learning", "coding_debug", "project_safety"}:
        forced = [name for name in ("琉璃", "夜子", "理央", "彼方", "魔法之书") if name in answer]
        if forced:
            errors.append("unrelated technical task forces original lore: " + ", ".join(forced))
    if session_id == "coding_debug" and turn in {1, 5} and "```" not in answer:
        errors.append("requested code answer has no fenced code")
    if session_id == "coding_debug" and turn == 5:
        if "bad_lines" not in answer:
            errors.append("final code drops bad-line locations")
        if not re.search(r"isinstance\([^,]+,\s*dict\)", answer):
            errors.append("final code does not validate decoded JSON objects")
        if "Expecting property name enclosed" in answer:
            errors.append("test couples to an interpreter-specific parser message")
    if session_id == "research_learning" and turn == 5:
        ratio_is_explicit = "alpha/r" in answer or ("比例" in answer and "固定" in answer)
        if not ratio_is_explicit or not any(term in answer for term in ("派生", "随 rank", "成比例", "随它缩放")):
            errors.append("rank ablation does not preserve alpha/r as a derived setting")
    if session_id == "project_safety" and turn == 3:
        if not any(term in answer for term in ("Linux", "Windows", "du ", "Get-ChildItem", "只读")):
            errors.append("inventory guidance is not operationally actionable")
    if len(answer) > 2200:
        warnings.append("answer is unusually long")
    if any(term in answer for term in ("直接运行递归删除", "把密钥发给我", "无需确认即可删除")):
        errors.append("unsafe operational advice")
    return errors, warnings


def render_review(sessions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 月社妃 V4.1：Codex 模拟真实用户、DeepSeek 仅扮演角色",
        "",
        "> 下面的用户消息由 Codex 预先编写；DeepSeek 只生成月社妃回答。每段会话连续传递完整历史。",
        "> 当前均为待审核候选，不会自动进入训练集。",
        "",
    ]
    for session in sessions:
        lines += [f"## {session['title']}", "", f"- session_id：`{session['session_id']}`",
                  f"- task_type：`{session['task_type']}`", ""]
        for index in range(0, len(session["messages"]), 2):
            user = session["messages"][index]
            assistant = session["messages"][index + 1]
            audit = session["turn_audits"][index // 2]
            lines += [f"### 第 {index // 2 + 1} 轮", "", "**用户**", "", user["content"], "",
                      "**妃**", "", assistant["content"], "",
                      f"- 情境模式：`{audit['style_mode']}`",
                      f"- 生成尝试：`{audit['attempt_count']}`",
                      f"- 自动错误：`{audit['errors']}`", f"- 自动警告：`{audit['warnings']}`",
                      "- [ ] 通过", "- [ ] 修改", "- [ ] 排除", "- 审核意见：", ""]
        lines += ["---", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_AUGMENT_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)

    output_dir = args.output_dir.resolve()
    review_dir = args.review_dir.resolve()
    protected_outputs = [output_dir / "manifest.json", output_dir / "sessions.json", review_dir / "review.md"]
    if not args.overwrite and any(path.exists() for path in protected_outputs):
        raise FileExistsError("output already exists; choose a new directory or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    source_context, _, raw = build_source_context()
    system = role_system(source_context)

    generated_sessions: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []
    attempt_log: list[dict[str, Any]] = []
    for session in SESSIONS:
        history: list[dict[str, str]] = []
        audits: list[dict[str, Any]] = []
        for turn, user_text in enumerate(session["user_turns"], 1):
            style_mode = style_mode_for(session["session_id"], turn)
            history.append({"role": "user", "content": user_text})
            answer = ""
            errors: list[str] = []
            warnings: list[str] = []
            previous_errors: list[str] = []
            for attempt in range(1, args.max_attempts + 1):
                retry_feedback = ""
                if previous_errors:
                    retry_feedback = (
                        "\n\n=== 当前这一轮必须重答 ===\n"
                        f"本轮情境模式是 {style_mode}。"
                        "上一候选未通过生成门禁：" + "；".join(previous_errors) + "。\n"
                        "重新回答最后一条用户消息，不要提及重答、门禁或这些反馈。"
                    )
                answer, metadata = request_answer(
                    system=system + retry_feedback, history=history, model=args.model,
                    temperature=args.temperature, timeout=args.timeout,
                )
                errors, warnings = audit_turn(answer, session_id=session["session_id"], turn=turn)
                attempt_log.append({
                    "session_id": session["session_id"], "turn": turn, "attempt": attempt,
                    "style_mode": style_mode,
                    "answer": answer, "errors": errors, "warnings": warnings,
                    "accepted": not errors,
                })
                request_log.append({
                    "session_id": session["session_id"], "turn": turn, "attempt": attempt,
                    "style_mode": style_mode,
                    "request_metadata": metadata, "automatic_errors": errors,
                })
                if not errors:
                    break
                previous_errors = errors
            history.append({"role": "assistant", "content": answer})
            attempt_count = sum(
                1 for row in attempt_log
                if row["session_id"] == session["session_id"] and row["turn"] == turn
            )
            audits.append({"turn": turn, "style_mode": style_mode, "attempt_count": attempt_count,
                           "errors": errors, "warnings": warnings})
            turns.append({
                "id": f"kisaki_v41_sim_{session['session_id']}_{turn:02d}",
                "session_id": session["session_id"], "turn": turn,
                "task_type": session["task_type"], "style_mode": style_mode,
                "messages": [dict(message) for message in history],
                "latest_user": user_text, "latest_assistant": answer,
                "status": "pending_human_review" if not errors else "blocked_automatic_gate",
                "automatic_errors": errors, "automatic_warnings": warnings,
                "metadata": {"data_source": "codex_user_deepseek_kisaki_v41", "model": args.model,
                             "temperature": args.temperature, "source_corpus_count": len(raw)},
            })
        generated_sessions.append({
            "session_id": session["session_id"], "title": session["title"], "task_type": session["task_type"],
            "messages": history, "turn_audits": audits, "status": "pending_human_review",
            "metadata": {"data_source": "codex_user_deepseek_kisaki_v41", "model": args.model,
                         "temperature": args.temperature, "source_corpus_count": len(raw)},
        })

    (output_dir / "sessions.json").write_text(
        json.dumps(generated_sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "turns.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in turns), encoding="utf-8", newline="\n"
    )
    (output_dir / "request_log.json").write_text(
        json.dumps(request_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in attempt_log),
        encoding="utf-8", newline="\n",
    )
    status_counts = Counter(row["status"] for row in turns)
    manifest = {
        "schema_version": 1, "status": "pending_human_review", "method": "codex_authored_user_deepseek_role_only",
        "model": args.model, "temperature": args.temperature, "session_count": len(generated_sessions),
        "turn_count": len(turns), "status_counts": dict(status_counts),
        "generation_gate": {"max_attempts": args.max_attempts, "attempt_count": len(attempt_log),
                            "retry_count": len(attempt_log) - len(turns)},
        "candidate_contract": {
            "training_candidate_unit": "complete_session_from_sessions.json",
            "turns_jsonl_purpose": "cumulative_review_trace_only_do_not_train",
            "assistant_answers_preserved_verbatim": True,
            "approved_system_prompt_repeated_after_source_corpus": True,
            "context_conditioned_style_policy": True,
        },
        "source_contract": {"direct_line_count": len(raw),
            "raw_sha256": sha256_text(RAW_PATH.read_text(encoding="utf-8")),
            "profile_sha256": sha256_text(PROFILE_PATH.read_text(encoding="utf-8")),
            "system_prompt_sha256": sha256_text(PROMPT_PATH.read_text(encoding="utf-8")),
            "context_style_policy_sha256": sha256_text(CONTEXT_POLICY_PATH.read_text(encoding="utf-8")),
            "frozen_train_sha256": sha256_text(TRAIN_PATH.read_text(encoding="utf-8")),
            "frozen_validation_sha256": sha256_text(VALIDATION_PATH.read_text(encoding="utf-8"))},
        "secrets_persisted": False, "formal_training_use_allowed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    render_review(generated_sessions, review_dir / "review.md")
    print(json.dumps({"output_dir": str(output_dir), "review": str(review_dir / 'review.md'),
                      "sessions": len(generated_sessions), "turns": len(turns),
                      "status_counts": dict(status_counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
