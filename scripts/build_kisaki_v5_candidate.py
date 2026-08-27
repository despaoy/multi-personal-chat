"""阶段 2：筛选 254 条长篇用户模拟（五维机器评分 + 人工审核材料）。

职责单一：读取 KISAKI-CANONICAL-V4 冻结数据，对两类用户模拟
（codex 251 + deepseek 3）生成审核材料。机器评分只辅助排序与预警，
不能代替最终审核；本脚本不修改 V4、不删除数据、不构建 V5。

五维评分（互不抵消，单一维度差不能被其他维度补偿）：
1. scenario_value        (0-5)  场景研究价值（承诺/冲突/边界/安全/不确定）
2. persona_fidelity      (0-5)  表达接近月社妃画像的程度（机器仅作长度/
                                服务语/讥讽风格代理估计，人工终审）
3. factual_grounding     三态：no_auto_flag（自动未发现问题，仍需人工
                                确认）/ needs_human（出现未引入原作元素，
                                需人工判断）/ auto_fail（自动判定虚构）。
                                语境分级："不是魔法"（否定）与"仿佛魔法"
                                （比喻）不罚；世界观事实声称与人物经历
                                声明判 auto_fail。
4. multiturn_coherence   (0-5)  多轮一致性（机器仅代理时间/物品线索，
                                矛盾检测需人工）
5. generic_assistant_risk (0-5，越低越好) 通用咨询助手/教程/客服风格

prefer_keep 门禁（全部条件缺一不可）：
  scenario_value >= 3 且 persona_fidelity >= 4 且
  factual_grounding = no_auto_flag 且 multiturn_coherence >= 4 且
  generic_assistant_risk <= 1 且 task_type 不在技术排除队列。

技术排除队列：优先使用 metadata.task_type（project_collaboration、
tool_usage、knowledge_explanation、code_debugging、api_* 等）；
正则关键词仅作补充。队内记录 scenario_value 封顶 2、建议
prefer_exclude，人工仍可决定少量保留。

场景重复簇：task_type 分组内，按"对象+冲突行为+目标"概念标签
（同义归一：屡次/频繁/总是→反复等）做 Jaccard 聚类。
仅提示人工确认（同簇建议保留 1–2 条），绝不自动排除。

预算：基线从当前 V4 records 动态计算（原作 + 短构造，监督口径），
不硬编码；短构造"假定全部保留"属初步口径，阶段 3 后必须重算。

产物（experiments/v5_candidate/）：
- simulation_review_packet.json   逐条审核数据（五维 + 三态事实 + 证据）
- review_batches/batch_01.md ...  分批人工审核材料（每批 ≤30 条，
                                  含完整对话、五维、问题回合、选择栏）
- simulation_review_summary.md    摘要（分带、三态、簇、预算初步估算）

决定文件门禁（simulation_review_decisions.json）：
- review_status 必须为 "approved"（reviewed_by 单独存在不代表已批准；
  collect 脚本产出的 draft 不算批准）；
- 决定 ID 集合必须与 254 条模拟 ID 完全相等（未知 ID / 缺失 ID 均报错）；
- 决定值只能是 "keep" 或 "exclude"（kepe/空串/pending 等一律报错，
  绝不静默当成 exclude）。
审核材料中的 revise（待改写）由 collect 脚本转换为 exclude 并标记
needs_revision，改写属后续工作。

--decisions 语义：仅校验决定并输出保留的模拟记录清单；
**不构建完整 V5 candidate**（那是阶段 5 的职责）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v4"
V5_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v5_candidate"
REVIEW_BATCH_DIR = V5_DIR / "review_batches"

SIMULATION_SOURCES = (
    "codex_user_simulation_v41_reviewed",
    "deepseek_user_simulation_v41_reviewed",
)

# 预算基线说明：数值在生成时从 V4 records 动态计算（不硬编码），
# 但短构造部分"假定全部保留"仍是初步口径——阶段 3 复查后需重算。
BUDGET_PRELIMINARY_NOTE = (
    "初步估算：基线数值从当前 V4 records 动态计算，但假定短构造"
    "（llm_v4_reviewed_constructed）全部保留；阶段 3 复查后必须以"
    "实际保留数据重算"
)

# ---------------- 技术排除队列（按 task_type，正则仅补充） ----------------
TECH_TASK_TYPES = frozenset(
    {
        "tool_usage",
        "knowledge_explanation",
        "project_collaboration",
        "technical_learning",
        "research_learning",
        "data_tool_collaboration",
        "api_design",
        "api_contract",
        "api_debugging",
        "code_debugging",
        "testing_debug",
        "data_migration",
        "observability",
        "security_rag",
        "security_upload",
        "performance_debug",
        "frontend_ux",
        "cache_consistency",
        "inference_performance",
        "reliability_design",
    }
)
_TECH_KEYWORD = re.compile(
    r"git|docker|redis|postgres(?:ql)?|sqlalchemy|vllm|api|sse|websocket|"
    r"oauth|webhook|http|css|typescript|python|jsonl?|yaml|k8s|代码|编程|"
    r"bug|调试|数据库|部署|服务器|缓存|lora|微调|复现|算法|前端|后端|"
    r"幂等|并发|回滚|索引|编译|测试用例|报错|traceback",
    re.IGNORECASE,
)

# ---------------- 场景价值正向信号（scene + 用户消息） ----------------
_POSITIVE_SCENARIO_RULES: tuple[tuple[str, str], ...] = (
    ("promise_plan", r"约好|答应|承诺|说好|约定|计划.*一起|一起.*计划"),
    ("conflict_repair", r"吵架|生气|道歉|和好|误会|对不起|闹翻|冷战"),
    (
        "relationship_boundary",
        r"拒绝|未经同意|没有经过|边界|越界|打扰|过分|擅自|占用|替你答应|替自己答应",
    ),
    ("safety", r"安全|危险|紧急|报警|受伤|走失|威胁|求助"),
    ("uncertainty", r"不确定|不知道|也许|可能吧|拿不准|没想好|纠结"),
)
_TIME_MARKERS = re.compile(r"今天|明天|后天|周末|下周|晚上|中午|早上|下午")
_PLACE_ITEM_MARKERS = re.compile(r"在家|宿舍|图书馆|电影院|公园|咖啡|带|买|拿|借|还")

# ---------------- 人物风格 / 通用助手风险（assistant 侧） ----------------
# 原作核心监督字符均值（阶段 1 实测 31.5）
ORIGINAL_AVG_SUPERVISED_CHARS = 31.5
_SARCASM_MARKERS = re.compile(r"哼|……|真是的|随便|无聊|笨蛋|啰嗦|省省|得了|啧")
_SERVICE_LANGUAGE = re.compile(
    r"您好|很高兴|为您服务|帮您|希望这些建议|希望对你有帮助|"
    r"还有什么可以帮|需要我帮|请放心|祝您|如果还有其他问题"
)
_STRUCTURED_REPLY = re.compile(
    r"^\s*(\d+[\.、]|[①②③]|[-*] |# )|首先.*其次|建议如下|第一步|总结一下|综上",
    re.MULTILINE,
)
_EMOTION_TEMPLATE = re.compile(r"加油|会好的|别难过|深呼吸|想开点|都会过去")
_ADVICE_WORD = re.compile(r"建议")

# ---------------- 原作元素分级（factual_grounding） ----------------
# 人物名（未由用户/场景引入即出现 = 至少标记；带经历语境 = 判 fail）
_CHARACTER_NAMES = ("琉璃",)
# 世界观元素（否定/比喻语境 = 不罚；事实声称 = 判 fail）
_WORLD_ELEMENTS = ("魔法",)
# 经历类元素（仅在与经历语境同现时判 fail）
_EXPERIENCE_ELEMENTS = ("点心", "社刊", "妹妹", "父母")
_EXPERIENCE_CONTEXT = re.compile(r"以前|曾|那时|那时候|记得|小时候|生病|上次|当時")
_NEGATION_BEFORE = re.compile(
    r"(?:不是|没有|并非|不带|算不上|谈不上|可不是什么|又不是|哪有什么|哪来的)[^，。！？\n]{0,6}$"
)
_METAPHOR_CONTEXT = re.compile(
    r"(?:解决一切|万能|什么都能解决|灵丹妙药|像.{0,6}|仿佛.{0,6}|如同.{0,6})"
)
_FACTUAL_CLAIM = re.compile(
    r"(?:我们的世界|这个世界|由魔法|魔法世界|魔法浓度|魔法使|魔法少女)"
)

# prefer_keep 门禁阈值
# factual_grounding 三态（no_auto_flag/needs_human/auto_fail）：
# 自动规则只能证明"没发现问题"或"发现问题"，不能证明事实正确。
GATE = {
    "scenario_value_min": 3,
    "persona_fidelity_min": 4,
    "factual_grounding_required": "no_auto_flag",
    "multiturn_coherence_min": 4,
    "generic_assistant_risk_max": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_records() -> tuple[list[dict], dict]:
    """读取 V4 train（只读），校验冻结哈希，返回 (记录列表, manifest)。"""
    manifest = json.loads(
        (V4_DIR / "canonical_dataset_manifest.json").read_text(encoding="utf-8")
    )
    train_path = V4_DIR / "train.jsonl"
    actual = sha256_file(train_path)
    if actual != manifest["train"]["sha256"]:
        raise SystemExit(
            f"[ABORT] train.jsonl sha256 与冻结 manifest 不一致:\n"
            f"  实际 {actual}\n  冻结 {manifest['train']['sha256']}"
        )
    records = []
    with open(train_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"train.jsonl 第 {line_no} 行解析失败: {e}")
    return records, manifest


def classify_record_source(record: dict) -> str:
    """统一来源分类（与阶段 1 清单一致的四类聚合）。"""
    data_source = str(record.get("metadata", {}).get("data_source", ""))
    if data_source == "game_extraction":
        return "game_extraction_current_sft"
    if data_source.startswith("llm_v4_"):
        return "llm_v4_reviewed_constructed"
    if data_source in SIMULATION_SOURCES:
        return data_source
    raise ValueError(f"未知 data_source: {data_source!r}")


def measure_record_exposure(record: dict) -> dict:
    """监督口径暴露量（与训练契约 all/last 一致）。"""
    messages = record.get("messages", [])
    meta = record.get("metadata", {})
    supervision = str(meta.get("assistant_supervision") or "all")
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    supervised = assistant_msgs[-1:] if supervision == "last" else assistant_msgs
    user_msgs = [m for m in messages if m["role"] == "user"]
    sup_chars = sum(len(m.get("content") or "") for m in supervised)
    return {
        "user_turns": len(user_msgs),
        "raw_assistant_messages": len(assistant_msgs),
        "supervised_assistant_targets": len(supervised),
        "supervised_assistant_chars": sup_chars,
        "avg_supervised_chars_per_target": (
            round(sup_chars / len(supervised), 1) if supervised else 0
        ),
    }


def compute_budget_baseline(records: list[dict]) -> dict:
    """从当前 V4 records 动态计算预算基线（监督口径）。

    - game_sup_chars / game_records：原作抽取（冻结核心）；
    - constructed_sup_chars / constructed_records：短构造
      （llm_v4_reviewed_constructed）——**假定全部保留**，
      属初步口径，阶段 3 复查后必须重算。
    未知来源直接报错，不静默计入其他。
    """
    game_chars = game_n = constructed_chars = constructed_n = 0
    for record in records:
        source = classify_record_source(record)  # 未知来源在此处即抛错
        if source in SIMULATION_SOURCES:
            continue  # 模拟数据是候选池，不属于基线
        exposure = measure_record_exposure(record)
        if source == "game_extraction_current_sft":
            game_chars += exposure["supervised_assistant_chars"]
            game_n += 1
        else:  # llm_v4_reviewed_constructed
            constructed_chars += exposure["supervised_assistant_chars"]
            constructed_n += 1
    return {
        "game_sup_chars": game_chars,
        "constructed_sup_chars": constructed_chars,
        "game_records": game_n,
        "constructed_records": constructed_n,
        "note": BUDGET_PRELIMINARY_NOTE,
    }


# ============================================
# 维度 1：场景价值 (0-5)
# ============================================


def score_scenario_value(record: dict) -> tuple[int, list[dict]]:
    meta = record.get("metadata", {})
    messages = record.get("messages", [])
    scene = str(meta.get("scene", ""))
    user_text = " ".join(m.get("content", "") for m in messages if m["role"] == "user")
    scene_user = f"{scene}\n{user_text}"
    task_type = str(meta.get("task_type", ""))

    hits: list[dict] = []
    for name, pattern in _POSITIVE_SCENARIO_RULES:
        found = sorted({m.group() for m in re.finditer(pattern, scene_user)})
        if found:
            hits.append({"signal": name, "evidence": found[:4]})

    score = min(len(hits), 5)
    if measure_record_exposure(record)["user_turns"] >= 4 and score >= 1:
        score = min(score + 1, 5)

    if task_type in TECH_TASK_TYPES:
        # 技术队列：场景对人物训练价值封顶 2，人工可救回
        score = min(score, 2)
        hits.append({"signal": "tech_task_type_capped", "evidence": [task_type]})
    return score, hits


# ============================================
# 维度 2：人物风格 (0-5)，机器代理估计
# ============================================


def score_persona_fidelity(record: dict) -> tuple[float, list[dict]]:
    """机器代理估计：回复长度接近原作均值、含讥讽/简短标记加分，
    服务型礼貌语言减分。不能替代人工对语言质感的判断。"""
    assistant_text = "\n".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "assistant"
    )
    exposure = measure_record_exposure(record)
    avg = exposure["avg_supervised_chars_per_target"]
    notes: list[dict] = []
    score = 3.0

    if avg <= 50:
        score += 1
        notes.append(
            {"signal": "reply_length_near_original", "evidence": [f"avg={avg}"]}
        )
    elif avg <= 80:
        score += 0.5
        notes.append({"signal": "reply_length_moderate", "evidence": [f"avg={avg}"]})
    elif avg > 150:
        score -= 1
        notes.append({"signal": "reply_length_long", "evidence": [f"avg={avg}"]})
        if avg > 250:
            score -= 1
            notes.append(
                {"signal": "reply_length_very_long", "evidence": [f"avg={avg}"]}
            )

    sarcasm = sorted(set(_SARCASM_MARKERS.findall(assistant_text)))[:4]
    if sarcasm:
        score += 0.5
        notes.append({"signal": "sarcasm_style_markers", "evidence": sarcasm})

    service = sorted(set(_SERVICE_LANGUAGE.findall(assistant_text)))[:4]
    if service:
        score -= 1.5
        notes.append({"signal": "service_language", "evidence": service})

    return max(0.0, min(5.0, score)), notes


# ============================================
# 维度 3：事实根基 (pass/fail)，原作元素分级
# ============================================


def _lore_occurrences(text: str, element: str) -> list[str]:
    """元素出现位置的前置语境片段（用于否定/比喻判定）。"""
    contexts = []
    start = 0
    while True:
        idx = text.find(element, start)
        if idx < 0:
            return contexts
        window = text[max(0, idx - 12) : idx + len(element) + 8]
        contexts.append(window)
        start = idx + len(element)


def check_factual_grounding(record: dict) -> tuple[str, list[dict]]:
    """原作元素分级检查（只看 assistant 侧未被用户/场景引入的提及）。

    返回三态 verdict（自动规则不能证明"事实正确"，只能给出以下状态）：
    - "no_auto_flag"：自动规则未发现问题，仍需人工确认；
    - "needs_human"：出现未由用户/场景引入的人物名或世界观元素，
      无明确虚构语境，需人工判断；
    - "auto_fail"：明确虚构——世界观事实声称或人物经历声称。

    语境分级：
    - 否定语境（"但不是魔法"）：不罚，记 no_auto_flag 备注；
    - 比喻语境（"这不是能解决一切的魔法"）：不罚，记 no_auto_flag 备注；
    - 世界观事实声称（"魔法浓度/我们的世界由魔法"）：auto_fail；
    - 人物名未引入提及（琉璃）：needs_human；
    - 人物名/经历元素 + 经历语境（"琉璃以前生病时"）：auto_fail。
    """
    meta = record.get("metadata", {})
    scene = str(meta.get("scene", ""))
    user_text = " ".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "user"
    )
    assistant_text = "\n".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "assistant"
    )
    introduced = scene + user_text
    notes: list[dict] = []
    # 状态优先级：no_auto_flag < needs_human < auto_fail
    verdict = "no_auto_flag"

    def escalate(new: str) -> None:
        nonlocal verdict
        if new == "auto_fail" or (new == "needs_human" and verdict == "no_auto_flag"):
            verdict = new

    for element in _WORLD_ELEMENTS:
        if element in introduced:
            continue
        for window in _lore_occurrences(assistant_text, element):
            before = window[: window.rfind(element)] if element in window else ""
            if _NEGATION_BEFORE.search(before) or _NEGATION_BEFORE.search(window):
                notes.append({"signal": "lore_negated", "evidence": [window]})
                continue
            if _METAPHOR_CONTEXT.search(window):
                notes.append({"signal": "lore_metaphor", "evidence": [window]})
                continue
            if _FACTUAL_CLAIM.search(assistant_text):
                escalate("auto_fail")
                notes.append({"signal": "world_fact_claim", "evidence": [window]})
                continue
            escalate("needs_human")
            notes.append({"signal": "lore_mention_needs_human", "evidence": [window]})

    for name in _CHARACTER_NAMES:
        if name in introduced:
            continue
        for window in _lore_occurrences(assistant_text, name):
            wider = (
                assistant_text[
                    max(0, assistant_text.find(window[:6]) - 10) : assistant_text.find(
                        window[:6]
                    )
                    + 40
                ]
                if window[:6] in assistant_text
                else window
            )
            if _EXPERIENCE_CONTEXT.search(wider):
                escalate("auto_fail")
                notes.append(
                    {"signal": "character_experience_claim", "evidence": [window]}
                )
            else:
                escalate("needs_human")
                notes.append(
                    {"signal": "character_name_uninvited", "evidence": [window]}
                )

    for element in _EXPERIENCE_ELEMENTS:
        if element in introduced:
            continue
        for window in _lore_occurrences(assistant_text, element):
            if _EXPERIENCE_CONTEXT.search(window):
                escalate("auto_fail")
                notes.append({"signal": "experience_claim", "evidence": [window]})

    return verdict, notes


# ============================================
# 维度 4：多轮一致性 (0-5)，机器代理估计
# ============================================


def score_multiturn_coherence(record: dict) -> tuple[int, list[dict]]:
    """机器代理：时间与地点/物品线索同现加分。矛盾检测（时间倒错、
    物品凭空出现、立场反转）机器无法可靠完成，需人工审核。"""
    user_text = " ".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "user"
    )
    exposure = measure_record_exposure(record)
    notes: list[dict] = []
    if exposure["user_turns"] < 2:
        return 3, [{"signal": "single_turn_machine_default", "evidence": ["n/a"]}]

    score = 3
    time_hits = sorted(set(_TIME_MARKERS.findall(user_text)))[:3]
    if time_hits:
        score += 1
        notes.append({"signal": "time_markers", "evidence": time_hits})
    place_hits = sorted(set(_PLACE_ITEM_MARKERS.findall(user_text)))[:3]
    if place_hits:
        score += 1
        notes.append({"signal": "place_item_markers", "evidence": place_hits})
    return min(score, 5), notes


# ============================================
# 维度 5：通用助手风险 (0-5，越低越好)
# ============================================


def score_generic_assistant_risk(record: dict) -> tuple[float, list[dict]]:
    assistant_text = "\n".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "assistant"
    )
    exposure = measure_record_exposure(record)
    avg = exposure["avg_supervised_chars_per_target"]
    notes: list[dict] = []
    score = 0.0

    structured = sorted(set(_STRUCTURED_REPLY.findall(assistant_text)))[:4]
    if structured:
        score += 2
        notes.append({"signal": "structured_reply", "evidence": structured})

    service = sorted(set(_SERVICE_LANGUAGE.findall(assistant_text)))[:4]
    if service:
        score += 1.5
        notes.append({"signal": "service_language", "evidence": service})

    advice_count = len(_ADVICE_WORD.findall(assistant_text))
    if advice_count >= 3:
        score += 1
        notes.append({"signal": "advice_density", "evidence": [f"建议×{advice_count}"]})

    if avg > 150:
        score += 1
        notes.append({"signal": "long_avg_reply", "evidence": [f"avg={avg}"]})
        if avg > 250:
            score += 1
            notes.append({"signal": "very_long_avg_reply", "evidence": [f"avg={avg}"]})

    emotion = sorted(set(_EMOTION_TEMPLATE.findall(assistant_text)))[:3]
    if emotion:
        score += 0.5
        notes.append({"signal": "emotion_template", "evidence": emotion})

    return min(5.0, score), notes


# ============================================
# 汇总与门禁
# ============================================


def score_simulation_value(record: dict) -> dict:
    """五维评分 + 门禁建议。机器维度互不抵消：
    prefer_keep 需要全部维度同时达标，任何一项不满足都不进入。"""
    meta = record.get("metadata", {})
    task_type = str(meta.get("task_type", ""))
    in_tech_queue = task_type in TECH_TASK_TYPES or bool(
        _TECH_KEYWORD.search(str(meta.get("scene", "")))
    )

    scenario, scenario_hits = score_scenario_value(record)
    persona, persona_notes = score_persona_fidelity(record)
    grounding, grounding_notes = check_factual_grounding(record)
    coherence, coherence_notes = score_multiturn_coherence(record)
    generic, generic_notes = score_generic_assistant_risk(record)

    gate_pass = (
        scenario >= GATE["scenario_value_min"]
        and persona >= GATE["persona_fidelity_min"]
        and grounding == GATE["factual_grounding_required"]
        and coherence >= GATE["multiturn_coherence_min"]
        and generic <= GATE["generic_assistant_risk_max"]
        and not in_tech_queue
    )

    if gate_pass:
        suggestion = "prefer_keep"
    elif in_tech_queue or grounding == "auto_fail" or generic >= 3 or persona <= 2:
        suggestion = "prefer_exclude"
    else:
        suggestion = "review_priority"

    return {
        "dimensions": {
            "scenario_value": scenario,
            "persona_fidelity": persona,
            "factual_grounding": grounding,
            "multiturn_coherence": coherence,
            "generic_assistant_risk": generic,
        },
        "gate_pass": gate_pass,
        "tech_queue": in_tech_queue,
        "task_type": task_type,
        "signals": {
            "scenario": scenario_hits,
            "persona": persona_notes,
            "grounding": grounding_notes,
            "coherence": coherence_notes,
            "generic": generic_notes,
        },
        "machine_suggestion": suggestion,
    }


# ============================================
# 场景重复簇（task_type 分组 + 对象/行为/目标概念标签 + Jaccard）
# ============================================

# 同义改写归一（"屡次/频繁/总是" 与 "反复" 不共享 bigram，必须先归一）
_SCENE_SYNONYMS = {
    "屡次": "反复",
    "频繁": "反复",
    "总是": "反复",
    "经常": "反复",
    "一再": "反复",
    "长期": "反复",
    "屡屡": "反复",
    "最后一刻": "临时",
    "临期": "临时",
    "突然": "临时",
}

# 场景概念词表（对象 + 冲突行为 + 频率/方式 + 目标），长词优先匹配。
# 未命中词表的场景概念集合过小，自然不会被聚类（宁可漏检、不可误并）。
_SCENE_CONCEPT_LEXICON: tuple[tuple[str, str], ...] = (
    # --- 对象 ---
    ("好友", "obj:朋友"),
    ("朋友", "obj:朋友"),
    ("室友", "obj:室友"),
    ("家人", "obj:家人"),
    ("父母", "obj:父母"),
    ("妹妹", "obj:妹妹"),
    ("亲戚", "obj:亲戚"),
    ("同事", "obj:同事"),
    ("群", "obj:群聊"),
    # --- 冲突行为（长词在前） ---
    ("替你答应", "act:替你答应"),
    ("取消约定", "act:取消约定"),
    ("取消", "act:取消约定"),
    ("借钱", "act:借钱"),
    ("借走", "act:借钱"),
    ("迟还", "act:借钱"),
    ("归还", "act:归还"),
    ("泄露", "act:泄露隐私"),
    ("私事", "act:泄露隐私"),
    ("换班", "act:换班"),
    ("调解", "act:调解"),
    ("争执", "act:争执"),
    ("吵架", "act:争执"),
    ("倾诉", "act:倾诉"),
    ("擅自", "act:擅自越界"),
    ("占用", "act:擅自越界"),
    ("替自己答应", "act:替你答应"),
    ("含糊", "act:含糊其辞"),
    ("道歉", "act:道歉"),
    ("和好", "act:道歉"),
    ("求助", "act:求助"),
    ("帮助", "act:帮助"),
    ("求帮忙", "act:求助"),
    ("爽约", "act:取消约定"),
    ("失约", "act:取消约定"),
    ("放鸽子", "act:取消约定"),
    ("打探", "act:泄露隐私"),
    ("转发", "act:泄露隐私"),
    ("安装", "act:安装软件"),
    ("关闭", "act:关闭防护"),
    ("告状", "act:告状"),
    ("比较", "act:比较打压"),
    ("说教", "act:说教"),
    ("指挥", "act:指挥"),
    # --- 频率/方式 ---
    ("反复", "freq:反复"),
    ("临时", "freq:临时"),
    ("深夜", "freq:深夜"),
    ("周期性", "freq:反复"),
    # --- 目标/情绪 ---
    ("边界", "goal:边界"),
    ("设限", "goal:边界"),
    ("期待", "goal:调整期待"),
    ("调整投入", "goal:调整期待"),
    ("失望", "goal:失望"),
    ("失落", "goal:失望"),
    ("信任", "goal:信任修复"),
    ("修复", "goal:信任修复"),
    ("负债感", "goal:负债感"),
    ("陪伴", "goal:陪伴"),
    ("职责", "goal:职责"),
    ("安全", "goal:安全"),
)


def scene_concepts(scene: str) -> set[str]:
    """场景 → 概念标签集合（先同义归一，再长词优先匹配词表）。"""
    text = str(scene)
    for src, dst in _SCENE_SYNONYMS.items():
        text = text.replace(src, dst)
    concepts: set[str] = set()
    for key, concept in sorted(_SCENE_CONCEPT_LEXICON, key=lambda kv: -len(kv[0])):
        if key in text:
            concepts.add(concept)
    return concepts


def cluster_scenes(entries: list[dict], threshold: float = 0.5) -> dict[str, int]:
    """场景重复簇（并查集）。

    规则（人工确认用，绝不自动排除）：
    1. 只在相同 task_type 内比较；
    2. 概念标签 Jaccard ≥ threshold 且共享至少一个 act: 行为标签；
    3. 同簇场景建议人工只保留最有价值的 1–2 条。

    返回 {entry_id: cluster_id}（含单元素簇；调用方按簇大小过滤展示）。
    """
    ids = [e["id"] for e in entries]
    concepts = {e["id"]: scene_concepts(e.get("scene", "")) for e in entries}
    task_types = {e["id"]: str(e.get("task_type", "")) for e in entries}
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            if task_types[a_id] != task_types[b_id]:
                continue
            a, b = concepts[a_id], concepts[b_id]
            if len(a) < 2 or len(b) < 2:
                continue
            shared_acts = {c for c in a & b if c.startswith("act:")}
            if not shared_acts:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                union(i, j)

    roots: dict[int, int] = {}
    mapping: dict[str, int] = {}
    for idx, rid in enumerate(ids):
        root = find(idx)
        if root not in roots:
            roots[root] = len(roots)
        mapping[rid] = roots[root]
    return mapping


# ============================================
# 审核材料
# ============================================


def build_review_packet(sim_records: list[dict]) -> dict:
    """逐条审核数据（JSON 基础），供分批 Markdown 生成与后续校验用。"""
    entries = []
    for record in sim_records:
        exposure = measure_record_exposure(record)
        scoring = score_simulation_value(record)
        meta = record.get("metadata", {})
        prior = meta.get("human_review", {})
        entries.append(
            {
                "id": record["id"],
                "source": meta.get("data_source"),
                "scene": meta.get("scene", ""),
                "task_type": meta.get("task_type", ""),
                "exposure": exposure,
                "scoring": scoring,
                "prior_review_status": prior.get("status", ""),
                "human_decision": None,
                "messages": record["messages"],
            }
        )
    # 排序：门禁通过 > 待复审 > 建议排除；同带内按场景价值降序
    band_order = {"prefer_keep": 0, "review_priority": 1, "prefer_exclude": 2}
    entries.sort(
        key=lambda e: (
            band_order[e["scoring"]["machine_suggestion"]],
            -e["scoring"]["dimensions"]["scenario_value"],
            e["id"],
        )
    )
    return entries


def _format_dialogue(messages: list[dict]) -> str:
    lines = []
    for turn, msg in enumerate(messages, 1):
        speaker = {"user": "用户", "assistant": "月社妃(模拟)"}.get(
            msg["role"], msg["role"]
        )
        lines.append(f"  - 第{turn}轮 [{speaker}]: {msg['content']}")
    return "\n".join(lines)


_FACTUAL_DISPLAY = {
    "no_auto_flag": "自动未发现问题（仍需人工确认）",
    "needs_human": "需人工确认（出现未引入原作元素）",
    "auto_fail": "**自动判定虚构 FAIL**",
}


def write_markdown_batches(
    entries: list[dict], cluster_map: dict[str, int], batch_size: int = 25
) -> list[Path]:
    """分批 Markdown 审核材料（每批 ≤ batch_size 条）。"""
    REVIEW_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    cluster_sizes = Counter(cluster_map.values())

    paths = []
    for batch_start in range(0, len(entries), batch_size):
        batch_no = batch_start // batch_size + 1
        batch = entries[batch_start : batch_start + batch_size]
        lines = [
            f"# 阶段 2 审核批次 {batch_no:02d}"
            f"（{batch_start + 1}–{batch_start + len(batch)} / {len(entries)}）",
            "",
            "每条记录请人工勾选：keep（保留）/ exclude（排除）/ revise（需改写，"
            "决定阶段按 exclude 处理，改写属后续工作）。",
            "",
        ]
        for rank, entry in enumerate(batch, batch_start + 1):
            d = entry["scoring"]["dimensions"]
            s = entry["scoring"]
            lines.append(f"## [{rank}] {entry['scene']}")
            lines.append(f"- ID: `{entry['id']}`")
            lines.append(f"- task_type: `{entry['task_type']}`")
            if s["tech_queue"]:
                lines.append("- **技术排除队列**: 是（场景价值封顶 2，人工可救回）")
            cluster = cluster_map.get(entry["id"])
            if cluster is not None and cluster_sizes.get(cluster, 0) > 1:
                lines.append(
                    f"- **场景重复簇**: 簇 {cluster}（共 {cluster_sizes[cluster]} 条同类场景，"
                    f"建议只保留最有价值的 1–2 条）"
                )
            lines.append(
                f"- 五维: 场景 {d['scenario_value']}/5 | 人物 {d['persona_fidelity']}/5 "
                f"| 事实: {_FACTUAL_DISPLAY.get(d['factual_grounding'], d['factual_grounding'])} "
                f"| 一致 {d['multiturn_coherence']}/5 | 通用助手风险 {d['generic_assistant_risk']}/5"
            )
            lines.append(
                f"- AI 建议: **{s['machine_suggestion']}**（门禁 {'通过' if s['gate_pass'] else '未通过'}）"
            )
            # 证据摘要
            for dim, key in (
                ("场景", "scenario"),
                ("人物", "persona"),
                ("事实", "grounding"),
                ("一致", "coherence"),
                ("通用", "generic"),
            ):
                signals = s["signals"][key]
                if signals:
                    ev = "; ".join(
                        f"{sig['signal']}:{'/'.join(sig['evidence'][:2])}"
                        for sig in signals[:4]
                    )
                    lines.append(f"  - {dim}证据: {ev}")
            lines.append("- 对话全文：")
            lines.append(_format_dialogue(entry["messages"]))
            lines.append("- **人工选择**: [ ] keep  [ ] exclude  [ ] revise")
            lines.append("- 人工备注: ______")
            lines.append("")
        path = REVIEW_BATCH_DIR / f"batch_{batch_no:02d}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _budget_simulation(entries: list[dict], baseline: dict) -> list[dict]:
    """keep-top-k 预算模拟（初步估算：假定 150 条短构造全部保留）。"""
    base_chars = baseline["game_sup_chars"] + baseline["constructed_sup_chars"]
    base_records = baseline["game_records"] + baseline["constructed_records"]
    table = []
    cumulative = 0
    for rank, entry in enumerate(entries, 1):
        cumulative += entry["exposure"]["supervised_assistant_chars"]
        total = base_chars + cumulative
        table.append(
            {
                "keep_top_k": rank,
                "record_id": entry["id"],
                "sup_chars": entry["exposure"]["supervised_assistant_chars"],
                "sim_sup_char_share_pct": round(cumulative / total * 100, 1),
                "sim_record_share_pct": round(rank / (base_records + rank) * 100, 1),
                "preliminary": True,
            }
        )
    return table


def write_review_packet(entries: list[dict], manifest: dict, baseline: dict) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    cluster_map = cluster_scenes(entries)

    packet = {
        "schema_version": 3,
        "packet_id": "KISAKI-V5-SIMULATION-REVIEW-PACKET",
        "generated_at": generated_at,
        "generated_from": {
            "dataset_id": manifest["dataset_id"],
            "train_sha256": manifest["train"]["sha256"],
        },
        "policy": {
            "machine_note": (
                "machine_suggestion 仅为五维规则评分的排序辅助；persona_fidelity "
                "与 multiturn_coherence 是机器代理估计，人工终审为准"
            ),
            "gate": GATE,
            "gate_note": "五维互不抵消：任何一项不达标都不进入 prefer_keep",
            "factual_grounding_note": (
                "三态（no_auto_flag/needs_human/auto_fail）：自动规则未发现问题"
                "≠事实已确认，no_auto_flag 仍需人工确认"
            ),
            "tech_queue_note": (
                "技术排除队列优先按 metadata.task_type 划分，正则仅补充；"
                "队内场景价值封顶 2，人工可决定少量保留"
            ),
            "cluster_note": (
                "场景重复簇按 task_type 分组 + 对象/行为/目标概念标签聚类，"
                "仅提示人工确认，绝不自动排除；同簇建议最多保留 1–2 条"
            ),
            "data_gap": (
                "254 条模拟的 interlocutor_kind 全部为 generic_user，"
                "『面对不同人物的差异化反应』维度缺失，需人工审核补判"
            ),
            "budget_note": baseline["note"],
        },
        "budget_baseline_computed": baseline,
        "budget_simulation_preliminary": _budget_simulation(entries, baseline),
        "scene_clusters": {
            str(cid): [e["id"] for e in entries if cluster_map.get(e["id"]) == cid]
            for cid in sorted(set(cluster_map.values()))
            if sum(1 for v in cluster_map.values() if v == cid) > 1
        },
        "entries": entries,
    }
    (V5_DIR / "simulation_review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    batch_paths = write_markdown_batches(entries, cluster_map)

    # 摘要
    suggestion_counts = Counter(e["scoring"]["machine_suggestion"] for e in entries)
    dim_medians = {}
    for dim in (
        "scenario_value",
        "persona_fidelity",
        "multiturn_coherence",
        "generic_assistant_risk",
    ):
        values = sorted(e["scoring"]["dimensions"][dim] for e in entries)
        dim_medians[dim] = values[len(values) // 2]
    factual_counts = Counter(
        e["scoring"]["dimensions"]["factual_grounding"] for e in entries
    )
    tech_count = sum(1 for e in entries if e["scoring"]["tech_queue"])

    lines = [
        "# 阶段 2：254 条长篇用户模拟审核材料摘要（v3 五维/三态事实）",
        "",
        f"生成时间：{generated_at}｜数据源：{manifest['dataset_id']}（sha256 已校验）",
        "",
        "## 建议带分布（五维门禁，互不抵消）",
        "",
        "| 建议带 | 数量 |",
        "|---|---:|",
    ]
    for band in ("prefer_keep", "review_priority", "prefer_exclude"):
        lines.append(f"| {band} | {suggestion_counts.get(band, 0)} |")
    lines += [
        "",
        "## 事实根基三态分布（自动规则 ≠ 事实确认）",
        "",
        "| 状态 | 数量 | 含义 |",
        "|---|---:|---|",
        f"| no_auto_flag | {factual_counts.get('no_auto_flag', 0)} | 自动未发现问题，仍需人工确认 |",
        f"| needs_human | {factual_counts.get('needs_human', 0)} | 出现未引入原作元素，需人工判断 |",
        f"| auto_fail | {factual_counts.get('auto_fail', 0)} | 自动判定虚构（世界观/经历声称） |",
        "",
        f"- 技术排除队列：{tech_count} 条",
        f"- 维度中位数：{json.dumps(dim_medians, ensure_ascii=False)}",
        f"- 场景重复簇（≥2 条，需人工确认，不自动排除）：{len(packet['scene_clusters'])} 簇",
        "",
        "## 预算模拟（初步估算，基线动态计算）",
        "",
        f"基线（从当前 V4 records 实测）：原作 {baseline['game_records']} 条 / "
        f"{baseline['game_sup_chars']} 监督字符；短构造 {baseline['constructed_records']} 条 / "
        f"{baseline['constructed_sup_chars']} 监督字符（假定全部保留，阶段 3 后重算）。",
        "",
        "| 保留条数 | 累计监督字符占比 | 记录占比 |",
        "|---:|---:|---:|",
    ]
    for row in packet["budget_simulation_preliminary"]:
        if row["keep_top_k"] in (1, 10, 15, 20, 25, 30) or row["keep_top_k"] % 10 == 0:
            lines.append(
                f"| {row['keep_top_k']} | {row['sim_sup_char_share_pct']}% "
                f"| {row['sim_record_share_pct']}% |"
            )
    lines += [
        "",
        "## 人工审核方式",
        "",
        f"- 分批 Markdown（每批 ≤30 条，含完整对话/五维/证据/选择栏）：`review_batches/` 共 "
        f"{len(batch_paths)} 批",
        "- revise = 需改写；决定阶段只接受 keep/exclude，revise 按 exclude 处理",
        "- 决定文件：`simulation_review_decisions.json`，格式：",
        '  `{"review_status": "approved", "reviewed_by": "owner", "decisions": {"<id>": "keep|exclude"}}`',
        "  （254 个 ID 全覆盖、值域严格校验、review_status 必须为 approved）",
        "",
        "## 数据缺口",
        "",
        "interlocutor_kind 全部为 generic_user——『面对不同人物的差异化反应』"
        "维度缺失，需人工审核补判。",
    ]
    (V5_DIR / "simulation_review_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return packet


# ============================================
# 决定门禁
# ============================================


def validate_decision_document(decision_doc: dict, sim_ids: set[str]) -> None:
    """决定文件严格校验，非法输入一律报错，绝不静默当成 exclude。"""
    if decision_doc.get("review_status") != "approved":
        raise SystemExit(
            '[ABORT] 决定文件缺少 review_status: "approved"'
            "（reviewed_by 单独存在不代表已人工批准）"
        )
    reviewed_by = decision_doc.get("reviewed_by")
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise SystemExit("[ABORT] 决定文件缺少 reviewed_by（人工批准者）")

    decisions = decision_doc.get("decisions")
    if not isinstance(decisions, dict) or not decisions:
        raise SystemExit("[ABORT] decisions 必须是非空对象")

    unknown = sorted(set(decisions) - sim_ids)
    if unknown:
        raise SystemExit(
            f"[ABORT] 决定文件包含未知 ID（{len(unknown)} 个）: {unknown[:5]}"
        )
    missing = sorted(sim_ids - set(decisions))
    if missing:
        raise SystemExit(
            f"[ABORT] {len(missing)} 条模拟记录未出现在决定文件中: {missing[:5]}"
        )
    invalid = sorted(
        rid for rid, value in decisions.items() if value not in ("keep", "exclude")
    )
    if invalid:
        samples = {rid: decisions[rid] for rid in invalid[:3]}
        raise SystemExit(f"[ABORT] 非法决定值（只能 keep/exclude）: {samples}")


def build_candidate_dataset(decisions_path: Path) -> dict:
    """校验决定并输出保留的模拟记录清单。

    注意：本函数不构建完整 V5 candidate（阶段 5 职责）；
    仅返回通过人工批准的模拟记录，供阶段 5 组装。"""
    decision_doc = json.loads(decisions_path.read_text(encoding="utf-8"))

    records, manifest = load_canonical_records()
    sim_records = [
        r for r in records if classify_record_source(r) in SIMULATION_SOURCES
    ]
    sim_ids = {r["id"] for r in sim_records}
    validate_decision_document(decision_doc, sim_ids)

    decisions = decision_doc["decisions"]
    keep_ids = {rid for rid, d in decisions.items() if d == "keep"}
    kept = [r for r in sim_records if r["id"] in keep_ids]

    sup_chars = sum(
        measure_record_exposure(r)["supervised_assistant_chars"] for r in kept
    )
    baseline = compute_budget_baseline(records)
    base = baseline["game_sup_chars"] + baseline["constructed_sup_chars"]
    return {
        "kept_records": len(kept),
        "excluded_records": len(sim_ids) - len(kept),
        "kept_supervised_chars": sup_chars,
        "sim_sup_char_share_pct_preliminary": round(
            sup_chars / (base + sup_chars) * 100, 1
        ),
        "records": kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help="人工批准决定文件（校验后仅输出保留模拟记录清单；不构建 V5，阶段 5 职责）",
    )
    args = parser.parse_args()

    if args.decisions is not None:
        result = build_candidate_dataset(args.decisions)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "records"},
                ensure_ascii=False,
            )
        )
        return 0

    records, manifest = load_canonical_records()
    sim_records = [
        r for r in records if classify_record_source(r) in SIMULATION_SOURCES
    ]
    if len(sim_records) != 254:
        print(
            f"[WARN] 预期 254 条用户模拟，实际 {len(sim_records)}",
            file=sys.stderr,
        )

    V5_DIR.mkdir(parents=True, exist_ok=True)
    baseline = compute_budget_baseline(records)
    entries = build_review_packet(sim_records)
    packet = write_review_packet(entries, manifest, baseline)

    counts = Counter(e["scoring"]["machine_suggestion"] for e in entries)
    factual = Counter(e["scoring"]["dimensions"]["factual_grounding"] for e in entries)
    print(f"审核材料已生成：{len(entries)} 条（五维门禁 + 三态事实）")
    print(
        f"  prefer_keep={counts.get('prefer_keep', 0)} "
        f"review_priority={counts.get('review_priority', 0)} "
        f"prefer_exclude={counts.get('prefer_exclude', 0)}"
    )
    print(
        f"  factual: no_auto_flag={factual.get('no_auto_flag', 0)} "
        f"needs_human={factual.get('needs_human', 0)} "
        f"auto_fail={factual.get('auto_fail', 0)}"
    )
    print(
        f"  tech_queue={sum(1 for e in entries if e['scoring']['tech_queue'])} "
        f"scene_clusters={len(packet['scene_clusters'])}"
    )
    print(
        f"  预算基线（动态实测）: 原作 {baseline['game_records']} 条/"
        f"{baseline['game_sup_chars']} 字符, 短构造 {baseline['constructed_records']} 条/"
        f"{baseline['constructed_sup_chars']} 字符（假定全部保留，初步估算）"
    )
    print("  分批 Markdown: review_batches/（每批 ≤30 条）")
    if sha256_file(V4_DIR / "train.jsonl") != manifest["train"]["sha256"]:
        print("[ABORT] 运行后 V4 train 哈希变化", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
