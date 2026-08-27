"""阶段 3：逐条复查 150 条短构造数据（llm_v4_reviewed_constructed）。

职责单一：读取 KISAKI-CANONICAL-V4 冻结数据（只读，sha256 校验），
对 150 条短构造记录生成复查材料。机器评分只辅助排序与预警，
不能代替最终审核；本脚本不修改 V4、不删除数据、不构建 V5。

与阶段 2 的差异（针对短构造数据特点的适配）：
1. 记录极短（138 条单轮 / 12 条 2–3 轮，assistant 均值 17.1 字符），
   曾通过 V4 三重审核（human_review + persona_review +
   full_dialogue_review）——本阶段是**复查**，重点是找可疑而非重证质量；
2. 无 metadata.task_type：技术排除队列退化为 scene/文本关键词正则；
3. 人物研究场景标签（scene 字段）是短构造的设计前提
   （人物关系/角色人设/夜子危机/琉璃斗嘴等），命中人物研究主线标签的
   记录 scenario_value 保底 3；factual/事实与安全类不保底，
   走事实根基重点检查；
4. 单轮记录无多轮一致性风险：multiturn_coherence 记 5 并标记
   single_turn_not_applicable（机器无法检查不存在的维度），
   多轮记录沿用阶段 2 代理规则；
5. 事实根基人物名单扩展为 琉璃/夜子/理央（短构造的原作互动人物），
   用户/场景引入不罚（scene 标签如"琉璃斗嘴"即视为已引入），
   assistant 侧未引入提及 → needs_human（角色问答类需人工确认
   人物关系是否符合原作设定），带经历语境 → auto_fail；
6. 历史人工审核信息（human_review.status/reason/改写前后文本）
   展示在复查材料中，辅助判断"曾改过的记录是否改对了"；
7. 场景重复簇：用户提问 bigram Jaccard ≥ 0.45（无 task_type 可分组，
   探查实测仅 7 个相似对，宁可漏检不可误并），仅提示不排除；
8. 复查排序：事实可疑（needs_human/auto_fail）优先，多轮记录优先，
   同带内按 scene 聚集便于横向对比。

五维评分与门禁沿用阶段 2（互不抵消，全部达标才 prefer_keep）。

产物（experiments/v5_candidate/）：
- constructed_review_packet.json      逐条复查数据（五维 + 三态事实 + 历史审核）
- constructed_review_batches/batch_01.md ... 分批复查材料（每批 ≤25 条）
- constructed_review_summary.md       摘要

决定文件门禁（constructed_review_decisions.json）：
- review_status 必须为 "approved"；决定 ID 集合必须与 150 条完全相等；
- 决定值只能是 "keep" 或 "exclude"（非法值一律报错，绝不静默）；
- revise（待改写）由 collect 脚本转换为 exclude + needs_revision。

--decisions 语义：校验决定 + 输出保留清单 + **阶段 3 后预算重算**
（原作/短构造/模拟三部分按实际批准数据动态计算，消除
"假定短构造全部保留"的初步口径）；不构建完整 V5（阶段 5 职责）。

用法：
  python scripts/build_kisaki_v5_constructed_review.py              # 生成复查材料
  python scripts/build_kisaki_v5_constructed_review.py --decisions \
      backend/data/character_dialogues/experiments/v5_candidate/constructed_review_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# 复用阶段 2 的加载/评分/门禁/正则（单一来源，避免规则漂移）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kisaki_v5_candidate import (
    _EXPERIENCE_CONTEXT,
    _EXPERIENCE_ELEMENTS,
    _FACTUAL_CLAIM,
    _METAPHOR_CONTEXT,
    _NEGATION_BEFORE,
    _WORLD_ELEMENTS,
    GATE,
    SIMULATION_SOURCES,
    V5_DIR,
    _lore_occurrences,
    classify_record_source,
    load_canonical_records,
    measure_record_exposure,
    score_generic_assistant_risk,
    score_persona_fidelity,
    score_scenario_value,
    validate_decision_document,
)

CONSTRUCTED_BATCH_DIR = V5_DIR / "constructed_review_batches"
DECISIONS_PATH = V5_DIR / "constructed_review_decisions.json"
SIMULATION_DECISIONS_PATH = V5_DIR / "simulation_review_decisions.json"

# 短构造涉及的原作人物（用户/场景引入不罚；assistant 未引入提及 → needs_human）
_CONSTRUCTED_CHARACTER_NAMES = ("琉璃", "夜子", "理央")

# 人物研究主线场景标签（短构造设计前提）：scenario_value 保底 3
_PERSONA_RESEARCH_SCENES = frozenset(
    {
        "人物关系",
        "角色人设",
        "不坦率理解",
        "请求帮助",
        "情感倾诉",
        "兴趣偏好",
        "问候闲聊",
        "夜子危机",
        "夜子日常",
        "琉璃斗嘴",
        "夸奖回避",
        "小恶魔",
        "温柔关心",
        "温柔反差",
        "自我否定",
        "深层感情",
        "日常温柔",
        "友情冲突",
        "出门邀请边界",
        "multiturn",
        "persona",
    }
)


# ============================================
# 适配后的评分维度
# ============================================


def score_scenario_value_constructed(record: dict) -> tuple[int, list[dict]]:
    """场景价值（短构造适配）：沿用阶段 2 规则 + 人物研究标签保底。

    保底逻辑：短构造的人物研究场景（scene 命中标签集）本身是
    人物 LoRA 的研究主线，scenario_value = max(规则分, 3)；
    factual/事实与安全类不保底（世界观事实声称重点走事实根基检查）。
    """
    score, hits = score_scenario_value(record)
    scene = str(record.get("metadata", {}).get("scene", ""))
    if scene in _PERSONA_RESEARCH_SCENES and score < 3:
        score = 3
        hits.append({"signal": "persona_research_scene_floor", "evidence": [scene]})
    return score, hits


def check_factual_grounding_constructed(record: dict) -> tuple[str, list[dict]]:
    """事实根基三态（短构造适配：人物名单扩展为琉璃/夜子/理央）。

    规则同阶段 2：
    - 用户/场景已引入（含 scene 标签，如"琉璃斗嘴"）→ 不罚；
    - assistant 侧未引入提及人物名 → needs_human
      （角色问答类需人工确认人物关系是否符合原作设定）；
    - 人物名 + 经历语境（"以前/生病时"等）→ auto_fail；
    - 世界观元素（魔法）按否定/比喻/事实声称分级。
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

    for name in _CONSTRUCTED_CHARACTER_NAMES:
        if name in introduced:
            continue
        for window in _lore_occurrences(assistant_text, name):
            if _EXPERIENCE_CONTEXT.search(window):
                escalate("auto_fail")
                notes.append(
                    {"signal": "character_experience_claim", "evidence": [window]}
                )
            else:
                escalate("needs_human")
                notes.append(
                    {
                        "signal": "character_name_uninvited",
                        "evidence": [window],
                        "hint": "角色问答类需人工确认人物关系是否符合原作设定",
                    }
                )

    for element in _EXPERIENCE_ELEMENTS:
        if element in introduced:
            continue
        for window in _lore_occurrences(assistant_text, element):
            if _EXPERIENCE_CONTEXT.search(window):
                escalate("auto_fail")
                notes.append({"signal": "experience_claim", "evidence": [window]})

    return verdict, notes


def score_multiturn_coherence_constructed(record: dict) -> tuple[int, list[dict]]:
    """多轮一致性（短构造适配）：单轮记 5 并标记不适用。

    单轮记录不存在多轮一致性风险，机器对不存在的维度不罚分；
    多轮记录沿用阶段 2 代理规则（时间/物品线索加分，矛盾需人工）。
    """
    exposure = measure_record_exposure(record)
    if exposure["user_turns"] < 2:
        return 5, [{"signal": "single_turn_not_applicable", "evidence": ["n/a"]}]

    user_text = " ".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "user"
    )
    from build_kisaki_v5_candidate import (
        _PLACE_ITEM_MARKERS,
        _TIME_MARKERS,
    )

    score = 3
    notes: list[dict] = [
        {
            "signal": "multi_turn_needs_human_check",
            "evidence": [f"{exposure['user_turns']}轮"],
        }
    ]
    if _TIME_MARKERS.search(user_text):
        score += 1
        notes.append(
            {
                "signal": "time_markers",
                "evidence": sorted(set(_TIME_MARKERS.findall(user_text)))[:3],
            }
        )
    if _PLACE_ITEM_MARKERS.search(user_text):
        score += 1
        notes.append(
            {
                "signal": "place_item_markers",
                "evidence": sorted(set(_PLACE_ITEM_MARKERS.findall(user_text)))[:3],
            }
        )
    return min(score, 5), notes


def score_constructed_record(record: dict) -> dict:
    """五维评分 + 门禁（沿用阶段 2 结构）。"""
    from build_kisaki_v5_candidate import _TECH_KEYWORD

    meta = record.get("metadata", {})
    scene = str(meta.get("scene", ""))
    user_text = " ".join(
        m.get("content", "") for m in record["messages"] if m["role"] == "user"
    )
    in_tech_queue = bool(_TECH_KEYWORD.search(scene + user_text))

    scenario, scenario_hits = score_scenario_value_constructed(record)
    persona, persona_notes = score_persona_fidelity(record)
    grounding, grounding_notes = check_factual_grounding_constructed(record)
    coherence, coherence_notes = score_multiturn_coherence_constructed(record)
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
        "task_type": "",
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
# 提问重复簇（用户文本 bigram Jaccard，无 task_type 分组）
# ============================================


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else {text}


def cluster_user_questions(
    entries: list[dict], threshold: float = 0.45
) -> dict[str, int]:
    """提问重复簇（并查集）：用户文本 bigram Jaccard ≥ threshold。

    探查实测仅 7 个相似对（"你觉得什么是X"系列、"夜子好像在X"系列），
    宁可漏检不可误并；仅提示人工同簇保留 1–2 条，绝不自动排除。
    """
    ids = [e["id"] for e in entries]
    grams = {e["id"]: _bigrams(e["user_text"]) for e in entries}
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
            a, b = grams[ids[i]], grams[ids[j]]
            if not a or not b:
                continue
            if len(a & b) / len(a | b) >= threshold:
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
# 复查材料
# ============================================


def build_constructed_packet(constructed_records: list[dict]) -> list[dict]:
    """逐条复查数据：五维 + 历史人工审核信息。"""
    entries = []
    for record in constructed_records:
        exposure = measure_record_exposure(record)
        scoring = score_constructed_record(record)
        meta = record.get("metadata", {})
        prior = meta.get("human_review", {}) or {}
        user_text = " ".join(
            m.get("content", "") for m in record["messages"] if m["role"] == "user"
        )
        entries.append(
            {
                "id": record["id"],
                "source": meta.get("data_source"),
                "scene": meta.get("scene", ""),
                "task_type": "",
                "user_text": user_text,
                "exposure": exposure,
                "scoring": scoring,
                "prior_review": {
                    "status": prior.get("status", ""),
                    "reviewed_by": prior.get("reviewed_by", ""),
                    "reviewed_at": prior.get("reviewed_at", ""),
                    "reason": prior.get("reason", ""),
                    "original_user_messages": prior.get("original_user_messages", []),
                    "original_assistant_messages": prior.get(
                        "original_assistant_messages", []
                    ),
                    "note": meta.get("note", ""),
                    "original_id": meta.get("original_id", ""),
                },
                "human_decision": None,
                "messages": record["messages"],
            }
        )
    # 复查排序：事实可疑优先 > 多轮优先 > 同带内按 scene 聚集
    factual_rank = {"auto_fail": 0, "needs_human": 1, "no_auto_flag": 2}
    band_order = {"prefer_keep": 0, "review_priority": 1, "prefer_exclude": 2}
    entries.sort(
        key=lambda e: (
            factual_rank[e["scoring"]["dimensions"]["factual_grounding"]],
            band_order[e["scoring"]["machine_suggestion"]],
            0 if e["exposure"]["user_turns"] >= 2 else 1,
            e["scene"],
            e["id"],
        )
    )
    return entries


_FACTUAL_DISPLAY = {
    "no_auto_flag": "自动未发现问题（仍需人工确认）",
    "needs_human": "需人工确认（未引入原作人物/元素）",
    "auto_fail": "**自动判定虚构 FAIL**",
}


def _format_dialogue(messages: list[dict]) -> str:
    lines = []
    for turn, msg in enumerate(messages, 1):
        speaker = {"user": "用户", "assistant": "月社妃"}.get(msg["role"], msg["role"])
        lines.append(f"  - 第{turn}轮 [{speaker}]: {msg['content']}")
    return "\n".join(lines)


def write_constructed_batches(
    entries: list[dict], cluster_map: dict[str, int], batch_size: int = 25
) -> list[Path]:
    """分批复查材料（每批 ≤ batch_size 条，含历史审核信息）。"""
    CONSTRUCTED_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    cluster_sizes = Counter(cluster_map.values())

    paths = []
    for batch_start in range(0, len(entries), batch_size):
        batch_no = batch_start // batch_size + 1
        batch = entries[batch_start : batch_start + batch_size]
        lines = [
            f"# 阶段 3 复查批次 {batch_no:02d}"
            f"（{batch_start + 1}–{batch_start + len(batch)} / {len(entries)}）",
            "",
            "每条记录请人工勾选：keep（保留）/ exclude（排除）/ revise（需改写，"
            "决定阶段按 exclude 处理，改写属后续工作）。",
            "",
            "复查重点：1) 回复是否像月社妃；2) 人物关系陈述是否符合原作设定"
            "（琉璃=兄、夜子/理央=朋友）；3) 世界观事实声称；4) 多轮记录一致性；"
            "5) 曾改写记录（历史审核 reason）是否改对。",
            "",
        ]
        for rank, entry in enumerate(batch, batch_start + 1):
            d = entry["scoring"]["dimensions"]
            s = entry["scoring"]
            prior = entry["prior_review"]
            lines.append(f"## [{rank}] {entry['scene']}")
            lines.append(f"- ID: `{entry['id']}`")
            lines.append(f"- data_source: `{entry['source']}`")
            if s["tech_queue"]:
                lines.append("- **技术关键词命中**: 是（场景价值不保底，人工判断）")
            cluster = cluster_map.get(entry["id"])
            if cluster is not None and cluster_sizes.get(cluster, 0) > 1:
                lines.append(
                    f"- **提问重复簇**: 簇 {cluster}（共 {cluster_sizes[cluster]} 条相似提问，"
                    f"建议只保留最有价值的 1–2 条）"
                )
            lines.append(
                f"- 五维: 场景 {d['scenario_value']}/5 | 人物 {d['persona_fidelity']}/5 "
                f"| 事实: {_FACTUAL_DISPLAY.get(d['factual_grounding'], d['factual_grounding'])} "
                f"| 一致 {d['multiturn_coherence']}/5 | 通用助手风险 {d['generic_assistant_risk']}/5"
            )
            lines.append(
                f"- AI 建议: **{s['machine_suggestion']}**"
                f"（门禁 {'通过' if s['gate_pass'] else '未通过'}）"
            )
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
            # 历史审核信息（阶段 3 特有：曾改写记录需重点复查）
            if prior["status"] or prior["reason"]:
                lines.append(
                    f"- **历史审核**: {prior['status']}（{prior['reviewed_by']}，"
                    f"{prior['reviewed_at']}）"
                )
                if prior["reason"]:
                    lines.append(f"  - 改写理由: {prior['reason']}")
                if prior["original_assistant_messages"]:
                    lines.append(
                        f"  - 改写前回复: {' / '.join(prior['original_assistant_messages'])}"
                    )
                if prior["original_user_messages"]:
                    lines.append(
                        f"  - 改写前提问: {' / '.join(prior['original_user_messages'])}"
                    )
            if prior["note"]:
                lines.append(f"- 构造备注: {prior['note']}")
            lines.append("- 对话全文：")
            lines.append(_format_dialogue(entry["messages"]))
            lines.append("- **人工选择**: [ ] keep  [ ] exclude  [ ] revise")
            lines.append("- 人工备注: ______")
            lines.append("")
        path = CONSTRUCTED_BATCH_DIR / f"batch_{batch_no:02d}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def write_constructed_summary(
    entries: list[dict],
    manifest: dict,
    cluster_map: dict[str, int],
    batch_paths: list[Path],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    suggestion_counts = Counter(e["scoring"]["machine_suggestion"] for e in entries)
    factual_counts = Counter(
        e["scoring"]["dimensions"]["factual_grounding"] for e in entries
    )
    tech_count = sum(1 for e in entries if e["scoring"]["tech_queue"])
    multi_turn = sum(1 for e in entries if e["exposure"]["user_turns"] >= 2)
    prior_revised = sum(
        1 for e in entries if e["prior_review"]["original_assistant_messages"]
    )
    clusters = {
        str(cid): [e["id"] for e in entries if cluster_map.get(e["id"]) == cid]
        for cid in sorted(set(cluster_map.values()))
        if sum(1 for v in cluster_map.values() if v == cid) > 1
    }
    total_sup_chars = sum(e["exposure"]["supervised_assistant_chars"] for e in entries)

    lines = [
        "# 阶段 3：150 条短构造复查材料摘要",
        "",
        f"生成时间：{generated_at}｜数据源：{manifest['dataset_id']}（sha256 已校验）",
        "",
        "## 建议带分布（五维门禁，互不抵消；复查视角）",
        "",
        "| 建议带 | 数量 |",
        "|---|---:|",
    ]
    for band in ("prefer_keep", "review_priority", "prefer_exclude"):
        lines.append(f"| {band} | {suggestion_counts.get(band, 0)} |")
    lines += [
        "",
        "## 事实根基三态分布",
        "",
        "| 状态 | 数量 | 含义 |",
        "|---|---:|---|",
        f"| no_auto_flag | {factual_counts.get('no_auto_flag', 0)} | 自动未发现问题，仍需人工确认 |",
        f"| needs_human | {factual_counts.get('needs_human', 0)} | 未引入原作人物（琉璃/夜子/理央）或元素，需人工判断 |",
        f"| auto_fail | {factual_counts.get('auto_fail', 0)} | 自动判定虚构（世界观/经历声称） |",
        "",
        f"- 技术关键词命中：{tech_count} 条",
        f"- 多轮记录（需人工查一致性）：{multi_turn} 条",
        f"- 曾改写记录（历史审核含改写前文本，重点复查）：{prior_revised} 条",
        f"- 提问重复簇（≥2 条，仅提示）：{len(clusters)} 簇",
        f"- 合计监督字符：{total_sup_chars}（当前全部保留口径；复查后按决定重算）",
        "",
        "## 复查重点",
        "",
        "1. needs_human 记录多为角色问答类（问妃本人的关系/偏好/恐惧），"
        "assistant 提及琉璃/夜子/理央——需人工确认人物关系符合原作设定；",
        "2. 曾改写记录（blindfix 子源 34 条含改写前文本）核对改写方向是否正确；",
        "3. 多轮记录（12 条）检查时间/物品/立场一致性；",
        "4. factual/事实与安全类（4 条）检查世界观事实声称（魔法之书等）。",
        "",
        "## 人工复查方式",
        "",
        f"- 分批 Markdown（每批 ≤25 条，含完整对话/五维/历史审核/选择栏）："
        f"`constructed_review_batches/` 共 {len(batch_paths)} 批",
        "- revise = 需改写；决定阶段只接受 keep/exclude，revise 按 exclude 处理",
        "- 决定文件：`constructed_review_decisions.json`，格式：",
        '  `{"review_status": "approved", "reviewed_by": "owner", "decisions": {"<id>": "keep|exclude"}}`',
        "  （150 个 ID 全覆盖、值域严格校验、review_status 必须为 approved）",
        "- 勾选回收复用：`python scripts/collect_kisaki_v5_simulation_decisions.py "
        "--packet <constructed_review_packet.json> --batches-dir "
        "<constructed_review_batches> --output <constructed_review_decisions.json>`",
    ]
    (V5_DIR / "constructed_review_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def generate_review_materials() -> None:
    records, manifest = load_canonical_records()
    constructed = [
        r for r in records if classify_record_source(r) == "llm_v4_reviewed_constructed"
    ]
    if len(constructed) != 150:
        raise SystemExit(
            f"[ABORT] 短构造记录数 {len(constructed)} != 150（数据集异常，拒绝生成）"
        )

    entries = build_constructed_packet(constructed)
    cluster_map = cluster_user_questions(entries)

    packet = {
        "schema_version": 1,
        "packet_id": "KISAKI-V5-CONSTRUCTED-REVIEW-PACKET",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_from": {
            "dataset_id": manifest["dataset_id"],
            "train_sha256": manifest["train"]["sha256"],
        },
        "policy": {
            "machine_note": (
                "复查视角：短构造曾通过 V4 三重审核，机器评分用于排序与预警，"
                "人工终审为准"
            ),
            "gate": GATE,
            "factual_grounding_note": (
                "人物名单扩展为琉璃/夜子/理央；用户/场景引入不罚，"
                "assistant 未引入提及 → needs_human"
            ),
            "single_turn_note": "单轮记录 multiturn_coherence 记 5（维度不适用）",
            "persona_scene_floor_note": "人物研究场景标签 scenario_value 保底 3",
            "cluster_note": "提问 bigram Jaccard ≥0.45 聚簇，仅提示不排除",
        },
        "scene_clusters": {
            str(cid): [e["id"] for e in entries if cluster_map.get(e["id"]) == cid]
            for cid in sorted(set(cluster_map.values()))
            if sum(1 for v in cluster_map.values() if v == cid) > 1
        },
        "entries": entries,
    }
    (V5_DIR / "constructed_review_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    batch_paths = write_constructed_batches(entries, cluster_map)
    write_constructed_summary(entries, manifest, cluster_map, batch_paths)

    print(f"复查材料生成完成：{len(entries)} 条 → {CONSTRUCTED_BATCH_DIR}")
    print(f"  建议带: {dict(Counter(e['scoring']['machine_suggestion'] for e in entries))}")
    print(
        f"  事实三态: {dict(Counter(e['scoring']['dimensions']['factual_grounding'] for e in entries))}"
    )


# ============================================
# --decisions：校验 + 阶段 3 后预算重算
# ============================================


def apply_decisions(decisions_path: Path) -> dict:
    """校验决定 + 输出保留清单 + 阶段 3 后预算重算。

    预算重算口径（消除"假定短构造全部保留"的初步口径）：
    - 原作：V4 实测（不变）；
    - 短构造：阶段 3 approved 决定的 keep 记录；
    - 模拟：阶段 2 approved 决定的 keep 记录（决定文件必须已存在且 approved）。
    """
    decision_doc = json.loads(decisions_path.read_text(encoding="utf-8"))
    records, manifest = load_canonical_records()

    constructed = [
        r for r in records if classify_record_source(r) == "llm_v4_reviewed_constructed"
    ]
    con_ids = {r["id"] for r in constructed}
    validate_decision_document(decision_doc, con_ids)

    # 阶段 2 approved 模拟决定必须已存在（预算重算依赖）
    if not SIMULATION_DECISIONS_PATH.exists():
        raise SystemExit(
            f"[ABORT] 缺少阶段 2 approved 决定文件: {SIMULATION_DECISIONS_PATH}"
            "（阶段 3 预算重算需要模拟保留数据）"
        )
    sim_doc = json.loads(SIMULATION_DECISIONS_PATH.read_text(encoding="utf-8"))
    sim_records = [
        r for r in records if classify_record_source(r) in SIMULATION_SOURCES
    ]
    sim_ids = {r["id"] for r in sim_records}
    validate_decision_document(sim_doc, sim_ids)

    decisions = decision_doc["decisions"]
    by_id = {r["id"]: r for r in records}

    def _kept_stats(ids: list[str]) -> dict:
        chars = n = 0
        for rid in ids:
            exposure = measure_record_exposure(by_id[rid])
            chars += exposure["supervised_assistant_chars"]
            n += 1
        return {"records": n, "sup_chars": chars}

    game_ids = [
        r["id"]
        for r in records
        if classify_record_source(r) == "game_extraction_current_sft"
    ]
    game = _kept_stats(game_ids)
    con_kept = _kept_stats([rid for rid, v in decisions.items() if v == "keep"])
    sim_kept = _kept_stats(
        [rid for rid, v in sim_doc["decisions"].items() if v == "keep"]
    )

    total_chars = game["sup_chars"] + con_kept["sup_chars"] + sim_kept["sup_chars"]
    total_records = game["records"] + con_kept["records"] + sim_kept["records"]
    result = {
        "dataset_id": manifest["dataset_id"],
        "budget_final_after_phase3": {
            "game_extraction": game,
            "constructed_kept": con_kept,
            "simulation_kept": sim_kept,
            "total_sup_chars": total_chars,
            "total_records": total_records,
            "share_pct": {
                "game": round(game["sup_chars"] / total_chars * 100, 1),
                "constructed": round(con_kept["sup_chars"] / total_chars * 100, 1),
                "simulation": round(sim_kept["sup_chars"] / total_chars * 100, 1),
            },
            "note": (
                "阶段 3 后正式口径：短构造按 approved 决定计入，"
                "模拟按阶段 2 approved 决定计入；不再有假定全留的初步口径"
            ),
        },
        "constructed_kept_records": sorted(
            rid for rid, v in decisions.items() if v == "keep"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="阶段 3：150 条短构造复查材料生成 / 决定校验与预算重算"
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help="人工批准决定文件（校验后输出保留清单与阶段 3 后预算重算；不构建 V5）",
    )
    args = parser.parse_args()

    if args.decisions:
        apply_decisions(args.decisions)
    else:
        generate_review_materials()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
