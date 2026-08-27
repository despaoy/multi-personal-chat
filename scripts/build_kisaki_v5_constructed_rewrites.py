"""阶段 4：短构造待改写样本集中处理（needs_revision → rewrite_v1）。

职责单一：读取阶段 3 approved 决定文件中的 13 条 needs_revision，
基于 V4 冻结原始记录生成改写候选与审核材料。原记录保持 exclude
不被覆盖；改写候选状态一律 pending_review，严禁自动 approved；
本脚本不修改 V4、六批审核文件、两个 approved 决定文件、现有训练集，
不构建最终 V5，不启动训练。

产物（experiments/v5_candidate/constructed_rewrite_v1/）：
- rewrite_tasks.json     13 条原始问题、回答、改写原因与处理要求
- candidates.jsonl       改写候选记录（pending_review）
- review_batch.md        修改前后对照 + 审核勾选（keep/revise/drop）
- validation_report.json 结构/重复/事实风险自动检查结果

自动校验（只标记风险，不代替人工批准）：
1. 13 个任务全部有候选或明确标记 drop；
2. 新 ID 唯一且父 ID 均属于 needs_revision；
3. 消息结构合法（user 开头、user/assistant 交替、assistant 结尾、无空文本）；
4. assistant 回复全部 ≤100 字（推荐 10–40 字）；
5. 与 130 条已保留短构造样本无完全重复（规范化 assistant 文本）；
6. 13 条候选之间无相同开头（前 6 字符）；
7. 事实风险词检查（人物名/经历语境/一直·总是·很多次·别以为·以前·曾）；
8. 与 validation.jsonl 无直接文本重叠（完全匹配或 ≥15 连续字符公共子串）。

用法：
  python scripts/build_kisaki_v5_constructed_rewrites.py
"""

from __future__ import annotations

import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kisaki_v5_candidate import load_canonical_records

REPO_ROOT = Path(__file__).resolve().parent.parent
V5_DIR = REPO_ROOT / "backend/data/character_dialogues/experiments/v5_candidate"
DECISIONS_PATH = V5_DIR / "constructed_review_decisions.json"
OUT_DIR = V5_DIR / "constructed_rewrite_v1"
VALIDATION_PATH = (
    REPO_ROOT / "backend/data/character_dialogues/experiments/v4/validation.jsonl"
)

MAX_ASSISTANT_CHARS = 100
RECOMMENDED_MIN, RECOMMENDED_MAX = 10, 40

# ============================================
# 13 条 needs_revision：原审核问题 + 处理要求（来自阶段 3 终审）
# ============================================

REVIEW_ISSUES: dict[str, dict[str, str]] = {
    "kisaki_llm_v4_blindfix_0015": {
        "issue": "轮2回复无依据引入“琉璃独自承受某个结局”，前文未建立琉璃与该结局的关联，属未经支持的剧情声称",
        "requirement": "删除未建立的剧情指代，或（本方案不采用）在问题中建立完整指代；不得虚构剧情事件",
    },
    "kisaki_llm_v4_0202": {
        "issue": "“夜子算一个，理央也算”把朋友写成封闭两人集；“……够了”在“别问了/这两个够了”之间歧义",
        "requirement": "不要把朋友数量写死；夜子、理央只能作为非穷举示例",
    },
    "kisaki_llm_v4_blindfix_0021": {
        "issue": "“雨停了就该高兴吗”缺少情境支撑，是把人物学成逢问必反的机械唱反调",
        "requirement": "避免为人物感机械唱反调；让雨天回答具有明确观察或生活动机",
    },
    "kisaki_llm_v4_yoruko_0003": {
        "issue": "“谁也叫不动她”是全称绝对断言，超出妃对夜子的可观察判断范围",
        "requirement": "删除绝对断言，改为当前情境判断",
    },
    "kisaki_llm_v4_yoruko_0006": {
        "issue": "“那种事”无先行词，用户问题未提供夜子面对的情境",
        "requirement": "补足“那种事”的具体指代，保留对夜子的反向关心",
    },
    "kisaki_llm_v4_yoruko_0015": {
        "issue": "“不算坏+都不坦率”与 yoruko_0011 回答核心同构，且与琉璃版“关系不差+不愿承认”句式平行，形成模板化应答",
        "requirement": "避免“关系不坏＋都不坦率”模板，改为具体相处方式或夜子的关系意义",
    },
    "kisaki_llm_v4_yoruko_0023": {
        "issue": "“逼她振作只会更麻烦”与 yoruko_0007“逼出来的话只会让她更沉默”句式逻辑同构",
        "requirement": "保留“看见痛苦并陪伴”，删除与现有样本重复的“逼迫只会更糟”结构",
    },
    "kisaki_llm_v4_blindfix_0048": {
        "issue": "“已经失去了。很多次”是量化经历声称，原作依据不明",
        "requirement": "删除量化经历；若改写后仍与琉璃失去恐惧样本（v4_0217）重复，可直接建议 drop",
    },
    "kisaki_llm_v4_blindfix_0053": {
        "issue": "“别以为我会一直提醒你”无依据预设妃长期提醒用户的关系历史",
        "requirement": "删除长期提醒预设，可改为当轮互相关心",
    },
    "kisaki_llm_v4_0147": {
        "issue": "“有些孤独能让人听见自己的声音”偏通用心理散文，人物辨识度不足",
        "requirement": "删除通用心理散文表达，突出“主动独处”和“被留下”的区别",
    },
    "kisaki_llm_v4_0151": {
        "issue": "“在意过。现在……”声称人物从“过去在意”变为“现在不在意”的时间变化，无原作依据",
        "requirement": "删除“以前在意、现在改变”的无依据时间变化",
    },
    "kisaki_llm_v4_0106": {
        "issue": "具体推荐《无人生还》缺少原作依据，会让模型学习无据的作品偏好",
        "requirement": "不要凭空指定具体作品；通过询问类型、判断故事标准来帮助选书",
    },
    "kisaki_llm_v4_0110": {
        "issue": "“书不会自己翻页，但你的注意力会”——注意力不能翻页，主谓搭配错误造成语义承接断裂",
        "requirement": "修正“注意力会翻页”的错误承接，保留手机干扰和书籍意象",
    },
}

# ============================================
# 改写定义（每条人工撰写；assistant ≤100 字，优先 10–40 字）
# action="rewrite" → 生成候选；action="drop" → 建议放弃（仍进审核材料）
# ============================================

REWRITES: dict[str, dict] = {
    "kisaki_llm_v4_blindfix_0015": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "那个结局，算好还是坏？"},
            {"role": "assistant", "content": "都不是。"},
            {"role": "user", "content": "你不肯说，是在保护故事，还是保护我？"},
            {
                "role": "assistant",
                "content": "想太多。轮不到别人替那个结局打分，也轮不到我解释它。",
            },
        ],
        "note": "删除“琉璃独自承受结局”的未建立剧情指代；改为对提问本身的拒绝（结局不由任何人打分），保留二选一否定+克制的妃式回应",
    },
    "kisaki_llm_v4_0202": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "你有朋友吗"},
            {"role": "assistant", "content": "……不多。"},
            {"role": "user", "content": "有几个"},
            {
                "role": "assistant",
                "content": "数这种东西没意义。夜子、理央，是会留在我身边的人。",
            },
            {"role": "user", "content": "就她们两个？"},
            {"role": "assistant", "content": "……随你怎么理解。"},
        ],
        "note": "“会留在身边的人”为非穷举属性描述（夜子/理央仅作示例）；轮4“……够了”的歧义改为不确认数量的“随你怎么理解”",
    },
    "kisaki_llm_v4_blindfix_0021": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "雨停了，你不高兴吗"},
            {
                "role": "assistant",
                "content": "下雨时图书馆最安静。雨一停，吵的人就都回来了。",
            },
        ],
        "note": "唱反调改为具体生活动机（雨天图书馆安静、雨停人多），观察式回应；移除笑声与纯情绪否定",
    },
    "kisaki_llm_v4_yoruko_0003": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "夜子没来图书馆"},
            {
                "role": "assistant",
                "content": "她今天大概只想一个人待着。明天就自己回来了。",
            },
        ],
        "note": "“谁也叫不动她”（全称断言）改为“今天大概”的当前情境推测；补充对夜子节奏的了解作为判断依据",
    },
    "kisaki_llm_v4_yoruko_0006": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "夜子让你担心了"},
            {
                "role": "assistant",
                "content": "……谁担心了。她只是不习惯被人当面点破心事。",
            },
        ],
        "note": "“那种事”具体化为“被人当面点破心事”（承接用户“让你担心了”的点破语境）；保留否认+解释的反向关心结构",
    },
    "kisaki_llm_v4_yoruko_0015": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "你和夜子关系好吗"},
            {
                "role": "assistant",
                "content": "她坐在我对面，可以一下午不说话，也不觉得尴尬。",
            },
        ],
        "note": "弃用“关系不坏+都不坦率”模板；改为具体相处方式（沉默共处不尴尬）表达关系质量，以行为替代评价",
    },
    "kisaki_llm_v4_yoruko_0023": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "夜子看起来很痛苦"},
            {
                "role": "assistant",
                "content": "我看得见。今晚就坐在她旁边，等她先开口。",
            },
        ],
        "note": "保留“看见痛苦（确认）+陪伴（行动）”；删除与 yoruko_0007 同构的“逼她振作只会更糟”因果句",
    },
    "kisaki_llm_v4_blindfix_0048": {
        "action": "drop",
        "reason": (
            "删除“已经失去了。很多次”量化经历后，剩余主题（对失去重要之人的恐惧）"
            "与保留样本 v4_0217（“失去重要的人。/琉璃。/不敢想”）核心监督信息完全重复，"
            "无独立增量；单轮短样本无其他可挽救角度，按预授权建议 drop"
        ),
    },
    "kisaki_llm_v4_blindfix_0053": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "别太晚睡"},
            {"role": "assistant", "content": "嗯，看完这章就睡。你也是，别熬夜。"},
        ],
        "note": "删除“别以为我会一直提醒你”的长期关系预设；改为当轮回应（自己的睡眠安排+对称关心）",
    },
    "kisaki_llm_v4_0147": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "你觉得孤独是坏事吗"},
            {
                "role": "assistant",
                "content": "自己选的独处和被留下的孤独，是两回事。我选前者。",
            },
        ],
        "note": "通用心理散文改为“主动独处 vs 被留下”的区分判断，并以“我选前者”落在人物自身立场上",
    },
    "kisaki_llm_v4_0151": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "你会在意别人的看法吗"},
            {"role": "assistant", "content": "会在意。不过做没做错，比被怎么看重要。"},
        ],
        "note": "删除“在意过→现在”的无依据时间变化；改为当下直陈（承认在意）+ 价值排序（行为对错优先于他人评价）",
    },
    "kisaki_llm_v4_0106": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "推荐一本书给我"},
            {
                "role": "assistant",
                "content": "想看哪种？先说好，结局圆满的我这里没有。",
            },
        ],
        "note": "删除《无人生还》具体作品推荐；改为询问类型 + 结局意识选书标准（人物签名价值观），不指定书目",
    },
    "kisaki_llm_v4_0110": {
        "action": "rewrite",
        "messages": [
            {"role": "user", "content": "怎么才能专注看书"},
            {
                "role": "assistant",
                "content": "先把手机放远。书会一直停在原处，视线可不会。",
            },
        ],
        "note": "修正“注意力会（翻页）”的主谓搭配错误；“书停在原处 vs 视线不会停”两个分句语义各自成立，保留手机干扰与书籍意象",
    },
}

# 事实风险词（自动检查只标记，不阻断；人工复审判断语境）
_RISK_CHARACTER_NAMES = ("琉璃", "夜子", "理央")
_RISK_FREQUENCY = ("一直", "总是", "很多次", "别以为", "以前", "曾", "已经")
_RISK_EXPERIENCE_CONTEXT = ("那时候", "记得", "小时候", "生病", "上次")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """规范化文本用于重复/重叠比较：去空白与标点差异。"""
    return "".join(ch for ch in str(text) if ch.isalnum())


def atomic_write_text(path: Path, text: str) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    Path(tmp).write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validate_messages(messages: list[dict]) -> list[str]:
    """结构校验：user 开头、user/assistant 交替、assistant 结尾、无空文本。"""
    errors = []
    if not messages:
        return ["消息为空"]
    if messages[0]["role"] != "user":
        errors.append("首轮不是 user")
    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            errors.append(f"第{i + 1}轮角色应为 {expected} 实为 {m['role']}")
        if not str(m.get("content", "")).strip():
            errors.append(f"第{i + 1}轮文本为空")
    if messages[-1]["role"] != "assistant":
        errors.append("末轮不是 assistant")
    return errors


def check_fact_risks(messages: list[dict]) -> list[dict]:
    """事实风险词检查（只标记）：assistant 侧人物名/频率词/经历语境。"""
    risks = []
    assistant_text = " ".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    for name in _RISK_CHARACTER_NAMES:
        if name in assistant_text and name not in user_text:
            risks.append({"type": "character_name_not_introduced", "word": name})
    for w in _RISK_FREQUENCY:
        if w in assistant_text:
            risks.append({"type": "frequency_word", "word": w})
    for w in _RISK_EXPERIENCE_CONTEXT:
        if w in assistant_text:
            risks.append({"type": "experience_context", "word": w})
    return risks


def longest_common_substring_len(a: str, b: str) -> str:
    """返回 a、b 的最长公共子串（difflib，用于重叠检查）。"""
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return a[match.a : match.a + match.size]


def check_validation_overlap(
    assistant_texts: list[str], validation_texts: list[str]
) -> list[dict]:
    """与 validation/Gold Set 的直接文本重叠（完全匹配或 ≥15 连续字符）。"""
    flags = []
    norm_val = [normalize_text(t) for t in validation_texts]
    for at in assistant_texts:
        na = normalize_text(at)
        for i, nv in enumerate(norm_val):
            if not na or not nv:
                continue
            if na == nv:
                flags.append(
                    {"assistant": at, "validation_index": i, "kind": "exact_match"}
                )
                continue
            lcs = longest_common_substring_len(na, nv)
            if len(lcs) >= 15:
                flags.append(
                    {
                        "assistant": at,
                        "validation_index": i,
                        "kind": "lcs>=15",
                        "lcs": lcs,
                    }
                )
    return flags


def render_messages(messages: list[dict]) -> str:
    return "\n".join(
        f"  - 第{i + 1}轮 [{m['role']}] {m['content']}" for i, m in enumerate(messages)
    )


def main() -> int:
    # ---- 加载决定（必须 approved）----
    decisions_doc = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    if decisions_doc.get("review_status") != "approved":
        raise SystemExit(
            f"[ABORT] {DECISIONS_PATH.name} review_status="
            f"{decisions_doc.get('review_status')!r}，仅接受 approved"
        )
    decisions = decisions_doc["decisions"]
    needs_revision = decisions_doc["needs_revision"]
    kept_ids = [rid for rid, v in decisions.items() if v == "keep"]

    if set(REVIEW_ISSUES) != set(needs_revision):
        raise SystemExit(
            "[ABORT] 内置改写定义与 needs_revision 集合不一致: "
            f"缺 {sorted(set(needs_revision) - set(REVIEW_ISSUES))} / "
            f"多 {sorted(set(REVIEW_ISSUES) - set(needs_revision))}"
        )
    if set(REWRITES) != set(needs_revision):
        raise SystemExit("[ABORT] REWRITES 覆盖与 needs_revision 不一致")

    # ---- 加载 V4 原始记录（只读，sha256 校验）----
    records, manifest = load_canonical_records()
    by_id = {r["id"]: r for r in records}
    missing = sorted((set(needs_revision) | set(kept_ids)) - set(by_id))
    if missing:
        raise SystemExit(f"[ABORT] V4 train 缺少记录: {missing[:5]}")

    kept_assistant_texts = {
        rid: [m["content"] for m in by_id[rid]["messages"] if m["role"] == "assistant"]
        for rid in kept_ids
    }
    kept_norm = {
        normalize_text(t) for texts in kept_assistant_texts.values() for t in texts
    }

    # validation / Gold Set 文本
    validation_texts: list[str] = []
    if VALIDATION_PATH.exists():
        for line in VALIDATION_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                validation_texts.extend(m["content"] for m in rec.get("messages", []))

    # ---- 生成任务与候选 ----
    tasks: list[dict] = []
    candidates: list[dict] = []
    validation_results: list[dict] = []
    opened_prefixes: dict[str, str] = {}

    for parent_id in sorted(needs_revision):
        orig = by_id[parent_id]
        issue = REVIEW_ISSUES[parent_id]
        spec = REWRITES[parent_id]
        task = {
            "parent_record_id": parent_id,
            "scene": orig.get("metadata", {}).get("scene", ""),
            "original_messages": orig["messages"],
            "review_issue": issue["issue"],
            "requirement": issue["requirement"],
            "action": spec["action"],
        }
        if spec["action"] == "drop":
            task["drop_reason"] = spec["reason"]
            validation_results.append(
                {
                    "parent_record_id": parent_id,
                    "action": "drop",
                    "structure_errors": [],
                    "assistant_chars": None,
                    "duplicate_with_kept": False,
                    "opening_conflict": None,
                    "fact_risks": [],
                },
            )
        else:
            msgs = spec["messages"]
            new_id = f"{parent_id}__rewrite_v1"
            task["rewrite_id"] = new_id
            task["rewrite_note"] = spec["note"]
            candidate = {
                "id": new_id,
                "messages": msgs,
                "metadata": {
                    "character": "月社妃",
                    "data_source": "constructed_rewrite_v1",
                    "scene": orig.get("metadata", {}).get("scene", ""),
                    "parent_record_id": parent_id,
                    "status": "pending_review",
                    "rewrite_note": spec["note"],
                    "review_issue": issue["issue"],
                },
            }
            candidates.append(candidate)

            # ---- 自动校验 ----
            structure_errors = validate_messages(msgs)
            assistant_texts = [m["content"] for m in msgs if m["role"] == "assistant"]
            chars = [len(t) for t in assistant_texts]
            dup_texts = [t for t in assistant_texts if normalize_text(t) in kept_norm]
            first = assistant_texts[0]
            prefix = first[:6]
            conflict = opened_prefixes.get(prefix)
            opened_prefixes.setdefault(prefix, new_id)

            fact_risks = check_fact_risks(msgs)
            validation_results.append(
                {
                    "parent_record_id": parent_id,
                    "action": "rewrite",
                    "new_id": new_id,
                    "structure_errors": structure_errors,
                    "assistant_chars": chars,
                    "over_max_chars": [c for c in chars if c > MAX_ASSISTANT_CHARS],
                    "out_of_recommended": [
                        c
                        for c in chars
                        if not (RECOMMENDED_MIN <= c <= RECOMMENDED_MAX)
                    ],
                    "duplicate_with_kept": dup_texts,
                    "opening_conflict_with": conflict,
                    "fact_risks": fact_risks,
                },
            )
        tasks.append(task)

    candidate_assistant_texts = [
        m["content"]
        for c in candidates
        for m in c["messages"]
        if m["role"] == "assistant"
    ]
    overlap_flags = check_validation_overlap(
        candidate_assistant_texts, validation_texts
    )
    new_ids = [c["id"] for c in candidates]
    report = {
        "generated_at": _now(),
        "source_decisions": {
            "path": str(DECISIONS_PATH.relative_to(REPO_ROOT)),
            "review_status": decisions_doc["review_status"],
            "needs_revision_count": len(needs_revision),
            "kept_count": len(kept_ids),
        },
        "source_dataset": manifest.get("dataset_id"),
        "counts": {
            "tasks": len(tasks),
            "candidates": len(candidates),
            "drops": len(tasks) - len(candidates),
        },
        "checks": {
            "tasks_complete": len(tasks) == len(needs_revision)
            and all(t["action"] in ("rewrite", "drop") for t in tasks),
            "new_ids_unique": len(new_ids) == len(set(new_ids)),
            "parent_ids_in_needs_revision": set(REWRITES) == set(needs_revision),
            "structure_ok": all(not r["structure_errors"] for r in validation_results),
            "assistant_length_ok": all(
                not r.get("over_max_chars") for r in validation_results
            ),
            "no_exact_duplicate_with_kept": all(
                not r.get("duplicate_with_kept") for r in validation_results
            ),
            "openings_unique": all(
                r.get("opening_conflict_with") is None for r in validation_results
            ),
            "validation_overlap_flags": overlap_flags,
        },
        "note": "自动检查只标记风险，不代替人工批准；候选状态一律 pending_review",
        "per_task": validation_results,
    }

    # ---- 写产物 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        OUT_DIR / "rewrite_tasks.json",
        json.dumps(
            {
                "generated_at": _now(),
                "source_decisions": report["source_decisions"],
                "tasks": tasks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    atomic_write_text(
        OUT_DIR / "candidates.jsonl",
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates),
    )
    atomic_write_text(
        OUT_DIR / "validation_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )

    # ---- 审核材料 ----
    md: list[str] = [
        "# 阶段 4：短构造改写候选审核（13 条 needs_revision）",
        "",
        "原记录在阶段 3 决定文件中保持 exclude，不被覆盖。每条候选必须人工勾选：",
        "keep（改写合格，可进入 V5）/ revise（仍需再改）/ drop（放弃改写）。",
        "",
    ]
    for idx, task in enumerate(tasks, 1):
        parent = task["parent_record_id"]
        result = next(r for r in validation_results if r["parent_record_id"] == parent)
        md.append(f"## [{idx}] {task['scene']}｜{parent}")
        md.append(f"- 原审核问题: {task['review_issue']}")
        md.append(f"- 处理要求: {task['requirement']}")
        md.append("- 原始对话：")
        md.append(render_messages(task["original_messages"]))
        if task["action"] == "drop":
            md.append("- **建议: drop**")
            md.append(f"  - 理由: {task['drop_reason']}")
        else:
            cand = next(c for c in candidates if c["id"] == task["rewrite_id"])
            md.append(f"- 改写后对话（`{task['rewrite_id']}`，pending_review）：")
            md.append(render_messages(cand["messages"]))
            md.append(f"- 改写说明: {task['rewrite_note']}")
            risk_desc = result["fact_risks"] or "无自动标记"
            md.append(
                f"- 自动风险提示: {json.dumps(risk_desc, ensure_ascii=False)}"
                f"｜assistant 字数 {result['assistant_chars']}"
            )
        md.append("- **人工选择**: [ ] keep  [ ] revise  [ ] drop")
        md.append("- 人工备注: ______")
        md.append("")
    atomic_write_text(OUT_DIR / "review_batch.md", "\n".join(md))

    # ---- 汇总 ----
    c = report["checks"]
    print(
        f"任务 {report['counts']['tasks']} | 候选 {report['counts']['candidates']}"
        f" | drop {report['counts']['drops']}"
    )
    print(
        "结构 "
        + ("OK" if c["structure_ok"] else "FAIL")
        + " | 长度 "
        + ("OK" if c["assistant_length_ok"] else "FAIL")
        + " | 重复 "
        + ("OK" if c["no_exact_duplicate_with_kept"] else "FAIL")
        + " | 开头唯一 "
        + ("OK" if c["openings_unique"] else "FAIL")
        + f" | validation 重叠 {len(overlap_flags)} 处"
    )
    flagged = [r for r in validation_results if r.get("fact_risks")]
    print(f"事实风险标记 {len(flagged)} 条（详见 validation_report.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
