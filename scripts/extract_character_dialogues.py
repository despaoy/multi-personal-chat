"""从游戏文本提取月社妃角色对话，生成可审计的 raw 和 SFT 数据集。

v2 改进（相对 v1）：
- 适配 gametext/纸上魔法使/*.txt 数据源结构
- 移除已清除的深白水波（ATRI）相关代码
- 添加过短台词过滤与占比控制（≤ 15%），缓解 v1 回复塌缩问题
- 保留 v1 的质量评分、去重、excluded 审计逻辑
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

TARGETS = {
    "tsukiyashiro_kisaki": {
        "display_name": "月社妃",
        "train_aliases": {"妃", "月社妃"},
        "joint_aliases": set(),
        "system": "你正在扮演月社妃。请依据给定角色设定和原作中的语言习惯，自然地回应。",
    },
}
SCRIPT_RE = re.compile(
    r"\[(?P<speaker>[^\]]+)\]\s*(?:"
    r"「(?P<corner_text>.*?)」|"
    r"『(?P<double_corner_text>.*?)』|"
    r"“(?P<curly_text>.*?)”"
    r")",
    re.DOTALL,
)
SPACE_RE = re.compile(r"\s+")
MEANING_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)

# v2 新增：过短回复阈值与占比上限
SHORT_REPLY_THRESHOLD = 5  # 有效字符数 < 5 视为过短
SHORT_REPLY_MAX_RATIO = 0.15  # 过短回复在最终 SFT 中占比 ≤ 15%
MAX_INTERVENING_NARRATION_LINES = 4
SCENE_RESET_RE = re.compile(
    r"^(?:"
    r"次日|翌日|第二天|数日后|几天后|隔天|"
    r"第二天早上|当天(?:早上|上午|下午|晚上)?|"
    r"放学后|清晨|早晨|傍晚|夜晚|深夜|"
    r"与此同时|另一方面|梦中|回忆的讲述(?:翻到下一页|已经结束)"
    r")"
)
UNCERTAIN_SPEAKER_RE = re.compile(r"^[?？]+$")
UNCERTAIN_SPEAKER_LABELS = {"声音", "某人", "不明", "众人", "全员"}


def clean(text):
    return SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def norm_key(text):
    text = unicodedata.normalize("NFKC", text).replace("……", "…").replace("--", "—")
    return SPACE_RE.sub("", text).strip()


def meaningful_len(text):
    return len(MEANING_RE.findall(text))


def stable_id(*parts):
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _separator_lines(separator):
    return [line.strip() for line in separator.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def scene_reset_in(separator):
    normalized = separator.replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"\n\s*\n", normalized):
        return True
    return any(SCENE_RESET_RE.match(line) for line in _separator_lines(normalized) if line)


def dialogue_link_reason(previous, current):
    """Return why two dialogue events cannot belong to one continuous block."""

    separator = current.get("separator_before", "")
    if scene_reset_in(separator):
        return "scene_boundary"
    narration_lines = sum(1 for line in _separator_lines(separator) if line)
    if narration_lines > MAX_INTERVENING_NARRATION_LINES:
        return "long_narration_gap"
    if current["line_start"] <= previous["line_end"]:
        return "overlapping_dialogue_events"
    return None


def reliable_speaker_label(label):
    normalized = clean(label)
    return bool(
        normalized
        and normalized not in UNCERTAIN_SPEAKER_LABELS
        and not UNCERTAIN_SPEAKER_RE.fullmatch(normalized)
    )


def read_script_events(path, source_id=None):
    """从纸上魔法使脚本文件提取对话事件。

    文件格式：[角色名] 「对话内容」
    同时保留台词物理行号及相邻台词之间的原文，用于识别长旁白和场景切换。
    """
    source_id = source_id or path.name
    text = path.read_text(encoding="utf-8")
    events = []
    previous_match_end = 0
    for match in SCRIPT_RE.finditer(text):
        line_start = text.count("\n", 0, match.start()) + 1
        line_end = text.count("\n", 0, match.end()) + 1
        dialogue = next(
            value
            for value in (
                match.group("corner_text"),
                match.group("double_corner_text"),
                match.group("curly_text"),
            )
            if value is not None
        )
        events.append({
            "speaker": clean(match.group("speaker")),
            "text": clean(dialogue),
            "source": f"{source_id}:line:{line_start}",
            "line_start": line_start,
            "line_end": line_end,
            "separator_before": text[previous_match_end:match.start()],
        })
        previous_match_end = match.end()

    block_start_line = None
    previous = None
    for event in events:
        if previous is None or dialogue_link_reason(previous, event):
            block_start_line = event["line_start"]
        event["scene_block_id"] = (
            f"{source_id}:scene-block:{block_start_line}"
        )
        previous = event
    return events


def source_scripts(root):
    """遍历 gametext/纸上魔法使/ 下所有 .txt 文件。

    每个文件视为 canonical 来源（完整剧情卷）。
    """
    for path in sorted(root.glob("*.txt")):
        yield path, path.name, "canonical"


def target_kind(speaker, target_key):
    target = TARGETS[target_key]
    if speaker in target["train_aliases"]:
        return "direct"
    if speaker in target["joint_aliases"]:
        return "joint"
    return None


def make_raw(groups, target_key):
    records = []
    for group in groups:
        for event in group["events"]:
            kind = target_kind(event["speaker"], target_key)
            if kind is None or not event["text"]:
                continue
            event_id = stable_id(target_key, event["source"], event["text"])
            records.append({
                "id": f"{target_key}_raw_{event_id}",
                "character": TARGETS[target_key]["display_name"],
                "speaker_label": event["speaker"],
                "speaker_kind": kind,
                "text": event["text"],
                "source": event["source"],
                "source_file": group["source_id"],
                "source_line_start": event["line_start"],
                "source_line_end": event["line_end"],
                "scene_block_id": event["scene_block_id"],
                "source_role": group["source_role"],
                "eligible_for_sft": kind == "direct" and group["source_role"] == "canonical",
            })
    return records


def quality_score(prompt, reply, lines):
    score = 50 + min(meaningful_len(prompt), 30) // 3
    score += min(meaningful_len(reply), 60) // 4
    score += min(max(lines - 1, 0), 3) * 2
    score += 3 if any(x in prompt for x in ("？", "?")) else 0
    score -= 8 if meaningful_len(reply) > 240 else 0
    return max(0, min(score, 100))


def build_candidates(groups, target_key):
    aliases = TARGETS[target_key]["train_aliases"]
    candidates = []
    for group in groups:
        if group["source_role"] != "canonical":
            continue
        events, index = group["events"], 0
        while index < len(events):
            event = events[index]
            if event["speaker"] not in aliases:
                index += 1
                continue
            start, reply_events = index, []
            while index < len(events) and events[index]["speaker"] in aliases:
                if reply_events and dialogue_link_reason(reply_events[-1], events[index]):
                    break
                if events[index]["text"]:
                    reply_events.append(events[index])
                index += 1
            cursor = start - 1
            boundary_reason = None
            if cursor >= 0:
                boundary_reason = dialogue_link_reason(events[cursor], events[start])
            context_source_speaker = (
                events[cursor]["speaker"] if cursor >= 0 and boundary_reason is None else None
            )
            context_speaker = (
                context_source_speaker
                if context_source_speaker and reliable_speaker_label(context_source_speaker)
                else None
            )
            context_events = []
            while cursor >= 0 and events[cursor]["speaker"] == context_source_speaker:
                if context_events and dialogue_link_reason(events[cursor], context_events[-1]):
                    break
                if events[cursor]["text"]:
                    context_events.append(events[cursor])
                cursor -= 1
            context_events.reverse()
            prompt = "\n".join(x["text"] for x in context_events).strip()
            reply = "\n".join(x["text"] for x in reply_events).strip()
            reasons = []
            if boundary_reason:
                reasons.append(boundary_reason)
            if not context_events:
                reasons.append("missing_context")
            if meaningful_len(prompt) < 2:
                reasons.append("low_information_prompt")
            if meaningful_len(reply) < 2:
                reasons.append("low_information_reply")
            if len(prompt) > 1200:
                reasons.append("prompt_too_long")
            if len(reply) > 1600:
                reasons.append("reply_too_long")
            # v2 新增：标记过短回复（不直接排除，后续按占比控制）
            is_short = meaningful_len(reply) < SHORT_REPLY_THRESHOLD
            if is_short:
                reasons.append("short_reply")
            source = reply_events[0]["source"] if reply_events else event["source"]
            ids = [f"{target_key}_raw_{stable_id(target_key, x['source'], x['text'])}" for x in reply_events]
            candidates.append({
                "id": f"{target_key}_turn_{stable_id(target_key, source, prompt, reply)}",
                "source": source,
                "source_file": group["source_id"],
                "source_speaker_label": event["speaker"],
                "context_speaker_label": context_speaker,
                "context_source_speaker_label": context_source_speaker,
                "prompt": prompt,
                "reply": reply,
                "target_event_ids": ids,
                "scene_block_id": event["scene_block_id"],
                "dialogue_block_id": (
                    f"{group['source_id']}:turn:"
                    f"{context_events[0]['line_start'] if context_events else event['line_start']}-"
                    f"{reply_events[-1]['line_end'] if reply_events else event['line_end']}"
                ),
                "source_line_start": context_events[0]["line_start"] if context_events else event["line_start"],
                "source_line_end": reply_events[-1]["line_end"] if reply_events else event["line_end"],
                "context_line_start": context_events[0]["line_start"] if context_events else None,
                "context_line_end": context_events[-1]["line_end"] if context_events else None,
                "response_line_start": reply_events[0]["line_start"] if reply_events else event["line_start"],
                "response_line_end": reply_events[-1]["line_end"] if reply_events else event["line_end"],
                "response_line_count": len(reply_events),
                "quality_score": quality_score(prompt, reply, len(reply_events)),
                "reasons": reasons,
                "is_short_reply": is_short,
            })
    return candidates


def as_sft(item, target_key, *, include_exclusion_reasons=False):
    metadata = {
        "character": TARGETS[target_key]["display_name"],
        "source": item["source"],
        "source_file": item["source_file"],
        "source_speaker_label": item["source_speaker_label"],
        "context_speaker_label": item["context_speaker_label"],
        "context_source_speaker_label": item.get("context_source_speaker_label"),
        "target_event_ids": item["target_event_ids"],
        "dialogue_block_id": item.get("dialogue_block_id"),
        "scene_block_id": item.get("scene_block_id"),
        "source_line_start": item.get("source_line_start"),
        "source_line_end": item.get("source_line_end"),
        "context_line_start": item.get("context_line_start"),
        "context_line_end": item.get("context_line_end"),
        "response_line_start": item.get("response_line_start"),
        "response_line_end": item.get("response_line_end"),
        "response_line_count": item["response_line_count"],
        "quality_score": item["quality_score"],
        "is_short_reply": item.get("is_short_reply", False),
        "extraction": "explicit-label-continuous-dialogue-block-v3",
    }
    if include_exclusion_reasons:
        metadata["exclusion_reasons"] = sorted(set(item.get("reasons", [])))
    return {
        "id": f"{target_key}_sft_{stable_id(target_key, item['source'], item['prompt'], item['reply'])}",
        "system": TARGETS[target_key]["system"],
        "conversations": [
            {"from": "human", "value": item["prompt"]},
            {"from": "assistant", "value": item["reply"]},
        ],
        "metadata": metadata,
    }


def make_sft(groups, target_key):
    """构建 SFT 数据集，控制过短回复占比。

    流程：
    1. 排除有质量问题的候选（除 short_reply 外的 reasons）
    2. 去重（同 prompt 取质量分最高的）
    3. 控制过短回复占比 ≤ SHORT_REPLY_MAX_RATIO
    """
    candidates = build_candidates(groups, target_key)

    # Step 1: 排除非 short_reply 的质量问题
    hard_excluded = []
    valid = []
    for item in candidates:
        non_short_reasons = [r for r in item["reasons"] if r != "short_reply"]
        if non_short_reasons:
            hard_excluded.append(item)
        else:
            valid.append(item)

    # Step 2: 去重（同 prompt 取质量分最高的）
    best = {}
    dedup_excluded = []
    for item in valid:
        key = norm_key(item["prompt"])
        old = best.get(key)
        rank = (item["quality_score"], meaningful_len(item["reply"]), item["id"])
        old_rank = None if old is None else (old["quality_score"], meaningful_len(old["reply"]), old["id"])
        if old is None or rank > old_rank:
            if old is not None:
                old_copy = dict(old)
                old_copy["reasons"] = old_copy["reasons"] + ["duplicate_prompt_lower_rank"]
                dedup_excluded.append(old_copy)
            best[key] = item
        else:
            item_copy = dict(item)
            item_copy["reasons"] = item_copy["reasons"] + ["duplicate_prompt_lower_rank"]
            dedup_excluded.append(item_copy)

    recommended = list(best.values())

    # Step 3: 控制过短回复占比
    short_items = [x for x in recommended if x.get("is_short_reply")]
    long_items = [x for x in recommended if not x.get("is_short_reply")]
    max_short = int(len(recommended) * SHORT_REPLY_MAX_RATIO)
    if len(short_items) > max_short:
        # 按质量分排序，保留质量分最高的 max_short 条短回复
        short_items.sort(key=lambda x: x["quality_score"], reverse=True)
        kept_short = short_items[:max_short]
        dropped_short = short_items[max_short:]
        for item in dropped_short:
            item_copy = dict(item)
            item_copy["reasons"] = item_copy["reasons"] + ["short_reply_ratio_exceeded"]
            dedup_excluded.append(item_copy)
        recommended = kept_short + long_items

    all_excluded = hard_excluded + dedup_excluded

    # full 集：仅去除完全相同问答，保留所有有效候选（含全部短回复）
    full_by_pair = {}
    full_excluded = []
    for item in valid:
        key = (norm_key(item["prompt"]), norm_key(item["reply"]))
        if key in full_by_pair:
            full_excluded.append(item)
        else:
            full_by_pair[key] = item

    return (
        [as_sft(x, target_key) for x in recommended],
        [as_sft(x, target_key) for x in full_by_pair.values()],
        [
            as_sft(x, target_key, include_exclusion_reasons=True)
            if "conversations" not in x
            else x
            for x in all_excluded
        ],
    )


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summary(raw, recommended, full, excluded):
    canonical = {x["id"] for x in raw if x["source_role"] == "canonical" and x["speaker_kind"] == "direct"}
    full_ids = {i for x in full for i in x["metadata"]["target_event_ids"]}
    recommended_ids = {i for x in recommended for i in x["metadata"]["target_event_ids"]}
    reasons = Counter(
        reason
        for item in excluded
        for reason in item.get("metadata", {}).get("exclusion_reasons", [])
    )
    short_in_recommended = sum(1 for x in recommended if x["metadata"].get("is_short_reply"))
    reply_lengths = [meaningful_len(x["conversations"][1]["value"]) for x in recommended]
    return {
        "all_supplied_attributable_occurrences": len(raw),
        "all_supplied_unique_utterances": len({x["text"] for x in raw}),
        "canonical_direct_occurrences": len(canonical),
        "raw_source_coverage_percent": 100.0,
        "full_sft_examples": len(full),
        "full_sft_covered_canonical_occurrences": len(full_ids & canonical),
        "recommended_sft_examples": len(recommended),
        "recommended_sft_unique_prompts": len({norm_key(x["conversations"][0]["value"]) for x in recommended}),
        "recommended_sft_covered_canonical_occurrences": len(recommended_ids & canonical),
        "short_reply_count_in_recommended": short_in_recommended,
        "short_reply_ratio_in_recommended": round(short_in_recommended / max(len(recommended), 1), 4),
        "short_reply_max_ratio": SHORT_REPLY_MAX_RATIO,
        "reply_length_meaningful": {
            "min": min(reply_lengths) if reply_lengths else 0,
            "max": max(reply_lengths) if reply_lengths else 0,
            "mean": round(sum(reply_lengths) / len(reply_lengths), 1) if reply_lengths else 0,
        },
        "excluded_turn_reasons": dict(sorted(reasons.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="提取月社妃角色对话（v2，Qwen3-8B）")
    parser.add_argument("--magic-root", type=Path, required=True,
                        help="纸上魔法使文本根目录（含 *.txt）")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出目录")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_key = "tsukiyashiro_kisaki"
    groups = [
        {"source_id": sid, "source_role": role, "events": read_script_events(path, sid)}
        for path, sid, role in source_scripts(args.magic_root)
    ]

    raw = make_raw(groups, target_key)
    recommended, full, excluded = make_sft(groups, target_key)

    write_jsonl(args.output_dir / f"{target_key}_raw.jsonl", raw)
    write_json(args.output_dir / f"{target_key}_sft.json", recommended)
    write_json(args.output_dir / f"{target_key}_sft_full.json", full)
    write_jsonl(args.output_dir / f"{target_key}_excluded.jsonl", excluded)

    char_summary = {
        "character": TARGETS[target_key]["display_name"],
        **summary(raw, recommended, full, excluded),
        "source_files": dict(Counter(x["source"].split(":line:", 1)[0] for x in raw)),
    }

    manifest = {
        "schema_version": 3,
        "extraction_version": "v3_continuous_dialogue_blocks",
        "extraction_policy": {
            "tsukiyashiro_kisaki": "只把显式[妃]/[月社妃]作为目标说话人；克丽索贝莉露是独立角色，即使冒用月社妃姓名也不并入。",
            "source_policy": "gametext/纸上魔法使/*.txt 全部视为 canonical 来源；只在同一连续对话块内配对，最多允许 4 行中间叙述，空行和明确场景重置会断开上下文。",
            "quality_policy": "排除无上下文、低信息、超长和重复样本；过短回复（<5字）按占比控制（≤15%）。",
            "v3_changes": "保留物理行号与对话块；拒绝跨长旁白或场景切换配对；未知说话者不注入 interlocutor；保留嵌套中文引号。",
        },
        "characters": {target_key: char_summary},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "coverage_report.json", {target_key: char_summary})

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
