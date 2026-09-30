"""Canonical game replies versus ordinary model candidates; never auto-label preference."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from training.persona_alignment import read_json, read_jsonl, write_json, write_jsonl
from training.persona_sampling import (
    chat_completion,
    digest,
    require_text,
    sample_candidates,
    sampling_contract,
    scene_messages,
    validate_scenes,
)
from training.preference_validation import fingerprint


def source_binding(row, raw_by_id):
    metadata = row["metadata"]
    ids = metadata.get("target_event_ids", [])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate original event IDs")
    events = [raw_by_id[event_id] for event_id in ids]
    if any(e.get("speaker_kind") != "direct" or e.get("source_role") != "canonical" for e in events):
        raise ValueError("target is not a canonical direct character utterance")
    if any(e["character"] != metadata["character"] or e["source_file"] != metadata["source_file"] for e in events):
        raise ValueError("source character/file mismatch")
    if len({e["scene_block_id"] for e in events}) != 1:
        raise ValueError("target crosses scene boundaries")
    if events[0]["scene_block_id"] != metadata.get("scene_block_id"):
        raise ValueError("source scene mismatch")
    if [e["source_line_start"] for e in events] != sorted(e["source_line_start"] for e in events):
        raise ValueError("source events are out of order")
    text = "\n".join(e["text"] for e in events)
    if row["messages"][-1]["content"] != text:
        raise ValueError("reply differs from canonical source; manually review normalization first")
    return {
        "event_ids": ids,
        "events_sha256": digest(events),
        "source_file": metadata["source_file"],
        "scene_block_id": metadata["scene_block_id"],
        "text": text,
        "text_sha256": digest(text),
    }


def prepare_sources(train, heldout, raw, profile):
    require_text(profile, "profile")
    raw_by_id = {r["id"]: r for r in raw}
    if len(raw_by_id) != len(raw):
        raise ValueError("duplicate raw source IDs")
    blocked_events, blocked_scenes, blocked_text = set(), set(), set()
    for row in heldout:
        metadata = row.get("metadata", {})
        blocked_events.update(metadata.get("target_event_ids", []))
        if metadata.get("scene_block_id"):
            blocked_scenes.add(metadata["scene_block_id"])
        for m in row.get("messages", []):
            if m.get("content"):
                blocked_text.add(fingerprint(m["content"]))
    drafts, excluded, seen = [], [], set()
    for row in train:
        metadata = row.get("metadata", {})
        reason = None
        if metadata.get("data_source") != "game_extraction":
            reason = "not_game_extraction"
        elif row.get("id") in seen:
            raise ValueError("duplicate training row ID")
        else:
            seen.add(row["id"])
            messages = row.get("messages", [])
            if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
                reason = "requires_manual_context_conversion"
            elif (
                set(metadata.get("target_event_ids", [])) & blocked_events
                or metadata.get("scene_block_id") in blocked_scenes
                or any(fingerprint(m["content"]) in blocked_text for m in messages)
            ):
                reason = "heldout_overlap"
            else:
                try:
                    binding = source_binding(row, raw_by_id)
                except (KeyError, ValueError, TypeError):
                    reason = "source_verification_failed"
        if reason:
            excluded.append({"id": row.get("id"), "reason": reason})
            continue
        scene = {
            "id": row["id"],
            "persona": metadata["character"],
            "split": "train",
            "review_status": "pending",
            "source_group": metadata["scene_block_id"],
            "source_ids": [metadata["scene_block_id"]],
            "persona_profile": profile,
            "relationship": "待审核：确认原作对话者及此时的关系，不能按现代聊天关系替换。",
            "situation": "待审核：补齐回复前已知的情境，不能加入目标台词或之后剧情。",
            "evidence": [],
            "history": [],
            "user_message": row["messages"][0]["content"],
        }
        drafts.append(
            {
                "scene": scene,
                "source_reply": binding,
                "source_row": row,
                "source_row_sha256": digest(row),
                "context_review": {
                    "reviewer": "",
                    "approved": False,
                    "sufficient_pre_reply_context": False,
                    "no_target_or_future_leakage": False,
                },
                "interlocutor": metadata.get("context_speaker_label"),
            }
        )
    return drafts, {
        "included_pending_review": len(drafts),
        "excluded": excluded,
        "exclusion_counts": dict(Counter(r["reason"] for r in excluded)),
        "heldout_sha256": digest(heldout),
        "raw_sha256": digest(raw),
        "note": "Drafts only; original-source status does not approve context or preference.",
    }


def sample_source_pairs(rows, raw, model, generation, *, call=chat_completion, on_record=None):
    raw_by_id = {r["id"]: r for r in raw}
    scenes, replies = [], {}
    for row in rows:
        scene, review = row["scene"], row["context_review"]
        if scene["split"] != "train":
            raise ValueError("source preferences only accept train scenes")
        if any(
            review.get(k) is not True
            for k in ("approved", "sufficient_pre_reply_context", "no_target_or_future_leakage")
        ):
            raise ValueError("source context requires explicit review")
        require_text(review.get("reviewer"), "context reviewer")
        if row["source_row_sha256"] != digest(row["source_row"]):
            raise ValueError("source row binding mismatch")
        bound = source_binding(row["source_row"], raw_by_id)
        if bound != row["source_reply"] or scene["id"] != row["source_row"]["id"]:
            raise ValueError("canonical reply binding mismatch")
        if scene["source_group"] != bound["scene_block_id"] or bound["scene_block_id"] not in scene["source_ids"]:
            raise ValueError("source provenance mismatch")
        messages = scene_messages(scene)
        normalized_answer = "".join(bound["text"].split()).casefold()
        if any(normalized_answer in "".join(m["content"].split()).casefold() for m in messages):
            raise ValueError("canonical answer appears in input; remove target leakage")
        scenes.append(scene)
        replies[digest(messages)] = bound
    validate_scenes(scenes)
    if len(replies) != len(scenes):
        raise ValueError("duplicate source prompts")
    original = {
        "name": "canonical_original",
        "model": "canonical-source-excerpt",
        "revision": digest([r["source_reply"] for r in rows]),
        "base_url": "http://source-excerpt.invalid",
    }
    if model["name"] == original["name"]:
        raise ValueError("reserved model name")
    models = [original, model]

    def generate(identity, messages, params):
        if identity["name"] == "canonical_original":
            return replies[digest(messages)]["text"]
        return call(identity, messages, params)

    def record(row):
        if row["model"]["name"] == "canonical_original":
            row["status"] = "source_excerpt"
            row["source_reply"] = replies[digest(row["messages"])]
        if on_record:
            on_record(row)

    records = sample_candidates(scenes, models, generation, samples_per_model=1, call=generate, on_record=record)
    return sampling_contract(scenes, models, generation, 1), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--heldout", type=Path, nargs="+", required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--reviewed", type=Path, required=True)
    sample.add_argument("--config", type=Path, required=True, help="model and generation objects")
    for command in (prepare, sample):
        command.add_argument("--raw", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = read_jsonl(args.raw)
    if args.command == "prepare":
        rows, report = prepare_sources(
            read_jsonl(args.train),
            [r for p in args.heldout for r in read_jsonl(p)],
            raw,
            args.profile.read_text(encoding="utf-8-sig"),
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / "source_contexts.pending.jsonl", rows)
        write_json(args.output_dir / "preparation_report.json", report)
    else:
        config = read_json(args.config)
        rows = read_jsonl(args.reviewed)
        # Validate everything before remote generation starts; journal each completed row.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        import json

        with (args.output_dir / "candidates.jsonl").open("x", encoding="utf-8") as handle:

            def append(row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

            contract, records = sample_source_pairs(rows, raw, config["model"], config["generation"], on_record=append)
        write_json(args.output_dir / "sampling_manifest.json", contract)
        errors = sum(r["status"] == "error" for r in records)
        write_json(
            args.output_dir / "summary.json", {"total": len(records), "errors": errors, "review_status": "pending"}
        )
        if errors:
            raise SystemExit("model sampling failed; incomplete runs cannot enter preference review")


if __name__ == "__main__":
    main()
