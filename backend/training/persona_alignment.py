"""CLI for sampling, blind review, preference export and reward-judge calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.persona_judge import calibration_from_review, judge_candidate
from training.persona_review import export_preferences, lock_decisions, make_packet, summarize_evaluation
from training.persona_sampling import sample_candidates, sampling_contract, validate_run


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_run(path):
    contract = read_json(path / "sampling_manifest.json")
    records = read_jsonl(path / "candidates.jsonl")
    validate_run(contract, records)
    return records


def review_inputs(args):
    return read_json(args.packet), read_json(args.locked), read_json(args.key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser(
        "sample", help="Sample real replies from reviewed scenes; never assign preferred labels"
    )
    sample.add_argument("--scenes", type=Path, required=True)
    sample.add_argument("--config", type=Path, required=True, help="models, generation and samples_per_model")
    sample.add_argument("--output-dir", type=Path, required=True)
    blind = commands.add_parser("blind", help="Create anonymous review packet, separate key and blank decisions")
    blind.add_argument("--run", type=Path, required=True)
    blind.add_argument("--purpose", choices=["preferences", "evaluation"], required=True)
    blind.add_argument("--baseline")
    blind.add_argument("--candidate")
    blind.add_argument("--seed", type=int, default=42)
    blind.add_argument("--output-dir", type=Path, required=True)
    lock = commands.add_parser("lock", help="Validate and lock completed human decisions without opening the key")
    lock.add_argument("--packet", type=Path, required=True)
    lock.add_argument("--decisions", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    for command in ("export", "report", "calibrate"):
        sub = commands.add_parser(command)
        for field in ("packet", "locked", "key"):
            sub.add_argument(f"--{field}", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "calibrate":
            sub.add_argument("--judgments", type=Path, required=True)
    judge = commands.add_parser("judge", help="Score candidates with an external model; outputs stay AI diagnostics")
    judge.add_argument("--run", type=Path, required=True)
    judge.add_argument("--model-config", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "sample":
        config, scenes = read_json(args.config), read_jsonl(args.scenes)
        if set(config) != {"models", "generation", "samples_per_model"}:
            raise ValueError("sampling config must contain models, generation, samples_per_model")
        contract = sampling_contract(scenes, config["models"], config["generation"], config["samples_per_model"])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "sampling_manifest.json", contract)
        with (args.output_dir / "candidates.jsonl").open("x", encoding="utf-8") as handle:

            def append(row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

            records = sample_candidates(
                scenes,
                config["models"],
                config["generation"],
                samples_per_model=config["samples_per_model"],
                on_record=append,
            )
        errors = sum(row["status"] != "generated" for row in records)
        write_json(
            args.output_dir / "summary.json", {"total": len(records), "errors": errors, "review_status": "pending"}
        )
        if errors:
            raise SystemExit("sampling contains failures; inspect error types, do not use as a complete evaluation")
    elif args.command == "blind":
        packet, key, template = make_packet(
            load_run(args.run), purpose=args.purpose, seed=args.seed, baseline=args.baseline, candidate=args.candidate
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "blind_review.json", packet)
        write_json(args.output_dir / "blind_key.json", key)
        write_json(args.output_dir / "decisions.template.json", template)
        lines = [
            "# 人物回复盲评",
            "",
            "只向审核者提供此文、blind_review.json 与 decisions.template.json。身份映射在 blind_key.json，锁定决定前不打开。",
            "",
        ]
        for item in packet["items"]:
            lines.extend(
                [
                    f"## {item['id']}",
                    "",
                    "```json",
                    json.dumps(item["messages"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "### A",
                    "",
                    item["A"],
                    "",
                    "### B",
                    "",
                    item["B"],
                    "",
                ]
            )
        (args.output_dir / "BLIND_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    elif args.command == "lock":
        write_json(args.output, lock_decisions(read_json(args.packet), read_json(args.decisions)))
    elif args.command == "judge":
        records, model = load_run(args.run), read_json(args.model_config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Failures stop scoring; no invented scores or placeholder approvals are written.
        with args.output.open("x", encoding="utf-8") as handle:
            for row in records:
                result = judge_candidate(row, model)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    else:
        packet, locked, key = review_inputs(args)
        if args.command == "export":
            result = export_preferences(packet, locked, key)
        elif args.command == "report":
            result = summarize_evaluation(packet, locked, key)
        else:
            result = calibration_from_review(packet, locked, key, read_jsonl(args.judgments))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if args.command == "export":
            write_jsonl(args.output_dir / "reviewed_preferences.jsonl", result.pop("pairs"))
        elif args.command == "calibrate":
            write_jsonl(args.output_dir / "calibration_pairs.jsonl", result.pop("rows"))
        write_json(args.output_dir / "report.json", result)


if __name__ == "__main__":
    main()
