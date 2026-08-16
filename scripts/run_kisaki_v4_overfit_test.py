#!/usr/bin/env python3
"""Run the prepared overfit training, generation, and review rendering."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OVERFIT = BACKEND / "data/character_dialogues/experiments/v4/overfit_20"
DEFAULT_CONFIG = OVERFIT / "config.json"
DEFAULT_CASES = OVERFIT / "cases.json"
DEFAULT_RESULTS = OVERFIT / "results.json"
DEFAULT_REVIEW = ROOT / "docs/research/review_packets/kisaki_v4/09_OVERFIT_TEST/review.md"
PROMPT = BACKEND / "data/character_dialogues/kisaki_system_prompt_v3.txt"


def run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.base_model:
        config["base_model_path"] = str(args.base_model.resolve())
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.resolve())
    resolved = OVERFIT / "resolved_config.json"
    resolved.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    run([sys.executable, "-m", "training.trainer", "--config", str(resolved)], cwd=BACKEND)
    adapter = Path(config["output_dir"]) / "final"
    run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_kisaki_v4_overfit_results.py"),
            "--base-model",
            config["base_model_path"],
            "--adapter",
            str(adapter),
            "--cases",
            str(args.cases),
            "--system-prompt",
            str(PROMPT),
            "--output",
            str(args.results),
        ],
        cwd=ROOT,
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/render_kisaki_v4_overfit_review.py"),
            "--results",
            str(args.results),
            "--output",
            str(args.review),
        ],
        cwd=ROOT,
    )
    print(json.dumps({"status": "pending_human_review", "review": str(args.review)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
