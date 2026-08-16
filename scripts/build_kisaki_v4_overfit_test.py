#!/usr/bin/env python3
"""Build the deterministic 20-record Kisaki V4 overfit smoke test."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = os.getenv("QQCHAT_LAB_ROOT", str(ROOT / "runtime")).rstrip("/\\")
V4 = ROOT / "backend/data/character_dialogues/experiments/v4"
TRAIN = V4 / "train.jsonl"
BASE_CONFIG = V4 / "r1v4_base_config.json"
PROMPT = ROOT / "backend/data/character_dialogues/kisaki_system_prompt_v3.txt"
OUTPUT = V4 / "overfit_20"
REVIEW = ROOT / "docs/research/review_packets/kisaki_v4/09_OVERFIT_TEST"
CONSTRUCTED_SOURCES = (
    "llm_v4_manual",
    "llm_v4_blindfix",
    "llm_v4_yoruko",
    "llm_v4_lifestyle",
    "llm_v4_riou",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_order(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"kisaki-v4-overfit20:{row['id']}".encode()).hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def select_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    game = sorted(
        (row for row in rows if row.get("metadata", {}).get("data_source") == "game_extraction"),
        key=stable_order,
    )[:10]
    constructed: list[dict[str, Any]] = []
    for source in CONSTRUCTED_SOURCES:
        candidates = sorted(
            (row for row in rows if row.get("metadata", {}).get("data_source") == source),
            key=stable_order,
        )
        if len(candidates) < 2:
            raise ValueError(f"overfit source has fewer than two records: {source}")
        constructed.extend(candidates[:2])
    selected = sorted(game + constructed, key=stable_order)
    if len(selected) != 20 or len({row["id"] for row in selected}) != 20:
        raise ValueError("overfit selection must contain 20 unique records")
    return selected


def evaluation_case(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"record must end with assistant: {row['id']}")
    return {
        "id": row["id"],
        "interlocutor": row.get("metadata", {}).get("context_speaker_label") or "用户",
        "messages": messages[:-1],
        "reference_answer": messages[-1]["content"],
        "data_source": row.get("metadata", {}).get("data_source"),
    }


def build(output: Path = OUTPUT, review: Path = REVIEW) -> dict[str, Any]:
    selected = select_records(load_jsonl(TRAIN))
    output.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    train_path = output / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
        newline="\n",
    )
    cases = [evaluation_case(row) for row in selected]
    (output / "cases.json").write_text(
        json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    config = load_json(BASE_CONFIG)
    config.update(
        {
            "train_data_path": portable_path(train_path),
            "eval_data_path": portable_path(train_path),
            "output_dir": f"{LAB_ROOT}/runtime/loras/kisaki/r1v4/overfit20",
            "system_prompt": PROMPT.read_text(encoding="utf-8").strip(),
            "system_prompt_policy": "replace",
            "num_train_epochs": 20,
            "gradient_accumulation_steps": 1,
            "eval_steps": 20,
            "save_steps": 20,
            "logging_steps": 5,
            "early_stopping_patience": 0,
            "load_best_model_at_end": True,
            "report_to": "none",
            "packing": False,
            "neftune_noise_alpha": 0.0,
        }
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    sources = Counter(case["data_source"] for case in cases)
    manifest = {
        "schema_version": 1,
        "status": "ready_to_run",
        "purpose": "Verify the 20-record train-to-adapter-to-generation path before R1V4.",
        "sample_count": 20,
        "source_distribution": dict(sorted(sources.items())),
        "train_path": portable_path(train_path),
        "cases_path": portable_path(output / "cases.json"),
        "config_path": portable_path(output / "config.json"),
        "results_path": portable_path(output / "results.json"),
        "human_review_path": portable_path(review / "review.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (review / "README.md").write_text(
        "# 20 条过拟合链路测试\n\n"
        "固定测试数据和配置已经生成。训练及生成完成后，运行结果渲染器生成 `review.md`；"
        "人工只审核该文件中的 20 条模型回答。\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
