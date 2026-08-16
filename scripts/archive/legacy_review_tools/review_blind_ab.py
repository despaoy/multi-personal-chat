"""Interactively review a blinded A/B benchmark without reading its key."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VALID_DECISIONS = {"A", "B", "tie", "invalid"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def counts(samples: list[dict]) -> Counter:
    return Counter(item.get("winner") or "pending" for item in samples)


def show(item: dict, index: int, total: int) -> None:
    print("\n" + "=" * 88)
    print(f"[{index + 1}/{total}] {item.get('id')}  category={item.get('category')}")
    print("-" * 88)
    print("PROMPT:\n" + item.get("prompt", ""))
    print("\nRESPONSE A:\n" + item.get("response_A", ""))
    print("\nRESPONSE B:\n" + item.get("response_B", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="可恢复的匿名 A/B 人工评测工具")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    output = args.output or args.input.with_name(args.input.stem + "_reviewed.json")
    payload = load(output if output.exists() else args.input)
    samples = payload.get("samples", [])
    if not samples:
        raise SystemExit("输入文件没有 samples")
    if not output.exists():
        payload["reviewer"] = args.reviewer
        atomic_write(output, payload)

    if args.summary:
        print(dict(counts(samples)))
        print(f"审核文件: {output}")
        return 0

    pending = [i for i, item in enumerate(samples) if item.get("winner") not in VALID_DECISIONS]
    if not pending:
        print(f"审核已完成: {dict(counts(samples))}")
        print(f"审核文件: {output}")
        return 0

    print("请只看 prompt、A 和 B；审核完成前不要打开 blind_key.json。")
    print("命令: a=A更好 b=B更好 t=平局 i=样本无效 s=跳过 q=保存退出")
    for index in pending:
        item = samples[index]
        show(item, index, len(samples))
        while True:
            command = input("\n决定 [a/b/t/i/s/q]: ").strip().lower()
            if command in {"a", "b", "t", "i"}:
                reason = input("理由（必填，写最关键的一点）: ").strip()
                if not reason:
                    print("必须填写具体理由")
                    continue
                item["winner"] = {"a": "A", "b": "B", "t": "tie", "i": "invalid"}[
                    command
                ]
                item["reason"] = reason
                item["reviewer"] = args.reviewer
                item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                atomic_write(output, payload)
                break
            if command == "s":
                break
            if command == "q":
                atomic_write(output, payload)
                print(f"已保存: {dict(counts(samples))}")
                print(f"下次使用同一命令会从 pending 样本继续: {output}")
                return 0
            print("未知命令，请输入 a/b/t/i/s/q")

    print(f"审核完成: {dict(counts(samples))}")
    print(f"审核文件: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
