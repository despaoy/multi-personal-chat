"""Prepare/run/score development probes; real calls require explicit --execute.

Run from repository root. Output is exclusive-create to preserve prior evidence.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from evaluation.narrative import packets, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, choices=[10, 20, 40], default=10)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["A", "B", "C", "D", "D-labels"],
        default=["A", "B", "C", "D"],
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new evidence filename")
    if args.annotations:
        rows = json.loads(args.annotations.read_text(encoding="utf-8"))["probes"]
        payload = {"kind": "human_annotation_summary", "metrics": score(rows)}
    else:
        rows = list(packets(args.turns, args.arms))
        digest = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        payload = {
            "kind": "real_model_run" if args.execute else "prepared_prompts_only",
            "synthetic": True,
            "split": "development",
            "data_sha256": digest,
            "git_revision": revision,
            "dirty_worktree": bool(dirty),
            "seed": args.seed,
            "model": args.model,
            "parameters": {"temperature": 0.2, "max_tokens": 256},
            "probes": rows,
        }
        if args.execute:
            if not args.model:
                parser.error("--execute requires --model")
            import httpx

            headers = {}
            key = os.getenv("NARRATIVE_EVAL_API_KEY")
            if key:
                headers["Authorization"] = "Bearer " + key
            with httpx.Client(timeout=180, headers=headers) as client:
                for row in rows:
                    start = time.perf_counter()
                    try:
                        response = client.post(
                            args.base_url.rstrip("/") + "/chat/completions",
                            json={
                                "model": args.model,
                                "messages": row["messages"],
                                "temperature": 0.2,
                                "max_tokens": 256,
                                "seed": args.seed,
                            },
                        )
                        response.raise_for_status()
                        body = response.json()
                        row.update(
                            reply=body["choices"][0]["message"]["content"],
                            usage=body.get("usage"),
                            served_model=body.get("model"),
                            system_fingerprint=body.get("system_fingerprint"),
                        )
                    except (
                        httpx.HTTPError,
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                    ) as exc:
                        row["error_type"] = type(exc).__name__
                    row["latency_seconds"] = time.perf_counter() - start
                    row["labels"] = {
                        k: None
                        for k in [
                            "correct",
                            "contaminated",
                            "cross_branch_leak",
                            "provenance_correct",
                            "abstained",
                        ]
                    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved {args.output}; no quality claim is inferred from unlabelled outputs.")


if __name__ == "__main__":
    main()
