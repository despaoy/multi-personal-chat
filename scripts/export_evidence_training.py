"""Export reviewed evidence-conditioned SFT/DPO data into separate split files.

This command never trains a model and refuses to overwrite an existing output
directory. It validates the complete corpus before creating output files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from training.evidence_dataset import SPLITS, build_evidence_training_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reviewed contextual-evidence-v1 JSONL corpus")
    parser.add_argument("--system-prompt", type=Path, required=True, help="Trusted, reviewed persona prompt file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        parser.error("input corpus is empty")
    bundle = build_evidence_training_bundle(
        rows,
        system_prompt=args.system_prompt.read_text(encoding="utf-8-sig"),
        max_evidence_chars=args.max_evidence_chars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in sorted(SPLITS):
        for name, records in (("sft", bundle.sft), ("preference", bundle.preference)):
            selected = [record for record in records if record["metadata"]["split"] == split]
            destination = args.output_dir / f"{split}.{name}.jsonl"
            with destination.open("x", encoding="utf-8") as output:
                for record in selected:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump(bundle.manifest, output, ensure_ascii=False, indent=2)
    print(json.dumps(bundle.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
