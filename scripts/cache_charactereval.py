"""Cache a pinned public benchmark outside the project; never execute its code.

This is a read-only research acquisition step, not training-data approval or an
automatic import into the project's character registry. Preserve the MIT notice;
the repository license does not settle all underlying novel/script rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "c3d44a6fc1790cc8c4b2fd7c01f0c72930655e0c"
BASE = f"https://raw.githubusercontent.com/morecry/CharacterEval/{REVISION}"
FILES = ("README.md", "LICENSE", "data/test_data.jsonl", "data/character_profiles.json", "data/id2metric.jsonl")
MAX_FILE_BYTES = 128 * 1024 * 1024


def acquire(output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"repository": "https://github.com/morecry/CharacterEval", "revision": REVISION, "files": {}}
    for name in FILES:
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=30) as response, path.open("xb") as destination:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ValueError("public benchmark file exceeds download budget; incomplete cache retained")
                digest.update(chunk)
                destination.write(chunk)
        manifest["files"][name] = {"bytes": size, "sha256": digest.hexdigest(), "url": f"{BASE}/{name}"}
        print(f"{name}: {size} bytes", flush=True)
    # The upstream .jsonl files are actually whole JSON values, not line records.
    cases = json.loads((output_dir / "data/test_data.jsonl").read_text(encoding="utf-8-sig"))
    profiles = json.loads((output_dir / "data/character_profiles.json").read_text(encoding="utf-8-sig"))
    metrics = json.loads((output_dir / "data/id2metric.jsonl").read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not isinstance(profiles, dict) or not isinstance(metrics, dict):
        raise ValueError("unexpected public benchmark structure")
    manifest["audit"] = {
        "cases": len(cases),
        "profile_count": len(profiles),
        "metric_annotation_count": len(metrics),
        "case_fields": sorted({key for row in cases for key in row}),
        "roles_in_cases": sorted({row["role"] for row in cases}),
        "subjective_metric_pairs": sorted({tuple(pair) for pairs in metrics.values() for pair in pairs}),
        "unique_case_ids": len({str(row["id"]) for row in cases}),
        "all_roles_have_profiles": all(row["role"] in profiles for row in cases),
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New external cache directory; never overwritten"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    result = acquire(args.output_dir)
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
