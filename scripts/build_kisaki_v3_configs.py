"""Generate V3 experiment configs (E1-E5) from V2 configs.

Reads each V2 ``kisaki_eN_canonical.json``, rewrites data/output/experiment
fields to point at the V3 dataset and V3 lora directories, and writes
``kisaki_eN_canonical_v3.json`` alongside the originals. V2 configs are
left untouched.

Run after ``build_kisaki_canonical_v3.py`` has produced the V3 dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "character_dialogues"
    / "experiments"
    / "configs"
)

# (v2_stem, v3_stem, v3_experiment_id, v3_comparison_role)
VARIANT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("kisaki_e1_canonical", "kisaki_e1_canonical_v3", "KISAKI-E1-V3", "v3_data_ablation_baseline"),
    ("kisaki_e2_canonical", "kisaki_e2_canonical_v3", "KISAKI-E2-V3", "v3_data_ablation_neftune_only"),
    ("kisaki_e3_canonical", "kisaki_e3_canonical_v3", "R1-E3-V3", "v3_data_ablation_dora_only"),
    ("kisaki_e4_canonical", "kisaki_e4_canonical_v3", "R1-E4-V3", "v3_data_ablation_rslora_only"),
    ("kisaki_e5_canonical", "kisaki_e5_canonical_v3", "R1-E5-V3", "v3_data_ablation_packing_only"),
)

V3_TRAIN_DATA = (
    "backend/data/character_dialogues/experiments/v3/tsukiyashiro_kisaki_train.json"
)
V3_EVAL_DATA = (
    "backend/data/character_dialogues/experiments/v3/tsukiyashiro_kisaki_eval.json"
)
V3_MANIFEST = (
    "backend/data/character_dialogues/experiments/v3/canonical_dataset_manifest.json"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_output_dir(original: str, v3_stem: str) -> str:
    """Point the lora output at the V3 directory tree.

    V2 path:   /home/szw/lhm2/runtime/loras/kisaki/canonical/eN/seed42
    V3 path:   /home/szw/lhm2/runtime/loras/kisaki/canonical_v3/eN/seed42

    The ``runtime`` prefix is resolved via QQCHAT_LAB_ROOT at training time
    (see run_kisaki_experiment.py), so we keep the same suffix pattern and
    only swap ``canonical`` → ``canonical_v3``.
    """
    if original is None:
        return original
    # Replace the canonical/ segment; works for both Windows and POSIX paths.
    replaced = original.replace("/canonical/", "/canonical_v3/")
    if replaced == original:
        # Fallback: append _v3 to the last meaningful segment.
        replaced = original.rstrip("/").rstrip("\\")
        if replaced.endswith("seed42"):
            replaced = replaced[:-6] + "_v3/seed42"
        else:
            replaced = replaced + "_v3"
    return replaced


def _build_v3_config(v2_config: dict[str, Any], v3_stem: str,
                     v3_experiment_id: str, v3_comparison_role: str) -> dict[str, Any]:
    v3 = deepcopy(v2_config)
    v3["train_data_path"] = V3_TRAIN_DATA
    v3["eval_data_path"] = V3_EVAL_DATA
    v3["output_dir"] = _rewrite_output_dir(v2_config.get("output_dir", ""), v3_stem)
    v3["_experiment_id"] = v3_experiment_id
    v3["_comparison_role"] = v3_comparison_role
    # Provenance: V2 base_model_path is kept verbatim so runtime override
    # (QQCHAT_LAB_ROOT / base_model_path override in run_kisaki_experiment.py)
    # still works on both servers.
    v3["_dataset_version"] = "KISAKI-CANONICAL-V3"
    v3["_dataset_manifest_path"] = V3_MANIFEST
    v3["_v2_config_source"] = f"{v3_stem.replace('_v3', '')}.json"
    return v3


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3 E1-E5 configs from V2")
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    args = parser.parse_args()

    written: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for v2_stem, v3_stem, exp_id, role in VARIANT_SPECS:
        v2_path = args.configs_dir / f"{v2_stem}.json"
        if not v2_path.exists():
            skipped.append({"v2_config": str(v2_path), "reason": "missing"})
            continue
        v2_config = _read_json(v2_path)
        v3_config = _build_v3_config(v2_config, v3_stem, exp_id, role)
        v3_path = args.configs_dir / f"{v3_stem}.json"
        _write_json(v3_path, v3_config)
        written.append({
            "v2_config": v2_path.name,
            "v3_config": v3_path.name,
            "experiment_id": exp_id,
            "comparison_role": role,
            "train_data_path": v3_config["train_data_path"],
            "output_dir": v3_config["output_dir"],
        })

    print(json.dumps({
        "written": written,
        "skipped": skipped,
        "total_written": len(written),
        "total_skipped": len(skipped),
    }, ensure_ascii=False, indent=2))
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
