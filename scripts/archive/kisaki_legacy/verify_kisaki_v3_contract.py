"""Verify KISAKI-CANONICAL-V3 data contract against V2.

Runs the five A.4 checks from the V2.1 spec:
  A.4.1  715 game_extraction training samples in V2 order
  A.4.2  84 game_extraction validation samples, no v3 contamination
  A.4.3  v3_provenance id-sequence hashes stable
  A.4.4  V2 files unchanged (re-hashes V2 archive index)
  A.4.5  manifest draft fields correct (current=715, target=826, accepted=0)

Also verifies V3 E1-E5 configs: hyperparameters preserved, paths point at v3.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import canonical_json_hash, sha256_text_file  # noqa: E402

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
V2_TRAIN = EXPERIMENT_DIR / "tsukiyashiro_kisaki_train.json"
V2_EVAL = EXPERIMENT_DIR / "tsukiyashiro_kisaki_eval.json"
V3_TRAIN = EXPERIMENT_DIR / "v3" / "tsukiyashiro_kisaki_train.json"
V3_EVAL = EXPERIMENT_DIR / "v3" / "tsukiyashiro_kisaki_eval.json"
V3_MANIFEST = EXPERIMENT_DIR / "v3" / "canonical_dataset_manifest.json"
V2_ARCHIVE_INDEX = EXPERIMENT_DIR / "v2_archive_index.json"
CONFIGS_DIR = EXPERIMENT_DIR / "configs"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    print("=== A.4 V3 contract verification ===")
    all_ok = True

    # A.4.1: 715 game_extraction training samples in V2 order
    v2_train = _read_json(V2_TRAIN)
    v3_train = _read_json(V3_TRAIN)
    v2_game = [x for x in v2_train if (x.get("metadata") or {}).get("data_source") == "game_extraction"]
    v2_ids = [x.get("id") for x in v2_game]
    v3_ids = [x.get("id") for x in v3_train]
    all_ok &= _check(
        "A.4.1 train order matches V2 game_extraction",
        v2_ids == v3_ids,
        f"v2_game={len(v2_ids)} v3_train={len(v3_ids)}",
    )

    # A.4.2: 84 validation samples, no v3 contamination, order matches V2
    v2_eval = _read_json(V2_EVAL)
    v3_eval = _read_json(V3_EVAL)
    v2_eval_game = [x for x in v2_eval if (x.get("metadata") or {}).get("data_source") == "game_extraction"]
    v3_eval_dist: dict[str, int] = {}
    for x in v3_eval:
        s = (x.get("metadata") or {}).get("data_source", "?")
        v3_eval_dist[s] = v3_eval_dist.get(s, 0) + 1
    no_v3 = v3_eval_dist.get("llm_v3_deepseek", 0) == 0
    order_match = [x.get("id") for x in v2_eval_game] == [x.get("id") for x in v3_eval]
    all_ok &= _check(
        "A.4.2 eval no v3 contamination",
        no_v3 and order_match and len(v3_eval) == 84,
        f"dist={v3_eval_dist} count={len(v3_eval)} order_match={order_match}",
    )

    # A.4.3: v3_provenance id-sequence hashes match manifest
    manifest = _read_json(V3_MANIFEST)
    prov = manifest.get("v3_provenance", {})
    train_hash = canonical_json_hash(v3_ids)
    eval_hash = canonical_json_hash([x.get("id") for x in v3_eval])
    train_match = prov.get("game_extraction_train", {}).get("id_sequence_sha256") == train_hash
    eval_match = prov.get("game_extraction_validation", {}).get("id_sequence_sha256") == eval_hash
    all_ok &= _check(
        "A.4.3 v3_provenance id-sequence hashes stable",
        train_match and eval_match,
        f"train={train_match} eval={eval_match}",
    )

    # A.4.4: V2 files unchanged (delegate to v2_archive_index --verify)
    r = subprocess.run(
        [sys.executable, "scripts/v2_archive_index.py", "--verify"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        v2_verify = json.loads(r.stdout)
    except json.JSONDecodeError:
        v2_verify = {"verified": False, "raw_stdout": r.stdout}
    all_ok &= _check(
        "A.4.4 V2 files unchanged since archive",
        v2_verify.get("verified") is True,
        f"exit={r.returncode} file_count={v2_verify.get('file_count')}",
    )

    # A.4.5: manifest draft fields correct
    status_ok = manifest.get("status") == "draft"
    current_ok = manifest.get("current_train_count") == 715
    target_ok = manifest.get("target_train_count") == 826
    accepted_ok = manifest.get("accepted_count") == 0
    pending_ok = manifest.get("pending_human_review_count") == 0
    all_ok &= _check(
        "A.4.5 manifest draft fields",
        status_ok and current_ok and target_ok and accepted_ok and pending_ok,
        f"status={manifest.get('status')} current={manifest.get('current_train_count')} "
        f"target={manifest.get('target_train_count')} accepted={manifest.get('accepted_count')} "
        f"pending={manifest.get('pending_human_review_count')}",
    )

    # Bonus: V3 E1-E5 configs preserve hyperparameters, paths point at v3
    print("  --- V3 E1-E5 configs ---")
    preserve_keys = (
        "use_dora", "use_rslora", "neftune_noise_alpha", "packing",
        "lora_r", "lora_alpha", "learning_rate", "num_train_epochs", "seed",
    )
    for n in range(1, 6):
        v2c = _read_json(CONFIGS_DIR / f"kisaki_e{n}_canonical.json")
        v3c = _read_json(CONFIGS_DIR / f"kisaki_e{n}_canonical_v3.json")
        diffs = [k for k in preserve_keys if v2c.get(k) != v3c.get(k)]
        train_ok = v3c.get("train_data_path", "").endswith("/v3/tsukiyashiro_kisaki_train.json")
        eval_ok = v3c.get("eval_data_path", "").endswith("/v3/tsukiyashiro_kisaki_eval.json")
        out_ok = v3c.get("output_dir", "").endswith(f"/canonical_v3/e{n}/seed42")
        exp_ok = v3c.get("_experiment_id", "").endswith("-V3") or v3c.get("_experiment_id", "").endswith("V3")
        ds_ok = v3c.get("_dataset_version") == "KISAKI-CANONICAL-V3"
        ok = not diffs and train_ok and eval_ok and out_ok and exp_ok and ds_ok
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] E{n}: "
              f"id={v3c.get('_experiment_id')} role={v3c.get('_comparison_role')} "
              f"paths_ok={train_ok and eval_ok and out_ok} hyperparams_preserved={not diffs}")

    print()
    print("=== Summary ===")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
