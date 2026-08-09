import json
import sys

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/archive/kisaki_legacy/validate_kisaki_experiments.py"
spec = importlib.util.spec_from_file_location("legacy_validate_kisaki_experiments", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


def test_custom_registry_output_does_not_modify_tracked_registry(tmp_path, monkeypatch):
    tracked_before = validator.REGISTRY_PATH.read_bytes()
    output = tmp_path / "runtime" / "canonical_experiment_registry.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_kisaki_experiments.py",
            "--write-registry",
            "--registry-output",
            str(output),
        ],
    )

    assert validator.main() == 0
    assert validator.REGISTRY_PATH.read_bytes() == tracked_before
    registry = json.loads(output.read_text(encoding="utf-8"))
    assert registry["schema_version"] == 3
    assert registry["series_id"] == "KISAKI-R1-CONTROLLED-PEFT"
