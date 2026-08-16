import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_kisaki_source_alignment.py"


def _module():
    spec = importlib.util.spec_from_file_location("audit_kisaki_source_alignment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_kisaki_source_assets_are_reproducible_and_attributable():
    result = _module().audit()

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["counts"] == {
        "source_files": 17,
        "raw_dialogues": 1598,
        "recommended_sft": 768,
        "full_sft": 784,
        "excluded": 189,
        "raw_with_disposition": 1598,
    }
    assert all(result["checks"].values())
