import importlib.util
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/build_kisaki_checkpoint_devset.py"
SOURCE = PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json"


def _module():
    spec = importlib.util.spec_from_file_location("build_kisaki_checkpoint_devset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_devset_is_deterministic_balanced_and_non_formal():
    module = _module()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))

    first = module.build_subset(source, seed=42)
    second = module.build_subset(source, seed=42)

    assert first == second
    assert first["formal_use_allowed"] is False
    assert first["evaluation_role"] == "development_checkpoint_selection"
    assert len(first["prompts"]) == 30
    assert Counter(row["category"] for row in first["prompts"]) == {
        "persona": 8,
        "factual": 5,
        "persona_knowledge": 2,
        "multiturn": 7,
        "safety": 8,
    }


def test_checkpoint_devset_has_cluster_coverage_and_all_safety_actions():
    module = _module()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = module.build_subset(source, seed=42)["prompts"]

    for category in ("persona", "factual", "persona_knowledge", "multiturn"):
        category_rows = [row for row in rows if row["category"] == category]
        assert len({row["cluster_id"] for row in category_rows}) == len(category_rows)

    safety = [row for row in rows if row["category"] == "safety"]
    assert {row["expected_action"] for row in safety} == {
        "allow",
        "allow_with_confirmation",
        "allow_with_redaction",
        "clarify",
        "clarify_supportive",
        "crisis_support",
        "refuse",
        "safe_alternative",
    }
    assert all(row["category"] != "rag_grounded" for row in rows)


def test_checkpoint_devset_interleaves_broad_groups():
    module = _module()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    categories = [row["category"] for row in module.build_subset(source)["prompts"]]
    broad = [
        "factual" if category in {"factual", "persona_knowledge"} else category
        for category in categories
    ]

    assert broad[:4] == ["persona", "factual", "multiturn", "safety"]
    assert max(
        sum(1 for category in broad[index : index + 4] if category == target)
        for index in range(len(broad) - 3)
        for target in {"persona", "factual", "multiturn", "safety"}
    ) <= 2
