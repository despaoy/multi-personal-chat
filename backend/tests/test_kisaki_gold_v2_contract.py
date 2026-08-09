import json
from pathlib import Path

from evaluation.experiment_contracts import normalized_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = PROJECT_ROOT / "backend" / "evaluation" / "kisaki_gold_set_v2_candidates.json"
REGISTRY_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "character_dialogues"
    / "experiments"
    / "research"
    / "research_program_registry_v4.json"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(item: dict) -> str:
    turns = item.get("turns") or [item["prompt"]]
    return "\x1f".join(normalized_text(str(turn)) for turn in turns)


def test_gold_v2_contains_150_unique_benchmark_items():
    prompts = _load(GOLD_PATH)["prompts"]
    signatures = [_signature(item) for item in prompts]
    assert len(prompts) == 150
    assert len(signatures) == len(set(signatures))


def test_gold_v2_multiturn_items_are_distinct_real_dialogues():
    prompts = _load(GOLD_PATH)["prompts"]
    multiturn = [item for item in prompts if item["category"] == "multiturn"]
    first_turns = [normalized_text(item["turns"][0]) for item in multiturn]
    assert len(multiturn) == 30
    assert all(len(item["turns"]) == 3 for item in multiturn)
    assert len(first_turns) == len(set(first_turns))


def test_gold_v2_is_registered_as_development_only():
    registry = _load(REGISTRY_PATH)
    gold_v2 = registry["active_assets"]["gold_v2"]
    assert gold_v2["role"] == "development_only"
