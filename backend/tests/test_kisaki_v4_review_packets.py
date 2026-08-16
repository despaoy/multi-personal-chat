import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "build_kisaki_v4_review_packets.py"
CANONICAL_SCRIPT = PROJECT_ROOT / "scripts" / "build_kisaki_v4_canonical_draft.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_kisaki_v4_review_packets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_review_packets_cover_all_current_human_review_inputs(tmp_path):
    output = tmp_path / "review"
    manifest = _module().build(output, 50)
    assert manifest["status"] == "pending_human_review"
    assert manifest["counts"] == {
        "source_lines": 1598,
        "game_train_candidates": 652,
        "constructed_train_candidates": 150,
        "v4_independent_validation": 77,
        "gold_v21": 150,
        "gold_v3": 150,
        "exclusions": 189,
    }
    stored = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert manifest["source_coverage"] == {
        "original_files_total": 17,
        "files_with_kisaki_lines": 13,
    }
    assert (output / "02_SOURCE_COVERAGE" / "00_SOURCE_FILE_INDEX.md").exists()
    assert (output / "01_PROFILE_PROMPT" / "01_character_profile.md").exists()
    prompt_v3 = (output / "01_PROFILE_PROMPT" / "02_system_prompt_v3.md").read_text(encoding="utf-8")
    assert "亲生哥哥" in prompt_v3
    assert "知识库" not in prompt_v3
    canonical_prompt = (PROJECT_ROOT / "backend/data/character_dialogues/kisaki_system_prompt_v3.txt").read_text(encoding="utf-8")
    assert canonical_prompt in prompt_v3
    assert "source_hashes" not in manifest
    first_gold = (output / "06_GOLD_V21" / "batch_01.md").read_text(encoding="utf-8")
    assert "required_facts" in first_gold
    assert "rubric" in first_gold
    assert "**assistant**" not in first_gold
    first_gold_v3 = (output / "07_GOLD_V3" / "batch_01.md").read_text(encoding="utf-8")
    assert "kisaki_v3_persona_001" in first_gold_v3


def test_review_packet_builder_refuses_to_overwrite_human_work(tmp_path):
    output = tmp_path / "review"
    output.mkdir()
    (output / "human_notes.md").write_text("keep", encoding="utf-8")
    try:
        _module().build(output, 50)
    except FileExistsError:
        pass
    else:
        raise AssertionError("builder overwrote a non-empty review directory")


def test_candidate_to_review_packet_rebuild_works_in_empty_directories(tmp_path):
    canonical_spec = importlib.util.spec_from_file_location(
        "build_kisaki_v4_canonical_for_pipeline", CANONICAL_SCRIPT
    )
    canonical = importlib.util.module_from_spec(canonical_spec)
    assert canonical_spec.loader is not None
    canonical_spec.loader.exec_module(canonical)

    candidates = tmp_path / "candidates"
    review = tmp_path / "review"
    canonical.build(candidates)
    manifest = _module().build(review, 50, candidates)

    assert manifest["counts"]["game_train_candidates"] == 652
    assert manifest["counts"]["v4_independent_validation"] == 77
    first_gold = (review / "06_GOLD_V21/batch_01.md").read_text(encoding="utf-8")
    current_gold = json.loads(
        (PROJECT_ROOT / "backend/evaluation/kisaki_gold_set_v21_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert current_gold["prompts"][0]["id"] in first_gold
