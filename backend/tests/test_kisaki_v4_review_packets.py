import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "build_kisaki_v4_review_packets.py"


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
        "game_train_candidates": 801,
        "constructed_train_candidates": 159,
        "legacy_validation": 92,
        "v5_draft_validation": 27,
        "gold_v2": 150,
        "gold_v3": 0,
        "exclusions": 117,
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
    assert manifest["source_hashes"]["prompt_v3"]
    first_gold = (output / "06_GOLD_V2" / "batch_01.md").read_text(encoding="utf-8")
    assert "**评分标准 expected_behavior**" in first_gold
    assert "**assistant**" not in first_gold
    assert "blocked_until_training_data_frozen" in (
        output / "07_GOLD_V3" / "README.md"
    ).read_text(encoding="utf-8")


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
