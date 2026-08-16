import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "enrich_kisaki_rag_evidence_lineage.py"


def _module():
    spec = importlib.util.spec_from_file_location("enrich_kisaki_rag_evidence_lineage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_segments_must_match_inside_one_continuous_window():
    module = _module()
    files = {
        "scene.txt": [
            "[妃] 「第一句。」",
            "短旁白。",
            "[妃] 「第二句。」",
        ]
    }

    source_file, spans = module.match_source_window("第一句。\n第二句。", files)

    assert source_file == "scene.txt"
    assert spans == ((1, 1), (3, 3))


def test_short_substring_does_not_count_as_exact_segment():
    module = _module()

    with pytest.raises(ValueError, match="all evidence segments"):
        module.match_source_window("啰嗦", {"scene.txt": ["[妃] 「多嘴啰嗦。」"]})


def test_distant_duplicate_is_not_added_to_lineage():
    module = _module()
    lines = ["[妃] 「呜——」", "旁白。", "[妃] 「你这个人真是糊涂虫呢！」", "旁白。", "[妃] 「这种事不用说我也明白……」"]
    lines.extend(["无关内容。"] * 100)
    lines.append("[妃] 「呜——」")

    _, spans = module.match_source_window(
        "呜——\n你这个人真是糊涂虫呢！\n这种事不用说我也明白……",
        {"scene.txt": lines},
    )

    assert spans == ((1, 1), (3, 3), (5, 5))


def test_segments_outside_window_are_rejected():
    module = _module()
    lines = ["[妃] 「第一句。」"] + ["旁白。"] * module.MAX_EVIDENCE_WINDOW_LINES + ["[妃] 「第二句。」"]

    with pytest.raises(ValueError, match="all evidence segments"):
        module.match_source_window("第一句。\n第二句。", {"scene.txt": lines})
