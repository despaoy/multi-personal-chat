import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "extract_character_dialogues.py"


def _module():
    spec = importlib.util.spec_from_file_location("extract_character_dialogues", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dialogue_parser_preserves_nested_curly_quotes(tmp_path):
    source = tmp_path / "scene.txt"
    source.write_text(
        "[妃] 「大家问“这是谁？”也没关系。」\n"
        "[妃] 「说“最喜欢哥哥！”的妹妹很稀奇哦。」\n",
        encoding="utf-8",
    )

    events = _module().read_script_events(source)

    assert [event["text"] for event in events] == [
        "大家问“这是谁？”也没关系。",
        "说“最喜欢哥哥！”的妹妹很稀奇哦。",
    ]


def test_dialogue_parser_preserves_multiline_outer_quote(tmp_path):
    source = tmp_path / "scene.txt"
    source.write_text(
        "[妃] 「第一行，\n第二行。」\n[琉璃] “知道了。”\n",
        encoding="utf-8",
    )

    events = _module().read_script_events(source)

    assert [event["text"] for event in events] == ["第一行， 第二行。", "知道了。"]
    assert [(event["line_start"], event["line_end"]) for event in events] == [
        (1, 2),
        (3, 3),
    ]


def _candidates(source, tmp_path):
    module = _module()
    path = tmp_path / "scene.txt"
    path.write_text(source, encoding="utf-8")
    groups = [{
        "source_id": path.name,
        "source_role": "canonical",
        "events": module.read_script_events(path),
    }]
    return module, module.build_candidates(groups, "tsukiyashiro_kisaki")


def test_adjacent_turn_with_short_narration_is_kept(tmp_path):
    module, candidates = _candidates(
        "[琉璃] 「你要一起回去吗？」\n"
        "他停下来等她回答。\n"
        "[妃] 「既然你都等了，我就陪你。」\n",
        tmp_path,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["prompt"] == "你要一起回去吗？"
    assert candidate["reply"] == "既然你都等了，我就陪你。"
    assert candidate["context_speaker_label"] == "琉璃"
    assert candidate["reasons"] == []
    assert candidate["source_line_start"] == 1
    assert candidate["source_line_end"] == 3
    assert candidate["dialogue_block_id"] == "scene.txt:turn:1-3"
    assert candidate["scene_block_id"] == "scene.txt:scene-block:1"
    assert module.MAX_INTERVENING_NARRATION_LINES == 4


def test_long_narration_breaks_pairing(tmp_path):
    _, candidates = _candidates(
        "[琉璃] 「你听见了吗？」\n"
        "第一行旁白。\n第二行旁白。\n第三行旁白。\n第四行旁白。\n第五行旁白。\n"
        "[妃] 「听见了。」\n",
        tmp_path,
    )

    assert len(candidates) == 1
    assert candidates[0]["prompt"] == ""
    assert "long_narration_gap" in candidates[0]["reasons"]
    assert "missing_context" in candidates[0]["reasons"]


def test_scene_reset_breaks_pairing_even_inside_line_window(tmp_path):
    _, candidates = _candidates(
        "[琉璃] 「明天再说吧。」\n"
        "第二天早上。\n"
        "[妃] 「早上好。」\n",
        tmp_path,
    )

    assert candidates[0]["prompt"] == ""
    assert "scene_boundary" in candidates[0]["reasons"]


def test_new_speaker_stops_context_collection(tmp_path):
    _, candidates = _candidates(
        "[琉璃] 「先前的问题呢？」\n"
        "[夜子] 「现在回答我就好。」\n"
        "[妃] 「……知道了。」\n",
        tmp_path,
    )

    assert candidates[0]["prompt"] == "现在回答我就好。"
    assert candidates[0]["context_speaker_label"] == "夜子"


def test_uncertain_speaker_is_not_injected(tmp_path):
    _, candidates = _candidates(
        "[？？？] 「你是谁？」\n"
        "[妃] 「先报上自己的名字吧。」\n",
        tmp_path,
    )

    assert candidates[0]["prompt"] == "你是谁？"
    assert candidates[0]["context_speaker_label"] is None
    assert candidates[0]["context_source_speaker_label"] == "？？？"


def test_excluded_sft_record_preserves_exclusion_reasons():
    module = _module()
    item = {
        "source": "scene.txt:line:1",
        "source_file": "scene.txt",
        "source_speaker_label": "妃",
        "context_speaker_label": None,
        "target_event_ids": ["raw-1"],
        "response_line_count": 1,
        "quality_score": 50,
        "is_short_reply": False,
        "prompt": "",
        "reply": "回答",
        "reasons": ["missing_context", "low_information_prompt"],
    }

    record = module.as_sft(item, "tsukiyashiro_kisaki", include_exclusion_reasons=True)

    assert record["metadata"]["exclusion_reasons"] == [
        "low_information_prompt",
        "missing_context",
    ]
