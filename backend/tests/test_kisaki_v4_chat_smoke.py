import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_kisaki_v4_chat_smoke.py"
RENDER_SCRIPT = ROOT / "scripts/render_kisaki_v4_chat_smoke_review.py"


def module():
    spec = importlib.util.spec_from_file_location("build_kisaki_v4_chat_smoke", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def test_chat_smoke_has_separate_held_out_and_contextual_sections():
    payload = module().build()
    assert len(payload["natural_chat"]) == 20
    assert len(payload["continuity"]["user_turns"]) == 6
    assert len(payload["contextual_story"]) == 8
    train = [
        json.loads(line)
        for line in (ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    train_prompts = {
        module().normalized(message["content"])
        for row in train
        for message in row["messages"]
        if message["role"] == "user"
    }
    assert not any(
        module().normalized(row["messages"][0]["content"]) in train_prompts
        for row in payload["natural_chat"]
    )
    assert all(row["source_line_end"] >= row["source_line_start"] for row in payload["contextual_story"])
    assert all(row["reference_answer"] for row in payload["contextual_story"])


def test_chat_smoke_renderer_keeps_three_review_scopes_separate(tmp_path):
    spec = importlib.util.spec_from_file_location("render_kisaki_v4_chat_smoke", RENDER_SCRIPT)
    render_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(render_module)
    natural = [
        {"id": f"n-{i}", "messages": [{"role": "user", "content": "问"}], "response": "答"}
        for i in range(20)
    ]
    continuity = {
        "turns": [{"turn": i, "user": "问", "response": "答"} for i in range(1, 7)]
    }
    contextual = [
        {
            "source_sample_id": f"c-{i}",
            "source_path": "story.txt",
            "source_line_start": 1,
            "response": "答",
            "reference_answer": "参考",
        }
        for i in range(8)
    ]
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {"variant": "e1", "natural_chat": natural, "continuity": continuity, "contextual_story": contextual},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.md"
    render_module.render(results, output)
    text = output.read_text(encoding="utf-8")
    assert "自然聊天（20 条）" in text
    assert "六轮连续对话" in text
    assert "带上下文原作场景（8 条）" in text
