import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts/build_kisaki_v4_overfit_test.py"
RENDER_SCRIPT = ROOT / "scripts/render_kisaki_v4_overfit_review.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_overfit_builder_selects_fixed_balanced_records(tmp_path):
    module = load_module(BUILD_SCRIPT, "build_kisaki_v4_overfit_test")
    output = tmp_path / "overfit"
    review = tmp_path / "review"
    first = module.build(output, review)
    first_train = (output / "train.jsonl").read_bytes()
    second = module.build(output, review)
    assert first == second
    assert first_train == (output / "train.jsonl").read_bytes()
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 20
    sources = Counter(row["metadata"]["data_source"] for row in rows)
    assert sources["game_extraction"] == 10
    assert all(sources[source] == 2 for source in module.CONSTRUCTED_SOURCES)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["train_data_path"] == config["eval_data_path"]
    assert config["num_train_epochs"] == 20
    assert config["system_prompt_policy"] == "replace"


def test_overfit_review_renderer_requires_and_renders_20_results(tmp_path):
    module = load_module(RENDER_SCRIPT, "render_kisaki_v4_overfit_review")
    rows = [
        {
            "id": f"case-{index:02d}",
            "interlocutor": "用户",
            "messages": [{"role": "user", "content": f"问题 {index}"}],
            "reference_answer": f"参考 {index}",
            "response": f"回答 {index}",
        }
        for index in range(20)
    ]
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"results": rows}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "review.md"
    module.render(results, output)
    text = output.read_text(encoding="utf-8")
    assert text.count("- [ ] 通过") == 20
    assert "回答 19" in text
