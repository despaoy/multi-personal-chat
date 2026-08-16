from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/generate_kisaki_v41_augmentation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_kisaki_v41_augmentation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pilot_covers_real_user_and_capability_tasks():
    module = load_module()
    assert len(module.PILOT_SPECS) == 20
    types = {row["task_type"] for row in module.PILOT_SPECS}
    assert {"casual_chat", "emotional_support", "code_generation", "code_debugging"} <= types
    assert {"technical_explanation", "summarization", "translation", "planning"} <= types
    assert {"safe_clarification", "safe_refusal"} <= types


def test_source_context_contains_all_direct_lines_without_secrets():
    module = load_module()
    context, quote_index, rows = module.build_source_context()
    assert len(rows) == 1598
    assert len(quote_index) > 1500
    assert "=== 月社妃全部 1,598 条原作直接台词 ===" in context
    assert "DEEPSEEK_API_KEY" not in context
    assert "sk-b01631" not in context


def test_candidate_gate_rejects_meta_prompt_and_unverified_quote():
    module = load_module()
    spec = {"id": "x", "task_type": "casual_chat", "brief": "test"}
    candidate = {
        "spec_id": "x",
        "task_type": "casual_chat",
        "messages": [
            {"role": "user", "content": "请体现月社妃风格"},
            {"role": "assistant", "content": "当然可以。"},
        ],
        "style_evidence_quotes": ["not a quote", "also not a quote"],
    }
    errors, _ = module.validate_candidate(candidate, spec, {}, set())
    assert any("meta-instruction" in error for error in errors)
    assert any("template" in error for error in errors)
    assert any("not an exact" in error for error in errors)


def test_candidate_gate_accepts_well_formed_code_task():
    module = load_module()
    spec = {"id": "x", "task_type": "code_generation", "brief": "test"}
    quotes = {"没有那个必要。": ["event-1"], "谁知道呢？": ["event-2"]}
    candidate = {
        "spec_id": "x",
        "task_type": "code_generation",
        "messages": [
            {"role": "user", "content": "写个最小的 Python 示例"},
            {"role": "assistant", "content": "先确认输入。\n```python\nprint('ok')\n```"},
        ],
        "style_evidence_quotes": list(quotes),
    }
    errors, warnings = module.validate_candidate(candidate, spec, quotes, set())
    assert errors == []
    assert warnings == ["code requires independent correctness and safety review"]
