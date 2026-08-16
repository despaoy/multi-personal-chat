from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_kisaki_v41_user_simulation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_kisaki_v41_user_simulation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_user_prompts_are_authored_and_sessions_are_multiturn():
    module = load_module()
    assert len(module.SESSIONS) == 4
    assert all(len(session["user_turns"]) == 5 for session in module.SESSIONS)
    assert sum(len(session["user_turns"]) for session in module.SESSIONS) == 20
    assert {session["session_id"] for session in module.SESSIONS} == {
        "daily_chat", "research_learning", "coding_debug", "project_safety"
    }
    assert len(module.STYLE_MODES) == 20


def test_user_prompts_do_not_contain_generation_meta_instructions():
    module = load_module()
    prompts = [text for session in module.SESSIONS for text in session["user_turns"]]
    forbidden = ("扮演月社妃", "体现月社妃", "训练数据", "模拟用户")
    assert all(not any(term in prompt for term in forbidden) for prompt in prompts)


def test_role_audit_blocks_meta_and_unrelated_lore():
    module = load_module()
    errors, _ = module.audit_turn("作为 AI，我想起了琉璃。", session_id="coding_debug", turn=2)
    assert "AI/role meta-reference" in errors
    assert any("forces original lore" in error for error in errors)


def test_approved_prompt_is_repeated_after_source_corpus():
    module = load_module()
    marker = "SOURCE_CORPUS_END"
    system = module.role_system(marker)
    assert system.index(marker) < system.rindex("你是月社妃")
    assert "熟悉用户" in system
    assert "保持 alpha/r 比例不变" in system
    assert "一个不必回答的愚蠢问题呢" in system


def test_generic_assistant_templates_are_rejected():
    module = load_module()
    errors, _ = module.audit_turn(
        "可以。这里有三个步骤，需要我继续补充吗？",
        session_id="daily_chat",
        turn=1,
    )
    assert "generic assistant opening" in errors
    assert "generic assistant closing" in errors


def test_sharp_persona_answer_passes_style_gate():
    module = load_module()
    errors, _ = module.audit_turn(
        "想一次改完所有参数？真是贪心。先固定其余条件，再谈结论。",
        session_id="research_learning",
        turn=2,
    )
    assert errors == []


def test_context_modes_do_not_require_every_answer_to_be_sharp():
    module = load_module()
    errors, _ = module.audit_turn(
        "今天不讲理的是实验，不是你。先休息吧，逞强也不会让结果自己变好。",
        session_id="daily_chat",
        turn=1,
    )
    assert errors == []


def test_restrained_care_rejects_inappropriate_harshness():
    module = load_module()
    errors, _ = module.audit_turn(
        "连这点失败都受不了，真是愚蠢。",
        session_id="daily_chat",
        turn=1,
    )
    assert "style mode restrained_care is inappropriately harsh" in errors
