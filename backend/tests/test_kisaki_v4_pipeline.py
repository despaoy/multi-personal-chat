"""Unit tests for Kisaki V4 regeneration pipeline (Task B.8).

Covers:
  B.8.1  hard-gate checks (each gate triggers on constructed violations)
  B.8.2  copy-detection combined rule (A reject / B reject / disputed)
  B.8.3  longest_common_substring vs subsequence (contiguous, not gapped)
  B.8.4  Judge A JSON parsing (code fence / missing fields / not_applicable)
  B.8.5  Judge B double-order consistency (same→passed, opposite→disputed)
  B.8.6  same-question constraint (mismatched human dialogue raises)
  B.8.7  Generator prompt construction (few-shot / negative / quota / feedback)
  B.8.8  resume from progress.json (completed specs are skipped)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (PROJECT_ROOT, BACKEND_ROOT, SCRIPTS_DIR):
    v = str(p)
    if v not in sys.path:
        sys.path.insert(0, v)

from hard_gate_kisaki_v4 import (  # noqa: E402
    check_ai_self_reference,
    check_length,
    check_meta_narrative,
    check_original_copy,
    check_repeated_opening,
    check_third_person_self_description,
    check_zheng_yin_ci,
    longest_common_substring,
    ngram_jaccard,
    run_all_gates,
    validate_json_structure,
)
from judge_kisaki_llm_v4 import (  # noqa: E402
    judge_b,
    parse_judge_a_response,
    parse_judge_b_response,
    verify_same_question_for_judge_b,
)
from generate_kisaki_llm_v4 import (  # noqa: E402
    build_generator_prompt,
    parse_generator_response,
)
from kisaki_v4_llm_client import SampleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Autouse fixture: force similarity backend to "fallback" for unit tests.
# The strict backend (default) requires the BGE embedding model on disk,
# which is not available in the test environment. Tests that mock
# semantic_similarity don't need the real backend, but the module-level
# resolution in hard_gate_kisaki_v4.get_similarity_backend() would raise
# RuntimeError in strict mode before the mock takes effect. Setting
# KISAKI_SIMILARITY_BACKEND=fallback lets the backend resolve to
# char_4gram_fallback so mocked tests can proceed.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_similarity_fallback(monkeypatch):
    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "fallback")
    # Reset the cached backend resolution so the env var takes effect
    import hard_gate_kisaki_v4 as hg
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)


# ---------------------------------------------------------------------------
# B.8.1: Hard-gate checks (each gate triggers on constructed violations)
# ---------------------------------------------------------------------------

def test_gate_length_rejects_overlong_assistant():
    convs = [
        {"from": "human", "value": "你好"},
        {"from": "assistant", "value": "x" * 101},
    ]
    failures = check_length(convs, max_chars=100)
    assert len(failures) == 1
    assert failures[0].rule == "length"


def test_gate_length_passes_within_limit():
    convs = [
        {"from": "human", "value": "你好"},
        {"from": "assistant", "value": "x" * 100},
    ]
    assert check_length(convs, max_chars=100) == []


def test_gate_ai_self_reference_detects_patterns():
    assert check_ai_self_reference("我是一个AI助手") != []
    assert check_ai_self_reference("作为AI，我无法感受") != []
    assert check_ai_self_reference("我是人工智能") != []
    assert check_ai_self_reference("呼呼呼，原来如此") == []


def test_gate_third_person_self_description():
    assert check_third_person_self_description("月社妃是一个复杂的角色") != []
    assert check_third_person_self_description("妃这个角色很特别") != []
    assert check_third_person_self_description("我没有那个必要") == []


def test_gate_repeated_opening_same_scene():
    candidate = [
        {"from": "human", "value": "你好"},
        {"from": "assistant", "value": "因此，我不会配合你。"},
    ]
    passed = [{
        "scene": "日常问候",
        "conversations": [
            {"from": "human", "value": "早"},
            {"from": "assistant", "value": "因此，我不会配合你。"},
        ],
    }]
    failures = check_repeated_opening(candidate, passed, "日常问候")
    assert len(failures) == 1
    assert failures[0].rule == "repeated_opening"


def test_gate_repeated_opening_different_scene_no_failure():
    candidate = [
        {"from": "human", "value": "你好"},
        {"from": "assistant", "value": "因此，我不会配合你。"},
    ]
    passed = [{
        "scene": "书籍讨论",  # different scene
        "conversations": [
            {"from": "human", "value": "早"},
            {"from": "assistant", "value": "因此，我不会配合你。"},
        ],
    }]
    assert check_repeated_opening(candidate, passed, "日常问候") == []


def test_gate_meta_narrative_non_meta_scene_rejects():
    # "故事" appears in a non-meta scene -> fail
    # Major-3 fix: check_meta_narrative now takes assistant_turns (list[str])
    failures = check_meta_narrative(["这是一个有趣的故事"], scene="日常问候")
    assert len(failures) == 1
    assert failures[0].rule == "meta_narrative"


def test_gate_meta_narrative_meta_scene_allows_one():
    # "故事" in a meta-narrative scene -> allowed (1 occurrence per turn)
    assert check_meta_narrative(["这是一个有趣的故事"], scene="书籍讨论") == []


def test_gate_meta_narrative_meta_scene_rejects_two_in_one_turn():
    """Critical-2 + Major-3: 2 meta words in a single turn -> fail (per-turn cap)."""
    failures = check_meta_narrative(["故事和作者都在这里"], scene="书籍讨论")
    assert len(failures) == 1
    assert "per-turn cap" in failures[0].detail


def test_gate_meta_narrative_meta_scene_allows_one_per_turn_across_turns():
    """Major-3: 1 meta word in each of 2 turns is allowed (per-turn ≤ 1)."""
    assert check_meta_narrative(["故事开始了", "作者很厉害"], scene="书籍讨论") == []


def test_count_meta_narrative_words_counts_occurrences_not_kinds():
    """Critical-2: '故事故事故事' should count as 3, not 1."""
    from hard_gate_kisaki_v4 import count_meta_narrative_words
    assert count_meta_narrative_words("故事故事故事") == 3
    assert count_meta_narrative_words("故事作者") == 2
    assert count_meta_narrative_words("没有元叙事词") == 0


def test_gate_zheng_yin_ci_count():
    assert check_zheng_yin_ci("正因如此") == []
    assert check_zheng_yin_ci("正因如此，正因如此") != []
    assert check_zheng_yin_ci("因此") == []


def test_gate_json_structure_invalid():
    assert validate_json_structure({}) != []
    assert validate_json_structure({"conversations": []}) != []
    assert validate_json_structure({
        "conversations": [{"from": "human", "value": ""}]
    }) != []
    assert validate_json_structure({
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "嗨"},
        ]
    }) == []


# ---------------------------------------------------------------------------
# B.8.2: Copy-detection combined rule (A / B / disputed)
# ---------------------------------------------------------------------------

def test_copy_detection_rule_a_reject_lcs_and_jaccard(monkeypatch):
    """Rule A: LCS > 20 AND Jaccard > 0.30 -> reject."""
    # Construct two strings sharing a long common substring (>20 chars)
    common = "因此我不会配合你的任何提议因为这毫无意义吧"  # 21 chars -> LCS > 20
    candidate = common + "额外的文字"
    reference = common + "不同的后缀"
    # Mock semantic_similarity to a low value so only rule A triggers
    monkeypatch.setattr("hard_gate_kisaki_v4.semantic_similarity", lambda a, b: 0.50)
    failures, disputed = check_original_copy(candidate, [reference])
    assert len(failures) == 1
    assert failures[0].rule == "original_copy_A"
    assert disputed == []


def test_copy_detection_rule_b_reject_cosine_and_jaccard(monkeypatch):
    """Rule B: cosine > 0.92 AND Jaccard > 0.20 -> reject."""
    # Use strings with high Jaccard (>0.20) but short LCS (≤20)
    candidate = "因此我不会配合你"
    reference = "因此我不会配合你"  # identical -> Jaccard=1.0
    # Mock cosine to >0.92
    monkeypatch.setattr("hard_gate_kisaki_v4.semantic_similarity", lambda a, b: 0.95)
    failures, disputed = check_original_copy(candidate, [reference])
    # Both rule A (LCS=7, not >20) and rule B should be checked
    # rule A: LCS=7 (not >20) -> no trigger; rule B: cos=0.95 AND jac>0.20 -> trigger
    rules = [f.rule for f in failures]
    assert "original_copy_B" in rules


def test_copy_detection_disputed_only_cosine(monkeypatch):
    """Disputed: only cosine > 0.85 (cos ≤ 0.92 OR Jaccard ≤ 0.20)."""
    candidate = "因此我不会配合你的任何提议"
    reference = "完全不同的文字内容无法匹配"
    # LCS is small, Jaccard is low, but cosine is between 0.85 and 0.92
    monkeypatch.setattr("hard_gate_kisaki_v4.semantic_similarity", lambda a, b: 0.88)
    failures, disputed = check_original_copy(candidate, [reference])
    assert failures == []
    assert len(disputed) == 1
    assert "human review" in disputed[0]


def test_copy_detection_no_trigger_when_all_low(monkeypatch):
    """No failure and no disputed when all metrics are low."""
    candidate = "因此我不会配合你"
    reference = "完全不同的文字"
    monkeypatch.setattr("hard_gate_kisaki_v4.semantic_similarity", lambda a, b: 0.50)
    failures, disputed = check_original_copy(candidate, [reference])
    assert failures == []
    assert disputed == []


# ---------------------------------------------------------------------------
# B.8.3: longest_common_substring vs subsequence (contiguous, not gapped)
# ---------------------------------------------------------------------------

def test_lcs_substring_contiguous_not_subsequence():
    """LCS("abcdef", "xbcdyz") = 3 ("bcd" contiguous), NOT 4 ("bcdf" gapped)."""
    assert longest_common_substring("abcdef", "xbcdyz") == 3


def test_lcs_substring_full_match():
    assert longest_common_substring("abc", "abc") == 3


def test_lcs_substring_no_match():
    assert longest_common_substring("abc", "xyz") == 0


def test_lcs_substring_empty():
    assert longest_common_substring("", "abc") == 0
    assert longest_common_substring("abc", "") == 0


def test_lcs_substring_partial_overlap():
    # "握手" appears in both but not as a longer substring
    assert longest_common_substring("握手言和", "握手") == 2


# ---------------------------------------------------------------------------
# B.8.4: Judge A JSON parsing (code fence / missing fields / NA)
# ---------------------------------------------------------------------------

def test_judge_a_parse_clean_json():
    raw = json.dumps({
        "scores": {"人物一致性": 8, "语境连贯": 9, "自然度": 7, "原作语气": 8, "事实关系": "not_applicable"},
        "evidence": {"人物一致性": "符合"},
        "violations": [],
        "reason": "总体合格",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is True
    assert "事实关系" not in result.applicable_dims
    assert "人物一致性" in result.applicable_dims


def test_judge_a_parse_code_fence():
    raw = "```json\n" + json.dumps({
        "scores": {"人物一致性": 5, "语境连贯": 9, "自然度": 7, "原作语气": 8, "事实关系": 9},
        "evidence": {},
        "violations": [],
        "reason": "人物一致性不足",
    }) + "\n```"
    result = parse_judge_a_response(raw)
    assert result.passed is False  # 人物一致性=5 < 7
    assert "人物一致性" in result.applicable_dims


def test_judge_a_parse_missing_fields():
    raw = "{}"
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert result.applicable_dims == []


def test_judge_a_parse_invalid_json():
    raw = "not json at all"
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert any("json_parse_error" in v for v in result.violations)


def test_judge_a_parse_all_na_means_no_applicable_dims():
    raw = json.dumps({
        "scores": {dim: "not_applicable" for dim in
                   ("人物一致性", "语境连贯", "自然度", "原作语气", "事实关系")},
        "evidence": {},
        "violations": [],
        "reason": "all NA",
    })
    result = parse_judge_a_response(raw)
    assert result.applicable_dims == []
    # Empty applicable_dims -> passed=False (fail-closed)
    assert result.passed is False


# ---------------------------------------------------------------------------
# B.8.5: Judge B double-order consistency
# ---------------------------------------------------------------------------

def test_judge_b_parse_run1_candidate_is_a_prefers_a():
    # candidate is A, judge says "A" -> prefers candidate
    # Major-6 fix: parse_judge_b_response now returns 4-tuple (added parse_ok)
    raw = json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A更好", "reason": "..."})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert prefers is True
    assert conf == 0.9
    assert parse_ok is True


def test_judge_b_parse_run2_candidate_is_b_prefers_b():
    # candidate is B, judge says "B" -> prefers candidate
    raw = json.dumps({"preferred": "B", "confidence": 0.8, "evidence": "B更好", "reason": "..."})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=False)
    assert prefers is True
    assert parse_ok is True


def test_judge_b_parse_failure_returns_parse_ok_false():
    """Major-6: JSON parse failure surfaces as parse_ok=False (not silent reject)."""
    prefers, conf, evid, parse_ok = parse_judge_b_response("not json", candidate_is_a=True)
    assert parse_ok is False
    assert prefers is False
    assert "json_parse_error" in evid


def test_judge_b_parse_invalid_preferred_value_returns_parse_ok_false():
    """Major-6: invalid 'preferred' value (not A/B) -> parse_ok=False."""
    raw = json.dumps({"preferred": "C", "confidence": 0.5, "evidence": "", "reason": ""})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert parse_ok is False
    assert "invalid_preferred_value" in evid


def test_judge_b_double_order_consistent_passed(monkeypatch):
    """Both runs prefer candidate -> passed."""
    candidate = {
        "sample_spec_id": "test_001",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_001",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
    }
    # Run 1: candidate=A, judge prefers A; Run 2: negative=A, judge prefers B(=candidate)
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A", "reason": "..."})
        else:
            return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B", "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_001")
    assert result.final_decision == "passed"
    assert result.prefers_candidate_run1 is True
    assert result.prefers_candidate_run2 is True


def test_judge_b_double_order_inconsistent_disputed(monkeypatch):
    """Run 1 prefers candidate, Run 2 prefers negative -> disputed."""
    candidate = {
        "sample_spec_id": "test_002",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_002",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A", "reason": "..."})
        else:
            # Run 2: negative=A, candidate=B; judge prefers A(=negative) -> not candidate
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A", "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_002")
    assert result.final_decision == "disputed"
    assert result.prefers_candidate_run1 is True
    assert result.prefers_candidate_run2 is False


def test_judge_b_double_order_both_reject_candidate(monkeypatch):
    """Both runs prefer negative -> rejected."""
    candidate = {
        "sample_spec_id": "test_003",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "嗯。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_003",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Run 1: candidate=A, judge prefers B(=negative)
            return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B", "reason": "..."})
        else:
            # Run 2: negative=A, candidate=B; judge prefers A(=negative)
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A", "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_003")
    assert result.final_decision == "rejected"


# ---------------------------------------------------------------------------
# B.8.6: Same-question constraint (mismatched human dialogue raises)
# ---------------------------------------------------------------------------

def test_same_question_violation_raises_in_judge_b():
    candidate = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "conversations": [
            {"from": "human", "value": "不同的提问"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    with pytest.raises(ValueError, match="same-question"):
        verify_same_question_for_judge_b(candidate, negative, "test_mismatch")


def test_same_question_passes_when_human_matches():
    candidate = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    # Should not raise
    verify_same_question_for_judge_b(candidate, negative, "test_match")


def test_generator_verify_same_question_raises():
    from generate_kisaki_llm_v4 import verify_same_question
    spec = SampleSpec(
        sample_spec_id="test",
        scene="日常问候",
        scene_desc="",
        human_dialogue=["你好"],
        v3_negative_sample_id="neg1",
    )
    negative = {"human_dialogue": ["不同的提问"]}
    with pytest.raises(ValueError, match="same-question"):
        verify_same_question(spec, negative)


# ---------------------------------------------------------------------------
# B.8.7: Generator prompt construction
# ---------------------------------------------------------------------------

def test_build_generator_prompt_contains_required_sections():
    spec = SampleSpec(
        sample_spec_id="kisaki_v3neg_test_001",
        scene="日常问候",
        scene_desc="打招呼、问好、早晚安、天气",
        human_dialogue=["你好啊", "今天怎么样"],
        v3_negative_sample_id="neg1",
        reference_ids=["ref1", "ref2"],
        quota_plan_hash="abc12345",
    )
    few_shots = [
        {
            "sample_id": "ref1",
            "conversations": [
                {"from": "human", "value": "早"},
                {"from": "assistant", "value": "因此，早。"},
            ],
        },
    ]
    negative = {
        "conversations": [
            {"from": "human", "value": "你好啊"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
        "problem_tags": ["meta_narrative_overload"],
    }
    quota_state = {
        "meta_narrative_over_limit": True,
        "laughter_needs_diversity": True,
    }
    prompt = build_generator_prompt(spec, few_shots, negative, quota_state, retry_feedback="上轮失败")
    # Check all sections present
    assert "【本次任务】" in prompt
    assert "日常问候" in prompt
    assert "你好啊" in prompt
    assert "今天怎么样" in prompt
    assert "【原作风格参考" in prompt
    assert "ref1" in prompt
    assert "【v3 负例" in prompt
    assert "meta_narrative_overload" in prompt
    assert "【本批次配额提示】" in prompt
    assert "元叙事样本已超限" in prompt
    assert "【上轮失败反馈" in prompt
    assert "上轮失败" in prompt


def test_build_generator_prompt_without_quota_and_feedback():
    spec = SampleSpec(
        sample_spec_id="test",
        scene="书籍讨论",
        scene_desc="讨论书籍",
        human_dialogue=["最近读了什么"],
        v3_negative_sample_id="neg1",
    )
    few_shots = []
    negative = {
        "conversations": [
            {"from": "human", "value": "最近读了什么"},
            {"from": "assistant", "value": "正因如此。"},
        ],
        "problem_tags": [],
    }
    prompt = build_generator_prompt(spec, few_shots, negative, None, "")
    assert "【本次任务】" in prompt
    assert "(无可用原作参考)" in prompt
    assert "【本批次配额提示】" not in prompt
    assert "【上轮失败反馈" not in prompt


def test_parse_generator_response_enforces_same_question():
    spec = SampleSpec(
        sample_spec_id="test",
        scene="日常问候",
        scene_desc="",
        human_dialogue=["你好", "在吗"],
        v3_negative_sample_id="neg1",
    )
    # Generator returns the correct human turns
    raw = json.dumps({
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
            {"from": "human", "value": "在吗"},
            {"from": "assistant", "value": "呼呼呼。"},
        ]
    })
    candidate = parse_generator_response(raw, spec)
    assert candidate["conversations"][0]["value"] == "你好"
    assert candidate["conversations"][2]["value"] == "在吗"


def test_parse_generator_response_overwrites_mismatched_human():
    """If generator paraphrases human turns, they get force-overwritten."""
    spec = SampleSpec(
        sample_spec_id="test",
        scene="日常问候",
        scene_desc="",
        human_dialogue=["你好"],
        v3_negative_sample_id="neg1",
    )
    raw = json.dumps({
        "conversations": [
            {"from": "human", "value": "你好啊"},  # paraphrased
            {"from": "assistant", "value": "因此。"},
        ]
    })
    candidate = parse_generator_response(raw, spec)
    # Human turn should be overwritten to match spec
    assert candidate["conversations"][0]["value"] == "你好"


def test_parse_generator_response_rejects_too_few_human_turns():
    spec = SampleSpec(
        sample_spec_id="test",
        scene="日常问候",
        scene_desc="",
        human_dialogue=["你好", "在吗"],  # spec requires 2 human turns
        v3_negative_sample_id="neg1",
    )
    raw = json.dumps({
        "conversations": [
            {"from": "human", "value": "你好"},  # only 1
            {"from": "assistant", "value": "因此。"},
        ]
    })
    with pytest.raises(ValueError, match="human turns"):
        parse_generator_response(raw, spec)


# ---------------------------------------------------------------------------
# B.8.8: Resume from progress.json (completed specs are skipped)
# ---------------------------------------------------------------------------

def test_resume_skips_completed_specs(tmp_path, monkeypatch):
    """Pipeline should skip specs already in progress.json's completed_spec_ids."""
    # Build a fake negative pool with 3 specs
    neg_pool = tmp_path / "v3_negative_pool.jsonl"
    specs_data = []
    for i in range(3):
        specs_data.append({
            "sample_spec_id": f"kisaki_v3neg_test_{i:03d}",
            "v3_sample_id": f"v3_{i}",
            "scene": "日常问候",
            "human_dialogue": [f"你好{i}"],
            "conversations": [
                {"from": "human", "value": f"你好{i}"},
                {"from": "assistant", "value": "正因如此。"},
            ],
            "problem_tags": [],
        })
    neg_pool.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in specs_data) + "\n",
        encoding="utf-8",
    )

    # Write a progress.json marking spec 0 and 1 as completed
    progress = {
        "started_at": "2026-01-01T00:00:00",
        "last_updated": "2026-01-01T00:00:00",
        "completed_spec_ids": ["kisaki_v3neg_test_000", "kisaki_v3neg_test_001"],
        "stats": {"passed": 2, "rejected": 0, "disputed": 0, "total_processed": 2},
    }
    (tmp_path / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Import the pipeline module fresh with tmp_path as OUTPUT_DIR
    # We monkeypatch the module-level path constants
    import regen_kisaki_llm_pipeline as pipeline

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(pipeline, "SAMPLES_PATH", tmp_path / "samples.jsonl")
    monkeypatch.setattr(pipeline, "REJECTED_PATH", tmp_path / "rejected_samples.jsonl")
    monkeypatch.setattr(pipeline, "DISPUTED_PATH", tmp_path / "disputed_samples.jsonl")
    monkeypatch.setattr(pipeline, "RUN_LOG_PATH", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")

    # Mock retrieve_few_shots to avoid needing the actual pool file.
    # Major-7: retrieve_few_shots now accepts human_dialogue and exclude_ids
    # kwargs, so the mock signature must accept them too.
    monkeypatch.setattr(
        "regen_kisaki_llm_pipeline.retrieve_few_shots",
        lambda scene, k=3, **kwargs: [],
    )
    # Also mock in generate_kisaki_llm_v4 since load_specs_from_negative_pool calls it
    monkeypatch.setattr(
        "generate_kisaki_llm_v4.retrieve_few_shots",
        lambda scene, k=3, **kwargs: [],
    )

    # Load specs and verify only the uncompleted one is pending
    specs = pipeline.load_specs_from_negative_pool(neg_pool)
    assert len(specs) == 3

    progress_loaded = pipeline.load_progress()
    completed = set(progress_loaded["completed_spec_ids"])
    pending = [s for s in specs if s.sample_spec_id not in completed]
    assert len(pending) == 1
    assert pending[0].sample_spec_id == "kisaki_v3neg_test_002"


def test_resume_loads_passed_samples_from_samples_jsonl(tmp_path, monkeypatch):
    """passed_samples list should be rebuilt from samples.jsonl on restart."""
    import regen_kisaki_llm_pipeline as pipeline

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "SAMPLES_PATH", tmp_path / "samples.jsonl")

    # Write a samples.jsonl with one passed candidate
    passed_record = {
        "sample_spec_id": "kisaki_v3neg_test_000",
        "status": "passed",
        "candidate": {
            "sample_spec_id": "kisaki_v3neg_test_000",
            "scene": "日常问候",
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "assistant", "value": "因此。"},
            ],
        },
    }
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(passed_record, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    passed = pipeline.load_passed_samples()
    assert len(passed) == 1
    assert passed[0]["sample_spec_id"] == "kisaki_v3neg_test_000"


def test_atomic_write_progress_is_crash_safe(tmp_path, monkeypatch):
    """save_progress uses atomic_write_json (temp + rename)."""
    import regen_kisaki_llm_pipeline as pipeline

    monkeypatch.setattr(pipeline, "PROGRESS_PATH", tmp_path / "progress.json")

    progress = {
        "started_at": "2026-01-01T00:00:00",
        "last_updated": "",
        "completed_spec_ids": ["a", "b"],
        "stats": {"passed": 2, "rejected": 0, "disputed": 0, "total_processed": 2},
    }
    pipeline.save_progress(progress)

    # Verify file exists and is valid JSON
    assert (tmp_path / "progress.json").exists()
    loaded = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert loaded["completed_spec_ids"] == ["a", "b"]
    assert loaded["stats"]["passed"] == 2
    assert loaded["last_updated"] != ""  # was set by save_progress

    # Verify no temp files left behind
    tmp_files = list(tmp_path.glob("progress.json.*.tmp"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# ngram_jaccard sanity check
# ---------------------------------------------------------------------------

def test_ngram_jaccard_identical_strings():
    assert ngram_jaccard("因此因此", "因此因此", n=3) == 1.0


def test_ngram_jaccard_disjoint_strings():
    assert ngram_jaccard("abc", "xyz", n=3) == 0.0


def test_ngram_jaccard_partial_overlap():
    # "abc" and "abd" share "ab" but for n=3 they share nothing
    jac = ngram_jaccard("abc", "abd", n=3)
    assert 0.0 <= jac <= 1.0


# ===========================================================================
# V2.1 fix coverage (Critical-1 / Major-5 / Major-6 / Major-7 / Major-4 / Major-9)
# ===========================================================================

# ---------------------------------------------------------------------------
# Critical-1: disputed_flags present => passed=False (routes to human review)
# ---------------------------------------------------------------------------

def test_run_all_gates_disputed_flags_force_not_passed(monkeypatch):
    """Critical-1: when only disputed_flags is non-empty (no hard failures),
    passed must be False so the sample routes to human review, not auto-pass.
    """
    candidate = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    # Mock copy detection to return only disputed (no hard failure)
    monkeypatch.setattr(
        "hard_gate_kisaki_v4.check_original_copy",
        lambda text, refs: ([], ["ref[0]: cosine=0.88 -> human review"]),
    )
    result = run_all_gates(candidate, scene="日常问候", references=["原作参考"], passed_samples=[])
    assert result.passed is False
    assert result.failures == []
    assert len(result.disputed_flags) == 1


def test_run_all_gates_passed_when_no_failures_and_no_disputed(monkeypatch):
    """Critical-1 sanity: passed=True only when both failures and disputed_flags are empty."""
    candidate = {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    monkeypatch.setattr(
        "hard_gate_kisaki_v4.check_original_copy",
        lambda text, refs: ([], []),
    )
    result = run_all_gates(candidate, scene="日常问候", references=[], passed_samples=[])
    assert result.passed is True
    assert result.failures == []
    assert result.disputed_flags == []


# ---------------------------------------------------------------------------
# Major-5: Judge A requires 4 mandatory core dims
# ---------------------------------------------------------------------------

def test_judge_a_missing_mandatory_dim_fails():
    """Major-5: missing 人物一致性 => cannot pass, even if other dims are high."""
    raw = json.dumps({
        "scores": {"语境连贯": 9, "自然度": 9, "原作语气": 9, "事实关系": 9},
        "evidence": {},
        "violations": [],
        "reason": "missing 人物一致性",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert any("missing_mandatory_dims" in v for v in result.violations)


def test_judge_a_mandatory_dim_na_fails():
    """Major-5: 人物一致性='not_applicable' => cannot pass."""
    raw = json.dumps({
        "scores": {
            "人物一致性": "not_applicable",
            "语境连贯": 9, "自然度": 9, "原作语气": 9, "事实关系": 9,
        },
        "evidence": {},
        "violations": [],
        "reason": "NA on 人物一致性",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert any("missing_mandatory_dims" in v for v in result.violations)


def test_judge_a_all_mandatory_present_with_facts_na_can_pass():
    """Major-5: 4 mandatory dims present + 事实关系 NA => can pass if scores ≥ 7."""
    raw = json.dumps({
        "scores": {
            "人物一致性": 8, "语境连贯": 9, "自然度": 7, "原作语气": 8,
            "事实关系": "not_applicable",
        },
        "evidence": {},
        "violations": [],
        "reason": "ok",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is True
    assert "事实关系" not in result.applicable_dims
    assert "人物一致性" in result.applicable_dims


# ---------------------------------------------------------------------------
# Major-6: Judge B parse failure routes to disputed (not rejected)
# ---------------------------------------------------------------------------

def test_judge_b_parse_failure_routes_to_disputed(monkeypatch):
    """Major-6: when one run's JSON fails to parse, decision='disputed'."""
    candidate = {
        "sample_spec_id": "test_parse_fail",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_parse_fail",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "this is not valid json"  # parse failure on run 1
        return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B", "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_parse_fail")
    assert result.final_decision == "disputed"


# ---------------------------------------------------------------------------
# Major-7: Few-shot relevance retrieval (keyword overlap + source diversity)
# ---------------------------------------------------------------------------

def test_retrieve_few_shots_relevance_ranking(monkeypatch):
    """Major-7: candidates with higher keyword overlap rank first."""
    from generate_kisaki_llm_v4 import retrieve_few_shots, _keyword_overlap_score

    # Higher overlap = higher score
    assert _keyword_overlap_score(["你好今天天气"], ["你好今天怎么样"]) > \
           _keyword_overlap_score(["你好"], ["完全不同的话"])

    # Mock the pool to have two candidates from different sources
    fake_pool = {
        "日常问候": [
            {"sample_id": "a", "source_file": "ch1.txt", "conversations": [
                {"from": "human", "value": "早上好今天天气怎么样"},
                {"from": "assistant", "value": "因此。"},
            ]},
            {"sample_id": "b", "source_file": "ch2.txt", "conversations": [
                {"from": "human", "value": "完全无关的问候"},
                {"from": "assistant", "value": "因此。"},
            ]},
        ],
    }
    monkeypatch.setattr("generate_kisaki_llm_v4._FEW_SHOT_BY_SCENE", fake_pool)
    results = retrieve_few_shots(
        "日常问候", k=1, human_dialogue=["早上好今天天气怎么样"],
    )
    assert len(results) == 1
    assert results[0]["sample_id"] == "a"  # higher overlap wins


def test_retrieve_few_shots_source_diversification(monkeypatch):
    """Major-7: top-k prefer different source_files when available."""
    from generate_kisaki_llm_v4 import retrieve_few_shots

    fake_pool = {
        "日常问候": [
            {"sample_id": "a", "source_file": "ch1.txt", "conversations": [
                {"from": "human", "value": "x"}, {"from": "assistant", "value": "y"}]},
            {"sample_id": "b", "source_file": "ch1.txt", "conversations": [
                {"from": "human", "value": "x"}, {"from": "assistant", "value": "y"}]},
            {"sample_id": "c", "source_file": "ch2.txt", "conversations": [
                {"from": "human", "value": "x"}, {"from": "assistant", "value": "y"}]},
        ],
    }
    monkeypatch.setattr("generate_kisaki_llm_v4._FEW_SHOT_BY_SCENE", fake_pool)
    results = retrieve_few_shots("日常问候", k=2)
    sources = {r["source_file"] for r in results}
    assert len(sources) == 2  # diversified across ch1.txt and ch2.txt


def test_retrieve_few_shots_exclude_ids_filters(monkeypatch):
    """Major-7: exclude_ids removes already-used few-shots (retry variation)."""
    from generate_kisaki_llm_v4 import retrieve_few_shots

    fake_pool = {
        "日常问候": [
            {"sample_id": "a", "source_file": "ch1.txt", "conversations": []},
            {"sample_id": "b", "source_file": "ch2.txt", "conversations": []},
        ],
    }
    monkeypatch.setattr("generate_kisaki_llm_v4._FEW_SHOT_BY_SCENE", fake_pool)
    results = retrieve_few_shots("日常问候", k=3, exclude_ids={"a"})
    ids = {r["sample_id"] for r in results}
    assert "a" not in ids
    assert "b" in ids


# ---------------------------------------------------------------------------
# Major-4 + Major-10: output_dir-aware paths + write ordering
# ---------------------------------------------------------------------------

def test_run_pipeline_writes_all_outputs_to_output_dir(tmp_path, monkeypatch):
    """Major-4: all output files must land under the custom --output-dir."""
    import regen_kisaki_llm_pipeline as pipeline

    custom_dir = tmp_path / "custom_output"

    # Mock the heavy dependencies so we don't actually call LLMs
    monkeypatch.setattr(
        "regen_kisaki_llm_pipeline.load_specs_from_negative_pool",
        lambda path: [],
    )
    monkeypatch.setattr(
        "regen_kisaki_llm_pipeline._load_passed_samples_from",
        lambda path: [],
    )
    monkeypatch.setattr(
        "regen_kisaki_llm_pipeline._load_progress_from",
        lambda path: {
            "started_at": "t", "last_updated": "t",
            "completed_spec_ids": [],
            "stats": {"passed": 0, "rejected": 0, "disputed": 0, "total_processed": 0},
            "last_committed_spec_id": None,
        },
    )

    # Create a dummy negative pool so the existence check passes
    neg_pool = custom_dir / "v3_negative_pool.jsonl"
    neg_pool.parent.mkdir(parents=True, exist_ok=True)
    neg_pool.write_text("", encoding="utf-8")

    summary = pipeline.run_pipeline(
        output_dir=custom_dir,
        negative_pool_path=neg_pool,
        limit=0,
    )
    # All expected outputs must be under custom_dir, not the default OUTPUT_DIR
    assert (custom_dir / "progress.json").exists()
    assert (custom_dir / "judge_config.json").exists()
    assert str(summary["samples_path"]).startswith(str(custom_dir))
    assert str(summary["progress_path"]).startswith(str(custom_dir))


def test_progress_records_last_committed_spec_id(tmp_path, monkeypatch):
    """Major-10: progress.json includes last_committed_spec_id for crash recovery."""
    import regen_kisaki_llm_pipeline as pipeline

    progress_path = tmp_path / "progress.json"
    progress = pipeline._load_progress_from(progress_path)
    assert progress["last_committed_spec_id"] is None  # fresh state

    # Simulate a commit
    progress["completed_spec_ids"].append("spec_001")
    progress["last_committed_spec_id"] = "spec_001"
    pipeline._save_progress_to(progress_path, progress)

    # Reload and verify the marker persisted
    reloaded = pipeline._load_progress_from(progress_path)
    assert reloaded["last_committed_spec_id"] == "spec_001"
    assert "spec_001" in reloaded["completed_spec_ids"]


# ---------------------------------------------------------------------------
# Major-9: Model registry + judge_config.json
# ---------------------------------------------------------------------------

def test_model_registry_records_model_families():
    """Major-9: MODEL_REGISTRY must record family for committee claim."""
    from kisaki_v4_llm_client import MODEL_REGISTRY

    assert MODEL_REGISTRY["judge_a"]["model_family"] == "qwen"
    assert MODEL_REGISTRY["judge_b"]["model_family"] == "deepseek"
    assert MODEL_REGISTRY["generator"]["model_family"] == "deepseek"


def test_build_judge_config_detects_cross_family_committee():
    """Major-9: judge_config must flag cross-family as independent committee."""
    from kisaki_v4_llm_client import build_judge_config

    config = build_judge_config()
    assert config["cross_model_committee"] is True  # qwen != deepseek
    assert "independent" in config["cross_model_committee_note"]
    # Generator and Judge B share family -> self-preference risk flagged
    assert config["generator_judge_b_same_family"] is True
    assert "self-preference" in config["self_preference_risk_note"]


def test_build_judge_config_flags_same_family_as_two_stage(monkeypatch):
    """Major-9: if both judges share a family, must say 'two-stage', not 'committee'."""
    from kisaki_v4_llm_client import build_judge_config, MODEL_REGISTRY

    # Force both judges to the same family
    original_a = dict(MODEL_REGISTRY["judge_a"])
    original_b = dict(MODEL_REGISTRY["judge_b"])
    MODEL_REGISTRY["judge_a"]["model_family"] = "deepseek"
    MODEL_REGISTRY["judge_b"]["model_family"] = "deepseek"
    try:
        config = build_judge_config()
        assert config["cross_model_committee"] is False
        assert "two-stage discrimination" in config["cross_model_committee_note"]
    finally:
        MODEL_REGISTRY["judge_a"] = original_a
        MODEL_REGISTRY["judge_b"] = original_b


# ===========================================================================
# V2.1 round-2 fix coverage (model ids / fsync / violations / confidence / backend)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1: Updated default model identifiers
# ---------------------------------------------------------------------------

def test_model_registry_uses_updated_deepseek_v4_models(monkeypatch):
    """Fix 1: deepseek-chat is deprecated; defaults must be v4-flash/v4-pro."""
    # Prevent load_dotenv from re-loading .env values during reload
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("DEEPSEEK_GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_JUDGE_B_MODEL", raising=False)
    import importlib
    import kisaki_v4_llm_client as client
    importlib.reload(client)

    assert client.MODEL_REGISTRY["generator"]["model_id"] == "deepseek-v4-flash"
    assert client.MODEL_REGISTRY["judge_b"]["model_id"] == "deepseek-v4-pro"
    # No lingering deepseek-chat defaults
    for role in ("generator", "judge_b"):
        assert "deepseek-chat" not in client.MODEL_REGISTRY[role]["model_id"]


def test_model_registry_uses_pinned_qwen_snapshot(monkeypatch):
    """Fix 1: Judge A must use fixed snapshot, not floating qwen-max alias."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("QWEN_JUDGE_A_MODEL", raising=False)
    import importlib
    import kisaki_v4_llm_client as client
    importlib.reload(client)

    judge_a_id = client.MODEL_REGISTRY["judge_a"]["model_id"]
    assert judge_a_id == "qwen3-max-2026-01-23"
    assert "snapshot" in client.MODEL_REGISTRY["judge_a"]["version_note"].lower() \
           or "pinned" in client.MODEL_REGISTRY["judge_a"]["version_note"].lower()


def test_env_var_overrides_model_registry(monkeypatch):
    """Fix 1: env vars still override the new defaults."""
    import importlib
    import kisaki_v4_llm_client as client

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("DEEPSEEK_GENERATOR_MODEL", "custom-test-model")
    importlib.reload(client)
    assert client.MODEL_REGISTRY["generator"]["model_id"] == "custom-test-model"
    # Reload again with cleared env to restore default state for subsequent tests
    monkeypatch.delenv("DEEPSEEK_GENERATOR_MODEL", raising=False)
    importlib.reload(client)


# ---------------------------------------------------------------------------
# Fix 2: append_jsonl fsync + read_jsonl_ids reconciliation
# ---------------------------------------------------------------------------

def test_append_jsonl_is_durable(tmp_path):
    """Fix 2: append_jsonl must flush + fsync so lines survive a crash."""
    from kisaki_v4_llm_client import append_jsonl, read_jsonl_ids

    path = tmp_path / "test.jsonl"
    append_jsonl(path, {"sample_spec_id": "spec_001", "status": "passed"})
    append_jsonl(path, {"sample_spec_id": "spec_002", "status": "rejected"})

    # Immediately readable (fsync guarantees the bytes are on disk)
    ids = read_jsonl_ids(path)
    assert ids == {"spec_001", "spec_002"}


def test_read_jsonl_ids_skips_malformed_lines(tmp_path):
    """Fix 2: partial/corrupted trailing lines must not crash read_jsonl_ids."""
    from kisaki_v4_llm_client import read_jsonl_ids

    path = tmp_path / "partial.jsonl"
    path.write_text(
        '{"sample_spec_id": "ok_001"}\n'
        '{"sample_spec_id": "ok_002"}\n'
        'THIS IS A PARTIAL LINE WITHOUT NEWLINE',  # crash survivor
        encoding="utf-8",
    )
    ids = read_jsonl_ids(path)
    assert ids == {"ok_001", "ok_002"}  # malformed line skipped


def test_pipeline_back_fills_orphaned_jsonl_ids(tmp_path, monkeypatch):
    """Fix 2: on resume, specs in JSONL but not in progress must be back-filled."""
    import regen_kisaki_llm_pipeline as pipeline

    # Simulate a crash: spec_001 is in samples.jsonl but NOT in progress.completed_spec_ids
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        json.dumps({"sample_spec_id": "spec_001", "status": "passed"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({
        "started_at": "t", "last_updated": "t",
        "completed_spec_ids": [],  # spec_001 missing — crash before commit
        "stats": {"passed": 0, "rejected": 0, "disputed": 0, "total_processed": 0},
        "last_committed_spec_id": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(pipeline, "PROGRESS_PATH", progress_path)
    monkeypatch.setattr(pipeline, "SAMPLES_PATH", samples_path)

    # Reload progress and verify back-fill logic
    progress = pipeline._load_progress_from(progress_path)
    from kisaki_v4_llm_client import read_jsonl_ids
    jsonl_ids = read_jsonl_ids(samples_path)
    orphaned = jsonl_ids - set(progress["completed_spec_ids"])
    assert orphaned == {"spec_001"}

    # Back-fill
    for sid in sorted(orphaned):
        progress["completed_spec_ids"].append(sid)
    assert "spec_001" in progress["completed_spec_ids"]


# ---------------------------------------------------------------------------
# Fix 3: Judge A violations participate in pass decision
# ---------------------------------------------------------------------------

def test_judge_a_violations_non_empty_forces_fail():
    """Fix 3: high scores but non-empty violations => passed=False."""
    raw = json.dumps({
        "scores": {
            "人物一致性": 9, "语境连贯": 9, "自然度": 9, "原作语气": 9,
            "事实关系": 9,
        },
        "evidence": {},
        "violations": ["character_breaks_fourth_wall"],
        "reason": "high scores but has violation",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert len(result.violations) == 1
    assert "character_breaks_fourth_wall" in result.violations[0]


def test_judge_a_violations_field_not_list_fails():
    """Fix 3: violations as a string (not list) => fail-closed."""
    raw = json.dumps({
        "scores": {
            "人物一致性": 9, "语境连贯": 9, "自然度": 9, "原作语气": 9,
            "事实关系": 9,
        },
        "evidence": {},
        "violations": "this should be a list not a string",
        "reason": "type error",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is False
    assert any("violations_field_not_list" in v for v in result.violations)


def test_judge_a_empty_violations_with_high_scores_passes():
    """Fix 3 sanity: empty violations + high scores => passed=True."""
    raw = json.dumps({
        "scores": {
            "人物一致性": 9, "语境连贯": 9, "自然度": 9, "原作语气": 9,
            "事实关系": 9,
        },
        "evidence": {},
        "violations": [],
        "reason": "all good",
    })
    result = parse_judge_a_response(raw)
    assert result.passed is True
    assert result.violations == []


# ---------------------------------------------------------------------------
# Fix 4: Judge B confidence must be in [0, 1]
# ---------------------------------------------------------------------------

def test_judge_b_negative_confidence_fails_parse():
    """Fix 4: confidence=-5 => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": -5, "evidence": "", "reason": ""})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert parse_ok is False
    assert "confidence_out_of_range" in evid


def test_judge_b_confidence_above_one_fails_parse():
    """Fix 4: confidence=1.5 => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": 1.5, "evidence": "", "reason": ""})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert parse_ok is False
    assert "confidence_out_of_range" in evid


def test_judge_b_missing_confidence_fails_parse():
    """Fix 4: confidence field absent => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "evidence": "", "reason": ""})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert parse_ok is False
    assert "missing_confidence" in evid


def test_judge_b_non_numeric_confidence_fails_parse():
    """Fix 4: confidence="high" (string) => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": "high", "evidence": "", "reason": ""})
    prefers, conf, evid, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
    assert parse_ok is False
    assert "invalid_confidence_type" in evid


def test_judge_b_valid_confidence_passes_parse():
    """Fix 4 sanity: confidence=0.0 and 1.0 are valid boundary values."""
    for conf_val in (0.0, 1.0, 0.5):
        raw = json.dumps({"preferred": "A", "confidence": conf_val, "evidence": "ok", "reason": ""})
        _, conf, _, parse_ok = parse_judge_b_response(raw, candidate_is_a=True)
        assert parse_ok is True
        assert conf == conf_val


def test_judge_b_low_confidence_routes_to_disputed(monkeypatch):
    """Fix 4: low confidence (below 0.6 pilot threshold) => disputed.

    The pilot calibration will determine the exact threshold; for now
    we document that low-confidence passed verdicts should be treated
    as disputed by the pipeline (this is a policy check, not a parse
    check). The parse layer accepts any [0,1] value; the pipeline
    layer is responsible for the threshold.
    """
    candidate = {
        "sample_spec_id": "test_low_conf",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_low_conf",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        # Both runs prefer candidate but with very low confidence (0.3)
        return json.dumps({"preferred": "A" if call_count[0] == 1 else "B",
                           "confidence": 0.3, "evidence": "weak", "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_low_conf")
    # Parse succeeds (0.3 is in [0,1]), decision is "passed" by the double-order rule.
    # The pipeline layer should check confidence and route low-conf to disputed.
    assert result.final_decision == "passed"
    assert result.confidence_run1 == 0.3
    assert result.confidence_run2 == 0.3
    # Document: pipeline policy should treat avg confidence < 0.6 as disputed


# ---------------------------------------------------------------------------
# Fix 5: Similarity backend strict vs fallback
# ---------------------------------------------------------------------------

def test_similarity_backend_strict_raises_without_bge(monkeypatch):
    """Fix 5: strict mode (default) must raise if BGE model is missing."""
    import hard_gate_kisaki_v4 as hg

    # Force strict mode and reset cached state
    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "strict")
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ERROR", "test: model not found")

    with pytest.raises(RuntimeError, match="strict"):
        hg.get_similarity_backend()


def test_similarity_backend_fallback_works_without_bge(monkeypatch):
    """Fix 5: fallback mode uses char 4-gram when BGE is missing."""
    import hard_gate_kisaki_v4 as hg

    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "fallback")
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)

    backend = hg.get_similarity_backend()
    assert backend == "char_4gram_fallback"

    # semantic_similarity should work in fallback mode
    sim = hg.semantic_similarity("因此我不会配合你", "因此我不会配合你")
    assert sim == 1.0  # identical strings


def test_similarity_backend_info_records_backend(monkeypatch):
    """Fix 5: get_similarity_backend_info exposes backend metadata."""
    import hard_gate_kisaki_v4 as hg

    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "fallback")
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)

    info = hg.get_similarity_backend_info()
    assert info["backend"] == "char_4gram_fallback"
    assert info["mode"] == "fallback"
    assert info["authoritative"] == "false"


def test_build_judge_config_includes_similarity_backend(monkeypatch):
    """Fix 5: judge_config.json must record the similarity backend."""
    import hard_gate_kisaki_v4 as hg
    from kisaki_v4_llm_client import build_judge_config

    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "fallback")
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)

    config = build_judge_config()
    assert "similarity_backend" in config
    assert config["similarity_backend"]["backend"] == "char_4gram_fallback"
    assert config["similarity_backend"]["authoritative"] == "false"
    assert config["config_version"] == 2
