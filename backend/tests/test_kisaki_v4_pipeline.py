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
    """Minor-5: force ALL tests to run without a real BGE model.

    Previously this fixture only set env=fallback and reset cached
    state, but left ``_load_embedder`` pointing at the real loader. On
    a host where BGE IS installed, ``_load_embedder`` would succeed,
    the backend would resolve to ``bge_embedding``, and every test that
    asserted ``char_4gram_fallback`` would fail. Tests that need to
    exercise the strict-mode raise path (e.g.
    ``test_similarity_backend_strict_raises_without_bge``) install
    their own ``_load_embedder`` mock on top of this one.
    """
    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "fallback")
    # Reset the cached backend resolution so the env var takes effect
    import hard_gate_kisaki_v4 as hg
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)

    # Minor-5: mock the loader so it never touches the real BGE model.
    # Returns None and records a load error, exactly as a missing-model
    # scenario would. Tests that need a successful load must override
    # ``_load_embedder`` themselves.
    def _fake_load_embedder():
        monkeypatch.setattr(hg, "_EMBEDDER", None)
        monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", True)
        monkeypatch.setattr(hg, "_EMBEDDER_MODEL_PATH", "./models/bge-small-zh-v1.5")
        monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ERROR", "test: model not available")
        return None

    monkeypatch.setattr(hg, "_load_embedder", _fake_load_embedder)


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

# Major-2: parse_judge_b_response now requires scores.A and scores.B (4
# dims each, 0-10) and returns a JudgeBParsed object. These constants
# provide valid score sets for various test scenarios.
_SCORES_A_HIGHER = {
    "A": {"人物一致性": 8, "原作语气": 8, "元叙事控制": 8, "自然度": 8},
    "B": {"人物一致性": 6, "原作语气": 6, "元叙事控制": 6, "自然度": 6},
}
_SCORES_B_HIGHER = {
    "A": {"人物一致性": 6, "原作语气": 6, "元叙事控制": 6, "自然度": 6},
    "B": {"人物一致性": 8, "原作语气": 8, "元叙事控制": 8, "自然度": 8},
}
_SCORES_TIE = {
    "A": {"人物一致性": 7, "原作语气": 7, "元叙事控制": 7, "自然度": 7},
    "B": {"人物一致性": 7, "原作语气": 7, "元叙事控制": 7, "自然度": 7},
}


def test_judge_b_parse_run1_candidate_is_a_prefers_a():
    # candidate is A, judge says "A" -> prefers candidate
    raw = json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A更好",
                      "scores": _SCORES_A_HIGHER, "reason": "..."})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is True
    assert p.prefers_candidate is True
    assert p.confidence == 0.9


def test_judge_b_parse_run2_candidate_is_b_prefers_b():
    # candidate is B, judge says "B" -> prefers candidate
    raw = json.dumps({"preferred": "B", "confidence": 0.8, "evidence": "B更好",
                      "scores": _SCORES_B_HIGHER, "reason": "..."})
    p = parse_judge_b_response(raw, candidate_is_a=False)
    assert p.parse_ok is True
    assert p.prefers_candidate is True


def test_judge_b_parse_failure_returns_parse_ok_false():
    """Major-6: JSON parse failure surfaces as parse_ok=False (not silent reject)."""
    p = parse_judge_b_response("not json", candidate_is_a=True)
    assert p.parse_ok is False
    assert p.prefers_candidate is False
    assert "json_parse_error" in p.evidence


def test_judge_b_parse_invalid_preferred_value_returns_parse_ok_false():
    """Major-6: invalid 'preferred' value (not A/B) -> parse_ok=False."""
    raw = json.dumps({"preferred": "C", "confidence": 0.5, "evidence": "", "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is False
    assert "invalid_preferred_value" in p.evidence


def test_judge_b_parse_tie_returns_is_tie_true():
    """Position-bias fix: 'tie' is a first-class verdict.

    parse_ok=True (valid parse), is_tie=True (no preference), and
    prefers_candidate=False (a tie does not favour the candidate).
    """
    raw = json.dumps({"preferred": "tie", "confidence": 0.5, "evidence": "两者接近",
                      "scores": _SCORES_TIE, "reason": "..."})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is True
    assert p.is_tie is True
    assert p.prefers_candidate is False


def test_judge_b_tie_in_run1_score_derived_passed(monkeypatch):
    """Major-3: tie no longer auto-routes to disputed. Decision is derived
    from averaged scores. When Run1 is tie (equal scores) but Run2's
    scores make the candidate clearly better on average, the sample passes.

    Run1: tie, _SCORES_TIE (A=28, B=28) → candidate=A gets 28
    Run2: prefers B(candidate), _SCORES_B_HIGHER (A=24, B=32) → candidate=B gets 32
    candidate_avg = (28+32)/2 = 30, negative_avg = (28+24)/2 = 26
    gap = 4 > JUDGE_B_SCORE_DERIVED_TIE_GAP(1.0) → passed
    """
    candidate = {
        "sample_spec_id": "test_tie1",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_tie1",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Run 1: tie (valid tie scores: close, both totals >= 12)
            return json.dumps({"preferred": "tie", "confidence": 0.7, "evidence": "接近",
                               "scores": _SCORES_TIE, "reason": "..."})
        else:
            # Run 2: prefers candidate (B), B scores higher
            return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B",
                               "scores": _SCORES_B_HIGHER, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_tie1")
    # Major-3: tie doesn't force disputed; scores decide
    assert result.final_decision == "passed"
    assert result.is_tie_run1 is True
    assert result.is_tie_run2 is False
    # Score-derived telemetry
    assert result.score_derived_gap > 1.0
    assert result.candidate_avg_scores  # non-empty


def test_judge_b_first_position_preference_telemetry(monkeypatch):
    """Position-bias telemetry: first_position_a_runN flags when judge
    preferred whichever response was shown as 'A'.

    Major-3: with score-derived decisions, position bias in ``preferred``
    no longer determines the verdict. But the telemetry still flags it
    for diagnostics. When the judge gives position-A higher scores in
    BOTH runs, the averaged scores cancel out → gap=0 → disputed.

    Run 1: candidate=A, _SCORES_A_HIGHER → candidate gets A's 32, negative gets B's 24
    Run 2: negative=A, candidate=B, _SCORES_A_HIGHER → candidate gets B's 24, negative gets A's 32
    candidate_avg = (32+24)/2 = 28, negative_avg = (24+32)/2 = 28
    gap = 0 → disputed (within tie_gap)
    """
    candidate = {
        "sample_spec_id": "test_pos",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_pos",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        # Both runs: judge always prefers A (position bias)
        if call_count[0] == 1:
            # Run 1: candidate=A, judge prefers A => prefers candidate
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
        else:
            # Run 2: negative=A, judge prefers A => prefers negative (not candidate)
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_pos")
    # Both runs chose position A — telemetry flags the bias
    assert result.first_position_a_run1 is True
    assert result.first_position_a_run2 is True
    # Major-3: averaged scores cancel the bias → gap=0 → disputed
    assert result.final_decision == "disputed"
    assert abs(result.score_derived_gap) < 0.01  # gap ≈ 0


def test_judge_b_score_derived_cancels_position_bias(monkeypatch):
    """Major-3 core test: position-biased scores that would cause a
    disputed verdict under the old logic now produce a correct verdict
    via score averaging.

    Scenario: candidate is genuinely better. But the judge has position
    bias — it always inflates position A's scores by 2 points per dim.

    Run 1: candidate=A (inflated), negative=B (deflated)
      scores: A={9,9,9,9}=36, B={7,7,7,7}=28
    Run 2: candidate=B (deflated), negative=A (inflated)
      scores: A={9,9,9,9}=36, B={7,7,7,7}=28
      (negative is at A=36, candidate is at B=28)

    Without averaging: Run1 prefers A(candidate)=passed, Run2 prefers
    A(negative)=rejected → disputed (old behavior).

    With averaging:
      candidate_avg = (36 + 28) / 2 = 32
      negative_avg  = (28 + 36) / 2 = 32
      gap = 0 → disputed

    Hmm — if the bias is perfectly symmetric, averaging gives gap=0.
    Let's instead test the case where candidate is genuinely better by
    2 points per dim, and the judge adds a +2 position bias to A:

    Run 1: candidate=A gets genuine(8)+bias(2)=10 per dim, negative=B gets genuine(6)+0=6
      scores: A={10,10,10,10}=40, B={6,6,6,6}=24
    Run 2: candidate=B gets genuine(8)+0=8, negative=A gets genuine(6)+bias(2)=8
      scores: A={8,8,8,8}=32, B={8,8,8,8}=32

    candidate_avg = (40 + 32) / 2 = 36
    negative_avg  = (24 + 32) / 2 = 28
    gap = 8 > 1 → passed

    Under old logic: Run1 prefers A(candidate), Run2 prefers A(negative,
    since A=32 >= B=32 but judge picks A) → disputed.
    Under new logic: passed (correct — candidate is genuinely better).
    """
    candidate = {
        "sample_spec_id": "test_bias_cancel",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此，没有那个必要。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_bias_cancel",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此，故事就是这样。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Run 1: candidate=A (inflated by +2 bias), negative=B
            return json.dumps({
                "preferred": "A", "confidence": 0.9, "evidence": "A",
                "scores": {"A": {"人物一致性": 10, "原作语气": 10, "元叙事控制": 10, "自然度": 10},
                           "B": {"人物一致性": 6, "原作语气": 6, "元叙事控制": 6, "自然度": 6}},
                "reason": "..."
            })
        else:
            # Run 2: negative=A (inflated by +2 bias), candidate=B
            # Judge still prefers A (position bias), but scores are closer
            return json.dumps({
                "preferred": "A", "confidence": 0.9, "evidence": "A",
                "scores": {"A": {"人物一致性": 8, "原作语气": 8, "元叙事控制": 8, "自然度": 8},
                           "B": {"人物一致性": 8, "原作语气": 8, "元叙事控制": 8, "自然度": 8}},
                "reason": "..."
            })
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_bias_cancel")
    # Old logic: disputed (Run1 prefers candidate, Run2 prefers negative)
    # New logic: passed (averaged scores show candidate is better)
    assert result.prefers_candidate_run1 is True   # Run1: A=candidate
    assert result.prefers_candidate_run2 is False  # Run2: A=negative
    assert result.final_decision == "passed"
    assert result.score_derived_gap > 1.0


def test_judge_b_score_derived_tie_gap_boundary(monkeypatch):
    """Major-3: when averaged scores are within TIE_GAP (1.0), the
    verdict is disputed (genuinely close, not a bias artifact)."""
    candidate = {
        "sample_spec_id": "test_tie_gap",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_tie_gap",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    # Both runs: candidate and negative get nearly equal scores
    # candidate_avg = (28+28)/2 = 28, negative_avg = (28+28)/2 = 28
    # gap = 0 → disputed
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        return json.dumps({
            "preferred": "A", "confidence": 0.9, "evidence": "A",
            "scores": _SCORES_TIE, "reason": "..."
        })
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_tie_gap")
    assert result.final_decision == "disputed"
    assert abs(result.score_derived_gap) < 1.0


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
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
        else:
            return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B",
                               "scores": _SCORES_B_HIGHER, "reason": "..."})
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
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
        else:
            # Run 2: negative=A, candidate=B; judge prefers A(=negative) -> not candidate
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
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
            return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B",
                               "scores": _SCORES_B_HIGHER, "reason": "..."})
        else:
            # Run 2: negative=A, candidate=B; judge prefers A(=negative)
            return json.dumps({"preferred": "A", "confidence": 0.9, "evidence": "A",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
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




# ---------------------------------------------------------------------------
# B.8.7: Generator prompt construction
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# B.8.8: Resume from progress.json (completed specs are skipped)
# ---------------------------------------------------------------------------







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
        return json.dumps({"preferred": "B", "confidence": 0.9, "evidence": "B",
                           "scores": _SCORES_B_HIGHER, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_parse_fail")
    assert result.final_decision == "disputed"


# ---------------------------------------------------------------------------
# Major-7: Few-shot relevance retrieval (keyword overlap + source diversity)
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Major-4 + Major-10: output_dir-aware paths + write ordering
# ---------------------------------------------------------------------------





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

# Major-2: parse_judge_b_response now requires scores.A and scores.B (4
# dims each, 0-10). Tests that only exercise confidence/preferred must
# still supply valid scores or parse will fail with scores_* errors.
_VALID_SCORES = {
    "A": {"人物一致性": 8, "原作语气": 8, "元叙事控制": 8, "自然度": 8},
    "B": {"人物一致性": 6, "原作语气": 6, "元叙事控制": 6, "自然度": 6},
}


def test_judge_b_negative_confidence_fails_parse():
    """Fix 4: confidence=-5 => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": -5, "evidence": "",
                      "scores": _VALID_SCORES, "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is False
    assert "confidence_out_of_range" in p.evidence


def test_judge_b_confidence_above_one_fails_parse():
    """Fix 4: confidence=1.5 => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": 1.5, "evidence": "",
                      "scores": _VALID_SCORES, "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is False
    assert "confidence_out_of_range" in p.evidence


def test_judge_b_missing_confidence_fails_parse():
    """Fix 4: confidence field absent => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "evidence": "",
                      "scores": _VALID_SCORES, "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is False
    assert "missing_confidence" in p.evidence


def test_judge_b_non_numeric_confidence_fails_parse():
    """Fix 4: confidence="high" (string) => parse_ok=False."""
    raw = json.dumps({"preferred": "A", "confidence": "high", "evidence": "",
                      "scores": _VALID_SCORES, "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is False
    assert "invalid_confidence_type" in p.evidence


def test_judge_b_valid_confidence_passes_parse():
    """Fix 4 sanity: confidence=0.0 and 1.0 are valid boundary values.

    Note: confidence=0.0 and 0.5 are below the pilot threshold 0.6, so
    low_confidence=True; confidence=1.0 is above. parse_ok is True for
    all three (threshold does not fail parse, only routes to disputed).
    """
    for conf_val in (0.0, 1.0, 0.5):
        raw = json.dumps({"preferred": "A", "confidence": conf_val, "evidence": "ok",
                          "scores": _VALID_SCORES, "reason": ""})
        p = parse_judge_b_response(raw, candidate_is_a=True)
        assert p.parse_ok is True
        assert p.confidence == conf_val
    # Boundary: 0.6 exactly is NOT low-confidence (< is strict).
    raw = json.dumps({"preferred": "A", "confidence": 0.6, "evidence": "ok",
                      "scores": _VALID_SCORES, "reason": ""})
    p = parse_judge_b_response(raw, candidate_is_a=True)
    assert p.parse_ok is True
    assert p.low_confidence is False


def test_judge_b_low_confidence_routes_to_disputed(monkeypatch):
    """Major-1: low confidence (below 0.6 pilot threshold) => disputed.

    Both runs prefer the candidate, but confidence=0.3 < 0.6 on both
    runs. parse_ok=True (0.3 is in [0,1]), scores are valid, no
    contradiction — but low_confidence=True on both runs, so the
    judge_b() decision logic routes to disputed instead of passed.
    The threshold can only be re-tuned after 30-sample calibration.
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
        if call_count[0] == 1:
            return json.dumps({"preferred": "A", "confidence": 0.3, "evidence": "weak",
                               "scores": _SCORES_A_HIGHER, "reason": "..."})
        else:
            return json.dumps({"preferred": "B", "confidence": 0.3, "evidence": "weak",
                               "scores": _SCORES_B_HIGHER, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_low_conf")
    # Major-1: low confidence routes to disputed, NOT passed
    assert result.final_decision == "disputed"
    assert result.confidence_run1 == 0.3
    assert result.confidence_run2 == 0.3
    assert result.low_confidence_run1 is True
    assert result.low_confidence_run2 is True


# ---------------------------------------------------------------------------
# Fix 5: Similarity backend strict vs fallback
# ---------------------------------------------------------------------------

def test_similarity_backend_strict_raises_without_bge(monkeypatch):
    """Fix 5: strict mode (default) must raise if BGE model is missing.

    Minor-5 fix: this test must NOT depend on whether the host machine
    has a real BGE model installed. Previously it reset
    ``_EMBEDDER_LOAD_ATTEMPTED=False`` and let ``_load_embedder`` run for
    real — on a machine with BGE installed the embedder would load
    successfully, strict mode would NOT raise, and the test would fail.
    Now we monkeypatch ``_load_embedder`` to return None and record a
    load error, so the strict-mode raise path is exercised regardless
    of the host environment.
    """
    import hard_gate_kisaki_v4 as hg

    # Force strict mode and reset cached state
    monkeypatch.setenv("KISAKI_SIMILARITY_BACKEND", "strict")
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND_RESOLVED", False)
    monkeypatch.setattr(hg, "_SIMILARITY_BACKEND", "")
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(hg, "_EMBEDDER", None)
    monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ERROR", None)

    # Minor-5: force the embedder load to fail without touching the real
    # model. This keeps the test isolated from the host environment.
    def _fake_load_embedder():
        monkeypatch.setattr(hg, "_EMBEDDER", None)
        monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ATTEMPTED", True)
        monkeypatch.setattr(hg, "_EMBEDDER_MODEL_PATH", "./models/bge-small-zh-v1.5")
        monkeypatch.setattr(hg, "_EMBEDDER_LOAD_ERROR", "test: model not available")
        return None

    monkeypatch.setattr(hg, "_load_embedder", _fake_load_embedder)

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


# ---------------------------------------------------------------------------
# Minor: regression tests for V2.1 contract fixes
# (score contradiction, quota trim bug, interleave, acceptance_check,
#  immutable manifest)
# ---------------------------------------------------------------------------

# Scores where B is clearly higher but preferred=A — a contradiction.
_SCORES_CONTRADICTORY_A = {
    "A": {"人物一致性": 3, "原作语气": 3, "元叙事控制": 3, "自然度": 3},
    "B": {"人物一致性": 9, "原作语气": 9, "元叙事控制": 9, "自然度": 9},
}

# Tie claimed but one dim differs by >2 — not a real tie.
_SCORES_FALSE_TIE = {
    "A": {"人物一致性": 7, "原作语气": 7, "元叙事控制": 7, "自然度": 7},
    "B": {"人物一致性": 7, "原作语气": 7, "元叙事控制": 7, "自然度": 2},
}


def test_judge_b_score_contradiction_routes_to_disputed(monkeypatch):
    """Major-2: preferred=A but B scores much higher => disputed."""
    candidate = {
        "sample_spec_id": "test_contradict",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_contradict",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        # Both runs prefer A (candidate is A in run1, B in run2) but
        # the other side's scores are much higher.
        if call_count[0] == 1:
            # Run 1: candidate is A, preferred=A, but B scores higher
            return json.dumps({"preferred": "A", "confidence": 0.8,
                               "evidence": "矛盾",
                               "scores": _SCORES_CONTRADICTORY_A, "reason": "..."})
        else:
            # Run 2: candidate is B, preferred=A (negative), but B scores higher
            # Swap A/B so the contradiction persists
            swapped = {"A": _SCORES_CONTRADICTORY_A["B"],
                       "B": _SCORES_CONTRADICTORY_A["A"]}
            return json.dumps({"preferred": "A", "confidence": 0.8,
                               "evidence": "矛盾",
                               "scores": swapped, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_contradict")
    assert result.final_decision == "disputed"
    assert bool(result.score_contradiction_run1) is True


def test_judge_b_false_tie_routes_to_disputed(monkeypatch):
    """Major-2: preferred=tie but a single dim gap > 2 => disputed."""
    candidate = {
        "sample_spec_id": "test_false_tie",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "因此。"},
        ],
    }
    negative = {
        "sample_spec_id": "test_false_tie",
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "assistant", "value": "正因如此。"},
        ],
    }
    call_count = [0]
    def mock_call(messages, **kwargs):
        call_count[0] += 1
        return json.dumps({"preferred": "tie", "confidence": 0.7,
                           "evidence": "假平局",
                           "scores": _SCORES_FALSE_TIE, "reason": "..."})
    monkeypatch.setattr("judge_kisaki_llm_v4.call_judge_b", mock_call)
    result = judge_b(candidate, negative, "test_false_tie")
    assert result.final_decision == "disputed"
    assert result.is_tie_run1 is True
    assert bool(result.score_contradiction_run1) is True


















    # processed_exactly is False (2 != 12), which is correct — the run
    # is incomplete. The point is expected_count stayed at 12 instead
    # of dropping to 0.
