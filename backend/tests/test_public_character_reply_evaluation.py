import json

from scripts.evaluate_public_character_replies import evaluate_public, public_blind_packet

from evaluation.charactereval_adapter import adapt_character_case


async def test_public_profile_uses_same_pipeline_without_metric_or_id_leakage(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_MEMORY_SELECTION_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_POLICY_ENABLED", "false")
    case = adapt_character_case(
        {"id": "hidden_case_marker", "role": "角色甲", "novel_name": "hidden_book_marker", "context": "角色乙：你好。"},
        {"角色甲": {"姓名": "角色甲", "人物性格": "克制"}},
        {"hidden_case_marker": [["hidden_metric_marker", "仅评审使用"]]},
    )
    calls = []

    async def reviewer(messages):
        calls.append(messages)
        serialized = json.dumps(messages, ensure_ascii=False)
        assert "hidden_metric_marker" not in serialized
        assert "hidden_case_marker" not in serialized
        assert "hidden_book_marker" not in serialized
        assert "月社妃" not in serialized
        if "state 必须包含" in messages[0]["content"]:
            return "invalid"
        assert "角色甲" in messages[0]["content"]
        assert "人物性格" in messages[0]["content"]
        return "嗯，来得正好。"

    report = await evaluate_public([case], reviewer)
    assert report["summary"]["responses"] == 2
    assert report["summary"]["generation_failures"] == 0
    assert report["summary"]["human_reviews_completed"] == 0
    assert report["benchmark_metadata"][0]["metric_ids"] == ["hidden_metric_marker"]
    assert len(calls) >= 3
    packet, key = public_blind_packet([case], report)
    assert set(packet[0]["outputs"]) == {"A", "B"}
    assert "query" not in packet[0] and "history" not in packet[0]
    assert "角色乙：你好" not in json.dumps(packet, ensure_ascii=False)
    assert {row["arm"] for row in key} == {"rules", "semantic_all"}
    assert "semantic_all" not in json.dumps(packet, ensure_ascii=False)
