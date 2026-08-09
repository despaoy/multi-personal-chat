from scripts.ai_prescreen_kisaki_gold_review import (
    PROFILE_BASIS,
    SUGGESTED_REVISIONS,
    prescreen_rows,
)


def _rows():
    rows = []
    categories = ("persona", "factual", "multiturn", "safety", "rag_grounded")
    for category in categories:
        for index in range(1, 31):
            rows.append(
                {
                    "id": f"kisaki_v2_{category}_{index:03d}",
                    "category": category,
                    "prompt": f"{category}-{index}",
                    "expected_behavior": "保持角色",
                    "human_decision": "pending",
                }
            )
    return rows


def test_ai_prescreen_is_separate_from_human_decisions():
    rows, counts = prescreen_rows(_rows())
    assert counts == {"needs_revision": 30, "approved": 120}
    assert {row["human_decision"] for row in rows} == {"pending"}
    assert all(row["ai_review_basis"] == PROFILE_BASIS for row in rows)


def test_ai_prescreen_provides_concrete_profile_based_revisions():
    rows, _ = prescreen_rows(_rows())
    by_id = {row["id"]: row for row in rows}
    sibling = by_id["kisaki_v2_factual_001"]
    assert sibling["ai_suggested_decision"] == "needs_revision"
    assert "亲生哥哥" in sibling["ai_suggested_expected_behavior"]
    miracle = by_id["kisaki_v2_persona_025"]
    assert "如何看待奇迹" in miracle["ai_suggested_prompt"]
    assert "仍会作出选择和许愿" in miracle["ai_suggested_expected_behavior"]
    assert by_id["kisaki_v2_rag_grounded_030"]["ai_suggested_decision"] == "approved"
    assert by_id["kisaki_v2_safety_001"]["ai_suggested_decision"] == "approved"


def test_ai_prescreen_preserves_original_content_in_suggestion_columns_when_approved():
    rows, _ = prescreen_rows(_rows())
    approved = next(row for row in rows if row["id"] == "kisaki_v2_persona_004")
    assert approved["ai_suggested_prompt"] == approved["prompt"]
    assert approved["ai_suggested_expected_behavior"] == approved["expected_behavior"]
    assert approved["ai_suggested_category"] == approved["category"]

def test_ai_prescreen_approves_items_after_suggestions_are_applied():
    rows = _rows()
    for row in rows:
        revision = SUGGESTED_REVISIONS.get(row["id"])
        if not revision:
            continue
        for key in ("category", "prompt", "expected_behavior"):
            if revision.get(key):
                row[key] = revision[key]
    screened, counts = prescreen_rows(rows)
    assert counts == {"approved": 150}
    assert all(row["ai_suggested_decision"] == "approved" for row in screened)