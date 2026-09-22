import copy

import pytest

from evaluation.longmemeval_adapter import adapt_case, recall_at_k, stable_subset


def _row():
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "Where do I live?",
        "question_date": "2026/01/01",
        "answer": "SECRET GOLD",
        "answer_session_ids": ["answer_private"],
        "haystack_session_ids": ["distractor", "answer_private"],
        "haystack_dates": ["2024", "2025"],
        "haystack_sessions": [
            [{"role": "assistant", "content": "You might live in Paris."}],
            [{"role": "user", "content": "I now live in London.", "has_answer": True}],
        ],
    }


def test_retrieval_docs_have_no_answers_labels_or_answer_id_prefix():
    row = _row()
    case = adapt_case(row)
    serialized = repr([doc.to_dict() for doc in case.documents])
    assert "SECRET GOLD" not in serialized
    assert "has_answer" not in serialized
    assert "answer_private" not in serialized
    assert "Speaker: assistant" in case.documents[0].content
    assert case.gold_session_ids == {"s0001"}
    assert recall_at_k(case, [0, 1], 1)["session_recall"] == 0
    assert recall_at_k(case, [1, 0], 1)["session_recall"] == 1
    assert row == _row()


def test_long_turn_chunks_cover_full_original_text():
    row = _row()
    content = "z" * 1000 + "late correction"
    row["haystack_sessions"][1][0]["content"] = content
    case = adapt_case(row, chunk_chars=150, overlap_chars=20)
    chunks = [doc for doc in case.documents if doc.metadata["session_id"] == "s0001"]
    covered = set()
    for doc in chunks:
        covered.update(range(doc.metadata["start"], doc.metadata["end"]))
    assert covered == set(range(len(content)))
    assert "late correction" in chunks[-1].content
    assert recall_at_k(case, list(range(1, len(case.documents))), 5)["turn_recall"] == 1


def test_abstention_has_no_fake_retrieval_gold():
    row = _row()
    row["question_id"] += "_abs"
    case = adapt_case(row)
    assert not case.gold_turn_ids and not case.gold_session_ids
    assert recall_at_k(case, [0, 1], 5)["session_recall"] is None


def test_token_bounded_splits_keep_all_characters_and_attribution():
    row = _row()
    content = "字" * 777 + "late correction"
    row["haystack_sessions"][1][0]["content"] = content
    case = adapt_case(row, token_counter=lambda text: len(text) * 2, max_tokens=512)
    covered = set()
    for doc in case.documents:
        assert len(doc.content) * 2 <= 512
        if doc.metadata["session_id"] == "s0001":
            covered.update(range(doc.metadata["start"], doc.metadata["end"]))
            assert "Speaker: user" in doc.content
    assert covered == set(range(len(content)))


def test_subset_is_order_independent_and_annotation_independent():
    rows = [{**_row(), "question_id": str(index)} for index in range(20)]
    selected = [row["question_id"] for row in stable_subset(rows, 3)]
    altered = copy.deepcopy(rows[::-1])
    for row in altered:
        row["answer"] = "different"
    assert [row["question_id"] for row in stable_subset(altered, 3)] == selected
    with pytest.raises(ValueError, match="duplicate"):
        stable_subset([_row(), _row()], 1)


def test_identical_repeated_sessions_preserve_dates_without_duplicate_recall_credit():
    row = _row()
    row["haystack_session_ids"].append(row["haystack_session_ids"][0])
    row["haystack_sessions"].append(copy.deepcopy(row["haystack_sessions"][0]))
    row["haystack_dates"].append("2026")
    case = adapt_case(row)
    assert len(case.documents) == 3
    assert len({doc.id for doc in case.documents}) == 3
    assert "2026" in case.documents[2].content
    assert case.repeated_session_occurrences == 1
    assert recall_at_k(case, [0, 2, 1], 2)["session_recall"] == 1
    row["haystack_sessions"][2][0]["content"] = "Different content under the same source ID"
    with pytest.raises(ValueError, match="conflicting"):
        adapt_case(row)
