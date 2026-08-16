from evaluation.retrieval_metrics import RetrievalMetrics
from evaluation.review_binding import bound_sample_review, structured_fact_score


def _review(response: str, judgment: bool):
    return {
        "evaluation_id": "model:rag",
        "model": "model",
        "samples": {
            "sample": {
                "sample_id": "sample",
                "response": response,
                "facts": {"琉璃没有忘记妃": judgment},
            }
        },
    }


def _score(document, response):
    supplied, status = bound_sample_review(
        document,
        evaluation_id="model:rag",
        model="model",
        sample_id="sample",
        response=response,
    )
    return status, structured_fact_score(["琉璃没有忘记妃"], supplied)


def test_affirmation_and_negation_receive_different_structured_fact_scores():
    _, affirmative = _score(_review("琉璃没有忘记妃。", True), "琉璃没有忘记妃。")
    _, negative = _score(_review("琉璃已经忘记妃。", False), "琉璃已经忘记妃。")
    assert affirmative["score"] == 1.0
    assert negative["score"] == 0.0


def test_ndcg_uses_all_expected_documents_for_ideal_ranking():
    score = RetrievalMetrics().ndcg(["doc-a"], ["doc-a", "doc-b", "doc-c"], k=3)
    assert 0.0 < score < 1.0


def test_changed_response_cannot_reuse_an_old_human_review():
    status, score = _score(_review("原回答", True), "新回答")
    assert status == "response_mismatch"
    assert score["status"] == "pending_human_review"
    assert score["score"] is None
