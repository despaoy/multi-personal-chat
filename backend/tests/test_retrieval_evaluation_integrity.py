"""Contract tests for real retrieval evaluation metrics."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import retrieval_eval
from db.schemas import RetrievalEvalQuestionCreate
from evaluation.retrieval_metrics import RetrievalMetrics
from knowledge import rag_helper


def test_retrieval_metrics_match_document_ids_and_title_fallbacks():
    questions = [
        {
            "id": "by-id",
            "question": "id question",
            "expected_doc_ids": ["42"],
            "expected_doc_titles": [],
        },
        {
            "id": "by-title",
            "question": "title question",
            "expected_doc_ids": [],
            "expected_doc_titles": ["Reference Guide"],
        },
        {
            "id": "negative",
            "question": "no answer",
            "expected_doc_ids": [],
            "expected_doc_titles": [],
        },
    ]

    def retrieve(query):
        if query == "id question":
            return [{"id": "doc_42_chunk_0", "document_id": 42, "title": "Other"}]
        if query == "title question":
            return [{"id": "chunk", "title": "reference guide"}]
        return []

    result = RetrievalMetrics().evaluate_questions(questions, retrieve, k=5)

    assert result["mock"] is False
    assert result["total"] == 3
    assert result["evaluated"] == 2
    assert result["avg_recall_at_k"] == 1.0
    assert result["avg_mrr"] == 1.0
    assert result["per_question"][0]["match_basis"] == "document_id"
    assert result["per_question"][1]["match_basis"] == "title"
    assert result["per_question"][2]["evaluable"] is False


def test_retrieval_api_uses_structured_results_without_mock(monkeypatch):
    class Database:
        def execute_sql(self, query, params=None):
            return [{
                "id": "q1",
                "question": "where",
                "expected_doc_ids": json.dumps(["7"]),
                "expected_doc_titles": json.dumps([]),
            }]

    class Helper:
        def retrieve_context(self, query, top_k, use_cache):
            return [{"document_id": 7, "id": "doc_7_chunk_0", "title": "Doc"}]

    monkeypatch.setattr(retrieval_eval, "db", Database())
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: Helper())

    response = asyncio.run(
        retrieval_eval.run_retrieval_eval(current_user={"role": "admin"})
    )

    assert response["metrics"]["mock"] is False
    assert response["metrics"]["avg_recall_at_k"] == 1.0


def test_missing_retrieval_metrics_module_returns_503(monkeypatch):
    monkeypatch.setitem(sys.modules, "evaluation.retrieval_metrics", None)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            retrieval_eval.run_retrieval_eval(current_user={"role": "admin"})
        )

    assert exc_info.value.status_code == 503


def test_retrieval_question_payload_is_bounded():
    with pytest.raises(ValidationError):
        RetrievalEvalQuestionCreate(question="")
    with pytest.raises(ValidationError):
        RetrievalEvalQuestionCreate(question="x" * 2001)
    with pytest.raises(ValidationError):
        RetrievalEvalQuestionCreate(
            question="q",
            expected_doc_ids=[str(index) for index in range(51)],
        )