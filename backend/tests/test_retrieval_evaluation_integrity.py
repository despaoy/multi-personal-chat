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

@pytest.mark.parametrize("error", [OSError("synthetic-private-index"), TimeoutError("synthetic-private-deadline")])
def test_retrieval_failure_is_not_a_zero_score(error, caplog):
    from unittest.mock import Mock

    retrieve = Mock(side_effect=error)
    question = dict(id='complete', question='文档42规定会议2026-12-20举行，问日期对应哪份文档？',
                    expected_doc_ids=['42'], expected_doc_titles=[])
    with pytest.raises(type(error)) as caught:
        RetrievalMetrics().evaluate_questions([question], retrieve)
    assert caught.value is error
    retrieve.assert_called_once_with(question['question'])
    assert 'synthetic-private' not in caplog.text


def test_missing_dataset_questions_is_not_an_empty_dataset(tmp_path):
    from unittest.mock import Mock

    path = tmp_path / 'dataset.json'
    retrieve = Mock(side_effect=AssertionError('No question exists'))
    path.write_text(json.dumps({'metadata': 'intentionally missing questions'}))
    with pytest.raises(KeyError, match='questions'):
        RetrievalMetrics().evaluate_dataset(str(path), retrieve)
    path.write_text(json.dumps({'questions': []}))
    assert RetrievalMetrics().evaluate_dataset(str(path), retrieve)['total'] == 0
    retrieve.assert_not_called()


@pytest.fixture
def complete_evaluation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from db.database import SQLiteDB

    database = SQLiteDB(tmp_path / 'complete-evaluation.sqlite')
    database.execute_sql(
        "INSERT INTO retrieval_eval_questions(id,question,expected_doc_ids,expected_doc_titles,gold_answer,category,created_at) "
        "VALUES ('complete','文档42规定会议2026-12-20举行，问日期对应哪份文档？','[\"42\"]','[]','2026-12-20','complete','2026-10-18')"
    )
    retrieve = Mock(return_value=[dict(document_id=42, title='会议安排', content='会议2026-12-20举行。')])
    factory = Mock(return_value=SimpleNamespace(retrieve_context=retrieve))
    monkeypatch.setattr(retrieval_eval, 'db', database)
    monkeypatch.setattr(rag_helper, 'get_rag_helper', factory)
    try:
        yield SimpleNamespace(database=database, retrieve=retrieve, factory=factory)
    finally:
        database.close_connection()


@pytest.mark.parametrize('operation,field,value', [
    ('list', 'expected_doc_ids', 'not-json'),
    ('run', 'expected_doc_titles', '{"unexpected":"field"}'),
    ('run', 'expected_doc_ids', '[7]'),
])
async def test_invalid_labels_fail_before_retrieval(complete_evaluation, operation, field, value):
    control = complete_evaluation
    control.database.execute_sql(f'UPDATE retrieval_eval_questions SET {field}=:value', {'value': value})
    method = retrieval_eval.list_questions if operation == 'list' else retrieval_eval.run_retrieval_eval
    with pytest.raises(HTTPException) as caught:
        await method(current_user={'role': 'admin'})
    assert caught.value.status_code == 500 and isinstance(caught.value.__cause__, ValueError)
    control.factory.assert_not_called()
    control.retrieve.assert_not_called()


async def test_nullable_labels_and_empty_dataset_keep_normal_business_results(complete_evaluation):
    control = complete_evaluation
    control.database.execute_sql('UPDATE retrieval_eval_questions SET expected_doc_ids=NULL, expected_doc_titles=NULL')
    listed = await retrieval_eval.list_questions(current_user={'role': 'admin'})
    assert listed['questions'][0]['expected_doc_ids'] == listed['questions'][0]['expected_doc_titles'] == []
    result = await retrieval_eval.run_retrieval_eval(current_user={'role': 'admin'})
    assert result['success'] and result['total'] == 1 and result['metrics']['evaluated'] == 0
    control.factory.reset_mock()
    control.database.execute_sql('DELETE FROM retrieval_eval_questions')
    empty = await retrieval_eval.run_retrieval_eval(current_user={'role': 'admin'})
    assert empty == dict(success=True, metrics={}, total=0, note='no questions in dataset')
    control.factory.assert_not_called()


def test_completed_retrieval_without_hits_is_a_genuine_zero():
    question = dict(id='known', question='完整题设：文档42记载会议2026-12-20举行，问日期对应的资料。',
                    expected_doc_ids=['42'], expected_doc_titles=[])
    result = RetrievalMetrics().evaluate_questions([question], lambda query: [])
    assert result['evaluated'] == 1 and result['avg_recall_at_k'] == 0
    assert result['per_question'][0]['error_type'] == '' and result['per_question'][0]['retrieved_count'] == 0
