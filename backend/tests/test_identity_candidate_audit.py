"""Offline auditing uses the current single index and restores its observers."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from evaluation import identity_candidate_audit as audit

from knowledge.multiscale_rag import service as service_module
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.retrieval_core.retrieval import RetrievalCandidate


def runtime_control():
    rows = []
    for i in range(4):
        kind = "relation" if i == 3 else "fact"
        body = (
            "林远与安宁登记为同事，不是亲属。" if kind == "relation" else "林远登记为虚构北潭馆档案员；雨天不受理新证。"
        )
        doc = KnowledgeIndexDocument(
            str(i),
            "synthetic",
            kind,
            "完整虚构登记",
            body,
            body,
            body,
            metadata=dict(subject="林远", target="安宁"),
            source=SourceReference("fiction.txt", 1, 1),
        )
        rows.append(RetrievalCandidate(i, doc, rerank_score=0.9 - i * 0.1))
    retriever = SimpleNamespace(search=Mock(return_value=rows))
    ranking = Mock(side_effect=lambda a, c, top_k: [replace(x, rerank_method="cross_encoder") for x in c[:top_k]])
    service = SimpleNamespace(
        retriever=retriever, reranker=SimpleNamespace(rerank=ranking), by_id={c.document.id: c.document for c in rows}
    )

    def retrieve(query, *, top_k):
        recalled = service.retriever.search(query)
        analysis = SimpleNamespace(original_query=query, normalized_query=query, entities=["林远"])
        ranked = service_module.rerank_with_title_frames(analysis, recalled, top_k=top_k, reranker=service.reranker)
        return dict(identity_scope=dict(subject="林远"), results=[dict(id=c.document.id) for c in ranked])

    service.retrieve = retrieve
    runtime = SimpleNamespace(_ensure_loaded=Mock(return_value=True), _service=service)
    return runtime, retriever.search, ranking


@pytest.mark.parametrize("enabled", [False, True])
def test_offline_audit_uses_single_retriever_and_restores_all_observers(tmp_path, monkeypatch, enabled):
    queries = tmp_path / "queries.json"
    output = tmp_path / "audit.jsonl"
    queries.write_text(json.dumps(["林远是谁？"]))
    runtime, search, rank = runtime_control()
    monkeypatch.setattr(audit, "MultiScaleRagRuntime", lambda: runtime)
    argv = ["audit", "--queries", str(queries), "--output", str(output)]
    if enabled:
        argv.append("--identity-coverage")
    monkeypatch.setattr("sys.argv", argv)
    original = service_module.rerank_with_title_frames
    audit.main()
    record = json.loads(output.read_text())
    assert record["experimental_identity_coverage"] is enabled
    assert [row["id"] for row in record["result"]["results"]] == (["0", "1", "3"] if enabled else ["0", "1", "2"])
    assert len(record["recalled"]) == 4 and len(record["ranked"]) == 4
    assert record["identity_relation_inventory"][0]["id"] == "3"
    assert runtime._service.retriever.search is search and runtime._service.reranker.rerank is rank
    assert service_module.rerank_with_title_frames is original
    runtime._ensure_loaded.assert_called_once_with()


def test_failed_load_cannot_write_a_success_audit(tmp_path, monkeypatch):
    queries = tmp_path / "queries.json"
    output = tmp_path / "audit.jsonl"
    queries.write_text(json.dumps(["林远是谁？"]))
    runtime, search, rank = runtime_control()
    runtime._ensure_loaded.return_value = False
    monkeypatch.setattr(audit, "MultiScaleRagRuntime", lambda: runtime)
    monkeypatch.setattr("sys.argv", ["audit", "--queries", str(queries), "--output", str(output)])
    with pytest.raises(RuntimeError, match="Index unavailable"):
        audit.main()
    assert not output.exists()
    search.assert_not_called()
    rank.assert_not_called()


def test_audit_failure_restores_observers_and_keeps_actual_partial_file(tmp_path, monkeypatch):
    queries = tmp_path / "queries.json"
    output = tmp_path / "audit.jsonl"
    queries.write_text(json.dumps(["林远是谁？"]))
    runtime, search, rank = runtime_control()
    error = OSError("Synthetic rank failure")
    rank.side_effect = error
    monkeypatch.setattr(audit, "MultiScaleRagRuntime", lambda: runtime)
    monkeypatch.setattr(
        "sys.argv", ["audit", "--queries", str(queries), "--output", str(output), "--identity-coverage"]
    )
    original = service_module.rerank_with_title_frames
    with pytest.raises(OSError) as caught:
        audit.main()
    assert caught.value is error and output.read_text() == ""
    assert runtime._service.retriever.search is search and runtime._service.reranker.rerank is rank
    assert service_module.rerank_with_title_frames is original
