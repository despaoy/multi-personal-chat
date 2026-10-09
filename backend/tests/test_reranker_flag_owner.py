"""Reranker flags share strict parsing and preserve separate opt-ins."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge import rag_helper, reranker
from knowledge.multiscale_rag import service
from knowledge.retrieval_core.rerank import PipelineReranker


@pytest.mark.parametrize("entry", ["generic", "pipeline", "character", "download"])
@pytest.mark.parametrize("value", ["true", "off", "misspelled", ""])
def test_reranker_flag_contract(monkeypatch, entry, value):
    name = {"character": "CHARACTER_RAG_RERANKER_ENABLED", "download": "RERANKER_ALLOW_DOWNLOAD"}.get(entry, "RERANKER_ENABLED")
    monkeypatch.setenv(name, value)
    encoder = object()
    factory = Mock(return_value=encoder)
    monkeypatch.setattr(rag_helper, "get_reranker", factory)
    monkeypatch.setattr(reranker, "get_reranker", factory)
    monkeypatch.setattr(service, "QueryAnalyzer", lambda configs: None)
    def resolve():
        if entry == "generic":
            return rag_helper.RAGHelper().enable_reranking
        if entry == "pipeline":
            return PipelineReranker()._resolve_cross_encoder() is encoder
        if entry == "character":
            return service.RoutedMultiScaleService(SimpleNamespace(domain_id="test"), {}, None, all_documents=[]).reranker._cross_encoder_enabled
        return reranker.RerankConfig().allow_download
    if value in {"misspelled", ""}:
        with pytest.raises(ValueError, match=name):
            resolve()
        factory.assert_not_called()
    else:
        assert resolve() is (value == "true")
        assert factory.call_count == int(value == "true" and entry in {"generic", "pipeline"})
