"""Only required character-index loads may initialize the embedding provider."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.multiscale_rag import runtime
from knowledge.retrieval_core.embedding import EmbeddingModelError
from services import chat_answer


@pytest.fixture
def controlled_load(monkeypatch):
    provider = SimpleNamespace(model_id="controlled", encoding_contract="complete", _model=None)
    factory = Mock(return_value=provider)
    bundles = Mock(
        return_value=(
            [],
            np.empty((0, 384)),
            dict(
                embedding_model=provider.model_id,
                embedding_encoding=provider.encoding_contract,
            ),
        )
    )
    service = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(runtime, "LocalMeanPoolingEmbeddingProvider", factory)
    monkeypatch.setattr(runtime, "_load_bundle", bundles)
    monkeypatch.setattr(runtime, "RoutedMultiScaleService", service)
    monkeypatch.setattr(runtime, "_retrieval_index", Mock(return_value=object()))
    return SimpleNamespace(provider=provider, factory=factory, bundles=bundles, service=service)


@pytest.mark.parametrize(
    "query,options",
    [
        ("", {}),
        ("今天天气怎么样", {}),
        ("月社妃是谁？", {"filters": {"knowledge_base_id": 7}}),
        ("月社妃是谁？", {"domain_id": "other"}),
    ],
)
def test_construct_inspect_and_out_of_scope_route_do_not_initialize_models(monkeypatch, query, options):
    forbidden = Mock(side_effect=AssertionError("No model or index I/O expected"))
    monkeypatch.setattr(runtime, "LocalMeanPoolingEmbeddingProvider", forbidden)
    monkeypatch.setattr(runtime, "_load_bundle", forbidden)
    instance = runtime.MultiScaleRagRuntime()
    assert not instance.is_warm()
    assert instance.retrieve_with_citations(query, **options) is None
    assert instance._provider is None and instance._service is None and instance._stats == {}
    forbidden.assert_not_called()


def test_cold_timeout_uses_real_lazy_runtime_without_model_or_index_io(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Timeout selection must not initialize models"))
    monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(runtime, "LocalMeanPoolingEmbeddingProvider", forbidden)
    monkeypatch.setattr(runtime, "_load_bundle", forbidden)
    assert chat_answer._rag_retrieval_timeout() == max(chat_answer._RAG_TIMEOUT, chat_answer._RAG_COLD_START_TIMEOUT)
    forbidden.assert_not_called()


def test_required_retrieval_propagates_missing_model_without_publishing(monkeypatch):
    failure = EmbeddingModelError("Required model unavailable")
    factory = Mock(side_effect=failure)
    load = Mock(side_effect=AssertionError("Failed provider must stop index initialization"))
    monkeypatch.setattr(runtime, "LocalMeanPoolingEmbeddingProvider", factory)
    monkeypatch.setattr(runtime, "_load_bundle", load)
    instance = runtime.MultiScaleRagRuntime()
    with pytest.raises(EmbeddingModelError) as caught:
        instance.retrieve_with_citations("月社妃是谁？")
    assert caught.value is failure
    assert instance._provider is None and instance._service is None and instance._stats == {}
    assert not instance.is_warm()
    load.assert_not_called()


def test_lazy_load_snapshots_explicit_model_path_and_initializes_once(monkeypatch, controlled_load):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "configured-at-construction")
    instance = runtime.MultiScaleRagRuntime()
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "different-later-path")
    controlled_load.factory.assert_not_called()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda _: instance.is_available(), range(4)))
    controlled_load.factory.assert_called_once_with(model_path="configured-at-construction")
    assert controlled_load.bundles.call_count == 3
    assert instance._provider is controlled_load.provider
    assert not instance.is_warm()
    controlled_load.provider._model = object()
    assert instance.is_warm()
    assert controlled_load.bundles.call_count == 3


def test_injected_provider_is_preserved(controlled_load):
    instance = runtime.MultiScaleRagRuntime()
    instance._provider = controlled_load.provider
    assert instance.is_available()
    controlled_load.factory.assert_not_called()
    assert controlled_load.service.call_args.args[2] is controlled_load.provider


def test_failed_service_load_does_not_publish_partial_provider_and_can_retry(controlled_load):
    instance = runtime.MultiScaleRagRuntime()
    failure = ValueError("Invalid service configuration")
    controlled_load.service.side_effect = failure
    with pytest.raises(ValueError) as caught:
        instance.is_available()
    assert caught.value is failure
    assert instance._provider is None and instance._service is None and instance._stats == {}
    controlled_load.service.side_effect = None
    assert instance.is_available()
    assert controlled_load.factory.call_count == 2
