"""Concurrent callers share initialized retrieval resources; constructors are controlled."""
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_reranker import _FakeModel, _FakeTokenizer

RESOURCES = [
    ("vector_db", "_vector_db", "VectorDatabase", "get_vector_db"),
    ("rag_helper", "_rag_helper", "RAGHelper", "get_rag_helper"),
    ("reranker", "_reranker_instance", "CrossEncoderReranker", "get_reranker"),
]


def concurrent(call):
    start = Barrier(8)

    def invoke(_):
        start.wait(timeout=5)
        return call()

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(invoke, range(8)))


def overlap(gate):
    try:
        gate.wait(timeout=1)
    except BrokenBarrierError:
        pass  # Release the synthetic overlap gate when initialization is serialized.


@pytest.mark.parametrize("name, slot, constructor, getter", RESOURCES)
def test_concurrent_first_use_publishes_one_resource(monkeypatch, name, slot, constructor, getter):
    module = importlib.import_module("knowledge." + name)
    gate = Barrier(8)

    def build(*args, **kwargs):
        overlap(gate)
        return object()

    factory = Mock(side_effect=build)
    monkeypatch.setattr(module, slot, None)
    monkeypatch.setattr(module, constructor, factory)
    values = concurrent(getattr(module, getter))
    assert factory.call_count == 1 and all(value is values[0] for value in values)
    assert getattr(module, getter)() is values[0] is getattr(module, slot)


@pytest.mark.parametrize("name, slot, constructor, getter", RESOURCES)
def test_failed_initialization_preserves_error_and_releases_ownership(monkeypatch, name, slot, constructor, getter):
    module = importlib.import_module("knowledge." + name)
    failure, value = OSError("synthetic initialization failure"), object()
    factory = Mock(side_effect=[failure, value])
    monkeypatch.setattr(module, slot, None)
    monkeypatch.setattr(module, constructor, factory)
    with pytest.raises(OSError) as raised:
        getattr(module, getter)()
    assert raised.value is failure and getattr(module, slot) is None
    assert getattr(module, getter)() is value and getattr(module, getter)() is value
    assert factory.call_count == 2


def test_shared_reranker_loads_one_tokenizer_and_model(monkeypatch):
    from knowledge.reranker import CrossEncoderReranker, RerankConfig

    gate = Barrier(8)

    def load_tokenizer(*args, **kwargs):
        overlap(gate)
        return _FakeTokenizer()

    tokenizer, model = Mock(side_effect=load_tokenizer), Mock(side_effect=_FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a, **k: model())))
    encoder = CrossEncoderReranker(RerankConfig(model_name="synthetic-concurrent-model", device="cpu"))
    assert all(concurrent(encoder._load_model))
    assert tokenizer.call_count == model.call_count == 1
    assert encoder._model_loaded and encoder.tokenizer is not None and encoder.model.eval_called
