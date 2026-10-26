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
    concurrent(encoder._load_model)
    assert tokenizer.call_count == model.call_count == 1
    assert encoder.model is not None and encoder.tokenizer is not None and encoder.model.eval_called



def test_private_memory_and_public_indexes_share_one_model(tmp_path, monkeypatch):
    import numpy as np

    from character.memory_service import CharacterMemoryService
    from knowledge.retrieval_core import embedding
    from knowledge.vector_db import IndexConfig, VectorDatabase

    gate = Barrier(8)

    def load(*args, **kwargs):
        overlap(gate)
        return object()

    constructor = Mock(side_effect=load)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(embedding, "resolve_local_model_path", lambda *args: "synthetic-shared-model")
    monkeypatch.setattr(embedding, "_cuda_available", lambda: False)
    provider = embedding.SentenceTransformerEmbeddingProvider()
    monkeypatch.setattr(embedding, "_default_embedding_provider", provider)
    encode = lambda model, texts, **kwargs: np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(embedding, "sentence_transformer_embeddings", encode)
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda self: False)
    indexes = [VectorDatabase(str(tmp_path / str(i)), IndexConfig(index_type="flat")) for i in range(2)]
    memory = CharacterMemoryService(None, semantic_enabled=True)
    records = [{"id": "drink", "content": "Only drink tea on Wednesday after work; drink water when traveling.",
                "updated_at": "2026-10-26T00:00:00+00:00"}]
    contents = ["Public fee is 68; appointment does not waive originals.",
                "Public service is free; verified same-day receipt waives originals."]
    start = Barrier(8)

    def invoke(index):
        start.wait(timeout=5)
        if index % 3 == 2:
            return memory._semantic_similarities("Wednesday tea and travel exception", records)
        indexes[index % 3].add_documents([dict(id=str(index), title="Complete public rule", content=contents[index % 3])])
        return "public_document_saved"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(8)))
    assert constructor.call_count == 1
    constructor.assert_called_once_with("synthetic-shared-model", device="cpu")
    assert memory._embedding_provider is embedding.get_default_embedding_provider() is provider
    assert [index.index.ntotal for index in indexes] == [3, 3]
    for index, result in enumerate(results):
        assert result == ({0: 1.0} if index % 3 == 2 else "public_document_saved")
