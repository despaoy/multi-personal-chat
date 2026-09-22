import sys
from types import SimpleNamespace

import torch

from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider


def test_embedding_initialization_preserves_calling_thread_grad_mode(monkeypatch):
    class Model:
        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

    # Stub the actual import boundary, not a collected reference to a lazy
    # Transformers module that another test may have subsequently reloaded.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        ),
    )
    provider = LocalMeanPoolingEmbeddingProvider(model_path="unused-test-path")
    with torch.enable_grad():
        provider._load()
        assert torch.is_grad_enabled()
