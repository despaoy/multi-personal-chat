"""Simple retrieval contracts must import without loading model/index engines."""

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def run_control(source):
    return subprocess.run([sys.executable, "-I", "-c", source, str(BACKEND)],
                          capture_output=True, text=True, timeout=15)


def test_index_error_contract_does_not_import_retrieval_implementations():
    result = run_control('''
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

class NoEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("knowledge.retrieval_core.")
                and fullname != "knowledge.retrieval_core.errors"):
            raise ImportError("Unexpected retrieval engine import: " + fullname)
        if fullname.split(".")[0] in {"numpy", "jieba", "faiss", "torch", "sentence_transformers", "dotenv"}:
            raise ImportError("Unexpected runtime dependency: " + fullname)

sys.meta_path.insert(0, NoEngines())
from knowledge.retrieval_core.errors import RetrievalIndexError

error = RetrievalIndexError("vector", "not_initialized")
assert error.public_detail()["stage"] == "retrieval_index"
assert error.public_detail()["channel"] == "vector"
assert error.public_detail()["reason"] == "not_initialized"
print("isolated-error-ok")
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated-error-ok"


def test_explicit_modules_remain_available_without_package_reexports():
    result = run_control('''
import sys
sys.path.insert(0, sys.argv[1])

from knowledge.retrieval_core import embedding, query
from knowledge.retrieval_core.index import DomainIndex
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.retrieval_core.rerank import PipelineReranker

assert callable(embedding.normalized_vector)
assert callable(query.QueryAnalyzer)
assert DomainIndex("fixture").count() == 0
assert callable(KnowledgeDomainConfig)
assert PipelineReranker(cross_encoder_enabled=False).text_view == "content"
print("explicit-imports-ok")
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "explicit-imports-ok"
