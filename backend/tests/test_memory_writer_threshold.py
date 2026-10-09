"""Writer candidate threshold errors fail instead of changing recall policy."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from character import memory_llm


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-0.1", "1.1", "invalid"])
def test_invalid_writer_threshold_stops_import(value):
    env = {**os.environ, "MEMORY_WRITE_SEMANTIC_THRESHOLD": value}
    result = subprocess.run([sys.executable, "-c", "import character.memory_llm"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "MEMORY_WRITE_SEMANTIC_THRESHOLD must be a finite number between 0 and 1" in result.stderr


@pytest.mark.parametrize("threshold", ["0", "1"])
def test_writer_threshold_accepts_exact_boundaries(threshold):
    env = {**os.environ, "MEMORY_WRITE_SEMANTIC_THRESHOLD": threshold}
    result = subprocess.run([sys.executable, "-c", "from character.memory_llm import _MEMORY_WRITE_SEMANTIC_THRESHOLD; assert _MEMORY_WRITE_SEMANTIC_THRESHOLD == " + threshold], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_writer_semantic_gate_keeps_related_and_excludes_unrelated(monkeypatch):
    monkeypatch.setattr(memory_llm, "_MEMORY_WRITE_SEMANTIC_THRESHOLD", 0.35)
    rows = tuple(dict(id=key, status="active", memory_type="user_fact", memory_key=key,
                     content=content, importance=0.8) for key, content in [("tea", "tea preference"), ("work", "office location")])
    provider = SimpleNamespace(embed_texts=lambda texts: np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    selected = memory_llm._search_existing_memories(rows, "drink", (), (), embedding_provider=provider)
    assert [row["id"] for row in selected] == ["tea"]
