"""Retrieval configuration errors must not silently change retrieval requirements."""
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from character.memory_service import CharacterMemoryService


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), "bad", True])
def test_invalid_score_fails_before_any_dependency_use(value):
    repository = Mock()
    embedding = Mock()
    with pytest.raises(ValueError, match="min_hybrid_score.*finite number between 0 and 1"):
        CharacterMemoryService(repository, embedding_provider=embedding, min_hybrid_score=value)
    assert repository.mock_calls == embedding.mock_calls == []


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_invalid_candidate_limit_fails_before_any_dependency_use(value):
    repository = Mock()
    with pytest.raises(ValueError, match="candidate_limit.*positive integer"):
        CharacterMemoryService(repository, candidate_limit=value)
    assert repository.mock_calls == []


@pytest.mark.parametrize("score,limit", [(0.0, None), (1.0, 1), (0.35, 100)])
async def test_valid_boundaries_preserve_requested_read_limit(score, limit):
    class Repository:
        async def list_memory_records(self, character_id, user_scope, *, limit, include_inactive):
            reads.append(limit)
            return []
    reads = []
    service = CharacterMemoryService(Repository(), semantic_enabled=False, min_hybrid_score=score, candidate_limit=limit)
    selected, count = await service.load_relevant_memories("test", None, "红茶偏好")
    assert (selected, count) == ((), 0)
    assert reads == [limit]


@pytest.mark.parametrize("name", ["MIN_HYBRID_MEMORY_SCORE", "MIN_MEMORY_CLAIM_CONFIDENCE"])
@pytest.mark.parametrize("value", ["NaN", "1.2", "-0.2", "invalid"])
def test_invalid_environment_threshold_stops_import(name, value):
    env = {**os.environ, name: value}
    result = subprocess.run([sys.executable, "-c", "import character.memory_service"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert f"{name} must be a finite number between 0 and 1" in result.stderr


def test_environment_boundaries_are_preserved():
    env = {**os.environ, "MIN_HYBRID_MEMORY_SCORE": "0", "MIN_MEMORY_CLAIM_CONFIDENCE": "1"}
    result = subprocess.run([sys.executable, "-c", "from character.memory_service import MIN_HYBRID_MEMORY_SCORE, MIN_CLAIM_CONFIDENCE; assert (MIN_HYBRID_MEMORY_SCORE, MIN_CLAIM_CONFIDENCE) == (0, 1)"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
