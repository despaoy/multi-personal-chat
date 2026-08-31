"""CAHM 混合语义检索、缓存和置信度门控。"""

from datetime import datetime, timezone

import numpy as np

from character.memory_service import CharacterMemoryService
from character.models import UserScope


class _Repo:
    def __init__(self, records):
        self.records = records
        self.limits = []

    async def list_memory_records(self, character_id, user_scope, limit=30):
        self.limits.append(limit)
        return self.records[:limit]


class _Embedding:
    model_id = "test-semantic"
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = {
            "我保研准备得怎么样？": (1.0, 0.0),
            "推免准备": (0.98, 0.02),
            "喜欢咖啡": (0.0, 1.0),
            "今天天气": (-1.0, 0.0),
        }
        return np.asarray([vectors.get(text, (0.0, 1.0)) for text in texts], dtype=np.float32)

    def embed_query(self, query):
        return self.embed_texts([query])[0]


def _scope():
    return UserScope("qq", "astrbot", "u1", "u1", "private")


def _row(index, key, content, importance=0.6):
    return {
        "id": index,
        "memory_key": key,
        "memory_type": "shared_event" if key.startswith("goal_") else "user_fact",
        "content": content,
        "importance": importance,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def test_synonym_retrieval_and_irrelevant_gate():
    repo = _Repo(
        [
            _row(1, "goal_推免准备", "用户正在进行或准备：推免准备", 0.7),
            _row(2, "preference_咖啡", "用户说喜欢咖啡", 0.9),
        ]
    )
    service = CharacterMemoryService(repo, embedding_provider=_Embedding(), semantic_enabled=True)

    selected, total = await service.load_relevant_memories("kisaki", _scope(), "我保研准备得怎么样？")

    assert total == 2
    assert [item.content for item in selected] == ["用户正在进行或准备：推免准备"]
    assert repo.limits == [100]


async def test_memory_embedding_cache_reuses_unchanged_vectors():
    embedding = _Embedding()
    repo = _Repo([_row(1, "goal_推免准备", "用户正在进行或准备：推免准备")])
    service = CharacterMemoryService(repo, embedding_provider=embedding, semantic_enabled=True)

    await service.load_relevant_memories("kisaki", _scope(), "我保研准备得怎么样？")
    await service.load_relevant_memories("kisaki", _scope(), "我保研准备得怎么样？")

    assert len(embedding.calls) == 2
    assert embedding.calls[0] == ["我保研准备得怎么样？", "推免准备"]
    assert embedding.calls[1] == ["我保研准备得怎么样？"]


async def test_gate_returns_empty_instead_of_filling_top_k():
    repo = _Repo([_row(1, "goal_天气", "用户正在进行或准备：今天天气")])
    service = CharacterMemoryService(
        repo,
        embedding_provider=_Embedding(),
        semantic_enabled=True,
        min_hybrid_score=0.9,
    )
    selected, _ = await service.load_relevant_memories("kisaki", _scope(), "我保研准备得怎么样？")
    assert selected == ()
