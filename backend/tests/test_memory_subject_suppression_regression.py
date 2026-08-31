"""语义检索不得绕过用户/角色主体抑制。"""

from datetime import datetime, timezone

import numpy as np

from character.memory_service import CharacterMemoryService
from character.models import UserScope


class _Repo:
    async def list_memory_records(self, character_id, user_scope, limit=100):
        return [
            {
                "id": 1,
                "memory_key": "user_name",
                "memory_type": "user_fact",
                "content": "用户说自己叫小明",
                "importance": 0.9,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": 2,
                "memory_key": "preference_咖啡",
                "memory_type": "user_fact",
                "content": "用户说喜欢咖啡",
                "importance": 0.8,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        ]


class _AllSimilar:
    model_id = "all-similar"
    dimension = 2

    def embed_texts(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_query(self, query):
        return np.asarray([1.0, 0.0], dtype=np.float32)


async def test_hybrid_subject_suppression_and_user_recall():
    service = CharacterMemoryService(_Repo(), embedding_provider=_AllSimilar(), semantic_enabled=True)
    scope = UserScope("qq", "astrbot", "u1", "u1", "private")

    names, _ = await service.load_relevant_memories("kisaki", scope, "我叫什么名字？")
    character_names, _ = await service.load_relevant_memories("kisaki", scope, "你叫什么名字？")
    preferences, _ = await service.load_relevant_memories("kisaki", scope, "我喜欢什么？")
    character_preferences, _ = await service.load_relevant_memories("kisaki", scope, "你喜欢什么？")

    assert any(item.content == "用户说自己叫小明" for item in names)
    assert all(item.content != "用户说自己叫小明" for item in character_names)
    assert any(item.content == "用户说喜欢咖啡" for item in preferences)
    assert all(item.content != "用户说喜欢咖啡" for item in character_preferences)
