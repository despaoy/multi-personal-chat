"""轻量角色上下文模块（第一步）。

统一表示人物信息（人物画像 + 当前关系 + 当前情景 + 相关记忆 + 本轮行为决策），
并安全、高效地整理成模型输入。暂时不改数据库、不调用额外 LLM、
不修改现有生成链路。
"""

from character.context_builder import (
    MAX_DECISION_FIELD_CHARS,
    MAX_MEMORY_ITEMS,
    MAX_MEMORY_TOTAL_CHARS,
    MAX_PROFILE_ITEM_CHARS,
    MAX_PROFILE_ITEMS_PER_CATEGORY,
    MAX_RELATIONSHIP_SUMMARY_CHARS,
    MAX_SINGLE_MEMORY_CHARS,
    MAX_SITUATION_FIELD_CHARS,
    MEMORY_REFERENCE_DISCLAIMER,
    build_user_scope,
    compile_character_context,
    compile_dynamic_context,
    compile_profile_context,
    compile_reference_context,
)
from character.profile_registry import (
    CharacterProfileNotFoundError,
    CharacterProfileRegistry,
    get_default_profile_registry,
)
from character.models import (
    CharacterContext,
    CharacterProfile,
    CompiledCharacterContext,
    DecisionPlan,
    MemoryItem,
    MemoryType,
    RelationshipStage,
    RelationshipState,
    SituationState,
    UserScope,
)

__all__ = [
    "MAX_DECISION_FIELD_CHARS",
    "MAX_MEMORY_ITEMS",
    "MAX_MEMORY_TOTAL_CHARS",
    "MAX_PROFILE_ITEM_CHARS",
    "MAX_PROFILE_ITEMS_PER_CATEGORY",
    "MAX_RELATIONSHIP_SUMMARY_CHARS",
    "MAX_SINGLE_MEMORY_CHARS",
    "MAX_SITUATION_FIELD_CHARS",
    "MEMORY_REFERENCE_DISCLAIMER",
    "CharacterProfileNotFoundError",
    "CharacterProfileRegistry",
    "get_default_profile_registry",
    "CharacterContext",
    "CharacterProfile",
    "CompiledCharacterContext",
    "DecisionPlan",
    "MemoryItem",
    "MemoryType",
    "RelationshipStage",
    "RelationshipState",
    "SituationState",
    "UserScope",
    "build_user_scope",
    "compile_character_context",
    "compile_dynamic_context",
    "compile_profile_context",
    "compile_reference_context",
]
