"""Stable names and compatibility identifiers for character knowledge retrieval."""

INDEX_FORMAT_VERSION = "character-knowledge-v3"
EMBEDDING_TEXT_VERSION = "character-knowledge-text-v1"
LEGACY_INDEX_FORMAT_VERSIONS = frozenset(
    {
        "multiscale-routed-v3-selected",
        "multiscale-semantic-v2",
    }
)
LEGACY_EMBEDDING_TEXT_VERSIONS = frozenset({"multiscale_semantic_v2_1"})
DEFAULT_INDEX_DIRECTORY = "character_knowledge_index_v3"
