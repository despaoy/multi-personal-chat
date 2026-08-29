# Changelog

All notable user-visible changes to this project are documented in this file.
The project follows [Semantic Versioning](https://semver.org/) for published
interfaces.

## Unreleased

### Added

- Multi-scale Character RAG as the supported character-domain knowledge
  retrieval feature.
- `retrievalMode` in knowledge-search responses with the stable values
  `evidence`, `hybrid`, `keyword` and `empty`.
- A production character-index builder, deployment documentation and read-only
  Compose index mounts.

### Changed

- Production retrieval symbols, files, logs and documentation now use stable
  functional names instead of experiment phase names.
- The character index format identifier is `character-knowledge-v3`; legacy
  promoted-index identifiers are normalized in memory without rewriting
  local index artifacts.
- Shared query, recall, fusion and reranking primitives live in
  `knowledge.retrieval_core` and are not exposed as an independent service.

### Deprecated

- `searchType` in knowledge-search responses. Existing values remain available
  for compatibility; clients should migrate to `retrievalMode` before the next
  API major version.
- `MULTISCALE_RAG_INDEX_ROOT` and `MULTISCALE_RAG_ABSTAIN_THRESHOLD`; use the
  corresponding `CHARACTER_RAG_*` variables.

### Archived

- The P6 orchestration, tests, scripts and design document are retained under
  `archive/p6_rag_pipeline/` for source-level audit and are not imported by
  production code. Generated vectors and raw comparison reports remain local.
