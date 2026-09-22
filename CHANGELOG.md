# Changelog

All notable user-visible changes to this project are documented in this file.
The project follows [Semantic Versioning](https://semver.org/) for published
interfaces.

## Unreleased

### Added

- Read-only release file audit in CI and local verification, including syntax,
  private runtime artifacts and exact-duplicate inventory.

- Durable AstrBot generation receipts and authenticated delivery acknowledgements
  (database migration `008_integration_receipts`). Update backend and gateway together.

- Multi-scale Character RAG as the supported character-domain knowledge
  retrieval feature.
- `retrievalMode` in knowledge-search responses with the stable values
  `evidence`, `hybrid`, `keyword` and `empty`.
- A production character-index builder, deployment documentation and read-only
  Compose index mounts.

### Changed

- Exclude local caches, test runtimes, environment files and database sidecars
  from container builds; keep personal comparison notes out of Git.
- Use the official npm registry with package-store integrity verification enabled.
- Verify Alembic using the selected Python interpreter and only clean the local
  verification run's own temporary directory.
- Clarify that third-party game text and derivative datasets require separate
  redistribution permission; the code license does not cover them automatically.

- Align gateway HTTP waiting and backend request budgets; failed requests can retry,
  and undelivered generated replies can be reused without another inference call.
- Exclude unacknowledged platform replies from character history and defer character
  state updates until delivery acknowledgement.
- Share retrieval/abstention policy across vLLM and fallback model transports. Low
  confidence and retrieval failures produce character-voiced uncertainty; failed
  uncertainty generation uses a fixed fallback while retaining failure metrics.

- Production retrieval symbols, files, logs and documentation now use stable
  functional names instead of experiment phase names.
- The character index format identifier is `character-knowledge-v3`; legacy
  promoted-index identifiers are normalized in memory without rewriting
  local index artifacts.
- Shared query, recall, fusion and reranking primitives live in
  `knowledge.retrieval_core` and are not exposed as an independent service.

### Fixed

- Preserve canonical LoRA configuration keys and fractional epochs in training
  tasks; prefer named JSONL training splits and reject ambiguous dataset folders.
- Connect cancellation to the training loop, avoid final export after observed
  cancellation, and block same-adapter restarts until the old worker exits.
- Retain the newest completed training tasks during in-memory cleanup.
- Allow relation queries to fall back when a dedicated relation index is absent.

- Correct the legacy file-send directory reference and the training configuration
  type annotation; remove a redundant legacy bot import.
- Update the authorization contract for the delivery acknowledgement endpoint,
  with rejection tests for missing tokens, invalid tokens and missing signatures.

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
