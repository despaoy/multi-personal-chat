# P6 retrieval archive

This directory preserves the retired P6 retrieval implementation for research
reproducibility. It is not a supported runtime package and must not be added to
`PYTHONPATH` or imported by application code.

## Contents

| Path | Description |
| --- | --- |
| `knowledge/` | Retired orchestration, service facade and context builder |
| `scripts/` | Former index builder and retrieval evaluator |
| `tests/` | Tests for the retired behavior |
| `docs/` | Historical P6 design document |
| `multiscale_preselection/` | Superseded V1/V2/V3 experiment implementations |
| `MANIFEST.json` | Original locations, byte sizes and SHA-256 checksums |

Generated vectors, indexed documents and raw evaluation reports are intentionally
not versioned. They can contain bulky, environment-specific research output and
must be rebuilt from approved source data when an old experiment is reproduced.

## Supported replacement

The supported production feature is Multi-scale Character RAG:

- runtime: `backend/knowledge/multiscale_rag/runtime.py`;
- index builder: `backend/scripts/build_character_rag_index.py`;
- architecture: `docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md`.

Shared retrieval primitives were extracted to `backend/knowledge/retrieval_core/`.
That package contains reusable query, recall, fusion and reranking components;
it is not the archived P6 orchestration under a new name.

## Integrity and use

Files listed in `MANIFEST.json` are immutable historical records. To reproduce
an old result, use an isolated environment and record the archived manifest,
dataset and model versions. Do not use archived scripts to build a production
index, and do not compare their outputs with current results without preserving
the original evaluation conditions.
