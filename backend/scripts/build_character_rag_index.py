"""Build the production multi-scale character knowledge index."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from knowledge.multiscale_rag.constants import (  # noqa: E402
    DEFAULT_INDEX_DIRECTORY,
    INDEX_FORMAT_VERSION,
)
from knowledge.multiscale_rag.index_builder import CharacterKnowledgeIndexBuilder  # noqa: E402
from knowledge.multiscale_rag.vector_runtime import (  # noqa: E402
    LocalMeanPoolingEmbeddingProvider,
    write_vector_artifacts,
)
from knowledge.retrieval_core.registry import get_default_registry  # noqa: E402

DEFAULT_OUTPUT = BACKEND_ROOT / "data" / "knowledge" / "tsukiyashiro_kisaki" / DEFAULT_INDEX_DIRECTORY
DEFAULT_SCENES = (
    BACKEND_ROOT / "data" / "knowledge" / "tsukiyashiro_kisaki" / "scene_metadata_enriched" / "enriched_scenes.jsonl"
)


def _write_partition(root: Path, name: str, documents: list, matrix, allowed: set[str], provider) -> None:
    rows = [row for row, document in enumerate(documents) if document.document_type in allowed]
    selected = [documents[row] for row in rows]
    vectors = matrix[rows]
    write_vector_artifacts(
        root / name,
        selected,
        vectors,
        {
            "version": INDEX_FORMAT_VERSION,
            "route_types": sorted(allowed),
            "document_count": len(selected),
            "vector_dimension": int(vectors.shape[1]),
            "embedding_model": provider.model_id,
            "normalized": True,
            "physical_scale_partition": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="构建生产多粒度角色知识索引")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scenes", default=str(DEFAULT_SCENES))
    parser.add_argument("--embedding-model-path", default="")
    parser.add_argument("--force", action="store_true", help="允许覆盖现有版本化产物")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    knowledge_root = (BACKEND_ROOT / "data" / "knowledge").resolve()
    if knowledge_root not in output.parents:
        raise ValueError(f"输出目录必须位于知识数据目录内: {output}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"索引已存在；确认重建时显式传入 --force: {output}")

    base = get_default_registry().require("tsukiyashiro_kisaki")
    provider = LocalMeanPoolingEmbeddingProvider(
        model_path=args.embedding_model_path or os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None
    )
    build = CharacterKnowledgeIndexBuilder(
        domain_id=base.domain_id,
        index_version=INDEX_FORMAT_VERSION,
        aliases=base.aliases,
        corpus_root=REPO_ROOT,
    ).build(base.source_root, Path(args.scenes))
    documents = list(build.documents)
    vectors = provider.embed_texts([document.embedding_text for document in documents])

    output.mkdir(parents=True, exist_ok=True)
    _write_partition(output, "card_index", documents, vectors, {"fact", "relation", "event"}, provider)
    _write_partition(output, "scene_story_index", documents, vectors, {"scene", "story"}, provider)
    _write_partition(output, "evidence_index", documents, vectors, {"evidence"}, provider)
    print(
        f"built character knowledge index: documents={len(documents)} counts={build.counts} "
        f"exact_evidence={build.exact_evidence_matches} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
